"""
Layer 4 + 5 – Evidence Retrieval & Re-ranking  (RAG-upgraded)
==============================================================
Full RAG pipeline:

  INDEXING  → chunk paper text → embed → store in ChromaDB (rag_store)
  RETRIEVAL → embed claim → ANN query ChromaDB → top-K candidates
  RE-RANK   → cross-encoder scores each (claim, passage) pair → top-K_rerank

Persistent vector store means each paper is embedded ONCE; subsequent runs
query the store directly — no re-embedding needed.

Falls back gracefully to in-memory cosine if ChromaDB is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from loguru import logger

from config import (
    EMBEDDING_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK, CHUNK_SENTENCES,
)

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class EvidencePassage:
    text: str
    bi_score: float     # cosine similarity from bi-encoder
    cross_score: float  # score from cross-encoder re-ranker
    rank: int           # final rank (1 = best)


# ── Lazy model loading (pure transformers, no sklearn) ────────────────────────

@lru_cache(maxsize=1)
def _get_tokenizer_model():
    """Load tokenizer + model for mean-pooled bi-encoder embeddings."""
    from transformers import AutoTokenizer, AutoModel
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _get_cross_encoder_pipeline():
    """Load a cross-encoder via transformers text-classification pipeline."""
    from transformers import pipeline
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logger.info(f"Loading cross-encoder: {model_name}")
    try:
        pipe = pipeline("text-classification", model=model_name, device=-1)
        return pipe
    except Exception as exc:
        logger.warning(f"Cross-encoder load failed ({exc}), using bi-score only.")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_evidence(
    claim: str,
    paper_text: str,
    paper_id: str = "",
) -> List[EvidencePassage]:
    """
    Full RAG two-stage retrieval for a single claim against a paper's text.

    Parameters
    ----------
    claim      : The sentence making a claim (the query).
    paper_text : Full paper body (or abstract if full text unavailable).
    paper_id   : Stable identifier used as ChromaDB document key.
                 If empty, falls back to in-memory retrieval.

    Returns passages ranked from most to least relevant.
    """
    if not paper_text.strip():
        return []

    passages = _chunk_text(paper_text)
    if not passages:
        return []

    # ── Stage 1: Bi-encoder retrieval ─────────────────────────────────────────
    # Try RAG store first (persistent ChromaDB); fall back to in-memory cosine.
    if paper_id:
        top_passages = _rag_retrieve(claim, passages, paper_id)
    else:
        top_passages = _bi_encode_retrieve(claim, passages)

    # ── Stage 2: Cross-encoder re-ranking ─────────────────────────────────────
    ranked = _cross_encode_rerank(claim, top_passages)
    return ranked


# ── Chunking ──────────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _chunk_text(text: str, chunk_size: int = CHUNK_SENTENCES) -> List[str]:
    """
    Split text into overlapping sentence-window chunks.
    chunk_size=3 → windows of 3 sentences with 1-sentence overlap.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [text[:1000]] if text.strip() else []

    chunks: List[str] = []
    step = max(1, chunk_size - 1)          # 1-sentence overlap
    for i in range(0, len(sentences), step):
        window = sentences[i: i + chunk_size]
        chunks.append(" ".join(window))
    return chunks


# ── Bi-encoder (mean-pooling, pure numpy cosine) ──────────────────────────────

def _mean_pool(token_embeddings: "np.ndarray", attention_mask: "np.ndarray") -> "np.ndarray":
    """Mean-pool token embeddings, ignoring padding tokens."""
    mask_expanded = attention_mask[:, :, None].astype(float)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _encode_texts(texts: List[str]) -> "np.ndarray":
    """Encode a list of texts → L2-normalised embeddings (N, D)."""
    import torch
    tokenizer, model = _get_tokenizer_model()

    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)

        token_embs  = output.last_hidden_state.numpy()
        attn_mask   = encoded["attention_mask"].numpy()
        embeddings  = _mean_pool(token_embs, attn_mask)

        # L2-normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


# ── RAG store retrieval (ChromaDB) ────────────────────────────────────────────

def _rag_retrieve(
    claim: str,
    passages: List[str],
    paper_id: str,
) -> List[Tuple[str, float]]:
    """
    Index paper chunks into ChromaDB on first call; query on subsequent calls.
    Returns top-K (passage_text, score) pairs.
    """
    from modules.rag_store import paper_indexed, index_paper, query as rag_query

    # Index once —————————————————————————————————————————————————————————————
    if not paper_indexed(paper_id):
        logger.debug(f"  [{paper_id}] indexing {len(passages)} chunks into RAG store…")
        try:
            all_texts  = passages
            embeddings = _encode_texts(all_texts)
            indexed = index_paper(paper_id, passages, embeddings)
            if indexed == 0 and not paper_indexed(paper_id):
                # ChromaDB failed silently; fall back to in-memory
                return _bi_encode_retrieve(claim, passages)
        except Exception as exc:
            logger.warning(f"RAG index error for [{paper_id}]: {exc}")
            return _bi_encode_retrieve(claim, passages)

    # Query ———————————————————————————————————————————————————————————————————
    try:
        claim_emb = _encode_texts([claim])[0]
        results   = rag_query(paper_id, claim_emb, top_k=TOP_K_RETRIEVAL)
        if results:
            logger.debug(f"  [{paper_id}] RAG store returned {len(results)} candidates")
            return results
    except Exception as exc:
        logger.warning(f"RAG query error for [{paper_id}]: {exc}")

    # Fallback to in-memory if store query returned nothing
    return _bi_encode_retrieve(claim, passages)


# ── In-memory bi-encoder fallback ─────────────────────────────────────────────

def _bi_encode_retrieve(
    claim: str, passages: List[str]
) -> List[Tuple[str, float]]:
    """Return top-K passages by cosine similarity (dot product of normalised vecs)."""
    try:
        all_texts  = [claim] + passages
        embeddings = _encode_texts(all_texts)

        claim_emb    = embeddings[0]        # (D,)
        passage_embs = embeddings[1:]       # (N, D)

        scores: np.ndarray = passage_embs @ claim_emb   # cosine (already normalised)

        k = min(TOP_K_RETRIEVAL, len(passages))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(passages[i], float(scores[i])) for i in top_indices]

    except Exception as exc:
        logger.warning(f"Bi-encoder failed ({exc}), returning all passages unranked.")
        return [(p, 0.5) for p in passages[:TOP_K_RETRIEVAL]]


# ── Cross-encoder re-ranking ──────────────────────────────────────────────────

def _cross_encode_rerank(
    claim: str, candidates: List[Tuple[str, float]]
) -> List[EvidencePassage]:
    """Re-rank candidates using cross-encoder; fall back to bi-score if unavailable."""
    if not candidates:
        return []

    pipe = _get_cross_encoder_pipeline()

    results: List[EvidencePassage] = []

    if pipe is not None:
        try:
            inputs = [f"{claim} [SEP] {text}" for text, _ in candidates]
            scores = pipe(inputs, batch_size=16)
            for (text, bi_score), score_dict in zip(candidates, scores):
                cs = score_dict.get("score", bi_score)
                results.append(EvidencePassage(
                    text=text, bi_score=bi_score, cross_score=float(cs), rank=0
                ))
        except Exception as exc:
            logger.warning(f"Cross-encoder inference failed ({exc}), using bi-scores.")
            pipe = None

    if pipe is None:
        for text, bi_score in candidates:
            results.append(EvidencePassage(
                text=text, bi_score=bi_score, cross_score=bi_score, rank=0
            ))

    # Sort by cross score descending, assign ranks
    results.sort(key=lambda e: e.cross_score, reverse=True)
    for rank, ev in enumerate(results[:TOP_K_RERANK], start=1):
        ev.rank = rank

    return results[:TOP_K_RERANK]
