"""
RAG Store – Persistent Vector Index
=====================================
Manages a ChromaDB collection that stores chunked paper passages as
dense embeddings so that:
  1. Each cited paper is indexed ONCE and reused across runs (cache/rag_index/).
  2. Evidence retrieval queries the persistent store instead of re-embedding
     everything in memory every time.

Public API
----------
  index_paper(paper_id, chunks, embeddings)  → store passages for a paper
  query(paper_id, query_embedding, top_k)    → retrieve top-K passages
  paper_indexed(paper_id)                    → check if already stored
  clear_paper(paper_id)                      → remove a paper's chunks
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from config import CACHE_DIR, TOP_K_RETRIEVAL

# ── ChromaDB collection ───────────────────────────────────────────────────────

_RAG_DIR = CACHE_DIR / "rag_index"
_COLLECTION_NAME = "citeguard_passages"

_client = None
_collection = None


def _get_collection():
    """Lazy-init ChromaDB persistent client + collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        _RAG_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(_RAG_DIR))
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"RAG store: ChromaDB collection '{_COLLECTION_NAME}' "
                    f"({_collection.count()} passages)")
    except Exception as exc:
        logger.warning(f"ChromaDB unavailable ({exc}). RAG store disabled.")
        _collection = None

    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def paper_indexed(paper_id: str) -> bool:
    """Return True if this paper already has chunks in the store."""
    col = _get_collection()
    if col is None:
        return False
    try:
        results = col.get(where={"paper_id": paper_id}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False


def index_paper(
    paper_id: str,
    chunks: List[str],
    embeddings: np.ndarray,
) -> int:
    """
    Store passages for one paper.
    Returns the number of passages stored (0 if already indexed or error).
    """
    col = _get_collection()
    if col is None:
        return 0

    if paper_indexed(paper_id):
        logger.debug(f"RAG store: [{paper_id}] already indexed, skipping.")
        return 0

    ids        = [_chunk_id(paper_id, i) for i in range(len(chunks))]
    metadatas  = [{"paper_id": paper_id, "chunk_idx": i} for i in range(len(chunks))]
    embeddings_list = embeddings.tolist()

    try:
        col.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )
        logger.info(f"RAG store: indexed {len(chunks)} chunks for [{paper_id}]")
        return len(chunks)
    except Exception as exc:
        logger.warning(f"RAG store index error for [{paper_id}]: {exc}")
        return 0


def query(
    paper_id: str,
    query_embedding: np.ndarray,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-K passages for a given paper.
    Returns list of (passage_text, cosine_similarity_score).
    Falls back to empty list if store unavailable.
    """
    col = _get_collection()
    if col is None:
        return []

    try:
        results = col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, _count_paper_chunks(col, paper_id)),
            where={"paper_id": paper_id},
            include=["documents", "distances"],
        )
        docs      = results["documents"][0]
        distances = results["distances"][0]   # cosine distance (0=identical)
        # Convert cosine distance → similarity score: similarity = 1 - distance
        return [(doc, float(1.0 - dist)) for doc, dist in zip(docs, distances)]
    except Exception as exc:
        logger.warning(f"RAG store query error for [{paper_id}]: {exc}")
        return []


def clear_paper(paper_id: str) -> None:
    """Remove all chunks for a paper (e.g. when re-indexing updated text)."""
    col = _get_collection()
    if col is None:
        return
    try:
        col.delete(where={"paper_id": paper_id})
        logger.debug(f"RAG store: cleared chunks for [{paper_id}]")
    except Exception as exc:
        logger.warning(f"RAG store clear error for [{paper_id}]: {exc}")


def store_stats() -> dict:
    """Return basic statistics about the current store."""
    col = _get_collection()
    if col is None:
        return {"available": False, "total_passages": 0}
    return {"available": True, "total_passages": col.count()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id(paper_id: str, idx: int) -> str:
    """Deterministic ID for a chunk so re-indexing is idempotent."""
    raw = f"{paper_id}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def _count_paper_chunks(col, paper_id: str) -> int:
    """Count how many chunks this paper has in the store."""
    try:
        res = col.get(where={"paper_id": paper_id}, include=[])
        return max(1, len(res["ids"]))
    except Exception:
        return TOP_K_RETRIEVAL
