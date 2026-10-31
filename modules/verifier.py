"""
Layer 6 – NLI Fact Verifier
============================
Takes a (claim, evidence_passage) pair and classifies:
  • SUPPORTS      – the passage entails the claim
  • NEUTRAL       – the passage neither confirms nor denies
  • CONTRADICTS   – the passage contradicts the claim

Uses DeBERTa-v3-large NLI model via Hugging Face transformers.
Falls back to a cosine-similarity-only heuristic if the model cannot be loaded
(useful for environments with limited RAM/GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple

from loguru import logger

from config import NLI_MODEL, NLI_THRESHOLD
from modules.aligner import EvidencePassage

# ── Labels ────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    SUPPORTS     = "SUPPORTS"
    NEUTRAL      = "NEUTRAL"
    CONTRADICTS  = "CONTRADICTS"
    NO_EVIDENCE  = "NO_EVIDENCE"   # cited paper not found / empty abstract


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    verdict: Verdict
    confidence: float                   # 0.0 – 1.0
    best_evidence: str                  # top evidence passage text
    entail_score: float   = 0.0
    neutral_score: float  = 0.0
    contra_score: float   = 0.0
    alignment_score: float = 0.0        # composite final score


# ── Model loader ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nli_pipeline():
    """Lazy-load the HuggingFace zero-shot / NLI pipeline."""
    try:
        from transformers import pipeline  # noqa: PLC0415
        logger.info(f"Loading NLI model: {NLI_MODEL}")
        # cross-encoder/nli-deberta-v3-large works as a direct NLI pipeline
        pipe = pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,          # CPU; set to 0 for GPU
        )
        return pipe
    except Exception as exc:
        logger.warning(f"NLI model load failed ({exc}). Falling back to cosine heuristic.")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def verify(
    claim: str,
    evidence_passages: List[EvidencePassage],
) -> VerificationResult:
    """
    Verify a single claim against ranked evidence passages.
    Uses the top-ranked passage for NLI inference.
    """
    if not evidence_passages:
        return VerificationResult(
            verdict=Verdict.NO_EVIDENCE,
            confidence=0.0,
            best_evidence="",
            alignment_score=0.0,
        )

    best = evidence_passages[0]   # highest cross-encoder score

    pipe = _load_nli_pipeline()
    if pipe is not None:
        return _verify_with_nli(claim, best, pipe)
    else:
        return _verify_with_cosine(claim, best)


# ── NLI-based verification ────────────────────────────────────────────────────

def _verify_with_nli(
    claim: str,
    evidence: EvidencePassage,
    pipe,
) -> VerificationResult:
    """
    Run zero-shot NLI classification.
    Label set: ["entailment", "neutral", "contradiction"]
    """
    try:
        result = pipe(
            evidence.text,
            candidate_labels=["entailment", "neutral", "contradiction"],
            hypothesis_template="{}",
            multi_label=False,
        )
        label_scores = dict(zip(result["labels"], result["scores"]))
        e = label_scores.get("entailment", 0.0)
        n = label_scores.get("neutral", 0.0)
        c = label_scores.get("contradiction", 0.0)
    except Exception as exc:
        logger.error(f"NLI inference failed: {exc}")
        return _verify_with_cosine(claim, evidence)

    verdict, confidence = _decide_verdict(e, n, c)
    alignment = _compute_alignment(verdict, confidence, evidence.bi_score)

    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        best_evidence=evidence.text,
        entail_score=e,
        neutral_score=n,
        contra_score=c,
        alignment_score=alignment,
    )


# ── Cosine-only fallback ──────────────────────────────────────────────────────

def _verify_with_cosine(
    claim: str,
    evidence: EvidencePassage,
) -> VerificationResult:
    """
    Simple heuristic when NLI model is unavailable.
    Uses bi-encoder cosine score to classify.
    """
    score = evidence.bi_score
    if score >= 0.65:
        verdict, confidence = Verdict.SUPPORTS, score
    elif score >= 0.35:
        verdict, confidence = Verdict.NEUTRAL, score
    else:
        verdict, confidence = Verdict.CONTRADICTS, 1.0 - score

    alignment = _compute_alignment(verdict, confidence, score)
    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        best_evidence=evidence.text,
        alignment_score=alignment,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decide_verdict(
    entail: float, neutral: float, contra: float
) -> Tuple[Verdict, float]:
    """Pick verdict from NLI label probabilities."""
    if entail >= NLI_THRESHOLD and entail >= neutral and entail >= contra:
        return Verdict.SUPPORTS, entail
    if contra >= NLI_THRESHOLD and contra >= neutral and contra >= entail:
        return Verdict.CONTRADICTS, contra
    return Verdict.NEUTRAL, neutral


def _compute_alignment(
    verdict: Verdict, nli_conf: float, bi_score: float
) -> float:
    """
    Composite alignment score ∈ [0, 1].
    Higher = stronger evidence alignment.
    """
    if verdict == Verdict.NO_EVIDENCE:
        return 0.0
    relevance_weight = 0.4
    nli_weight       = 0.6
    base = relevance_weight * bi_score + nli_weight * nli_conf
    if verdict == Verdict.CONTRADICTS:
        base *= 0.2   # heavily penalise contradictions
    elif verdict == Verdict.NEUTRAL:
        base *= 0.5
    return round(min(1.0, base), 4)
