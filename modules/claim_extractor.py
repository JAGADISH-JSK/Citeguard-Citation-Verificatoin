"""
Layer 2b – Claim Extractor
Extracts the specific claim sentence(s) that a citation is backing.

Strategy
--------
1. For each InlineCitation, look at its `context` window.
2. Use sentence tokenisation to isolate the exact sentence(s) containing
   the citation marker.
3. Strip the citation marker itself, leaving a clean claim string.
4. Return a list of ClaimCitationPair objects ready for the aligner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from loguru import logger

from modules.citation_parser import CitationMap, InlineCitation

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ClaimCitationPair:
    citation_key: str           # e.g. "3" or "Smith2020"
    raw_citation: str           # original inline marker, e.g. "[3]"
    claim: str                  # clean sentence making the claim
    context: str                # wider context window (for display)


# ── Sentence splitter ─────────────────────────────────────────────────────────

# Simple sentence boundary regex (no heavy NLP dependency required).
# Splits on ". ", "! ", "? " preceded by a non-digit (to avoid "Fig. 3").
_SENT_SPLIT = re.compile(r'(?<=[^0-9A-Z])[.!?]\s+(?=[A-Z])')

# All common citation marker patterns
_CITE_STRIP = re.compile(
    r'\[[\d,\s\-]+\]'                   # [1], [1,2], [1-3]
    r'|\([A-Z][a-zÀ-ÿ]+[^)]*\d{4}[a-z]?\)'   # (Smith et al., 2020)
    r'|\d+',                             # bare superscript numbers
    re.MULTILINE,
)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_claims(citation_map: CitationMap, full_text: str) -> List[ClaimCitationPair]:
    """
    For every inline citation, extract the claim sentence(s) it supports.
    """
    pairs: List[ClaimCitationPair] = []

    for inline in citation_map.inline:
        claim = _extract_claim_sentence(inline, full_text)
        if not claim:
            logger.debug(f"No claim sentence found for citation {inline.raw}")
            continue

        for key in inline.keys:
            pairs.append(ClaimCitationPair(
                citation_key=key,
                raw_citation=inline.raw,
                claim=claim,
                context=inline.context,
            ))

    logger.info(f"Extracted {len(pairs)} claim-citation pairs")
    return pairs


# ── Internals ─────────────────────────────────────────────────────────────────

def _extract_claim_sentence(inline: InlineCitation, full_text: str) -> str:
    """
    Isolate the sentence that contains the citation marker.

    Algorithm
    ---------
    1. Take the context window stored in InlineCitation.
    2. Split into sentences.
    3. Find the sentence(s) that contain the raw citation marker.
    4. Strip the marker and clean up whitespace.
    """
    context = inline.context
    sentences = _split_sentences(context)

    for sent in sentences:
        if inline.raw in sent or _citation_in_sent(inline.raw, sent):
            clean = _clean_claim(sent)
            if len(clean.split()) >= 4:   # must be a real sentence
                return clean

    # Fallback: strip marker from the whole context window
    return _clean_claim(context)[:400]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using the regex splitter."""
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _citation_in_sent(raw: str, sentence: str) -> bool:
    """Check if any citation-like token from raw appears in sentence."""
    # Extract digits from raw citation
    nums = re.findall(r"\d+", raw)
    for n in nums:
        if re.search(rf"\[{n}[,\-\]]|\[[\d,\s]*{n}[\d,\s]*\]", sentence):
            return True
    return False


def _clean_claim(text: str) -> str:
    """Remove citation markers and normalise whitespace."""
    cleaned = _CITE_STRIP.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = cleaned.strip(".,;: ")
    return cleaned
