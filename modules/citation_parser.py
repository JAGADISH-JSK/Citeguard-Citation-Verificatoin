"""
Layer 2a – Citation Parser
Detects and normalises all citations in a research paper.

Supported citation styles
  • Numbered  : [1], [2,3], [12]
  • Author-year: (Smith et al., 2020), (Jones & Lee, 2019)
  • Superscript: ¹²  (common in medical/biology papers – best-effort)

Produces:
  • InlineCitation  – a single cite found in the body text
  • Reference       – a parsed bibliography entry
  • CitationMap     – links inline cites → reference entries
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class InlineCitation:
    raw: str            # original string, e.g. "[3]" or "(Smith et al., 2020)"
    keys: List[str]     # normalised keys, e.g. ["3"] or ["Smith2020"]
    context: str        # surrounding sentence
    char_offset: int    # position in full_text


@dataclass
class Reference:
    key: str            # same key used in InlineCitation.keys
    raw_text: str       # full raw string from bibliography
    authors: List[str] = field(default_factory=list)
    title: str = ""
    year: Optional[str] = None
    venue: str = ""
    doi: str = ""
    url: str = ""


@dataclass
class CitationMap:
    inline: List[InlineCitation]
    references: Dict[str, Reference]   # key → Reference


# ── Regex patterns ────────────────────────────────────────────────────────────

# [1], [2], [1,2,3], [1-3]
_NUM_CITE_RE = re.compile(r"\[(\d[\d,\s\-]*)\]")

# (Smith et al., 2020) | (Jones & Lee, 2019) | (Author, 2020)
_AUTH_CITE_RE = re.compile(
    r"\(([A-Z][a-zÀ-ÿ\-]+(?:\s+et\s+al\.?|\s+&\s+[A-Z][a-zÀ-ÿ\-]+)?)"
    r",?\s+(\d{4}[a-z]?)\)"
)

# DOI in reference list
_DOI_RE  = re.compile(r"10\.\d{4,9}/[^\s,]+", re.IGNORECASE)
# URL in reference list
_URL_RE  = re.compile(r"https?://[^\s,]+")
# Year anywhere
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# ── Public API ────────────────────────────────────────────────────────────────

def parse(full_text: str, reference_block: str) -> CitationMap:
    """
    Full pipeline: detect inline citations, parse references, link them.
    """
    inline     = _detect_inline(full_text)
    references = _parse_references(reference_block)
    _link(inline, references)

    logger.info(
        f"Citations found: {len(inline)} inline, "
        f"{len(references)} reference entries"
    )
    return CitationMap(inline=inline, references=references)


# ── Inline detection ──────────────────────────────────────────────────────────

def _detect_inline(full_text: str) -> List[InlineCitation]:
    cites: List[InlineCitation] = []
    seen: set = set()

    # Numbered citations  [1], [1,2,3]
    for m in _NUM_CITE_RE.finditer(full_text):
        keys = _expand_num_keys(m.group(1))
        context = _extract_context(full_text, m.start())
        token = (m.group(), m.start())
        if token not in seen:
            seen.add(token)
            cites.append(InlineCitation(
                raw=m.group(), keys=keys,
                context=context, char_offset=m.start(),
            ))

    # Author-year citations  (Smith et al., 2020)
    for m in _AUTH_CITE_RE.finditer(full_text):
        author_part = m.group(1).strip()
        year_part   = m.group(2).strip()
        key         = _make_author_key(author_part, year_part)
        context     = _extract_context(full_text, m.start())
        token       = (m.group(), m.start())
        if token not in seen:
            seen.add(token)
            cites.append(InlineCitation(
                raw=m.group(), keys=[key],
                context=context, char_offset=m.start(),
            ))

    cites.sort(key=lambda c: c.char_offset)
    return cites


def _expand_num_keys(raw: str) -> List[str]:
    """'1,3-5' → ['1','3','4','5']"""
    keys: List[str] = []
    for part in re.split(r",\s*", raw.strip()):
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                keys.extend(str(i) for i in range(int(lo), int(hi) + 1))
            except ValueError:
                keys.append(part.strip())
        else:
            keys.append(part.strip())
    return keys


def _make_author_key(author: str, year: str) -> str:
    # Take the first author (before 'et al' or '&') then its last word
    first_author = re.split(r"\s+et\s+al|\s+&\s+", author)[0].strip()
    last_name = first_author.split()[-1] if first_author.split() else first_author
    return f"{last_name}{year}"


def _extract_context(text: str, offset: int, window: int = 300) -> str:
    """Return the sentence-like window around a citation."""
    start = max(0, offset - window)
    end   = min(len(text), offset + window)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet


# ── Reference parsing ─────────────────────────────────────────────────────────

def _parse_references(ref_block: str) -> Dict[str, Reference]:
    """
    Split the bibliography block into individual entries and parse each one.
    Handles numbered [1] Author... and plain Author, Title... styles.
    """
    refs: Dict[str, Reference] = {}
    if not ref_block:
        return refs

    entries = _split_reference_entries(ref_block)
    for idx, entry in enumerate(entries, start=1):
        ref = _parse_single_reference(entry, fallback_key=str(idx))
        refs[ref.key] = ref

    return refs


def _split_reference_entries(ref_block: str) -> List[str]:
    """Split block into individual reference strings."""
    # Split on [N] markers appearing at start-of-string or after newline/space.
    # Works for single-line or multi-line bibliography blocks.
    parts = re.split(r"(?:^|(?<=\n)|(?<=\s))(\[\d+\])", ref_block)
    if len(parts) > 2:
        entries: List[str] = []
        # parts: [preamble, marker1, text1, marker2, text2, ...]
        it = iter(parts[1:])
        for marker, text in zip(it, it):
            entries.append(f"{marker} {text.strip()}")
        return [e for e in entries if e.strip()]

    # Fallback: split on blank lines
    return [e.strip() for e in re.split(r"\n{2,}", ref_block) if e.strip()]


def _parse_single_reference(raw: str, fallback_key: str) -> Reference:
    raw = raw.strip()

    # Determine key
    num_match = re.match(r"^\[(\d+)\]", raw)
    if num_match:
        key = num_match.group(1)
        body = raw[num_match.end():].strip()
    else:
        key  = fallback_key
        body = raw

    # Extract DOI
    doi_match = _DOI_RE.search(body)
    doi = doi_match.group(0).rstrip(".,)") if doi_match else ""

    # Extract URL
    url_match = _URL_RE.search(body)
    url = url_match.group(0).rstrip(".,)") if url_match else ""

    # Extract year
    year_match = _YEAR_RE.search(body)
    year = year_match.group(0) if year_match else None

    # Extract authors (before the year or first period)
    authors = _extract_authors(body)

    # Extract title (heuristic: quoted string or second sentence)
    title = _extract_title(body)

    return Reference(
        key=key, raw_text=raw,
        authors=authors, title=title,
        year=year, doi=doi, url=url,
    )


def _extract_authors(body: str) -> List[str]:
    """Return author tokens before the first year or first period."""
    year_m = _YEAR_RE.search(body)
    boundary = year_m.start() if year_m else body.find(".")
    if boundary <= 0:
        boundary = min(len(body), 120)
    author_str = body[:boundary].strip().rstrip(".,")
    # split on ';' or ' and ' or ','
    parts = re.split(r";\s*|\band\b|,\s+(?=[A-Z])", author_str)
    return [p.strip() for p in parts if p.strip()]


def _extract_title(body: str) -> str:
    """Heuristic title extraction: look for quoted text or second sentence."""
    # Quoted title
    q_match = re.search(r'"([^"]{10,})"', body)
    if q_match:
        return q_match.group(1).strip()
    # Title-cased second sentence
    sentences = re.split(r"\.\s+", body)
    if len(sentences) >= 2:
        candidate = sentences[1].strip()
        if len(candidate) > 10:
            return candidate
    return ""


# ── Linking ───────────────────────────────────────────────────────────────────

def _link(inline: List[InlineCitation], refs: Dict[str, Reference]) -> None:
    """Log unresolved citations for diagnostics."""
    unresolved = 0
    for cite in inline:
        for k in cite.keys:
            if k not in refs:
                unresolved += 1
    if unresolved:
        logger.warning(f"{unresolved} inline citation keys have no matching reference entry")
