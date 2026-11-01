"""
Layer 3 – Metadata Resolver
Given a parsed Reference, queries external APIs to:
  1. Confirm the paper exists (existence check)
  2. Fetch its abstract (needed for evidence alignment)
  3. Attempt to download full-text (arXiv HTML or S2 open-access PDF)
  4. Flag retracted or suspicious papers

API priority:
  Semantic Scholar  →  CrossRef  →  arXiv  →  unresolved

Full-text strategy (RAG upgrade)
  • arXiv papers: download the HTML abstract+body from ar5iv.org (no PDF parsing needed)
  • Semantic Scholar papers with openAccessPdf: download + extract text via PyMuPDF
  • Falls back gracefully to abstract-only if full text is unavailable
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json
import re

import httpx
from loguru import logger

from config import (
    SEMANTIC_SCHOLAR_API, CROSSREF_API, ARXIV_API,
    S2_API_KEY, CACHE_DIR, RAG_FULL_TEXT, RAG_MAX_CHARS,
)
from modules.citation_parser import Reference

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ResolvedCitation:
    key: str
    found: bool = False
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    abstract: str = ""
    doi: str = ""
    venue: str = ""
    citation_count: int = 0
    source: str = ""          # "semantic_scholar" | "crossref" | "arxiv"
    warning: str = ""         # e.g. "retracted", "not found"
    # RAG additions ──────────────────────────────────────────────────
    arxiv_id: str = ""        # e.g. "2005.14165"
    open_access_pdf: str = "" # URL if S2 reports an open-access PDF
    full_text: str = ""       # full paper body text (may be empty)


# ── Caching helpers ───────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", key)[:80]
    return CACHE_DIR / f"{safe}.json"


def _load_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_cache(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── HTTP client factory ───────────────────────────────────────────────────────

def _build_client() -> httpx.AsyncClient:
    headers = {"User-Agent": "CiteGuard/1.0 (citation-verification-tool)"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    return httpx.AsyncClient(headers=headers, timeout=15.0, follow_redirects=True)


# ── Main resolve function ─────────────────────────────────────────────────────

async def resolve_all(references: Dict[str, Reference]) -> Dict[str, ResolvedCitation]:
    """
    Resolve every reference concurrently.
    Returns a dict  key → ResolvedCitation.
    """
    async with _build_client() as client:
        tasks = {
            key: _resolve_one(client, ref)
            for key, ref in references.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    resolved: Dict[str, ResolvedCitation] = {}
    for key, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.error(f"Failed to resolve citation [{key}]: {result}")
            resolved[key] = ResolvedCitation(key=key, found=False, warning=str(result))
        else:
            resolved[key] = result
    return resolved


def resolve_all_sync(references: Dict[str, Reference]) -> Dict[str, ResolvedCitation]:
    """Synchronous wrapper around resolve_all (useful for scripts/tests)."""
    return asyncio.run(resolve_all(references))


# ── Single reference resolver ─────────────────────────────────────────────────

async def _resolve_one(client: httpx.AsyncClient, ref: Reference) -> ResolvedCitation:
    cache_key = ref.doi or ref.title or ref.key
    cached = _load_cache(cache_key)
    if cached:
        logger.debug(f"Cache hit for [{ref.key}]")
        return ResolvedCitation(**cached)

    rc = ResolvedCitation(key=ref.key)

    # 1 – Semantic Scholar (best coverage for CS/NLP papers)
    if not rc.found:
        rc = await _query_semantic_scholar(client, ref)

    # 2 – CrossRef (DOI-authoritative for published journals)
    if not rc.found and ref.doi:
        rc = await _query_crossref(client, ref)

    # 3 – arXiv (preprints)
    if not rc.found and ref.title:
        rc = await _query_arxiv(client, ref)

    if not rc.found:
        rc.warning = "not found in any database"
        logger.warning(f"Unresolved: [{ref.key}] \"{ref.title}\"")

    _save_cache(cache_key, rc.__dict__)
    return rc


# ── Semantic Scholar ──────────────────────────────────────────────────────────

async def _query_semantic_scholar(
    client: httpx.AsyncClient, ref: Reference
) -> ResolvedCitation:
    rc = ResolvedCitation(key=ref.key)
    fields = "title,abstract,authors,year,externalIds,venue,citationCount,openAccessPdf"

    # Try by DOI first
    if ref.doi:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{ref.doi}?fields={fields}"
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return _parse_s2_paper(resp.json(), ref.key)
        except httpx.RequestError:
            pass

    # Try by title search
    if ref.title:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
        params = {"query": ref.title, "fields": fields, "limit": 3}
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                if items:
                    return _parse_s2_paper(items[0], ref.key)
        except httpx.RequestError:
            pass

    return rc


def _parse_s2_paper(data: dict, key: str) -> ResolvedCitation:
    authors = [a.get("name", "") for a in data.get("authors", [])]
    ext_ids = data.get("externalIds", {}) or {}
    oa_pdf  = (data.get("openAccessPdf") or {}).get("url", "")
    arxiv_id = ext_ids.get("ArXiv", "")
    return ResolvedCitation(
        key=key,
        found=True,
        title=data.get("title", ""),
        authors=authors,
        year=str(data.get("year", "")) if data.get("year") else None,
        abstract=data.get("abstract", "") or "",
        doi=ext_ids.get("DOI", ""),
        venue=data.get("venue", "") or "",
        citation_count=data.get("citationCount", 0) or 0,
        source="semantic_scholar",
        arxiv_id=arxiv_id,
        open_access_pdf=oa_pdf,
    )


# ── CrossRef ──────────────────────────────────────────────────────────────────

async def _query_crossref(
    client: httpx.AsyncClient, ref: Reference
) -> ResolvedCitation:
    rc = ResolvedCitation(key=ref.key)
    url = f"{CROSSREF_API}/{ref.doi}"
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            msg  = resp.json().get("message", {})
            title_list = msg.get("title", [""])
            authors = [
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in msg.get("author", [])
            ]
            date_parts = msg.get("published-print", {}).get("date-parts", [[None]])
            year = str(date_parts[0][0]) if date_parts and date_parts[0][0] else None
            rc = ResolvedCitation(
                key=ref.key,
                found=True,
                title=title_list[0] if title_list else "",
                authors=authors,
                year=year,
                abstract=msg.get("abstract", "") or "",
                doi=ref.doi,
                venue=msg.get("container-title", [""])[0],
                source="crossref",
            )
    except httpx.RequestError as exc:
        logger.debug(f"CrossRef request error for [{ref.key}]: {exc}")
    return rc


# ── arXiv ─────────────────────────────────────────────────────────────────────

async def _query_arxiv(
    client: httpx.AsyncClient, ref: Reference
) -> ResolvedCitation:
    rc = ResolvedCitation(key=ref.key)
    params = {
        "search_query": f"ti:{ref.title}",
        "max_results": 3,
        "sortBy": "relevance",
    }
    try:
        resp = await client.get(ARXIV_API, params=params)
        if resp.status_code == 200:
            entry = _parse_arxiv_xml(resp.text)
            if entry:
                rc = ResolvedCitation(
                    key=ref.key, found=True, source="arxiv", **entry
                )
    except httpx.RequestError as exc:
        logger.debug(f"arXiv request error for [{ref.key}]: {exc}")
    return rc


def _parse_arxiv_xml(xml_text: str) -> Optional[dict]:
    """Very lightweight XML parse (avoids lxml dependency)."""
    import xml.etree.ElementTree as ET
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title   = (entry.findtext("atom:title", "", ns) or "").strip()
        summary = (entry.findtext("atom:summary", "", ns) or "").strip()
        authors = [
            a.findtext("atom:name", "", ns) or ""
            for a in entry.findall("atom:author", ns)
        ]
        published = entry.findtext("atom:published", "", ns) or ""
        year = published[:4] if published else None

        # Extract arXiv ID from the entry <id> tag, e.g.
        # "http://arxiv.org/abs/2005.14165v1" → "2005.14165"
        entry_id = (entry.findtext("atom:id", "", ns) or "").strip()
        arxiv_id = ""
        if "arxiv.org/abs/" in entry_id:
            arxiv_id = entry_id.split("arxiv.org/abs/")[-1].split("v")[0]

        return dict(
            title=title, abstract=summary, authors=authors, year=year,
            doi="", venue="arXiv", citation_count=0, warning="",
            arxiv_id=arxiv_id, open_access_pdf="",
        )
    except ET.ParseError:
        return None


# ── Full-text fetching (RAG upgrade) ──────────────────────────────────────────

async def fetch_full_text(client: httpx.AsyncClient, rc: ResolvedCitation) -> str:
    """
    Attempt to retrieve full paper body text.

    Strategy (in order):
      1. arXiv paper → ar5iv.org HTML (clean, no PDF parsing)
      2. S2 open-access PDF URL → download + PyMuPDF text extraction
      3. Fall back to empty string (caller uses abstract instead)

    The result is plain text, ready to be chunked by the aligner.
    """
    # Strategy 1: ar5iv HTML (arXiv papers only, no PDF needed)
    arxiv_id = rc.arxiv_id
    if arxiv_id:
        text = await _fetch_ar5iv_html(client, arxiv_id)
        if text:
            logger.info(f"  [{rc.key}] full text via ar5iv ({len(text):,} chars)")
            return text

    # Strategy 2: S2 open-access PDF
    if rc.open_access_pdf:
        text = await _fetch_pdf_text(client, rc.open_access_pdf)
        if text:
            logger.info(f"  [{rc.key}] full text via S2 PDF ({len(text):,} chars)")
            return text

    return ""


async def _fetch_ar5iv_html(client: httpx.AsyncClient, arxiv_id: str) -> str:
    """
    Fetch paper body from ar5iv.org — a LaTeX-to-HTML mirror of arXiv.
    Strips navigation/figures, returns clean body paragraphs.
    """
    url = f"https://ar5iv.org/html/{arxiv_id}"
    try:
        resp = await client.get(url, timeout=20.0)
        if resp.status_code != 200:
            return ""
        return _extract_ar5iv_text(resp.text)
    except Exception as exc:
        logger.debug(f"ar5iv fetch failed for {arxiv_id}: {exc}")
        return ""


def _extract_ar5iv_text(html: str) -> str:
    """
    Extract paragraph text from ar5iv HTML without any heavy HTML parser.
    Uses simple regex to pull <p>...</p> content and strips tags.
    Keeps only paragraphs with ≥20 characters (removes captions/headers).
    """
    import re
    # Pull all <p> content
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE)
    # Strip HTML tags from each paragraph
    tag_re = re.compile(r"<[^>]+>")
    cleaned = []
    for p in paragraphs:
        text = tag_re.sub(" ", p)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 20:
            cleaned.append(text)
    return "\n\n".join(cleaned)


async def _fetch_pdf_text(client: httpx.AsyncClient, pdf_url: str) -> str:
    """
    Download a PDF and extract text via PyMuPDF.
    Returns plain text or empty string on any error.
    """
    try:
        resp = await client.get(pdf_url, timeout=30.0)
        if resp.status_code != 200 or not resp.content:
            return ""
        return _extract_pdf_bytes(resp.content)
    except Exception as exc:
        logger.debug(f"PDF fetch failed ({pdf_url}): {exc}")
        return ""


def _extract_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from raw PDF bytes using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except Exception as exc:
        logger.debug(f"PyMuPDF extraction error: {exc}")
        return ""


async def enrich_full_text(resolved: Dict[str, "ResolvedCitation"]) -> None:
    """
    Attempt full-text enrichment for all resolved citations in-place.
    Called after resolve_all so that full_text is populated before the
    aligner runs.

    Skipped entirely when RAG_FULL_TEXT=false (abstract-only mode).
    """
    if not RAG_FULL_TEXT:
        logger.info("   Full-text fetching disabled (RAG_FULL_TEXT=false).")
        return

    async with _build_client() as client:
        for key, rc in resolved.items():
            if not rc.found or rc.full_text:
                continue   # skip unfound or already enriched
            text = await fetch_full_text(client, rc)
            if text:
                # Cap to RAG_MAX_CHARS to avoid enormous passages
                rc.full_text = text[:RAG_MAX_CHARS]
                # Re-save enriched cache entry
                cache_key = rc.doi or rc.title or rc.key
                _save_cache(cache_key, rc.__dict__)


def enrich_full_text_sync(resolved: Dict[str, "ResolvedCitation"]) -> None:
    """Synchronous wrapper for enrich_full_text."""
    asyncio.run(enrich_full_text(resolved))
