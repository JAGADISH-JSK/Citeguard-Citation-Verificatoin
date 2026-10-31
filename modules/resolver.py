"""
Layer 3 – Metadata Resolver
Given a parsed Reference, queries external APIs to:
  1. Confirm the paper exists (existence check)
  2. Fetch its abstract (needed for evidence alignment)
  3. Flag retracted or suspicious papers

API priority:
  Semantic Scholar  →  CrossRef  →  arXiv  →  unresolved
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
    S2_API_KEY, CACHE_DIR,
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
    fields = "title,abstract,authors,year,externalIds,venue,citationCount"

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
        return dict(title=title, abstract=summary, authors=authors, year=year, doi="", venue="arXiv", citation_count=0, warning="")
    except ET.ParseError:
        return None
