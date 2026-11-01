"""
CiteGuard – Main Pipeline Entry Point
======================================
Usage:
  python main.py <path-to-paper.pdf>

  or via the FastAPI server:
  uvicorn api:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

import config  # noqa – ensures .env is loaded before any module import

from modules.extractor       import extract, ExtractedDocument
from modules.citation_parser import parse, CitationMap
from modules.claim_extractor import extract_claims, ClaimCitationPair
from modules.resolver        import resolve_all_sync, enrich_full_text_sync, ResolvedCitation
from modules.aligner         import retrieve_evidence
from modules.verifier        import verify, Verdict
from modules.reporter        import (
    CitationRecord, build_report, save_report,
)
from modules.rag_store       import store_stats

# ── Logger setup ──────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(pdf_path: str) -> dict:
    """
    Full CiteGuard pipeline for a single paper.
    Returns a dict with 'json' and 'html' output file paths.
    """
    path = Path(pdf_path)
    stem = path.stem

    # ── Layer 1: Extract document text ────────────────────────────
    logger.info("━━ Layer 1: Document Extraction")
    doc: ExtractedDocument = extract(pdf_path)
    logger.info(f"   {len(doc.full_text):,} characters extracted, "
                f"{len(doc.sections)} sections detected")

    # ── Layer 2a: Parse citations ──────────────────────────────────
    logger.info("━━ Layer 2a: Citation Parsing")
    citation_map: CitationMap = parse(doc.full_text, doc.reference_block)

    if not citation_map.references:
        logger.warning("No reference entries found. Check if the PDF has a bibliography section.")

    # ── Layer 2b: Extract claims ───────────────────────────────────
    logger.info("━━ Layer 2b: Claim Extraction")
    claim_pairs: list[ClaimCitationPair] = extract_claims(citation_map, doc.full_text)

    # ── Layer 3: Resolve metadata ──────────────────────────────────
    logger.info("━━ Layer 3: Metadata Resolution (API calls…)")
    resolved: dict[str, ResolvedCitation] = resolve_all_sync(citation_map.references)

    # ── Layer 3b: Full-text enrichment (RAG upgrade) ───────────────
    logger.info("━━ Layer 3b: Full-text Enrichment (arXiv / open-access PDF)")
    enrich_full_text_sync(resolved)
    full_text_count = sum(1 for r in resolved.values() if r.full_text)
    logger.info(f"   {full_text_count}/{len(resolved)} papers enriched with full text")

    # ── RAG store status ───────────────────────────────────────────
    stats = store_stats()
    if stats["available"]:
        logger.info(f"   RAG store: {stats['total_passages']} passages indexed")

    # ── Layers 4-6: Retrieve evidence + verify ─────────────────────
    logger.info("━━ Layers 4-6: Evidence Retrieval + NLI Verification (RAG)")
    records: list[CitationRecord] = []

    # Deduplicate: one record per (citation_key) — use first claim seen
    seen_keys: set[str] = set()

    for pair in claim_pairs:
        key = pair.citation_key
        if key in seen_keys:
            continue
        seen_keys.add(key)

        res = resolved.get(key)
        if res is None:
            logger.debug(f"  [{key}] no resolver result, skipping evidence check")
            continue

        # RAG upgrade: prefer full_text (full paper body) over abstract
        paper_text = res.full_text or res.abstract or ""

        # Stable paper ID for the vector store — prefer DOI, else title hash
        paper_id = res.doi or res.arxiv_id or key

        # Retrieve evidence passages via RAG store
        passages = retrieve_evidence(pair.claim, paper_text, paper_id=paper_id) if paper_text else []

        # Run NLI verification
        vr = verify(pair.claim, passages)

        # Build record
        ref = citation_map.references.get(key)
        records.append(CitationRecord(
            key=key,
            raw_citation=pair.raw_citation,
            claim=pair.claim,
            paper_title=res.title or (ref.title if ref else ""),
            paper_authors=res.authors or (ref.authors if ref else []),
            paper_year=res.year or (ref.year if ref else None),
            paper_found=res.found,
            verdict=vr.verdict.value,
            confidence=vr.confidence,
            alignment_score=vr.alignment_score,
            best_evidence=vr.best_evidence,
            doi=res.doi or (ref.doi if ref else ""),
            source=res.source,
            warning=res.warning,
        ))
        logger.info(
            f"  [{key}] {vr.verdict.value:12s} "
            f"align={vr.alignment_score:.2f}  \"{pair.claim[:60]}...\""
        )

    # ── Layer 7: Generate report ───────────────────────────────────
    logger.info("━━ Layer 7: Report Generation")
    report = build_report(input_file=path.name, citation_records=records)
    output_paths = save_report(report, stem=stem)

    logger.info(
        f"✅ Done.  {report.total_citations} citations processed. "
        f"Supported={report.verified_count}  "
        f"Neutral={report.neutral_count}  "
        f"Contradicts={report.contradicts_count}  "
        f"NotFound={report.not_found_count}"
    )
    return {k: str(v) for k, v in output_paths.items()}


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-paper.pdf>")
        sys.exit(1)

    result = run(sys.argv[1])
    print("\nOutputs:")
    for fmt, path in result.items():
        print(f"  {fmt.upper():5s} → {path}")
