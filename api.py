"""
CiteGuard – FastAPI Backend
============================
Endpoints:
  GET  /                      → serves index.html
  POST /verify                → upload PDF, start background job
  GET  /status/{job_id}       → poll job progress
  GET  /report/{stem}         → serve generated HTML report
  GET  /download/{stem}       → download JSON report
  GET  /static/{path}         → static files (css, js)

Run with:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import sys
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor

import config
from main import run as citeguard_run

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="CiteGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# ── Job store (in-memory) ─────────────────────────────────────────────────────
# Each job: { status, stage, progress, result, error }
_jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for running the blocking pipeline
_executor = ThreadPoolExecutor(max_workers=2)

STAGES = [
    "Uploading PDF",
    "Extracting text",
    "Parsing citations",
    "Resolving metadata",
    "Retrieving evidence",
    "Verifying claims",
    "Generating report",
    "Done",
]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main single-page frontend."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """
    Accept a PDF upload, save it, kick off background processing.
    Returns { job_id } immediately.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate file size (max 20 MB)
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20 MB.")

    # Save to input directory
    job_id = str(uuid.uuid4())[:8]
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = config.INPUT_DIR / f"{job_id}_{file.filename}"
    pdf_path.write_bytes(content)

    # Initialise job record
    _jobs[job_id] = {
        "status": "running",
        "stage": STAGES[1],
        "stage_index": 1,
        "total_stages": len(STAGES) - 1,
        "filename": file.filename,
        "result": None,
        "error": None,
    }

    # Run pipeline in thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, job_id, str(pdf_path))

    return JSONResponse({"job_id": job_id, "filename": file.filename})


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll job progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(job)


@app.get("/report/{stem}", response_class=HTMLResponse)
async def report(stem: str):
    """Serve the generated HTML report."""
    report_path = config.OUTPUT_DIR / f"{stem}_report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))


@app.get("/documents")
async def documents():
    """Return list of all successfully completed jobs (for My Documents page)."""
    docs = []
    for job_id, job in _jobs.items():
        if job["status"] == "done" and job.get("result"):
            r = job["result"]
            docs.append({
                "job_id":          job_id,
                "filename":        job.get("filename", "unknown.pdf"),
                "stem":            r.get("stem", ""),
                "total_citations": r.get("total_citations", 0),
                "verified_count":  r.get("verified_count", 0),
                "overall_score":   r.get("overall_score", 0),
            })
    return JSONResponse(docs)


@app.get("/download/{stem}")
async def download(stem: str):
    """Download the JSON report."""
    json_path = config.OUTPUT_DIR / f"{stem}_report.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        path=str(json_path),
        media_type="application/json",
        filename=f"{stem}_citeguard_report.json",
    )


@app.get("/download-pdf/{job_id}")
async def download_pdf(job_id: str):
    """Download the original uploaded PDF."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    filename = job.get("filename", "paper.pdf")
    stem = job.get("result", {}).get("stem", f"{job_id}_")
    # Find the PDF in INPUT_DIR
    pdf_candidates = list(config.INPUT_DIR.glob(f"{job_id}_*.pdf"))
    if not pdf_candidates:
        raise HTTPException(status_code=404, detail="PDF file not found.")
    return FileResponse(
        path=str(pdf_candidates[0]),
        media_type="application/pdf",
        filename=filename,
    )


# ── Background pipeline runner ────────────────────────────────────────────────

def _run_pipeline(job_id: str, pdf_path: str) -> None:
    """
    Runs in a thread. Updates _jobs[job_id] as the pipeline progresses.
    We monkey-patch the logger to update stage info.
    """
    job = _jobs[job_id]

    def _set_stage(idx: int) -> None:
        job["stage_index"] = idx
        job["stage"] = STAGES[min(idx, len(STAGES) - 1)]

    try:
        _set_stage(1)   # Extracting text

        # Import pipeline layers individually for stage tracking
        from modules.extractor       import extract
        from modules.citation_parser import parse
        from modules.claim_extractor import extract_claims
        from modules.resolver        import resolve_all_sync
        from modules.aligner         import retrieve_evidence
        from modules.verifier        import verify as nli_verify
        from modules.reporter        import CitationRecord, build_report, save_report
        from pathlib import Path as _Path

        path = _Path(pdf_path)
        stem = f"{job_id}_{path.stem}"

        # Layer 1
        _set_stage(1)
        doc = extract(pdf_path)

        # Layer 2
        _set_stage(2)
        citation_map = parse(doc.full_text, doc.reference_block)
        claim_pairs  = extract_claims(citation_map, doc.full_text)

        # Layer 3
        _set_stage(3)
        resolved = resolve_all_sync(citation_map.references)

        # Layers 4-6
        _set_stage(4)
        records = []
        seen_keys: set = set()
        total = len(claim_pairs)

        for i, pair in enumerate(claim_pairs):
            key = pair.citation_key
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Update progress within stage 4-6
            progress_stage = 4 + int((i / max(total, 1)) * 2)
            _set_stage(min(progress_stage, 6))

            res = resolved.get(key)
            if res is None:
                continue

            paper_text = res.abstract or ""
            passages   = retrieve_evidence(pair.claim, paper_text) if paper_text else []
            vr         = nli_verify(pair.claim, passages)

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

        # Layer 7
        _set_stage(6)
        report_obj  = build_report(input_file=path.name, citation_records=records)
        output_paths = save_report(report_obj, stem=stem)

        _set_stage(7)
        job["status"] = "done"
        job["result"] = {
            "stem": stem,
            "total_citations":   report_obj.total_citations,
            "verified_count":    report_obj.verified_count,
            "neutral_count":     report_obj.neutral_count,
            "contradicts_count": report_obj.contradicts_count,
            "not_found_count":   report_obj.not_found_count,
            "overall_score":     report_obj.overall_score,
            "records": [
                {
                    "key":             r.key,
                    "raw_citation":    r.raw_citation,
                    "claim":           r.claim,
                    "paper_title":     r.paper_title,
                    "paper_year":      r.paper_year,
                    "paper_found":     r.paper_found,
                    "verdict":         r.verdict,
                    "confidence":      round(r.confidence * 100, 1),
                    "alignment_score": round(r.alignment_score * 100, 1),
                    "best_evidence":   r.best_evidence,
                    "doi":             r.doi,
                    "warning":         r.warning,
                }
                for r in records
            ],
        }

    except Exception as exc:
        job["status"] = "error"
        job["error"]  = str(exc)
        traceback.print_exc(file=sys.stderr)
