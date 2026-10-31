"""
Layer 7 – Report Generator
===========================
Combines all pipeline outputs into a structured CitationReport and
serialises it to:
  • JSON  (machine-readable, always written)
  • HTML  (human-readable, rendered from Jinja2 template)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from config import OUTPUT_DIR, TMPL_DIR
from modules.verifier import VerificationResult, Verdict

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class CitationRecord:
    key: str
    raw_citation: str
    claim: str
    paper_title: str
    paper_authors: List[str]
    paper_year: Optional[str]
    paper_found: bool
    verdict: str                  # Verdict enum value
    confidence: float
    alignment_score: float
    best_evidence: str
    doi: str
    source: str
    warning: str


@dataclass
class CitationReport:
    input_file: str
    generated_at: str
    total_citations: int
    verified_count: int
    neutral_count: int
    contradicts_count: int
    not_found_count: int
    overall_score: float          # mean alignment score across all
    records: List[CitationRecord] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def build_report(
    input_file: str,
    citation_records: List[CitationRecord],
) -> CitationReport:
    """Assemble the final CitationReport from individual records."""
    total      = len(citation_records)
    verified   = sum(1 for r in citation_records if r.verdict == Verdict.SUPPORTS.value)
    neutral    = sum(1 for r in citation_records if r.verdict == Verdict.NEUTRAL.value)
    contradicts= sum(1 for r in citation_records if r.verdict == Verdict.CONTRADICTS.value)
    not_found  = sum(1 for r in citation_records if r.verdict == Verdict.NO_EVIDENCE.value)

    scores = [r.alignment_score for r in citation_records]
    overall = round(sum(scores) / len(scores), 4) if scores else 0.0

    return CitationReport(
        input_file=input_file,
        generated_at=datetime.utcnow().isoformat() + "Z",
        total_citations=total,
        verified_count=verified,
        neutral_count=neutral,
        contradicts_count=contradicts,
        not_found_count=not_found,
        overall_score=overall,
        records=citation_records,
    )


def save_report(report: CitationReport, stem: str) -> Dict[str, Path]:
    """
    Write JSON + HTML reports to OUTPUT_DIR.
    Returns dict with 'json' and 'html' keys pointing to output paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # ── JSON ──────────────────────────────────────────────────────
    json_path = OUTPUT_DIR / f"{stem}_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2, ensure_ascii=False)
    logger.info(f"JSON report: {json_path}")
    paths["json"] = json_path

    # ── HTML ──────────────────────────────────────────────────────
    html_path = OUTPUT_DIR / f"{stem}_report.html"
    html = _render_html(report)
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report: {html_path}")
    paths["html"] = html_path

    return paths


# ── HTML rendering ────────────────────────────────────────────────────────────

_INLINE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CiteGuard Report – {{ report.input_file }}</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system,"Segoe UI",system-ui,sans-serif;
         background:#f1f5fb; color:#0d2137; margin:0; padding:32px; }
  .report-header { display:flex; align-items:flex-start; justify-content:space-between;
                   margin-bottom:8px; gap:16px; flex-wrap:wrap; }
  .report-header h1 { font-size:1.6rem; font-weight:700; margin:0 0 4px; color:#0f172a; }
  .eyebrow { font-size:.68rem; font-weight:700; letter-spacing:.1em; color:#6366f1;
             text-transform:uppercase; margin-bottom:6px; }
  .meta { color:#64748b; font-size:.82rem; margin-bottom:28px; line-height:1.8; }
  .meta strong { color:#0f172a; }
  .pdf-btn { display:inline-flex; align-items:center; gap:7px;
             background:#6366f1; color:#fff; border:none; padding:10px 20px;
             border-radius:8px; font-size:.85rem; font-weight:600; cursor:pointer;
             text-decoration:none; white-space:nowrap; }
  .pdf-btn:hover { background:#4f46e5; }
  .summary { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:32px; }
  .card { background:#fff; border:1px solid #e2e8f0; border-radius:10px;
          padding:16px 22px; min-width:110px; text-align:center;
          box-shadow:0 1px 4px rgba(0,0,0,.06); }
  .card .num { font-size:2rem; font-weight:700; line-height:1.1; }
  .card .lbl { font-size:.68rem; color:#64748b; text-transform:uppercase;
               letter-spacing:.06em; margin-top:3px; }
  .supports    { color:#22c55e; }
  .neutral     { color:#f59e0b; }
  .contradicts { color:#ef4444; }
  .no_evidence { color:#94a3b8; }
  table { width:100%; border-collapse:collapse; background:#fff;
          border:1px solid #e2e8f0; border-radius:10px; overflow:hidden;
          box-shadow:0 1px 4px rgba(0,0,0,.06); }
  th { background:#f8fafc; font-size:.72rem; text-transform:uppercase;
       letter-spacing:.06em; padding:11px 14px; text-align:left;
       border-bottom:1px solid #e2e8f0; color:#64748b; font-weight:700; }
  td { padding:10px 14px; border-bottom:1px solid #f1f5f9;
       font-size:.84rem; vertical-align:top; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#fafbff; }
  code { font-size:.78rem; background:#f1f5f9; padding:1px 5px; border-radius:4px; }
  .badge { display:inline-block; padding:3px 9px; border-radius:99px;
           font-size:.72rem; font-weight:700; }
  .badge-supports    { background:#dcfce7; color:#15803d; }
  .badge-neutral     { background:#fef3c7; color:#b45309; }
  .badge-contradicts { background:#fee2e2; color:#b91c1c; }
  .badge-no_evidence { background:#f1f5f9; color:#4b5563; }
  .evidence { font-style:italic; color:#475569; max-width:280px;
              overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .align-bar-wrap { display:flex; align-items:center; gap:7px; }
  .align-track { width:52px; height:6px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .align-fill  { height:100%; background:#6366f1; border-radius:99px; }
  .align-pct   { font-size:.78rem; font-weight:700; color:#6366f1; min-width:32px; }
  footer { margin-top:32px; text-align:center; font-size:.72rem;
           color:#94a3b8; border-top:1px solid #e2e8f0; padding-top:14px; }

  /* ── Print / PDF styles ─────────────────────────────────────── */
  @media print {
    body { background:#fff !important; padding:16px; font-size:11px; }
    .pdf-btn { display:none !important; }
    table { box-shadow:none; border:1px solid #ccc; }
    th { background:#f0f0f0 !important; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    .badge { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    .align-fill { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    @page { margin:1.5cm; size:A4 landscape; }
    tr { page-break-inside:avoid; }
  }
</style>
</head>
<body>
<div class="report-header">
  <div>
    <div class="eyebrow">CiteGuard · Citation Verification Report</div>
    <h1>{{ report.input_file }}</h1>
  </div>
  <button class="pdf-btn" onclick="window.print()">
    🖨 Save as PDF
  </button>
</div>

<p class="meta">
  <strong>Generated:</strong> {{ report.generated_at }} &nbsp;&nbsp;
  <strong>Overall Alignment Score:</strong> {{ "%.1f"|format(report.overall_score * 100) }}% &nbsp;&nbsp;
  <strong>Total Citations:</strong> {{ report.total_citations }}
</p>

<div class="summary">
  <div class="card">
    <div class="num">{{ report.total_citations }}</div>
    <div class="lbl">Total</div>
  </div>
  <div class="card">
    <div class="num supports">{{ report.verified_count }}</div>
    <div class="lbl">✅ Supported</div>
  </div>
  <div class="card">
    <div class="num neutral">{{ report.neutral_count }}</div>
    <div class="lbl">⚠ Neutral</div>
  </div>
  <div class="card">
    <div class="num contradicts">{{ report.contradicts_count }}</div>
    <div class="lbl">❌ Contradicts</div>
  </div>
  <div class="card">
    <div class="num no_evidence">{{ report.not_found_count }}</div>
    <div class="lbl">🔍 Not Found</div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Citation</th>
      <th>Paper Title</th>
      <th>Year</th>
      <th>Claim</th>
      <th>Verdict</th>
      <th>Alignment</th>
      <th>Best Evidence</th>
    </tr>
  </thead>
  <tbody>
  {% for r in report.records %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><code>{{ r.raw_citation }}</code></td>
      <td>
        {% if r.doi %}
          <a href="https://doi.org/{{ r.doi }}" target="_blank">{{ r.paper_title[:55] }}{% if r.paper_title|length > 55 %}…{% endif %}</a>
        {% else %}
          {{ r.paper_title[:55] }}{% if r.paper_title|length > 55 %}…{% endif %}
        {% endif %}
        {% if r.warning %}<br/><small style="color:#b91c1c">⚠ {{ r.warning }}</small>{% endif %}
      </td>
      <td>{{ r.paper_year or "—" }}</td>
      <td style="max-width:180px">{{ r.claim[:100] }}{% if r.claim|length > 100 %}…{% endif %}</td>
      <td>
        <span class="badge badge-{{ r.verdict.lower() }}">{{ r.verdict.replace("_"," ") }}</span>
        <div style="font-size:.72rem;color:#64748b;margin-top:2px">conf {{ "%.0f"|format(r.confidence * 100) }}%</div>
      </td>
      <td>
        <div class="align-bar-wrap">
          <div class="align-track"><div class="align-fill" style="width:{{ "%.0f"|format(r.alignment_score * 100) }}%"></div></div>
          <span class="align-pct">{{ "%.0f"|format(r.alignment_score * 100) }}%</span>
        </div>
      </td>
      <td class="evidence" title="{{ r.best_evidence }}">{{ r.best_evidence[:100] if r.best_evidence else "—" }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<footer>CiteGuard v1.0 &nbsp;·&nbsp; Made with IBM Bob &nbsp;·&nbsp; {{ report.generated_at }}</footer>
</body>
</html>
"""


def _render_html(report: CitationReport) -> str:
    env = Environment(autoescape=select_autoescape(["html"]))
    tmpl = env.from_string(_INLINE_TEMPLATE)
    return tmpl.render(report=report)
