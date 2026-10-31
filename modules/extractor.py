"""
Layer 1 – Document Processing
Extracts raw text and section-level structure from PDF research papers.
Uses pdfplumber as primary; falls back to PyMuPDF (fitz) if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Section:
    title: str
    text: str
    page_start: int
    page_end: int


@dataclass
class ExtractedDocument:
    file_path: str
    full_text: str
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    reference_block: str = ""   # raw bibliography text


# ── Section heading patterns ──────────────────────────────────────────────────

_SECTION_PATTERNS = re.compile(
    r"^(?:\d+\.?\s+)?"                              # optional numbering
    r"(abstract|introduction|related work|"
    r"background|methodology|methods|"
    r"experiments?|results?|discussion|"
    r"conclusion|references?|bibliography)",
    re.IGNORECASE | re.MULTILINE,
)

_REF_SECTION = re.compile(
    r"(references|bibliography)\s*\n",
    re.IGNORECASE,
)


# ── Core extractor ────────────────────────────────────────────────────────────

def extract(file_path: str | Path) -> ExtractedDocument:
    """
    Main entry point.  Accepts a PDF path and returns an ExtractedDocument.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Extracting: {path.name}")

    try:
        return _extract_pdfplumber(path)
    except Exception as exc:
        logger.warning(f"pdfplumber failed ({exc}), trying PyMuPDF…")
        return _extract_pymupdf(path)


# ── pdfplumber implementation ─────────────────────────────────────────────────

def _extract_pdfplumber(path: Path) -> ExtractedDocument:
    import pdfplumber

    pages_text: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            pages_text.append(text)

    full_text = "\n".join(pages_text)
    sections  = _split_sections(pages_text)
    ref_block = _extract_reference_block(full_text)

    return ExtractedDocument(
        file_path=str(path),
        full_text=full_text,
        sections=sections,
        reference_block=ref_block,
    )


# ── PyMuPDF fallback ──────────────────────────────────────────────────────────

def _extract_pymupdf(path: Path) -> ExtractedDocument:
    import fitz  # PyMuPDF

    pages_text: List[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pages_text.append(page.get_text("text") or "")

    full_text = "\n".join(pages_text)
    sections  = _split_sections(pages_text)
    ref_block = _extract_reference_block(full_text)

    return ExtractedDocument(
        file_path=str(path),
        full_text=full_text,
        sections=sections,
        reference_block=ref_block,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sections(pages_text: List[str]) -> List[Section]:
    """
    Naively splits full text into sections based on heading patterns.
    Returns a list of Section objects.
    """
    sections: List[Section] = []
    current_title = "preamble"
    current_lines: List[str] = []
    current_page_start = 1

    for page_num, page_text in enumerate(pages_text, start=1):
        for line in page_text.splitlines():
            if _SECTION_PATTERNS.match(line.strip()):
                # Save previous section
                if current_lines:
                    sections.append(Section(
                        title=current_title,
                        text="\n".join(current_lines).strip(),
                        page_start=current_page_start,
                        page_end=page_num,
                    ))
                current_title      = line.strip()
                current_lines      = []
                current_page_start = page_num
            else:
                current_lines.append(line)

    # flush last section
    if current_lines:
        sections.append(Section(
            title=current_title,
            text="\n".join(current_lines).strip(),
            page_start=current_page_start,
            page_end=len(pages_text),
        ))

    return sections


def _extract_reference_block(full_text: str) -> str:
    """Return everything after the first 'References' heading."""
    match = _REF_SECTION.search(full_text)
    if match:
        return full_text[match.start():].strip()
    return ""


def get_section(doc: ExtractedDocument, name: str) -> Optional[str]:
    """Convenience: return text of a named section (case-insensitive)."""
    for sec in doc.sections:
        if name.lower() in sec.title.lower():
            return sec.text
    return None
