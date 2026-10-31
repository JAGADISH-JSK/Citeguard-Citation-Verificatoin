"""
Basic unit tests for CiteGuard modules.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── citation_parser tests ─────────────────────────────────────────────────────

from modules.citation_parser import parse, _expand_num_keys, _make_author_key


def test_expand_num_keys_simple():
    assert _expand_num_keys("3") == ["3"]


def test_expand_num_keys_range():
    assert _expand_num_keys("1-3") == ["1", "2", "3"]


def test_expand_num_keys_mixed():
    assert _expand_num_keys("1, 3-5") == ["1", "3", "4", "5"]


def test_make_author_key():
    assert _make_author_key("Smith et al.", "2020") == "Smith2020"
    assert _make_author_key("Jones & Lee", "2019") == "Jones2019"   # first author before &


def test_parse_numbered_inline():
    text = "Transformers outperform RNNs [1] on many tasks [2,3]."
    ref_block = "[1] Vaswani et al. Attention is All You Need. 2017.\n[2] Devlin et al. BERT. 2019.\n[3] Brown et al. GPT-3. 2020."
    cm = parse(text, ref_block)
    assert any("1" in c.keys for c in cm.inline)
    assert "1" in cm.references


def test_parse_author_year_inline():
    text = "As shown by (Vaswani et al., 2017), attention is powerful."
    ref_block = ""
    cm = parse(text, ref_block)
    assert any("Vaswani2017" in c.keys for c in cm.inline)


# ── claim_extractor tests ─────────────────────────────────────────────────────

from modules.claim_extractor import extract_claims, _clean_claim


def test_clean_claim_strips_brackets():
    result = _clean_claim("Transformers outperform RNNs [1] on NLP tasks.")
    assert "[1]" not in result
    assert "Transformers" in result


def test_clean_claim_strips_author_year():
    result = _clean_claim("As shown by (Smith et al., 2020), accuracy improves.")
    assert "(Smith et al., 2020)" not in result


# ── aligner tests ─────────────────────────────────────────────────────────────

from modules.aligner import _chunk_text


def test_chunk_text_basic():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = _chunk_text(text, chunk_size=2)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_empty():
    assert _chunk_text("", chunk_size=3) == []


# ── verifier tests ────────────────────────────────────────────────────────────

from modules.verifier import _decide_verdict, _compute_alignment, Verdict


def test_decide_verdict_supports():
    v, conf = _decide_verdict(0.85, 0.10, 0.05)
    assert v == Verdict.SUPPORTS


def test_decide_verdict_contradicts():
    v, conf = _decide_verdict(0.05, 0.10, 0.85)
    assert v == Verdict.CONTRADICTS


def test_decide_verdict_neutral():
    v, conf = _decide_verdict(0.30, 0.40, 0.30)
    assert v == Verdict.NEUTRAL


def test_compute_alignment_no_evidence():
    score = _compute_alignment(Verdict.NO_EVIDENCE, 0.0, 0.0)
    assert score == 0.0


def test_compute_alignment_supports():
    score = _compute_alignment(Verdict.SUPPORTS, 0.9, 0.8)
    assert score > 0.5


def test_compute_alignment_contradicts():
    score = _compute_alignment(Verdict.CONTRADICTS, 0.9, 0.8)
    assert score < 0.3
