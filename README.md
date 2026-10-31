# CiteGuard 🔍

> **Automated Citation Verification and Evidence Alignment for Research Papers**

CiteGuard automatically checks every citation in a research paper to verify:
1. **Existence** – does the cited paper actually exist?
2. **Alignment** – does it actually support the claim being made?
3. **Verdict** – Supports / Neutral / Contradicts

---

## Architecture

```
PDF Input
  │
  ├─ Layer 1: Document Extraction     (pdfplumber / PyMuPDF)
  ├─ Layer 2a: Citation Parsing       (Regex + NER)
  ├─ Layer 2b: Claim Extraction       (sentence tokenisation)
  ├─ Layer 3: Metadata Resolution     (Semantic Scholar / CrossRef / arXiv)
  ├─ Layer 4: Evidence Retrieval      (Bi-encoder – sentence-transformers)
  ├─ Layer 5: Re-ranking              (Cross-encoder)
  ├─ Layer 6: NLI Verification        (DeBERTa-Large-MNLI)
  └─ Layer 7: Report Generation       (JSON + HTML)
```

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd citeguard

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Configure environment
copy .env.example .env
# (edit .env if you have a Semantic Scholar API key)

# 5. Run on a paper
python main.py input/your_paper.pdf
```

Outputs are written to `output/`:
- `your_paper_report.json` – machine-readable full results
- `your_paper_report.html` – human-readable report

---

## Project Structure

```
citeguard/
├── modules/
│   ├── extractor.py         # Layer 1 – PDF parsing
│   ├── citation_parser.py   # Layer 2a – citation detection
│   ├── claim_extractor.py   # Layer 2b – claim sentence extraction
│   ├── resolver.py          # Layer 3 – API metadata lookup
│   ├── aligner.py           # Layers 4+5 – bi-encoder + cross-encoder
│   ├── verifier.py          # Layer 6 – NLI fact verification
│   └── reporter.py          # Layer 7 – report generation
├── input/                   # Drop your PDFs here
├── output/                  # Reports written here
├── cache/                   # API response cache (JSON)
├── templates/               # (optional) Custom Jinja2 templates
├── tests/                   # Unit tests
├── config.py                # Central configuration
├── main.py                  # Pipeline entry point
├── requirements.txt
└── .env.example
```

---

## Configuration

All settings can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L12-v2` | Bi-encoder model |
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-large` | NLI model |
| `TOP_K_RETRIEVAL` | `50` | Bi-encoder candidate count |
| `TOP_K_RERANK` | `10` | Cross-encoder output count |
| `NLI_THRESHOLD` | `0.5` | Minimum confidence to assign verdict |
| `S2_API_KEY` | _(empty)_ | Semantic Scholar API key |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Verdict Labels

| Label | Meaning |
|---|---|
| ✅ SUPPORTS | Cited paper's content entails the claim |
| ⚠️ NEUTRAL | Weak or tangential relationship |
| ❌ CONTRADICTS | Cited paper's content contradicts the claim |
| 🔍 NO_EVIDENCE | Paper not found or abstract unavailable |

---

## Swapping to IBM Watson / Granite

To use IBM's embedding models, update `EMBEDDING_MODEL` in `.env`:

```env
EMBEDDING_MODEL=ibm-granite/granite-embedding-278m-multilingual
```

For full IBM watsonx integration, replace the sentence-transformers calls
in `modules/aligner.py` with the watsonx.ai embeddings API.
