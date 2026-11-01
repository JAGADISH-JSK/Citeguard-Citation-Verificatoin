# CiteGuard 🔍

> **AI-Powered Citation Verification & Evidence Analysis for Research Papers — with RAG Pipeline**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.133+-009688?style=flat&logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG%20Store-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-6366f1?style=flat)
![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat)

CiteGuard automatically checks every citation in a research paper to verify:
1. **Existence** – does the cited paper actually exist?
2. **Alignment** – does it actually support the claim being made?
3. **Verdict** – Supports / Neutral / Contradicts / No Evidence

---

## Screenshots

### 🏠 Dashboard
![CiteGuard Dashboard](docs/screenshots/CiteGuard_Dashboard.png)

### 📤 Upload Document
![CiteGuard Upload](docs/screenshots/CiteGuard_Upload.png)

### 📊 Citation Analysis Results
![CiteGuard Results](docs/screenshots/CiteGuard_Results.png)

---

## Architecture Diagram

```mermaid
flowchart LR
    A([📄 PDF Input]) --> B

    subgraph PIPELINE ["⚙️ CiteGuard 8-Layer RAG Pipeline"]
        B["🗂️ Layer 1\nDocument Extraction\npdfplumber · PyMuPDF"]
        B --> C["🔗 Layer 2a\nCitation Parsing\nRegex + NER"]
        B --> D["💬 Layer 2b\nClaim Extraction\nSentence Tokenisation"]
        C --> E["🌐 Layer 3\nMetadata Resolution\nSemantic Scholar · CrossRef · arXiv"]
        D --> E
        E --> F["📖 Layer 3b ✨ NEW\nFull-text Enrichment\nar5iv HTML · S2 open-access PDF"]
        F --> G["🗄️ RAG Store ✨ NEW\nChromaDB Vector Index\nPersistent chunk embeddings"]
        G --> H["🔍 Layer 4\nEvidence Retrieval\nBi-Encoder · all-MiniLM-L12-v2"]
        H --> I["📊 Layer 5\nRe-ranking\nCross-Encoder · ms-marco-MiniLM"]
        I --> J["🧠 Layer 6\nNLI Verification\nDeBERTa-Large-MNLI"]
        J --> K["📝 Layer 7\nReport Generation\nJSON + HTML"]
    end

    K --> L(["✅ SUPPORTS"])
    K --> M(["⚠️ NEUTRAL"])
    K --> N(["❌ CONTRADICTS"])
    K --> O(["🔍 NO EVIDENCE"])
```

---

## RAG Pipeline (Retrieval-Augmented Generation)

CiteGuard uses a full **3-phase RAG architecture** to verify citations against evidence:

```mermaid
flowchart TD
    subgraph INDEXING ["📚 INDEXING — resolver.py + rag_store.py"]
        I1["Fetch full paper text\nar5iv.org HTML or S2 open-access PDF"]
        I2["Chunk into 3-sentence\noverlapping windows"]
        I3["Embed each chunk\nall-MiniLM-L12-v2 → 384-dim vector"]
        I4["Store in ChromaDB\ncache/rag_index/ — persistent"]
        I1 --> I2 --> I3 --> I4
    end

    subgraph RETRIEVAL ["🔍 RETRIEVAL — aligner.py"]
        R1["Embed the claim\n(same bi-encoder)"]
        R2["ANN cosine search\nChromaDB → top-50 passages"]
        R3["Cross-encoder re-rank\nms-marco-MiniLM → top-10"]
        R1 --> R2 --> R3
    end

    subgraph GENERATION ["🧠 GENERATION — verifier.py"]
        G1["NLI inference\nDeBERTa-Large-MNLI\n(premise=passage, hypothesis=claim)"]
        G2["Verdict\nSUPPORTS / NEUTRAL\nCONTRADICTS / NO_EVIDENCE"]
        G1 --> G2
    end

    INDEXING --> RETRIEVAL --> GENERATION
```

### RAG vs Previous (abstract-only)

| | Before | After (RAG) |
|---|---|---|
| Evidence corpus | Abstract only (~200 words) | Full paper body (~8,000–15,000 words) |
| Chunks per paper | 3–5 | 400–800 |
| Vector store | In-memory (re-computed every run) | ChromaDB (persistent, index once) |
| Speed (2nd run) | Slow (re-embeds) | Fast (query only) |

---

## Pipeline Overview

| Layer | Name | Tools / Models |
|---|---|---|
| **1** | Document Extraction | `pdfplumber`, `PyMuPDF` |
| **2a** | Citation Parsing | Regex + spaCy NER |
| **2b** | Claim Extraction | Sentence tokenisation |
| **3** | Metadata Resolution | Semantic Scholar API, CrossRef, arXiv |
| **3b** ✨ | Full-text Enrichment | ar5iv.org HTML, S2 open-access PDF |
| **RAG** ✨ | Vector Store | ChromaDB (persistent, `cache/rag_index/`) |
| **4** | Evidence Retrieval | Bi-encoder (`all-MiniLM-L12-v2`) |
| **5** | Re-ranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| **6** | NLI Verification | DeBERTa-Large-MNLI |
| **7** | Report Generation | Jinja2 → JSON + HTML |

---

## Verdict Labels

| Label | Meaning |
|---|---|
| ✅ **SUPPORTS** | Cited paper's content entails the claim |
| ⚠️ **NEUTRAL** | Weak or tangential relationship |
| ❌ **CONTRADICTS** | Cited paper's content contradicts the claim |
| 🔍 **NO_EVIDENCE** | Paper not found or abstract unavailable |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/JAGADISH-JSK/CiteGuard-Citation-Verification.git
cd CiteGuard-Citation-Verification

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# (edit .env if you have a Semantic Scholar API key)

# 5a. Run on a paper (CLI)
python main.py input/your_paper.pdf

# 5b. Run the web dashboard
uvicorn api:app --reload --port 8000
# Open http://127.0.0.1:8000
```

---

## Project Structure

```
citeguard/
├── modules/
│   ├── extractor.py         # Layer 1  – PDF parsing
│   ├── citation_parser.py   # Layer 2a – citation detection
│   ├── claim_extractor.py   # Layer 2b – claim sentence extraction
│   ├── resolver.py          # Layer 3  – API metadata lookup + full-text fetch ✨
│   ├── rag_store.py         # RAG Store – ChromaDB vector index ✨ NEW
│   ├── aligner.py           # Layers 4+5 – bi-encoder + cross-encoder (RAG-upgraded) ✨
│   ├── verifier.py          # Layer 6  – NLI fact verification
│   └── reporter.py          # Layer 7  – report generation
├── static/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Styles
│   └── app.js               # Frontend logic
├── docs/
│   └── screenshots/         # Project screenshots
├── input/                   # Drop your PDFs here
├── output/                  # Reports written here
├── cache/
│   ├── *.json               # API response cache
│   └── rag_index/           # ChromaDB persistent vector store ✨ NEW
├── tests/                   # Unit tests
├── config.py                # Central configuration
├── main.py                  # CLI pipeline entry point
├── api.py                   # FastAPI web server
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
| `RAG_FULL_TEXT` | `true` | Fetch full paper text (set `false` for abstract-only mode) ✨ |
| `RAG_MAX_CHARS` | `120000` | Max characters of full text fed to chunker ✨ |
| `S2_API_KEY` | _(empty)_ | Semantic Scholar API key |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| PDF Parsing | pdfplumber · PyMuPDF |
| NLP | sentence-transformers |
| NLI Model | DeBERTa-Large-MNLI (HuggingFace) |
| **RAG Vector Store** | **ChromaDB (persistent)** ✨ |
| **Full-text Source** | **ar5iv.org · S2 open-access PDF** ✨ |
| Metadata APIs | Semantic Scholar · CrossRef · arXiv |
| Frontend | HTML · CSS · Vanilla JS |
| Report | Jinja2 · JSON · HTML |

---

## IBM Watson / Granite Integration

To use IBM's embedding models, update `EMBEDDING_MODEL` in `.env`:

```env
EMBEDDING_MODEL=ibm-granite/granite-embedding-278m-multilingual
```

For full IBM watsonx integration, replace the sentence-transformers calls
in `modules/aligner.py` with the watsonx.ai embeddings API.

---

## License

MIT License — free to use, modify and distribute.
