"""
CiteGuard – Central configuration.
All tuneable constants live here; override via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = BASE_DIR / "cache"
TMPL_DIR   = BASE_DIR / "templates"

# ── Embedding model ───────────────────────────────────────────────
# Swap to "ibm-granite/granite-embedding-278m-multilingual" for IBM stack
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L12-v2"
)

# ── NLI / fact-verification model ─────────────────────────────────
NLI_MODEL = os.getenv(
    "NLI_MODEL",
    "cross-encoder/nli-deberta-v3-large"
)

# ── Evidence retrieval ────────────────────────────────────────────
TOP_K_RETRIEVAL  = int(os.getenv("TOP_K_RETRIEVAL", 50))   # bi-encoder candidates
TOP_K_RERANK     = int(os.getenv("TOP_K_RERANK", 10))      # after cross-encoder
NLI_THRESHOLD    = float(os.getenv("NLI_THRESHOLD", 0.5))  # min entailment score

# ── External APIs ─────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
CROSSREF_API         = "https://api.crossref.org/works"
ARXIV_API            = "http://export.arxiv.org/api/query"

# Optional: Semantic Scholar API key (higher rate-limit)
S2_API_KEY = os.getenv("S2_API_KEY", "")

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SENTENCES = int(os.getenv("CHUNK_SENTENCES", 3))  # sentences per passage chunk

# ── Logging ───────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
