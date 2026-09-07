"""Central configuration.

Every knob that affects retrieval quality lives here rather than being scattered
through the code, so that the evaluation harness can sweep them and so that a
deployment can be re-tuned without a code change.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Paths -------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    """Where the built index and the raw corpus live."""

    # ---- Models ------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    """384-dim bi-encoder. 33M params, runs on CPU in ~10ms/chunk.

    Chosen over `all-MiniLM-L6-v2` because BGE is meaningfully stronger on
    BEIR retrieval benchmarks at the same size, and over `bge-base` because
    the +2 points of nDCG were not worth 3x the latency on a free CPU host.
    """

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """Cross-encoder used to re-score the fused candidate list.

    A cross-encoder reads (query, passage) jointly instead of comparing two
    independently-computed vectors, so it resolves the cases a bi-encoder
    cannot: negation, and queries whose answer depends on a term that is
    semantically close but factually wrong.
    """

    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    """BGE models are trained with an asymmetric instruction prefix on the
    query side only. Omitting it costs several points of recall — a subtle
    bug that is easy to ship and hard to notice."""

    # ---- Chunking ----------------------------------------------------------
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 48
    """Chunks below this are merged into their neighbour rather than indexed;
    stray fragments (page numbers, orphan headings) otherwise pollute BM25."""

    # ---- Retrieval ---------------------------------------------------------
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    """Reciprocal Rank Fusion damping constant. 60 is the value from the
    original Cormack et al. paper and is remarkably insensitive in practice."""

    rerank_candidates: int = 30
    final_top_k: int = 6
    """How many chunks actually reach the prompt. Beyond ~6 the model starts
    losing the middle of the context and answer quality drops."""

    min_rerank_score: float = -6.0
    """Groundedness gate. ms-marco cross-encoders emit unbounded logits;
    below roughly -6 the passage is unrelated to the query. If nothing clears
    the bar we refuse to answer instead of hallucinating from noise."""

    # ---- Generation --------------------------------------------------------
    llm_provider: str = "groq"
    """One of: groq | gemini | openai | echo. `echo` is a dependency-free
    stub used by the tests and by anyone who wants to see retrieval working
    before signing up for an API key."""

    llm_model: str = "openai/gpt-oss-120b"
    """Groq's model catalogue includes Enterprise-only entries that a free-tier
    key cannot call — `llama-3.3-70b-versatile` among them, which fails at
    generation time with a 404 that looks like an application bug. This default
    is verified available on the free tier. Run `python scripts/list_models.py`
    to see exactly what your own key can call."""
    llm_api_key: str = ""
    llm_base_url: str = ""
    """Only needed for provider=openai, e.g. a local Ollama or vLLM server."""

    llm_temperature: float = 0.1
    llm_max_tokens: int = 900
    llm_timeout_s: float = 60.0

    # ---- Uploads -----------------------------------------------------------
    max_upload_mb: int = 20
    max_session_docs: int = 10
    session_ttl_minutes: int = 60
    """Uploaded documents are indexed into a per-session in-memory store and
    evicted on a timer. Nothing a visitor uploads is written to disk or made
    visible to another visitor."""

    # ---- Service -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 7860  # Hugging Face Spaces convention
    cors_origins: str = "*"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"


settings = Settings()
