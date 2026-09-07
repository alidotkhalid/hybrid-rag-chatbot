# syntax=docker/dockerfile:1
#
# Two stages. The first resolves and installs Python dependencies; the second
# copies only the resulting site-packages plus the application. The build
# toolchain (gcc, headers) never reaches the runtime image, which cuts roughly
# 400MB and removes a compiler from the deployed container.

# ── Stage 1: dependencies ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CPU-only torch. The default wheel pulls ~5GB of CUDA libraries that are dead
# weight on a CPU host — this is the single biggest lever on image size.
RUN pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

# Hugging Face Spaces runs containers as uid 1000 and mounts nothing writable
# except the home directory. Model caches must therefore live under $HOME or
# the first download fails with a permission error — a confusing crash that is
# entirely avoidable by setting these up front.
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/sentence-transformers \
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

COPY --from=builder --chown=app:app /root/.local /home/app/.local

WORKDIR /home/app/service
COPY --chown=app:app ragcore/ ./ragcore/
COPY --chown=app:app app/ ./app/
COPY --chown=app:app evaluation/ ./evaluation/
COPY --chown=app:app data/index/ ./data/index/
COPY --chown=app:app pyproject.toml README.md ./

# Bake the models into the image rather than downloading them on first request.
# A cold start that has to fetch 150MB of weights takes 30-60s, during which
# every request times out; this moves that cost to build time, where it is
# paid once.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512); \
print('models cached')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:7860/api/health',timeout=4).json()['index_ready'] else 1)"

# One worker on purpose: each worker loads its own copy of the embedding and
# reranker models (~600MB resident), and the free tier has neither the RAM nor
# the CPU for two. Concurrency comes from the thread pool inside the process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]
