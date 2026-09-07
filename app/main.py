"""FastAPI service.

Endpoints:
    GET  /                  the chat UI
    GET  /api/health        readiness + index statistics
    GET  /api/documents     what is in the preloaded corpus
    POST /api/chat          Server-Sent Events: query, sources, tokens, done
    POST /api/upload        index a document into the caller's session
    POST /api/session/clear drop a session's uploads
    GET  /api/debug/retrieve  retrieval only, no generation (for tuning)

**Why SSE and not WebSockets.** The stream is one-directional and short-lived,
SSE reconnects and buffers without any client-side protocol work, and it
survives the HTTP proxies in front of most free hosting tiers. WebSockets would
add a stateful connection for no gain.

**Threading.** Model inference is synchronous and CPU-bound. The generator that
produces the stream is therefore run in a worker thread and its items are
handed to the event loop through a queue, so one slow request cannot block the
whole server. The alternative — `def` endpoints — would serialise the entire
response including the network wait on the LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ragcore import __version__
from ragcore.config import settings
from ragcore.llm import LLMConfigError, build_llm
from ragcore.loaders import SUPPORTED_SUFFIXES, load_file
from ragcore.pipeline import RAGPipeline
from ragcore.reranker import CrossEncoderReranker, IdentityReranker
from ragcore.retriever import HybridRetriever
from ragcore.store import ChunkStore

from .schemas import ChatRequest, HealthOut
from .sessions import SessionManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    """Everything loaded once at startup and shared across requests."""

    store: ChunkStore | None = None
    embedder = None
    reranker = None
    pipeline: RAGPipeline | None = None
    sessions: SessionManager | None = None
    error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and the index once, before serving.

    Everything here is fail-soft: a missing index or an unreachable LLM
    degrades the service rather than preventing it from starting, so that
    /api/health can report *why* it is unhealthy instead of the container
    crash-looping with the reason buried in a log.
    """
    log.info("loading index from %s", settings.index_dir)
    try:
        state.store = ChunkStore.load(settings.index_dir)
        log.info("index: %s", state.store.stats())
    except Exception as exc:  # noqa: BLE001
        state.error = f"index unavailable: {exc}"
        log.error(state.error)

    try:
        from ragcore.embedder import build_embedder

        log.info("loading embedder %s", settings.embedding_model)
        state.embedder = build_embedder(settings)
        if state.store is not None:
            state.store.assert_compatible_with(state.embedder)
    except Exception as exc:  # noqa: BLE001
        state.error = f"embedder unavailable: {exc}"
        log.error(state.error)

    try:
        log.info("loading reranker %s", settings.reranker_model)
        state.reranker = CrossEncoderReranker(settings.reranker_model)
    except Exception as exc:  # noqa: BLE001
        log.warning("reranker unavailable (%s); falling back to fusion order", exc)
        state.reranker = IdentityReranker()

    try:
        llm = build_llm(settings)
    except LLMConfigError as exc:
        from ragcore.llm import EchoLLM

        log.warning("LLM not configured (%s); running retrieval-only", exc)
        llm = EchoLLM()

    if state.store is not None and state.embedder is not None:
        retriever = HybridRetriever(state.store, state.embedder, state.reranker, settings)
        state.pipeline = RAGPipeline(retriever, llm, settings)
        state.sessions = SessionManager(settings, state.embedder)
        log.info("ready — %s chunks, llm=%s", state.store.size, getattr(llm, "model", "?"))

    evictor = asyncio.create_task(_evict_loop())
    try:
        yield
    finally:
        evictor.cancel()


async def _evict_loop() -> None:
    while True:
        await asyncio.sleep(300)
        if state.sessions:
            n = state.sessions.evict_expired()
            if n:
                log.info("evicted %d expired session(s)", n)


app = FastAPI(
    title="Hybrid RAG Chatbot",
    version=__version__,
    description="Hybrid retrieval (BM25 + dense) with cross-encoder reranking "
    "and citation-validated generation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- SSE plumbing -----------------------------------------------------------

_SENTINEL = object()


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_in_thread(make_generator):
    """Run a blocking generator in a thread, yielding its items as they arrive.

    `asyncio.to_thread` would only work for a single value; this keeps the
    stream incremental so the first token reaches the browser as soon as the
    model produces it.
    """
    q: queue.Queue = queue.Queue(maxsize=64)

    def worker() -> None:
        try:
            for item in make_generator():
                q.put(item)
        except Exception as exc:  # noqa: BLE001
            q.put(("error", str(exc)))
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _SENTINEL:
            return
        yield item


# --- Routes -----------------------------------------------------------------


@app.get("/api/health", response_model=HealthOut)
async def health() -> HealthOut:
    ready = state.pipeline is not None
    return HealthOut(
        status="ok" if ready else "degraded",
        version=__version__,
        index_ready=ready,
        llm_provider=settings.llm_provider,
        llm_model=getattr(getattr(state.pipeline, "llm", None), "model", "n/a"),
        stats=(state.store.stats() if state.store else {"error": state.error or "no index"}),
    )


@app.get("/api/documents")
async def documents() -> dict:
    if state.store is None:
        raise HTTPException(503, state.error or "index not loaded")
    return {"documents": state.store.documents(), "count": state.store.doc_count}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if state.pipeline is None:
        raise HTTPException(503, state.error or "service not ready")

    pipeline = state.pipeline
    history = [(t.role, t.content) for t in req.history]
    extra_store = state.sessions.store_for(req.session_id) if state.sessions else None

    async def event_stream():
        try:
            async for event, payload in _stream_in_thread(
                lambda: pipeline.stream_answer(
                    req.question, history=history, extra_store=extra_store
                )
            ):
                if event == "done":
                    yield _sse(
                        "done",
                        {
                            "answer": payload.answer,
                            "citations": [asdict(c) for c in payload.citations],
                            "grounded": payload.grounded,
                            "search_query": payload.search_query,
                            "timings_ms": {
                                k: round(v, 1) for k, v in payload.timings_ms.items()
                            },
                        },
                    )
                else:
                    yield _sse(event, payload)
        except Exception as exc:  # noqa: BLE001
            log.exception("chat stream failed")
            yield _sse("error", str(exc))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx-style proxies in front of most PaaS hosts
            # buffer the whole response and the stream arrives all at once.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), session_id: str = Form("")):
    if state.sessions is None:
        raise HTTPException(503, state.error or "service not ready")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix or 'none'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB.")

    if not session_id or state.sessions.get(session_id) is None:
        session_id = state.sessions.new_session()

    # Parsers need a real path; the temp file is removed immediately after.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        document = await asyncio.to_thread(load_file, tmp_path, Path(file.filename).stem)
        # The parsers need a real path, so the upload is written to a temp file
        # first — and `load_file` faithfully records the path it read, which is
        # something like "tmpfimp_7gc.pdf". Restore the name the visitor
        # actually uploaded, because this string is displayed on every source
        # card the document produces.
        document.source = Path(file.filename).name
        info = await asyncio.to_thread(state.sessions.add_document, session_id, document)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("upload failed")
        raise HTTPException(500, f"Could not process that file: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"session_id": session_id, **info}


@app.post("/api/session/clear")
async def clear_session(payload: dict) -> dict:
    session_id = payload.get("session_id")
    if state.sessions and session_id:
        state.sessions.clear(session_id)
    return {"ok": True}


@app.get("/api/debug/retrieve")
async def debug_retrieve(
    q: str,
    k: int = 6,
    dense: bool = True,
    sparse: bool = True,
    rerank: bool = True,
) -> dict:
    """Retrieval without generation.

    This is the endpoint used to tune the system: it makes the effect of
    turning each stage on and off directly observable, and it is how the
    ablation numbers in the README were produced.
    """
    if state.pipeline is None:
        raise HTTPException(503, state.error or "service not ready")
    result = state.pipeline.retriever.retrieve(
        q, top_k=k, use_dense=dense, use_sparse=sparse, use_rerank=rerank
    )
    return {
        "query": q,
        "grounded": result.grounded,
        "timings_ms": {k_: round(v, 2) for k_, v in result.timings_ms.items()},
        "results": [
            {
                "title": sc.chunk.title,
                "section": sc.chunk.section,
                "page": sc.chunk.page,
                "retriever": sc.retriever_path,
                "dense_rank": sc.dense_rank,
                "sparse_rank": sc.sparse_rank,
                "fused_score": round(sc.fused_score, 5) if sc.fused_score else None,
                "rerank_score": round(sc.rerank_score, 3) if sc.rerank_score else None,
                "text": sc.chunk.body[:400],
            }
            for sc in result.chunks
        ],
    }


# --- Static UI --------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
