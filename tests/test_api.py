"""HTTP-level tests.

The app's real lifespan downloads a 130MB embedding model and a cross-encoder.
These tests replace it with a no-op and inject the stub components instead, so
the whole HTTP surface — SSE framing, upload handling, error codes — is
exercised in under a second with no network.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest
from conftest import ScriptedLLM
from fastapi.testclient import TestClient

from app import main as main_module
from app.sessions import SessionManager
from ragcore.pipeline import RAGPipeline


@pytest.fixture
def client(store, embedder, retriever, settings, monkeypatch):
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    monkeypatch.setattr(main_module.app.router, "lifespan_context", noop_lifespan)

    state = main_module.state
    state.store = store
    state.embedder = embedder
    state.reranker = retriever.reranker
    state.pipeline = RAGPipeline(retriever, ScriptedLLM("Grounded [1] claim [2]."), settings)
    state.sessions = SessionManager(settings, embedder)
    state.error = None

    with TestClient(main_module.app) as c:
        yield c

    for attr in ("store", "embedder", "reranker", "pipeline", "sessions"):
        setattr(state, attr, None)


def parse_sse(text: str) -> list[tuple[str, object]]:
    out = []
    for frame in text.split("\n\n"):
        event, data = None, []
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data.append(line[6:])
        if event and data:
            out.append((event, json.loads("\n".join(data))))
    return out


class TestHealth:
    def test_reports_ok_when_ready(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["index_ready"] is True
        assert body["stats"]["chunks"] > 0

    def test_reports_degraded_without_a_pipeline(self, client):
        main_module.state.pipeline = None
        body = client.get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["index_ready"] is False


class TestDocuments:
    def test_lists_the_corpus(self, client):
        body = client.get("/api/documents").json()
        assert body["count"] == 3
        assert all({"doc_id", "title", "source"} <= set(d) for d in body["documents"])


class TestChat:
    def test_streams_sse_frames(self, client):
        resp = client.post("/api/chat", json={"question": "multi-head attention"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        frames = parse_sse(resp.text)
        names = [name for name, _ in frames]
        assert names[0] == "query"
        assert "sources" in names
        assert "token" in names
        assert names[-1] == "done"

    def test_done_frame_carries_the_full_payload(self, client):
        resp = client.post("/api/chat", json={"question": "multi-head attention"})
        done = dict(parse_sse(resp.text))["done"]
        assert {"answer", "citations", "grounded", "search_query", "timings_ms"} <= set(done)
        assert done["answer"]
        assert done["timings_ms"]["total_ms"] > 0

    def test_sources_carry_retrieval_provenance(self, client):
        resp = client.post("/api/chat", json={"question": "attention heads"})
        sources = dict(parse_sse(resp.text))["sources"]
        assert sources
        for s in sources:
            assert s["retriever"] in {"dense", "sparse", "both"}
            assert {"n", "title", "chunk_id", "snippet"} <= set(s)

    def test_accepts_history(self, client):
        resp = client.post(
            "/api/chat",
            json={
                "question": "How many does it use?",
                "history": [
                    {"role": "user", "content": "Tell me about the transformer"},
                    {"role": "assistant", "content": "It is an architecture."},
                ],
            },
        )
        assert resp.status_code == 200

    def test_rejects_an_empty_question(self, client):
        assert client.post("/api/chat", json={"question": ""}).status_code == 422

    def test_rejects_an_overlong_question(self, client):
        assert client.post("/api/chat", json={"question": "x" * 3000}).status_code == 422

    def test_returns_503_when_not_ready(self, client):
        main_module.state.pipeline = None
        assert client.post("/api/chat", json={"question": "hi"}).status_code == 503


class TestUpload:
    def test_indexes_a_markdown_file(self, client):
        resp = client.post(
            "/api/upload",
            files={"file": ("notes.md", b"# Notes\n\n" + b"Some content here. " * 40, "text/markdown")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"]
        assert body["chunks"] > 0
        assert body["documents"] == ["notes"]

    def test_uploaded_content_becomes_retrievable(self, client):
        """The property that matters: after upload, the document is reachable
        through the normal chat endpoint using the returned session id."""
        upload = client.post(
            "/api/upload",
            files={
                "file": (
                    "secret.md",
                    b"# Internal\n\n" + b"Project ZEPHYRQUARK reached 91.4 percent accuracy. " * 12,
                    "text/markdown",
                )
            },
        ).json()

        resp = client.post(
            "/api/chat",
            json={"question": "What accuracy did ZEPHYRQUARK reach?", "session_id": upload["session_id"]},
        )
        sources = dict(parse_sse(resp.text))["sources"]
        assert any("ZEPHYRQUARK" in s["snippet"] for s in sources)

    def test_sources_show_the_uploaded_filename_not_a_temp_path(self, client):
        """Uploads are staged to a temp file so the parsers can read a real
        path. That temp name must not reach the UI — it is displayed on every
        source card the document produces."""
        upload = client.post(
            "/api/upload",
            files={
                "file": (
                    "My Research Paper.md",
                    b"# Findings\n\n" + b"The KRILLVANE metric reached 88.2 percent. " * 14,
                    "text/markdown",
                )
            },
        ).json()

        resp = client.post(
            "/api/chat",
            json={"question": "What did KRILLVANE reach?", "session_id": upload["session_id"]},
        )
        sources = dict(parse_sse(resp.text))["sources"]
        mine = [s for s in sources if "KRILLVANE" in s["snippet"]]
        assert mine, "uploaded document was not retrieved"
        for s in mine:
            assert s["source"] == "My Research Paper.md"
            assert not s["source"].startswith("tmp")

    def test_uploads_are_isolated_between_sessions(self, client):
        """A public demo requirement: one visitor's upload must not leak into
        another visitor's retrieval."""
        first = client.post(
            "/api/upload",
            files={"file": ("a.md", b"# A\n\n" + b"Token QUOKKAVERSE appears here. " * 15, "text/markdown")},
        ).json()
        second = client.post(
            "/api/upload",
            files={"file": ("b.md", b"# B\n\n" + b"Different content entirely here. " * 15, "text/markdown")},
        ).json()
        assert first["session_id"] != second["session_id"]

        resp = client.post(
            "/api/chat",
            json={"question": "QUOKKAVERSE", "session_id": second["session_id"]},
        )
        sources = dict(parse_sse(resp.text))["sources"]
        assert not any("QUOKKAVERSE" in s["snippet"] for s in sources)

    def test_reuses_a_session_across_uploads(self, client):
        first = client.post(
            "/api/upload",
            files={"file": ("a.md", b"# A\n\n" + b"Content one here now. " * 15, "text/markdown")},
        ).json()
        second = client.post(
            "/api/upload",
            data={"session_id": first["session_id"]},
            files={"file": ("b.md", b"# B\n\n" + b"Content two here now. " * 15, "text/markdown")},
        ).json()
        assert second["session_id"] == first["session_id"]
        assert second["documents"] == ["a", "b"]
        assert second["total_chunks"] > second["chunks"]

    def test_rejects_an_unsupported_type(self, client):
        resp = client.post("/api/upload", files={"file": ("x.exe", b"MZ\x00\x00", "application/octet-stream")})
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_rejects_an_oversized_file(self, client, settings):
        blob = b"x" * (settings.max_upload_mb * 1024 * 1024 + 10)
        resp = client.post("/api/upload", files={"file": ("big.txt", blob, "text/plain")})
        assert resp.status_code == 413

    def test_rejects_an_empty_document(self, client):
        resp = client.post("/api/upload", files={"file": ("empty.txt", b"   ", "text/plain")})
        assert resp.status_code == 400

    def test_enforces_the_per_session_document_cap(self, client, settings):
        session_id = None
        for i in range(settings.max_session_docs):
            data = {"session_id": session_id} if session_id else {}
            resp = client.post(
                "/api/upload",
                data=data,
                files={"file": (f"f{i}.md", f"# D{i}\n\n".encode() + b"Body content here. " * 15, "text/markdown")},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]

        resp = client.post(
            "/api/upload",
            data={"session_id": session_id},
            files={"file": ("over.md", b"# Over\n\n" + b"Body content here. " * 15, "text/markdown")},
        )
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"].lower()


class TestSessionClear:
    def test_clears_uploads(self, client):
        upload = client.post(
            "/api/upload",
            files={"file": ("a.md", b"# A\n\n" + b"Token BANDICOOTLY here now. " * 15, "text/markdown")},
        ).json()
        client.post("/api/session/clear", json={"session_id": upload["session_id"]})

        resp = client.post(
            "/api/chat",
            json={"question": "BANDICOOTLY", "session_id": upload["session_id"]},
        )
        sources = dict(parse_sse(resp.text))["sources"]
        assert not any("BANDICOOTLY" in s["snippet"] for s in sources)


class TestDebugRetrieve:
    def test_returns_retrieval_without_generation(self, client):
        body = client.get("/api/debug/retrieve", params={"q": "attention heads"}).json()
        assert body["query"] == "attention heads"
        assert body["results"]
        assert "rerank_score" in body["results"][0]

    def test_ablation_flags_are_honoured(self, client):
        dense_only = client.get(
            "/api/debug/retrieve", params={"q": "attention", "sparse": False}
        ).json()
        assert all(r["sparse_rank"] is None for r in dense_only["results"])

        sparse_only = client.get(
            "/api/debug/retrieve", params={"q": "attention", "dense": False}
        ).json()
        assert all(r["dense_rank"] is None for r in sparse_only["results"])

    def test_respects_k(self, client):
        body = client.get("/api/debug/retrieve", params={"q": "attention", "k": 2}).json()
        assert len(body["results"]) <= 2


class TestUI:
    def test_serves_the_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        # Assert on structure the app actually depends on, not on display
        # copy — a wording or capitalisation change is not a regression.
        assert resp.headers["content-type"].startswith("text/html")
        assert '<div class="app">' in resp.text
        assert '/static/app.js' in resp.text
        assert '/static/styles.css' in resp.text

    def test_index_page_wires_up_every_element_the_client_needs(self, client):
        """app.js looks these up by id at load. A markup change that drops one
        breaks the UI silently — the page renders, the feature just stops."""
        html = client.get("/").text
        for element_id in (
            "messages", "input", "send", "status", "status-text",
            "debug-toggle", "corpus-list", "doc-count", "index-stats",
            "file-input", "upload-label", "upload-text", "uploaded-list",
            "sources-body", "close-sources", "suggestions",
        ):
            assert f'id="{element_id}"' in html, f"missing #{element_id}"

    def test_serves_static_assets(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/styles.css").status_code == 200
