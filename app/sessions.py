"""Per-visitor upload sessions.

Design constraint: this is a public demo, so anything a visitor uploads must be
invisible to every other visitor and must not persist. Uploaded documents are
therefore chunked and indexed into an **in-memory** `ChunkStore` keyed by an
opaque session id, never written to disk, and evicted on a TTL.

The uploaded store is a full `ChunkStore` — same chunking, same embeddings,
same BM25 — so an uploaded document is retrieved by the same code path as the
preloaded corpus and competes with it on equal terms.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from ragcore.chunking import chunk_documents
from ragcore.config import Settings
from ragcore.store import ChunkStore
from ragcore.types import Chunk, Document


@dataclass
class Session:
    session_id: str
    created_at: float
    last_seen: float
    chunks: list[Chunk] = field(default_factory=list)
    doc_titles: list[str] = field(default_factory=list)
    store: ChunkStore | None = None


class SessionManager:
    def __init__(self, settings: Settings, embedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def new_session(self) -> str:
        session_id = secrets.token_urlsafe(16)
        now = time.time()
        with self._lock:
            self._sessions[session_id] = Session(session_id, now, now)
        return session_id

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = time.time()
            return session

    def evict_expired(self) -> int:
        cutoff = time.time() - self.settings.session_ttl_minutes * 60
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in stale:
                del self._sessions[sid]
        return len(stale)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- indexing ------------------------------------------------------------

    def add_document(self, session_id: str, document: Document) -> dict[str, object]:
        """Chunk, embed and index one uploaded document into a session.

        The whole session store is rebuilt rather than appended to. At the
        scale a single visitor can upload (capped at `max_session_docs`) a
        rebuild is milliseconds, and it keeps BM25's corpus statistics — IDF
        and average document length — correct, which an append could not.
        """
        session = self.get(session_id)
        if session is None:
            raise KeyError("unknown or expired session")

        if len(session.doc_titles) >= self.settings.max_session_docs:
            raise ValueError(
                f"Upload limit reached ({self.settings.max_session_docs} documents "
                "per session). Clear your uploads to add more."
            )

        new_chunks = chunk_documents(
            [document],
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
            min_tokens=self.settings.chunk_min_tokens,
        )
        if not new_chunks:
            raise ValueError("No extractable text found in that file.")

        with self._lock:
            session.chunks.extend(new_chunks)
            session.doc_titles.append(document.title)
            session.store = ChunkStore.build(
                session.chunks,
                self.embedder,
                embedding_model=self.settings.embedding_model,
            )
            session.last_seen = time.time()

        return {
            "title": document.title,
            "chunks": len(new_chunks),
            "total_chunks": len(session.chunks),
            "documents": list(session.doc_titles),
        }

    def store_for(self, session_id: str | None) -> ChunkStore | None:
        session = self.get(session_id)
        return session.store if session else None
