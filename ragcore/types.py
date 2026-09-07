"""Core data types shared across ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    title: str
    text: str
    source: str = ""
    """URL or filename the document came from."""
    authors: str = ""
    year: str = ""
    page_map: list[tuple[int, int]] = field(default_factory=list)
    """(char_offset, page_number) breakpoints, ascending by offset. Lets a
    chunk report the page it came from without re-parsing the PDF."""

    def page_for_offset(self, offset: int) -> int:
        page = 1
        for start, num in self.page_map:
            if start <= offset:
                page = num
            else:
                break
        return page


@dataclass(slots=True)
class Chunk:
    """An indexed unit of retrieval."""

    chunk_id: str
    doc_id: str
    text: str
    """The text that gets embedded and BM25-indexed. Includes the section
    heading prefix (see `chunking.py`) so that a chunk carries its own
    context — 'we use a batch size of 512' is useless without knowing which
    experiment it belongs to."""

    body: str
    """The chunk text *without* the heading prefix, for display."""

    title: str = ""
    source: str = ""
    section: str = ""
    page: int = 1
    ordinal: int = 0
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def retriever_path(self) -> str:
        """Which retriever(s) surfaced this chunk. Shown in the UI's debug
        panel — being able to point at 'BM25 found this, vectors missed it'
        is the clearest way to justify the hybrid design."""
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.sparse_rank is not None:
            return "sparse"
        return "unknown"


@dataclass(slots=True)
class Citation:
    marker: int
    """The [n] the model wrote."""
    chunk_id: str
    title: str
    source: str
    section: str
    page: int
    snippet: str
    retriever_path: str = "unknown"
    rerank_score: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    grounded: bool
    """False when nothing cleared `min_rerank_score` — the pipeline then
    refuses rather than answering from unrelated context."""
    timings_ms: dict[str, float] = field(default_factory=dict)
