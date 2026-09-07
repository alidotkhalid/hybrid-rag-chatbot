"""Request and response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    session_id: str | None = None


class SourceOut(BaseModel):
    n: int
    title: str
    section: str = ""
    page: int = 1
    source: str = ""
    chunk_id: str
    retriever: str = "unknown"
    rerank_score: float | None = None
    snippet: str = ""


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    title: str
    source: str = ""
    section: str = ""
    page: int = 1
    snippet: str = ""
    retriever_path: str = "unknown"
    rerank_score: float | None = None


class HealthOut(BaseModel):
    status: str
    version: str
    index_ready: bool
    llm_provider: str
    llm_model: str
    stats: dict = Field(default_factory=dict)


class UploadOut(BaseModel):
    session_id: str
    title: str
    chunks: int
    total_chunks: int
    documents: list[str]
