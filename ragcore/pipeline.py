"""End-to-end answer pipeline: question in, streamed grounded answer out."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .citations import validate_and_rewrite
from .config import Settings
from .llm import LLM, LLMError
from .prompts import (
    NO_ANSWER_MESSAGE,
    SYSTEM_PROMPT,
    build_condense_prompt,
    build_user_prompt,
)
from .retriever import HybridRetriever
from .store import ChunkStore
from .types import Citation, ScoredChunk

log = logging.getLogger(__name__)

# Cheap test for whether a follow-up actually depends on the history. Running
# the condense step on every turn would add an LLM round-trip to every request
# for no benefit — most follow-ups in practice are already self-contained.
_ANAPHORA = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|there|"
    r"the (?:same|former|latter|above|paper|authors?|approach|method|model))\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class AnswerResult:
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    grounded: bool = True
    search_query: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # -- query understanding -------------------------------------------------

    def _needs_condensing(self, question: str, history: list[tuple[str, str]]) -> bool:
        if not history:
            return False
        # Short questions ("why?", "and for BERT?") are almost always
        # elliptical; long ones almost always stand alone.
        if len(question.split()) <= 4:
            return True
        return bool(_ANAPHORA.search(question))

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Rewrite a follow-up into a standalone search query.

        Retrieval has no memory: "what about the decoder?" embeds to something
        close to nothing useful. Condensing against the last few turns is what
        makes multi-turn RAG work at all. Failure here is non-fatal — we fall
        back to the raw question rather than dropping the request.
        """
        if not self._needs_condensing(question, history):
            return question
        try:
            prompt = build_condense_prompt(history[-4:], question)
            rewritten = "".join(
                self.llm.stream(
                    "You rewrite follow-up questions into standalone ones. "
                    "Output only the rewritten question.",
                    prompt,
                )
            ).strip()
        except LLMError as exc:  # pragma: no cover - network path
            log.warning("condense failed, using raw question: %s", exc)
            return question
        rewritten = rewritten.strip().strip('"')
        # Guard against a model that decides to answer instead of rewrite.
        if not rewritten or len(rewritten) > 400:
            return question
        return rewritten

    # -- main entry point ----------------------------------------------------

    def stream_answer(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        extra_store: ChunkStore | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Yield ('event_name', payload) tuples for the transport to serialise.

        Events, in order:
            ('query',     the standalone search query actually used)
            ('sources',   the retrieved passages, sent before generation so
                          the UI can render evidence while tokens stream)
            ('token',     a text delta)  — repeated
            ('done',      AnswerResult with validated citations)
            ('error',     message)       — terminal, in place of 'done'
        """
        history = history or []
        result = AnswerResult()
        t_start = time.perf_counter()

        # 1. Resolve the question against conversation history.
        t0 = time.perf_counter()
        search_query = self.condense(question, history)
        result.search_query = search_query
        result.timings_ms["condense_ms"] = (time.perf_counter() - t0) * 1000
        yield "query", search_query

        # 2. Retrieve.
        retrieval = self.retriever.retrieve(search_query, extra_store=extra_store)
        result.retrieved = retrieval.chunks
        result.grounded = retrieval.grounded
        result.timings_ms.update(retrieval.timings_ms)

        yield "sources", [
            {
                "n": i,
                "title": sc.chunk.title,
                "section": sc.chunk.section,
                "page": sc.chunk.page,
                "source": sc.chunk.source,
                "chunk_id": sc.chunk.chunk_id,
                "retriever": sc.retriever_path,
                "rerank_score": sc.rerank_score,
                "snippet": sc.chunk.body[:280],
            }
            for i, sc in enumerate(retrieval.chunks, start=1)
        ]

        # 3. Refuse rather than hallucinate when nothing cleared the gate.
        if not retrieval.grounded or not retrieval.chunks:
            result.answer = NO_ANSWER_MESSAGE
            result.grounded = False
            result.timings_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
            yield "token", NO_ANSWER_MESSAGE
            yield "done", result
            return

        # 4. Generate.
        raw = []
        t0 = time.perf_counter()
        first_token_at: float | None = None
        try:
            for delta in self.llm.stream(
                SYSTEM_PROMPT, build_user_prompt(question, retrieval.chunks)
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                raw.append(delta)
                yield "token", delta
        except LLMError as exc:
            result.error = str(exc)
            log.error("generation failed: %s", exc)
            yield "error", (
                "The language model could not be reached. Retrieved sources are "
                "shown above — they came from the local index and are unaffected."
            )
            return

        result.timings_ms["generation_ms"] = (time.perf_counter() - t0) * 1000
        if first_token_at is not None:
            result.timings_ms["ttft_ms"] = (first_token_at - t_start) * 1000

        # 5. Validate citations against what was actually retrieved.
        answer, citations = validate_and_rewrite("".join(raw), retrieval.chunks)
        result.answer = answer
        result.citations = citations
        result.timings_ms["total_ms"] = (time.perf_counter() - t_start) * 1000
        yield "done", result

    def answer(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        extra_store: ChunkStore | None = None,
    ) -> AnswerResult:
        """Blocking convenience wrapper, used by the evaluation harness."""
        final = AnswerResult()
        for event, payload in self.stream_answer(
            question, history=history, extra_store=extra_store
        ):
            if event == "done":
                final = payload  # type: ignore[assignment]
            elif event == "error":
                final.error = str(payload)
        return final
