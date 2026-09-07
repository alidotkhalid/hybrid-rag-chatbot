"""Retrieval-pipeline behaviour.

These use `HashEmbedder`, which is lexical-ish rather than semantic, so they
assert on *structural* properties of the pipeline — provenance, diversification,
the groundedness gate, ablation wiring — rather than on semantic ranking
quality. Semantic quality is measured with the real model by the evaluation
harness in `evaluation/`, which is the right tool for it: a unit test that
asserts "this paraphrase ranks first" is a flaky benchmark wearing a test's
clothes.
"""

from ragcore.reranker import IdentityReranker
from ragcore.retriever import HybridRetriever
from ragcore.types import Chunk, ScoredChunk


class TestHybridRetrieval:
    def test_returns_results(self, retriever):
        result = retriever.retrieve("multi-head attention heads")
        assert result.chunks
        assert result.grounded

    def test_respects_top_k(self, retriever):
        assert len(retriever.retrieve("attention", top_k=2).chunks) <= 2

    def test_records_stage_timings(self, retriever):
        timings = retriever.retrieve("attention").timings_ms
        assert {"dense_ms", "sparse_ms", "fusion_ms"} <= set(timings)
        assert timings["total_ms"] > 0

    def test_bm25_finds_an_exact_rare_term(self, retriever):
        """The case that motivates hybrid retrieval: a rare literal token."""
        result = retriever.retrieve("WMT 2014 English-German 4.5 million")
        assert result.chunks
        assert any("WMT" in c.chunk.body for c in result.chunks)

    def test_provenance_is_recorded(self, retriever):
        result = retriever.retrieve("low-rank decomposition matrices")
        paths = {c.retriever_path for c in result.chunks}
        assert paths <= {"dense", "sparse", "both"}
        assert any(c.dense_rank or c.sparse_rank for c in result.chunks)


class TestAblations:
    """Each retriever must work alone — this is what the eval harness sweeps."""

    def test_dense_only(self, retriever):
        result = retriever.retrieve("attention", use_sparse=False)
        assert result.chunks
        assert all(c.sparse_rank is None for c in result.chunks)

    def test_sparse_only(self, retriever):
        result = retriever.retrieve("attention", use_dense=False)
        assert result.chunks
        assert all(c.dense_rank is None for c in result.chunks)

    def test_both_disabled_returns_nothing(self, retriever):
        result = retriever.retrieve("attention", use_dense=False, use_sparse=False)
        assert result.chunks == []
        assert result.grounded is False

    def test_rerank_can_be_disabled(self, retriever):
        result = retriever.retrieve("attention", use_rerank=False)
        assert result.chunks
        assert all(c.rerank_score is None for c in result.chunks)


class TestGroundednessGate:
    def test_gate_opens_for_relevant_queries(self, retriever):
        assert retriever.retrieve("multi-head attention").grounded

    def test_gate_closes_when_nothing_clears_the_threshold(self, store, embedder, settings):
        class AlwaysIrrelevant:
            def score(self, query, passages):
                return [-99.0] * len(passages)

        retriever = HybridRetriever(store, embedder, AlwaysIrrelevant(), settings)
        result = retriever.retrieve("the mating habits of the emperor penguin")
        assert result.grounded is False
        # The passages are still returned — the *pipeline* decides to refuse,
        # so the UI can still show what was considered.
        assert result.chunks

    def test_empty_store_is_not_grounded(self, embedder, settings):
        from ragcore.sparse import BM25Index
        from ragcore.store import ChunkStore, VectorIndex

        empty = ChunkStore([], VectorIndex(embedder.dimension), BM25Index().fit([]))
        retriever = HybridRetriever(empty, embedder, IdentityReranker(), settings)
        result = retriever.retrieve("anything at all")
        assert result.chunks == []
        assert result.grounded is False


class TestDiversification:
    def test_caps_chunks_per_document(self):
        candidates = [
            ScoredChunk(
                chunk=Chunk(chunk_id=f"a::{i}", doc_id="a", text="t", body="t"),
                score=10.0 - i,
            )
            for i in range(8)
        ] + [
            ScoredChunk(
                chunk=Chunk(chunk_id=f"b::{i}", doc_id="b", text="t", body="t"),
                score=1.0 - i,
            )
            for i in range(4)
        ]
        selected = HybridRetriever._diversify(candidates, top_k=5, per_doc_cap=3)
        from collections import Counter

        counts = Counter(c.chunk.doc_id for c in selected)
        assert counts["a"] == 3
        assert counts["b"] == 2

    def test_cap_is_lifted_rather_than_under_filling(self):
        """If only one document has anything relevant, returning three chunks
        when six were asked for would be worse than exceeding the cap."""
        candidates = [
            ScoredChunk(
                chunk=Chunk(chunk_id=f"a::{i}", doc_id="a", text="t", body="t"),
                score=10.0 - i,
            )
            for i in range(6)
        ]
        assert len(HybridRetriever._diversify(candidates, top_k=6, per_doc_cap=3)) == 6

    def test_preserves_score_order_within_the_cap(self):
        candidates = [
            ScoredChunk(
                chunk=Chunk(chunk_id=f"{d}::{i}", doc_id=d, text="t", body="t"),
                score=score,
            )
            for d, i, score in [("a", 0, 9.0), ("b", 0, 8.0), ("a", 1, 7.0)]
        ]
        selected = HybridRetriever._diversify(candidates, top_k=3, per_doc_cap=3)
        assert [c.score for c in selected] == [9.0, 8.0, 7.0]


class TestSessionUploads:
    def test_uploaded_chunks_compete_with_the_corpus(self, retriever, embedder, settings):
        """An uploaded document must be reachable through the same ranking,
        not appended as a second-class source."""
        from ragcore.chunking import chunk_documents
        from ragcore.store import ChunkStore
        from ragcore.types import Document

        uploaded = Document(
            doc_id="mine",
            title="My Private Benchmark Report",
            source="report.pdf",
            text=(
                "# Findings\n\n"
                "Our internal benchmark ZEPHYRQUARK scored 91.4 on the held-out "
                "split, outperforming every published baseline we tested."
            ),
        )
        extra = ChunkStore.build(chunk_documents([uploaded]), embedder)

        result = retriever.retrieve("ZEPHYRQUARK benchmark score", extra_store=extra)
        assert any(c.chunk.doc_id == "mine" for c in result.chunks)

    def test_corpus_still_works_with_an_empty_extra_store(self, retriever, embedder):
        from ragcore.sparse import BM25Index
        from ragcore.store import ChunkStore, VectorIndex

        empty = ChunkStore([], VectorIndex(embedder.dimension), BM25Index().fit([]))
        result = retriever.retrieve("attention", extra_store=empty)
        assert result.chunks
