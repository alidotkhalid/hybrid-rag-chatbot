import numpy as np
import pytest

from ragcore.embedder import HashEmbedder
from ragcore.store import ChunkStore, VectorIndex
from ragcore.types import Chunk


class TestVectorIndex:
    def test_returns_the_nearest_vector(self):
        index = VectorIndex(dimension=3)
        index.add(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
        results = index.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=1)
        assert results[0][0] == 0

    def test_results_are_score_ordered(self):
        index = VectorIndex(dimension=2)
        index.add(np.array([[1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32))
        scores = [s for _, s in index.search(np.array([1, 0], dtype=np.float32), top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_clamped_to_size(self):
        index = VectorIndex(dimension=2)
        index.add(np.array([[1, 0]], dtype=np.float32))
        assert len(index.search(np.array([1, 0], dtype=np.float32), top_k=50)) == 1

    def test_empty_index_returns_nothing(self):
        assert VectorIndex(dimension=4).search(np.zeros(4, dtype=np.float32), 5) == []

    def test_dimension_mismatch_is_rejected(self):
        index = VectorIndex(dimension=3)
        with pytest.raises(ValueError, match="dimension"):
            index.add(np.zeros((2, 5), dtype=np.float32))

    def test_add_is_incremental(self):
        index = VectorIndex(dimension=2)
        index.add(np.array([[1, 0]], dtype=np.float32))
        index.add(np.array([[0, 1]], dtype=np.float32))
        assert index.size == 2

    def test_roundtrips_through_disk(self, tmp_path):
        index = VectorIndex(dimension=3)
        vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        index.add(vectors)
        index.save(tmp_path)
        loaded = VectorIndex.load(tmp_path)
        assert loaded.size == 2
        query = np.array([1, 0, 0], dtype=np.float32)
        assert index.search(query, 2) == loaded.search(query, 2)


class TestChunkStore:
    def test_build_indexes_every_chunk(self, store):
        assert store.size == len(store.chunks)
        assert store.vectors.size == store.size
        assert store.bm25.doc_count == store.size

    def test_counts_distinct_documents(self, store):
        assert store.doc_count == 3

    def test_documents_are_listed_once_each(self, store):
        docs = store.documents()
        assert len(docs) == 3
        assert len({d["doc_id"] for d in docs}) == 3

    def test_stats_are_populated(self, store):
        stats = store.stats()
        assert stats["chunks"] > 0
        assert stats["documents"] == 3
        assert stats["vector_dimension"] == 96
        assert stats["mean_chunk_tokens"] > 0

    def test_mismatched_counts_are_rejected(self, embedder):
        chunks = [Chunk(chunk_id="a", doc_id="d", text="x", body="x")]
        index = VectorIndex(dimension=embedder.dimension)  # deliberately empty
        from ragcore.sparse import BM25Index

        with pytest.raises(ValueError, match="mismatch"):
            ChunkStore(chunks, index, BM25Index().fit(["x"]))

    def test_roundtrips_through_disk(self, store, tmp_path, embedder):
        store.save(tmp_path / "idx")
        loaded = ChunkStore.load(tmp_path / "idx")

        assert loaded.size == store.size
        assert loaded.doc_count == store.doc_count
        assert loaded.embedding_model == store.embedding_model
        assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in store.chunks]
        assert [c.section for c in loaded.chunks] == [c.section for c in store.chunks]

        # And it must still retrieve identically — the point of persisting.
        query = embedder.embed_query("multi-head attention")
        assert store.vectors.search(query, 3) == loaded.vectors.search(query, 3)
        assert store.bm25.search("attention", 3) == loaded.bm25.search("attention", 3)

    def test_missing_index_gives_an_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ragcore.ingest"):
            ChunkStore.load(tmp_path / "nope")


class TestCompatibilityGuard:
    """The guard against the silent-failure mode this project cares most about:
    querying an index with a different embedding model than built it."""

    def test_accepts_the_matching_embedder(self, store, embedder):
        store.embedding_model = ""  # HashEmbedder has no model_name
        store.assert_compatible_with(embedder)

    def test_rejects_a_different_dimension(self, store):
        with pytest.raises(ValueError, match="Rebuild the index"):
            store.assert_compatible_with(HashEmbedder(dimension=32))

    def test_rejects_a_different_model_name(self, store, embedder):
        store.embedding_model = "BAAI/bge-small-en-v1.5"
        embedder.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        with pytest.raises(ValueError, match="all-MiniLM"):
            store.assert_compatible_with(embedder)
