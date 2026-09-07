"""Persistent chunk store: vectors + BM25 + metadata, saved and loaded as a unit.

**Index choice.** This uses FAISS `IndexFlatIP` — an exact, brute-force inner
product scan — not HNSW or IVF. That is a deliberate call, not a shortcut. The
preloaded corpus is on the order of 10^3 chunks; an exact scan over 1500 × 384
floats is well under a millisecond, while an approximate index would add a
build step, tuning parameters, and a recall ceiling below 100% in exchange for
nothing. Approximate search earns its complexity somewhere north of ~10^6
vectors. The `VectorIndex` interface below is where you would swap it, and the
README records the threshold at which you should.

Vectors are L2-normalised upstream, so inner product is exactly cosine
similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sparse import BM25Index
from .types import Chunk

try:  # pragma: no cover - exercised implicitly by whichever branch is installed
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


class VectorIndex:
    """Exact inner-product search, backed by FAISS when available.

    The numpy fallback exists so the test suite and the evaluation harness run
    in environments without FAISS wheels, and so a contributor on an
    unsupported platform is not blocked. Results are identical up to floating
    point ordering.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._faiss_index = faiss.IndexFlatIP(dimension) if _HAS_FAISS else None
        self._matrix: np.ndarray = np.zeros((0, dimension), dtype=np.float32)

    @property
    def backend(self) -> str:
        return "faiss-IndexFlatIP" if self._faiss_index is not None else "numpy-exact"

    @property
    def size(self) -> int:
        return int(self._matrix.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected dimension {self.dimension}, got {vectors.shape[1]}"
            )
        self._matrix = (
            vectors if self.size == 0 else np.vstack([self._matrix, vectors])
        )
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.size == 0:
            return []
        k = min(top_k, self.size)
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(q, k)
            return [
                (int(i), float(s))
                for i, s in zip(ids[0], scores[0], strict=True)
                if i != -1
            ]
        sims = self._matrix @ q[0]
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        # The raw matrix is the source of truth and is portable across FAISS
        # versions and across the faiss/numpy backends; a FAISS binary index
        # is not. Rebuilding from the matrix at load time costs milliseconds.
        np.save(path / "vectors.npy", self._matrix)

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        matrix = np.load(path / "vectors.npy")
        index = cls(dimension=int(matrix.shape[1]) if matrix.size else 0)
        index.add(matrix)
        return index


class ChunkStore:
    """Everything needed to answer a query, persisted together.

    Keeping the vector index, the BM25 index and the chunk metadata in one
    directory written by one call removes a whole class of production bug:
    an index and a metadata file that disagree about what document 417 is.
    """

    MANIFEST = "manifest.json"

    def __init__(
        self,
        chunks: list[Chunk],
        vector_index: VectorIndex,
        bm25: BM25Index,
        embedding_model: str = "",
    ) -> None:
        self.chunks = chunks
        self.vectors = vector_index
        self.bm25 = bm25
        self.embedding_model = embedding_model
        if len(chunks) != vector_index.size:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {vector_index.size}"
            )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, embedding_model: str = "") -> ChunkStore:
        texts = [c.text for c in chunks]
        vectors = embedder.embed_documents(texts)
        index = VectorIndex(dimension=embedder.dimension)
        index.add(vectors)
        bm25 = BM25Index().fit(texts)
        return cls(chunks, index, bm25, embedding_model=embedding_model)

    # -- accessors -----------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def doc_count(self) -> int:
        return len({c.doc_id for c in self.chunks})

    def documents(self) -> list[dict[str, str]]:
        seen: dict[str, dict[str, str]] = {}
        for c in self.chunks:
            if c.doc_id not in seen:
                seen[c.doc_id] = {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "source": c.source,
                }
        return sorted(seen.values(), key=lambda d: d["title"])

    def stats(self) -> dict[str, object]:
        tokens = [c.token_count for c in self.chunks]
        return {
            "chunks": self.size,
            "documents": self.doc_count,
            "vector_backend": self.vectors.backend,
            "vector_dimension": self.vectors.dimension,
            "embedding_model": self.embedding_model,
            "bm25_vocabulary": len(self.bm25.vocab),
            "mean_chunk_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        }

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vectors.save(path)
        self.bm25.save(path)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (path / self.MANIFEST).write_text(
            json.dumps(
                {
                    "version": 1,
                    "embedding_model": self.embedding_model,
                    "stats": self.stats(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ChunkStore:
        path = Path(path)
        manifest_path = path / cls.MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build one first:  python -m ragcore.ingest build"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(
            chunks=chunks,
            vector_index=VectorIndex.load(path),
            bm25=BM25Index.load(path),
            embedding_model=manifest.get("embedding_model", ""),
        )

    def assert_compatible_with(self, embedder) -> None:
        """Guard against querying an index with a different embedding model.

        This is the single most common silent failure in a deployed RAG
        system: the index was built with model A, the service loads model B,
        dimensions happen to match, and retrieval quietly returns noise. Fail
        loudly instead.
        """
        if self.vectors.dimension and embedder.dimension != self.vectors.dimension:
            raise ValueError(
                f"Index was built with {self.vectors.dimension}-dim vectors "
                f"({self.embedding_model or 'unknown model'}) but the loaded embedder "
                f"produces {embedder.dimension} dims. Rebuild the index."
            )
        model_name = getattr(embedder, "model_name", "")
        if self.embedding_model and model_name and model_name != self.embedding_model:
            raise ValueError(
                f"Index was built with '{self.embedding_model}' but the service is "
                f"configured for '{model_name}'. Rebuild the index or set "
                f"RAG_EMBEDDING_MODEL back."
            )
