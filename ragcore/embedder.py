"""Embedding models.

The rest of the system depends on the `Embedder` protocol, not on
sentence-transformers. That keeps torch out of the import path for the tests
and the evaluation harness's retrieval-logic checks, and it means swapping to a
hosted embedding API later is a one-file change.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """BGE-family bi-encoder running locally on CPU.

    Two details that are easy to get wrong and expensive to debug:

    1. **Asymmetric prefixing.** BGE is trained with an instruction prefix on
       queries only. Applying it to documents too, or to neither, costs
       several points of recall. The prefix is applied here and nowhere else,
       so it cannot drift out of sync between ingestion and query time.
    2. **Normalisation.** Vectors are L2-normalised at encode time, which lets
       the FAISS inner-product index behave as exact cosine similarity. If
       normalisation is skipped, inner product silently rewards long vectors
       and retrieval quality degrades in a way that looks like a chunking bug.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # heavy; import lazily

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self._lock = threading.Lock()
        """sentence-transformers is not thread-safe for concurrent encode()
        calls on the same module; the API serves requests from a thread pool,
        so encoding is serialised. At our throughput this is not the
        bottleneck — the LLM call is."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        with self._lock:
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 256,
            )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            vec = self._model.encode(
                [self.query_prefix + text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return np.asarray(vec, dtype=np.float32)[0]


class HashEmbedder:
    """Deterministic, dependency-free stand-in used by the test suite.

    It is a hashed bag-of-words projection: not semantic, but it is stable,
    normalised, and gives similar text similar vectors, which is enough to
    exercise index construction, persistence and fusion without downloading
    a model. It is never used in production paths.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            h = int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
            )
            v[h % self.dimension] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def build_embedder(settings) -> Embedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
    )
