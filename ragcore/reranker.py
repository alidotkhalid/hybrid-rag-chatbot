"""Cross-encoder reranking.

A bi-encoder must compress a passage into a fixed vector *before it has seen
the query*. That is a hard information bottleneck: the passage's vector has to
be simultaneously good for every possible question. A cross-encoder instead
reads the query and passage together in one forward pass and outputs a single
relevance logit, so it can attend to the exact terms of the query.

The cost is that it cannot be precomputed — it is O(candidates) forward passes
per query, not one vector lookup. Hence the standard two-stage shape used here:
cheap recall-oriented retrieval to ~30 candidates, expensive precision-oriented
reranking down to the ~6 that reach the prompt.

On this corpus the reranker is what fixes the "retrieved the right paper, wrong
section" failure — see `evaluation/` for the measured ablation.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder  # heavy; import lazily

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device, max_length=512)
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            scores = self._model.predict(
                [(query, p) for p in passages],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        return [float(s) for s in scores]


class IdentityReranker:
    """No-op reranker: preserves the fused order.

    Used by the tests, by the `--no-rerank` ablation in the evaluation
    harness, and as a graceful degradation path if the cross-encoder fails to
    load on a constrained host.
    """

    model_name = "identity"

    def score(self, query: str, passages: list[str]) -> list[float]:
        # Descending so that the existing order is preserved by a stable sort,
        # and above `min_rerank_score` so the groundedness gate does not fire
        # spuriously when reranking is disabled.
        return [float(len(passages) - i) for i in range(len(passages))]
