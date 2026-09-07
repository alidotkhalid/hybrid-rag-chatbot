"""Reciprocal Rank Fusion.

The problem: dense scores are cosine similarities in [-1, 1] with a narrow,
corpus-dependent spread; BM25 scores are unbounded sums of IDF weights whose
scale depends on query length and vocabulary rarity. They are not comparable,
and normalising them (min-max, z-score) makes the combination depend on the
score distribution of whichever candidates happened to be retrieved — which
changes per query, so a weight tuned on one query set silently mis-generalises.

RRF sidesteps the calibration problem entirely by discarding scores and using
only *ranks*:

    RRF(d) = Σ over retrievers r of  1 / (k + rank_r(d))

`k` damps the influence of the top ranks; with k=60 the difference between
rank 1 and rank 2 is small, so a document that both retrievers place in their
top 10 outranks one that a single retriever places first. That is precisely the
behaviour we want from a hybrid system: agreement is stronger evidence than any
one retriever's confidence.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods".
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists into one.

    Args:
        ranked_lists: each inner sequence is a retriever's result, best first.
        k: RRF damping constant.
        weights: optional per-retriever weight. Defaults to uniform. Useful
            when one retriever is known to be stronger on a given corpus —
            but note that uniform weighting is a genuinely strong baseline
            and the evaluation harness in `evaluation/` did not find a
            weighting that beat it here.

    Returns:
        [(id, fused_score), ...] sorted by descending score. Ties are broken
        by best (lowest) rank achieved in any list, then by id, so the output
        is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must be the same length as ranked_lists")

    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}

    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank[item[0]], item[0]),
    )


def rank_positions(ranked: Sequence[int]) -> dict[int, int]:
    """Map id -> 1-based rank, for annotating results with provenance."""
    return {doc_id: rank for rank, doc_id in enumerate(ranked, start=1)}
