"""Retrieval and answer metrics.

Deliberately implemented here rather than imported: these definitions are short,
and knowing exactly what "recall@5" means in a given report — recall over
*documents* or over *chunks*, counted how — matters more than saving twenty
lines. Every metric below is document-level, because the gold labels are
documents; a chunk-level label set would need per-chunk annotation that would
go stale on every re-chunk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate(retrieved_sources: Sequence[str], gold_sources: Sequence[str]) -> float:
    """1.0 if any gold document appears anywhere in the retrieved list.

    The most permissive metric, and the one that best predicts whether the
    generator *can* answer at all — an answer needs one good passage, not all
    of them.
    """
    return 1.0 if set(retrieved_sources) & set(gold_sources) else 0.0


def recall_at_k(retrieved_sources: Sequence[str], gold_sources: Sequence[str], k: int) -> float:
    """Fraction of gold documents present in the top k."""
    if not gold_sources:
        return 0.0
    found = set(retrieved_sources[:k]) & set(gold_sources)
    return len(found) / len(set(gold_sources))


def precision_at_k(retrieved_sources: Sequence[str], gold_sources: Sequence[str], k: int) -> float:
    if not retrieved_sources[:k]:
        return 0.0
    gold = set(gold_sources)
    return sum(1 for s in retrieved_sources[:k] if s in gold) / len(retrieved_sources[:k])


def reciprocal_rank(retrieved_sources: Sequence[str], gold_sources: Sequence[str]) -> float:
    """1/rank of the first gold document, 0 if absent.

    MRR is the metric that actually tracks answer quality here, because the
    generator's attention is heavily front-loaded: a gold passage at rank 1
    is worth much more than the same passage at rank 5.
    """
    gold = set(gold_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sources: Sequence[str], gold_sources: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    Unlike MRR it rewards finding *several* gold documents, which is what
    distinguishes a good multi-source answer from one that found the easiest
    citation and stopped.
    """
    gold = set(gold_sources)
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(retrieved_sources[:k], start=1)
        if source in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def mean_with_ci(values: Sequence[float]) -> tuple[float, float]:
    """Mean and the half-width of a 95% normal-approximation interval.

    Reported because a 35-question eval set is small: a 3-point difference
    between two configurations on this set is well inside the noise, and a
    report that hides that invites over-fitting to it.
    """
    if len(values) < 2:
        return mean(values), 0.0
    return mean(values), 1.96 * stdev(values) / math.sqrt(len(values))
