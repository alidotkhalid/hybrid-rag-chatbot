"""The hybrid retriever: dense + sparse → RRF → cross-encoder → groundedness gate.

    query
      ├── dense  (bi-encoder → exact IP scan)   → top 30
      └── sparse (BM25)                         → top 30
             │
             ├── Reciprocal Rank Fusion (k=60)  → top 30 candidates
             │
             ├── cross-encoder rerank            → sorted by relevance logit
             │
             ├── per-document diversification    → no single paper monopolises
             │
             └── groundedness gate               → top 6, or "not in corpus"

Each stage exists to fix a failure the previous stage cannot:

* Dense alone misses exact identifiers and rare tokens.
* Sparse alone misses paraphrases and any query sharing no vocabulary with the
  answer.
* Fusion recovers recall but leaves the ordering mediocre, because rank
  agreement is a weak relevance signal.
* Reranking fixes the ordering but will happily rank the least-bad of six
  irrelevant passages first, which is how a RAG system ends up confidently
  answering a question its corpus cannot support — hence the gate.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from .config import Settings
from .fusion import rank_positions, reciprocal_rank_fusion
from .reranker import Reranker
from .store import ChunkStore
from .types import RetrievalResult, ScoredChunk


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_rerank: bool = True,
        extra_store: ChunkStore | None = None,
    ) -> RetrievalResult:
        """Run the full pipeline.

        `extra_store` holds documents a visitor uploaded in this session. It is
        searched with the same code path and its results are fused into the
        same candidate pool, so an uploaded document competes with the
        preloaded corpus on equal terms rather than being appended as a
        second-class source.

        The `use_*` flags exist for the ablation study in `evaluation/`.
        """
        settings = self.settings
        top_k = top_k or settings.final_top_k
        timings: dict[str, float] = {}

        stores: list[ChunkStore] = [self.store]
        if extra_store is not None and extra_store.size:
            stores.append(extra_store)

        # Global index space across the (up to two) stores, so fusion and
        # reranking never need to know which store a candidate came from.
        offsets: list[int] = []
        running = 0
        for store in stores:
            offsets.append(running)
            running += store.size
        all_chunks = [c for store in stores for c in store.chunks]
        if not all_chunks:
            return RetrievalResult(chunks=[], grounded=False, timings_ms=timings)

        # --- Stage 1: dense ------------------------------------------------
        dense_ranked: list[int] = []
        if use_dense:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            merged: list[tuple[int, float]] = []
            for store, offset in zip(stores, offsets, strict=True):
                merged.extend(
                    (idx + offset, score)
                    for idx, score in store.vectors.search(qvec, settings.dense_top_k)
                )
            merged.sort(key=lambda x: -x[1])
            dense_ranked = [i for i, _ in merged[: settings.dense_top_k]]
            timings["dense_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 2: sparse -----------------------------------------------
        sparse_ranked: list[int] = []
        if use_sparse:
            t0 = time.perf_counter()
            merged = []
            for store, offset in zip(stores, offsets, strict=True):
                merged.extend(
                    (idx + offset, score)
                    for idx, score in store.bm25.search(query, settings.sparse_top_k)
                )
            merged.sort(key=lambda x: -x[1])
            sparse_ranked = [i for i, _ in merged[: settings.sparse_top_k]]
            timings["sparse_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 3: fusion -----------------------------------------------
        lists: list[Sequence[int]] = [lst for lst in (dense_ranked, sparse_ranked) if lst]
        if not lists:
            return RetrievalResult(chunks=[], grounded=False, timings_ms=timings)

        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(lists, k=settings.rrf_k)[: settings.rerank_candidates]
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        dense_pos = rank_positions(dense_ranked)
        sparse_pos = rank_positions(sparse_ranked)

        candidates = [
            ScoredChunk(
                chunk=all_chunks[idx],
                score=fused_score,
                fused_score=fused_score,
                dense_rank=dense_pos.get(idx),
                sparse_rank=sparse_pos.get(idx),
            )
            for idx, fused_score in fused
        ]

        # --- Stage 4: rerank -----------------------------------------------
        if use_rerank and candidates:
            t0 = time.perf_counter()
            scores = self.reranker.score(query, [c.chunk.text for c in candidates])
            for cand, score in zip(candidates, scores, strict=True):
                cand.rerank_score = score
                cand.score = score
            candidates.sort(key=lambda c: -c.score)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 5: diversify --------------------------------------------
        selected = self._diversify(candidates, top_k)

        # --- Stage 6: groundedness gate ------------------------------------
        grounded = bool(selected) and (
            not use_rerank
            or max((c.rerank_score or float("-inf")) for c in selected)
            >= settings.min_rerank_score
        )

        timings["total_ms"] = sum(
            v for k, v in timings.items() if k.endswith("_ms") and k != "total_ms"
        )
        return RetrievalResult(chunks=selected, grounded=grounded, timings_ms=timings)

    @staticmethod
    def _diversify(candidates: list[ScoredChunk], top_k: int, per_doc_cap: int = 3) -> list[ScoredChunk]:
        """Cap how many chunks a single document contributes.

        Long papers dominate an unconstrained top-k: six chunks from the same
        section of one paper produce an answer that is narrow and impossible
        to cross-check. Capping at 3 preserves the ability to synthesise
        across sources, which is the thing a RAG system is actually for.
        The cap is lifted if it would leave the context under-filled.
        """
        selected: list[ScoredChunk] = []
        per_doc: dict[str, int] = {}
        overflow: list[ScoredChunk] = []

        for cand in candidates:
            doc = cand.chunk.doc_id
            if per_doc.get(doc, 0) < per_doc_cap:
                selected.append(cand)
                per_doc[doc] = per_doc.get(doc, 0) + 1
            else:
                overflow.append(cand)
            if len(selected) == top_k:
                return selected

        for cand in overflow:
            if len(selected) == top_k:
                break
            selected.append(cand)
        return selected
