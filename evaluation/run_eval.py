"""Evaluation harness and ablation runner.

    python -m evaluation.run_eval                 # retrieval ablation (no LLM, no API key)
    python -m evaluation.run_eval --generate      # add answer-quality metrics (needs a key)
    python -m evaluation.run_eval --config hybrid_rerank --generate
    python -m evaluation.run_eval --out evaluation/results.json

**What this measures, and what it does not.** Retrieval metrics are computed
against hand-labelled gold documents and are objective. Answer metrics are
proxies: citation coverage detects *missing* evidence, not *wrong* evidence,
and keyword recall rewards saying the right words rather than being correct.
An LLM-as-judge pass would close some of that gap and is the obvious next
step; it is not here because a judge that shares a family with the generator
is a biased judge, and because a metric nobody can reproduce without an API
key is worse than an honest proxy.

The refusal rate on the `unanswerable` split is the metric worth watching. It
is objective, it is the failure mode that actually matters for a system people
will trust, and almost nothing else in a RAG evaluation catches it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (  # noqa: E402
    hit_rate,
    mean,
    mean_with_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragcore.citations import citation_coverage  # noqa: E402
from ragcore.config import settings  # noqa: E402
from ragcore.pipeline import RAGPipeline  # noqa: E402
from ragcore.retriever import HybridRetriever  # noqa: E402
from ragcore.store import ChunkStore  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"

# The ablation grid. Each entry isolates one stage so its contribution is
# attributable rather than assumed.
CONFIGS: dict[str, dict[str, bool]] = {
    "dense_only":     {"use_dense": True,  "use_sparse": False, "use_rerank": False},
    "sparse_only":    {"use_dense": False, "use_sparse": True,  "use_rerank": False},
    "hybrid":         {"use_dense": True,  "use_sparse": True,  "use_rerank": False},
    "hybrid_rerank":  {"use_dense": True,  "use_sparse": True,  "use_rerank": True},
}


@dataclass
class Question:
    id: str
    question: str
    gold_sources: list[str]
    answer_keywords: list[str] = field(default_factory=list)
    type: str = "semantic"
    note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold_sources)


def load_questions() -> list[Question]:
    questions = []
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            questions.append(
                Question(
                    id=data["id"],
                    question=data["question"],
                    gold_sources=data.get("gold_sources", []),
                    answer_keywords=data.get("answer_keywords", []),
                    type=data.get("type", "semantic"),
                    note=data.get("note", ""),
                )
            )
    return questions


# --- Retrieval evaluation ---------------------------------------------------


def evaluate_retrieval(
    retriever: HybridRetriever, questions: list[Question], config: dict[str, bool], k: int
) -> dict:
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]

    per_question = []
    latencies = []

    for q in answerable:
        t0 = time.perf_counter()
        result = retriever.retrieve(q.question, top_k=k, **config)
        latencies.append((time.perf_counter() - t0) * 1000)

        # De-duplicate to document level, preserving rank order.
        sources: list[str] = []
        for sc in result.chunks:
            if sc.chunk.source not in sources:
                sources.append(sc.chunk.source)

        per_question.append(
            {
                "id": q.id,
                "type": q.type,
                "hit": hit_rate(sources, q.gold_sources),
                "recall@k": recall_at_k(sources, q.gold_sources, k),
                "precision@k": precision_at_k(sources, q.gold_sources, k),
                "mrr": reciprocal_rank(sources, q.gold_sources),
                "ndcg@k": ndcg_at_k(sources, q.gold_sources, k),
                "grounded": result.grounded,
                "retrieved": sources[:k],
                "gold": q.gold_sources,
            }
        )

    # Refusal behaviour on out-of-corpus questions.
    refused = 0
    false_refusals = 0
    for q in unanswerable:
        if not retriever.retrieve(q.question, top_k=k, **config).grounded:
            refused += 1
    for row in per_question:
        if not row["grounded"]:
            false_refusals += 1

    hit_ci = mean_with_ci([r["hit"] for r in per_question])
    mrr_ci = mean_with_ci([r["mrr"] for r in per_question])

    by_type: dict[str, dict] = {}
    for qtype in sorted({r["type"] for r in per_question}):
        rows = [r for r in per_question if r["type"] == qtype]
        by_type[qtype] = {
            "n": len(rows),
            "hit_rate": round(mean([r["hit"] for r in rows]), 3),
            "mrr": round(mean([r["mrr"] for r in rows]), 3),
        }

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "hit_rate": round(hit_ci[0], 3),
        "hit_rate_ci95": round(hit_ci[1], 3),
        f"recall@{k}": round(mean([r["recall@k"] for r in per_question]), 3),
        f"precision@{k}": round(mean([r["precision@k"] for r in per_question]), 3),
        "mrr": round(mrr_ci[0], 3),
        "mrr_ci95": round(mrr_ci[1], 3),
        f"ndcg@{k}": round(mean([r["ndcg@k"] for r in per_question]), 3),
        "refusal_rate_unanswerable": round(refused / len(unanswerable), 3) if unanswerable else None,
        "false_refusal_rate": round(false_refusals / len(per_question), 3) if per_question else None,
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "latency_mean_ms": round(mean(latencies), 1),
        "by_type": by_type,
        "per_question": per_question,
    }


# --- Answer evaluation ------------------------------------------------------


def evaluate_answers(pipeline: RAGPipeline, questions: list[Question]) -> dict:
    rows = []
    for q in questions:
        result = pipeline.answer(q.question)
        answer_lower = (result.answer or "").lower()

        keyword_recall = (
            sum(1 for kw in q.answer_keywords if kw.lower() in answer_lower) / len(q.answer_keywords)
            if q.answer_keywords
            else None
        )

        rows.append(
            {
                "id": q.id,
                "type": q.type,
                "answerable": q.answerable,
                "grounded": result.grounded,
                "n_citations": len(result.citations),
                "citation_coverage": round(citation_coverage(result.answer, result.retrieved), 3),
                "keyword_recall": round(keyword_recall, 3) if keyword_recall is not None else None,
                "refused": not result.grounded,
                "latency_ms": round(result.timings_ms.get("total_ms", 0), 1),
                "answer": result.answer,
            }
        )

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]

    return {
        "citation_coverage": round(mean([r["citation_coverage"] for r in answerable_rows]), 3),
        "mean_citations": round(mean([float(r["n_citations"]) for r in answerable_rows]), 2),
        "keyword_recall": round(
            mean([r["keyword_recall"] for r in answerable_rows if r["keyword_recall"] is not None]), 3
        ),
        "uncited_answer_rate": round(
            mean([1.0 if r["n_citations"] == 0 else 0.0 for r in answerable_rows]), 3
        ),
        "correct_refusal_rate": round(
            mean([1.0 if r["refused"] else 0.0 for r in unanswerable_rows]), 3
        )
        if unanswerable_rows
        else None,
        "false_refusal_rate": round(
            mean([1.0 if r["refused"] else 0.0 for r in answerable_rows]), 3
        ),
        "latency_p50_ms": round(
            sorted(r["latency_ms"] for r in rows)[len(rows) // 2], 1
        ) if rows else 0,
        "per_question": rows,
    }


# --- Reporting --------------------------------------------------------------


def print_ablation_table(results: dict[str, dict], k: int) -> None:
    header = f"{'config':<16} {'hit':>7} {'recall':>8} {'MRR':>7} {'nDCG':>7} {'refuse':>8} {'p50 ms':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<16} "
            f"{r['hit_rate']:>7.3f} "
            f"{r[f'recall@{k}']:>8.3f} "
            f"{r['mrr']:>7.3f} "
            f"{r[f'ndcg@{k}']:>7.3f} "
            f"{(r['refusal_rate_unanswerable'] or 0):>8.3f} "
            f"{r['latency_p50_ms']:>8.1f}"
        )
    # Report the interval per configuration, not just for whichever one
    # happened to run last: a config sitting at a perfect 1.000 has zero
    # variance and therefore a zero-width interval, which makes the summary
    # line read as "differences smaller than 0.000 are not meaningful" —
    # true, vacuous, and actively misleading about the others.
    print("\n95% CI on hit rate (n=%d):" % next(iter(results.values()))["n_answerable"])
    for name, r in results.items():
        ceiling = " (at ceiling — zero variance)" if r["hit_rate"] >= 1.0 else ""
        print(f"  {name:<16} {r['hit_rate']:.3f} ± {r['hit_rate_ci95']:.3f}{ceiling}")
    widest = max(r["hit_rate_ci95"] for r in results.values())
    print(
        f"  Differences smaller than ±{widest:.3f} between configurations are "
        "not statistically meaningful at this sample size."
    )

    # The refusal column is only interpretable where the gate can actually
    # fire. Saying so in the output stops it being read as a comparison.
    if any(not CONFIGS[name]["use_rerank"] for name in results):
        print(
            "\nNote: the groundedness gate keys off cross-encoder scores, so\n"
            "      refusal is structurally impossible without reranking. Only\n"
            "      configurations with rerank enabled have a meaningful\n"
            "      refusal rate."
        )

    print("\nBy question type (hit rate):")
    types = sorted({t for r in results.values() for t in r["by_type"]})
    print(f"{'config':<16} " + " ".join(f"{t:>13}" for t in types))
    for name, r in results.items():
        cells = " ".join(
            f"{r['by_type'].get(t, {}).get('hit_rate', float('nan')):>13.3f}" for t in types
        )
        print(f"{name:<16} {cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the retrieval pipeline.")
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--k", type=int, default=6, help="top-k passed to the retriever")
    parser.add_argument("--generate", action="store_true", help="also evaluate generated answers")
    parser.add_argument("--out", type=Path, help="write full results as JSON")
    args = parser.parse_args(argv)

    questions = load_questions()
    print(
        f"{len(questions)} questions "
        f"({sum(q.answerable for q in questions)} answerable, "
        f"{sum(not q.answerable for q in questions)} out-of-corpus)"
    )

    from ragcore.embedder import build_embedder
    from ragcore.reranker import CrossEncoderReranker

    store = ChunkStore.load(settings.index_dir)
    print(f"index: {store.size} chunks over {store.doc_count} documents")

    embedder = build_embedder(settings)
    store.assert_compatible_with(embedder)
    reranker = CrossEncoderReranker(settings.reranker_model)
    retriever = HybridRetriever(store, embedder, reranker, settings)

    selected = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    results: dict[str, dict] = {}
    for name, config in selected.items():
        print(f"\nrunning {name} …", flush=True)
        results[name] = evaluate_retrieval(retriever, questions, config, args.k)

    print_ablation_table(results, args.k)

    payload: dict = {
        "settings": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "chunk_target_tokens": settings.chunk_target_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "rrf_k": settings.rrf_k,
            "min_rerank_score": settings.min_rerank_score,
            "k": args.k,
        },
        "index": store.stats(),
        "retrieval": results,
    }

    if args.generate:
        from ragcore.llm import build_llm

        print("\nevaluating generated answers …", flush=True)
        pipeline = RAGPipeline(retriever, build_llm(settings), settings)
        answers = evaluate_answers(pipeline, questions)
        payload["generation"] = answers
        print("\nAnswer quality:")
        for key in (
            "citation_coverage",
            "mean_citations",
            "keyword_recall",
            "uncited_answer_rate",
            "correct_refusal_rate",
            "false_refusal_rate",
            "latency_p50_ms",
        ):
            print(f"  {key:<26} {answers[key]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
