"""Tests for the evaluation harness itself.

A metrics implementation that is wrong produces confident, plausible,
completely misleading numbers — the worst failure mode in the whole project,
because it is invisible. So the metrics are tested against hand-computed
values rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    hit_rate,
    mean_with_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.run_eval import CONFIGS, Question, evaluate_retrieval, load_questions


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["a.pdf", "b.pdf"], ["b.pdf"]) == 1.0

    def test_miss(self):
        assert hit_rate(["a.pdf"], ["z.pdf"]) == 0.0

    def test_empty_retrieval(self):
        assert hit_rate([], ["a.pdf"]) == 0.0


class TestRecall:
    def test_finds_all_gold(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_finds_half(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_k_truncates(self):
        # "b" is present but at rank 3, outside k=2.
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_no_gold_is_zero(self):
        assert recall_at_k(["a"], [], k=3) == 0.0


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], ["a"], k=2) == 0.5

    def test_empty(self):
        assert precision_at_k([], ["a"], k=3) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [
            (["a", "x", "y"], 1.0),
            (["x", "a", "y"], 0.5),
            (["x", "y", "a"], 1 / 3),
            (["x", "y", "z"], 0.0),
        ],
    )
    def test_rank_positions(self, retrieved, expected):
        assert reciprocal_rank(retrieved, ["a"]) == pytest.approx(expected)

    def test_uses_the_first_gold_hit(self):
        assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == pytest.approx(0.5)


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_reversed_gold_still_perfect_for_binary_relevance(self):
        # Both gold docs are in the top 2; order between two equally-relevant
        # documents does not matter under binary relevance.
        assert ndcg_at_k(["b", "a"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_penalises_a_lower_rank(self):
        good = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
        worse = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
        assert good > worse

    def test_hand_computed_value(self):
        # gold={a}; a is at rank 2 -> DCG = 1/log2(3) = 0.6309; ideal = 1.0
        assert ndcg_at_k(["x", "a"], ["a"], k=3) == pytest.approx(0.63093, abs=1e-4)

    def test_no_gold(self):
        assert ndcg_at_k(["a"], [], k=3) == 0.0


class TestConfidenceInterval:
    def test_zero_variance_gives_zero_width(self):
        m, ci = mean_with_ci([1.0, 1.0, 1.0])
        assert m == 1.0 and ci == 0.0

    def test_single_value_has_no_interval(self):
        assert mean_with_ci([0.5]) == (0.5, 0.0)

    def test_interval_shrinks_with_more_samples(self):
        small = mean_with_ci([0.0, 1.0] * 5)[1]
        large = mean_with_ci([0.0, 1.0] * 50)[1]
        assert large < small


class TestQuestionSet:
    def test_loads_and_parses(self):
        questions = load_questions()
        assert len(questions) >= 30
        assert all(q.id and q.question for q in questions)

    def test_ids_are_unique(self):
        ids = [q.id for q in load_questions()]
        assert len(ids) == len(set(ids))

    def test_has_both_answerable_and_unanswerable_splits(self):
        questions = load_questions()
        assert sum(q.answerable for q in questions) >= 25
        assert sum(not q.answerable for q in questions) >= 3

    def test_answerable_questions_have_keywords(self):
        for q in load_questions():
            if q.answerable:
                assert q.answer_keywords, f"{q.id} has no answer keywords"

    def test_gold_sources_are_real_corpus_entries(self):
        """Catches the commonest gold-set bug: a typo'd filename that silently
        makes a question unscorable and drags every metric down."""
        from ragcore.corpus import CORPUS

        valid = {p.filename for p in CORPUS}
        for q in load_questions():
            for source in q.gold_sources:
                assert source in valid, f"{q.id} references unknown source {source}"

    def test_covers_multiple_question_types(self):
        types = {q.type for q in load_questions()}
        assert {"semantic", "lexical", "unanswerable"} <= types

    def test_file_is_valid_jsonl(self):
        path = Path(__file__).resolve().parent.parent / "evaluation" / "questions.jsonl"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"questions.jsonl line {i} is not valid JSON: {exc}")


class TestEvaluateRetrieval:
    """The harness runs against the stub retriever, so this checks the
    plumbing and the shape of the report — not model quality."""

    @pytest.fixture
    def questions(self):
        return [
            Question(
                id="a1",
                question="How many attention heads are used?",
                gold_sources=["1706.03762.pdf"],
                answer_keywords=["8"],
                type="lexical",
            ),
            Question(
                id="a2",
                question="What does LoRA do to trainable parameters?",
                gold_sources=["2106.09685.pdf"],
                answer_keywords=["reduce"],
                type="semantic",
            ),
            Question(id="u1", question="Who won the world cup?", gold_sources=[], type="unanswerable"),
        ]

    def test_report_has_every_metric(self, retriever, questions):
        report = evaluate_retrieval(retriever, questions, CONFIGS["hybrid"], k=4)
        for key in (
            "hit_rate",
            "recall@4",
            "precision@4",
            "mrr",
            "ndcg@4",
            "refusal_rate_unanswerable",
            "false_refusal_rate",
            "latency_p50_ms",
            "by_type",
            "per_question",
        ):
            assert key in report, f"missing {key}"

    def test_counts_the_splits(self, retriever, questions):
        report = evaluate_retrieval(retriever, questions, CONFIGS["hybrid"], k=4)
        assert report["n_answerable"] == 2
        assert report["n_unanswerable"] == 1
        assert len(report["per_question"]) == 2

    def test_metrics_are_in_range(self, retriever, questions):
        report = evaluate_retrieval(retriever, questions, CONFIGS["hybrid"], k=4)
        for key in ("hit_rate", "recall@4", "precision@4", "mrr", "ndcg@4"):
            assert 0.0 <= report[key] <= 1.0, f"{key} out of range"

    def test_breaks_down_by_question_type(self, retriever, questions):
        report = evaluate_retrieval(retriever, questions, CONFIGS["hybrid"], k=4)
        assert set(report["by_type"]) == {"lexical", "semantic"}
        assert report["by_type"]["lexical"]["n"] == 1

    def test_every_ablation_config_runs(self, retriever, questions):
        for name, config in CONFIGS.items():
            report = evaluate_retrieval(retriever, questions, config, k=4)
            assert report["n_answerable"] == 2, f"{name} failed"

    def test_retrieval_finds_the_gold_document(self, retriever, questions):
        """With the real corpus fixture loaded, the lexical question should
        actually hit — a sanity check that gold labels and sources line up."""
        report = evaluate_retrieval(retriever, questions, CONFIGS["hybrid"], k=4)
        rows = {r["id"]: r for r in report["per_question"]}
        assert rows["a1"]["hit"] == 1.0
        assert rows["a2"]["hit"] == 1.0
