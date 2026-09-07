
from conftest import ScriptedLLM

from ragcore.llm import LLMTransientError
from ragcore.pipeline import RAGPipeline
from ragcore.prompts import NO_ANSWER_MESSAGE
from ragcore.retriever import HybridRetriever


def events(pipeline, question, **kwargs) -> dict:
    """Collect a stream into {event_name: payload | [payloads]}."""
    out: dict = {"token": []}
    for name, payload in pipeline.stream_answer(question, **kwargs):
        if name == "token":
            out["token"].append(payload)
        else:
            out[name] = payload
    return out


class TestStreamShape:
    def test_emits_events_in_order(self, pipeline):
        names = [name for name, _ in pipeline.stream_answer("multi-head attention")]
        assert names[0] == "query"
        assert names[1] == "sources"
        assert names[-1] == "done"
        assert "token" in names

    def test_sources_precede_the_first_token(self, pipeline):
        """The UI renders evidence while the answer streams, so sources must
        arrive first — otherwise the citation chips have nothing to link to."""
        names = [name for name, _ in pipeline.stream_answer("attention")]
        assert names.index("sources") < names.index("token")

    def test_tokens_concatenate_into_the_answer(self, pipeline):
        result = events(pipeline, "multi-head attention")
        streamed = "".join(result["token"])
        assert "grounded" in streamed
        assert result["done"].answer


class TestCitationValidation:
    def test_valid_citations_are_kept(self, retriever, settings):
        pipeline = RAGPipeline(retriever, ScriptedLLM("Grounded claim [1]."), settings)
        done = events(pipeline, "multi-head attention")["done"]
        assert len(done.citations) == 1
        assert "[1]" in done.answer

    def test_hallucinated_citations_are_stripped(self, retriever, settings):
        """End-to-end version of the guarantee: whatever the model writes,
        the answer the user sees cannot reference a passage that was never
        retrieved."""
        pipeline = RAGPipeline(retriever, ScriptedLLM("Claim [1] and claim [99]."), settings)
        done = events(pipeline, "multi-head attention")["done"]
        assert "[99]" not in done.answer
        assert all(c.marker <= len(done.retrieved) for c in done.citations)

    def test_citations_point_at_retrieved_chunks(self, retriever, settings):
        pipeline = RAGPipeline(retriever, ScriptedLLM("A [1]. B [2]."), settings)
        done = events(pipeline, "attention architecture")["done"]
        retrieved_ids = {sc.chunk.chunk_id for sc in done.retrieved}
        assert all(c.chunk_id in retrieved_ids for c in done.citations)


class TestRefusal:
    def test_refuses_when_nothing_is_grounded(self, store, embedder, settings):
        class AlwaysIrrelevant:
            def score(self, query, passages):
                return [-99.0] * len(passages)

        pipeline = RAGPipeline(
            HybridRetriever(store, embedder, AlwaysIrrelevant(), settings),
            ScriptedLLM("I will confidently make something up [1]."),
            settings,
        )
        result = events(pipeline, "who won the 1998 world cup")
        assert result["done"].grounded is False
        assert result["done"].answer == NO_ANSWER_MESSAGE

    def test_refusal_does_not_call_the_model(self, store, embedder, settings):
        """The gate must fire *before* generation — otherwise it costs a
        round-trip and the model has already been given misleading context."""

        class AlwaysIrrelevant:
            def score(self, query, passages):
                return [-99.0] * len(passages)

        llm = ScriptedLLM()
        pipeline = RAGPipeline(
            HybridRetriever(store, embedder, AlwaysIrrelevant(), settings), llm, settings
        )
        events(pipeline, "unrelated question")
        assert llm.calls == []


class TestFollowUps:
    def test_standalone_questions_skip_condensing(self, retriever, settings):
        llm = ScriptedLLM()
        pipeline = RAGPipeline(retriever, llm, settings)
        events(pipeline, "How many attention heads does the base model use?")
        # One call: generation only, no condense round-trip.
        assert len(llm.calls) == 1

    def test_pronoun_follow_ups_are_condensed(self, retriever, settings):
        llm = ScriptedLLM("How many heads does the transformer base model use?")
        pipeline = RAGPipeline(retriever, llm, settings)
        history = [("user", "Tell me about the transformer"), ("assistant", "It is an architecture.")]
        result = events(pipeline, "How many heads does it use?", history=history)
        assert len(llm.calls) == 2  # condense + generate
        assert result["query"] != "How many heads does it use?"

    def test_very_short_follow_ups_are_condensed(self, retriever, settings):
        llm = ScriptedLLM("What is the rank r in LoRA?")
        pipeline = RAGPipeline(retriever, llm, settings)
        history = [("user", "Explain LoRA"), ("assistant", "It uses low-rank matrices.")]
        events(pipeline, "and the rank?", history=history)
        assert len(llm.calls) == 2

    def test_no_history_means_no_condensing(self, retriever, settings):
        llm = ScriptedLLM()
        pipeline = RAGPipeline(retriever, llm, settings)
        result = events(pipeline, "What is it?")
        assert result["query"] == "What is it?"
        assert len(llm.calls) == 1

    def test_condense_failure_falls_back_to_the_raw_question(self, retriever, settings):
        class BrokenCondenser:
            model = "broken"
            calls = 0

            def stream(self, system, user):
                BrokenCondenser.calls += 1
                if BrokenCondenser.calls == 1:
                    raise LLMTransientError("provider down")
                yield "fallback answer [1]"

        pipeline = RAGPipeline(retriever, BrokenCondenser(), settings)
        history = [("user", "Explain LoRA"), ("assistant", "Low-rank matrices.")]
        result = events(pipeline, "and it?", history=history)
        assert result["query"] == "and it?"
        assert result["done"].answer  # still answered


class TestGenerationFailure:
    def test_llm_error_yields_an_error_event_not_a_crash(self, retriever, settings):
        class BrokenLLM:
            model = "broken"

            def stream(self, system, user):
                raise LLMTransientError("provider unreachable")
                yield  # pragma: no cover

        pipeline = RAGPipeline(retriever, BrokenLLM(), settings)
        names = [n for n, _ in pipeline.stream_answer("attention")]
        assert "error" in names
        assert "sources" in names  # retrieval succeeded and is still reported


class TestBlockingWrapper:
    def test_answer_returns_the_final_result(self, pipeline):
        result = pipeline.answer("multi-head attention")
        assert result.answer
        assert result.retrieved
        assert "total_ms" in result.timings_ms

    def test_timings_are_recorded(self, pipeline):
        timings = pipeline.answer("attention").timings_ms
        assert timings["total_ms"] > 0
        assert "generation_ms" in timings
