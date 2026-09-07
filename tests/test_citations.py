from ragcore.citations import (
    citation_coverage,
    extract_markers,
    normalize_markers,
    validate_and_rewrite,
)
from ragcore.types import Chunk, ScoredChunk


def make_chunks(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"doc::{i}",
                doc_id="doc",
                text=f"prefix {i}",
                body=f"Body text of passage {i}. " * 20,
                title=f"Paper {i}",
                section="Method",
                page=i + 1,
            ),
            score=1.0,
            dense_rank=i + 1,
            rerank_score=2.0 - i,
        )
        for i in range(n)
    ]


class TestNormalizeMarkers:
    """gpt-oss models emit OpenAI's own citation syntax regardless of the
    prompt. If it is not normalised, every marker fails validation, no chips
    render, and the answer still *reads* fine — a silent failure that only
    shows up by looking at the UI."""

    def test_gpt_oss_dagger_format(self):
        assert normalize_markers("tasks 【1†L9-L12】") == "tasks [1]"

    def test_multiple_dagger_markers(self):
        assert normalize_markers("a 【1†L9-L12】 b 【2†L1-L4】") == "a [1] b [2]"

    def test_full_width_brackets_without_dagger(self):
        assert normalize_markers("claim 【3】") == "claim [3]"

    def test_ascii_brackets_with_dagger(self):
        assert normalize_markers("claim [2†source]") == "claim [2]"

    def test_plain_markers_are_untouched(self):
        assert normalize_markers("claim [1] and [2].") == "claim [1] and [2]."

    def test_ordinary_text_is_untouched(self):
        assert normalize_markers("no markers at all here") == "no markers at all here"

    def test_normalised_markers_survive_validation(self):
        """End to end: the alternate syntax must produce real citations."""
        chunks = make_chunks(3)
        answer, cites = validate_and_rewrite(
            "The token aggregates the sequence 【1†L9-L12】 【2†L1-L4】", chunks
        )
        assert "【" not in answer
        assert len(cites) == 2
        assert [c.marker for c in cites] == [1, 2]

    def test_out_of_range_alternate_markers_are_still_stripped(self):
        answer, cites = validate_and_rewrite("Claim 【9†L1-L2】", make_chunks(2))
        assert "9" not in answer
        assert cites == []

    def test_coverage_counts_normalised_markers(self):
        chunks = make_chunks(2)
        text = "This is a properly substantive sentence about the method 【1†L1-L4】."
        assert citation_coverage(text, chunks) == 1.0


class TestExtractMarkers:
    def test_finds_markers_in_order(self):
        assert extract_markers("A [2] B [1] C") == [2, 1]

    def test_deduplicates(self):
        assert extract_markers("[1] and again [1]") == [1]

    def test_ignores_non_numeric_brackets(self):
        assert extract_markers("see [note] and [x1]") == []

    def test_no_markers(self):
        assert extract_markers("plain prose") == []


class TestValidateAndRewrite:
    def test_valid_citations_survive(self):
        chunks = make_chunks(3)
        answer, cites = validate_and_rewrite("Claim one [1]. Claim two [2].", chunks)
        assert "[1]" in answer and "[2]" in answer
        assert [c.marker for c in cites] == [1, 2]

    def test_out_of_range_markers_are_stripped(self):
        """The failure this function exists for: a model citing passage [7]
        when only three passages were retrieved."""
        chunks = make_chunks(3)
        answer, cites = validate_and_rewrite("Real [1]. Invented [7].", chunks)
        assert "[7]" not in answer
        assert len(cites) == 1

    def test_survivors_are_renumbered_densely(self):
        """If the model only cited passages 2 and 5, the reader should see
        [1] and [2] — a citation list starting at [2] looks like a bug."""
        chunks = make_chunks(6)
        answer, cites = validate_and_rewrite("First [2]. Second [5].", chunks)
        assert "[1]" in answer and "[2]" in answer
        assert "[5]" not in answer
        assert [c.marker for c in cites] == [1, 2]
        assert cites[0].chunk_id == "doc::1"  # old [2] -> chunk index 1
        assert cites[1].chunk_id == "doc::4"  # old [5] -> chunk index 4

    def test_renumbering_follows_first_appearance(self):
        chunks = make_chunks(4)
        answer, cites = validate_and_rewrite("A [3]. B [1]. C [3] again.", chunks)
        assert answer.startswith("A [1]")
        assert "B [2]" in answer
        assert "C [1] again" in answer
        assert len(cites) == 2

    def test_consecutive_duplicate_markers_are_collapsed(self):
        """Models sometimes emit "[1][1]", which renders as two identical
        chips and reads as a rendering bug."""
        chunks = make_chunks(3)
        answer, cites = validate_and_rewrite("The ratio is 1:1 [1][1]. Next [2].", chunks)
        assert "[1][1]" not in answer
        assert answer.count("[1]") == 1
        assert len(cites) == 2

    def test_duplicates_separated_by_whitespace_are_collapsed(self):
        answer, _ = validate_and_rewrite("Claim [1] [1]. Next.", make_chunks(2))
        assert answer.count("[1]") == 1

    def test_distinct_adjacent_markers_are_preserved(self):
        """[1][2] is a legitimate multi-source citation and must survive."""
        answer, cites = validate_and_rewrite("Claim [1][2]. Next.", make_chunks(3))
        assert "[1][2]" in answer
        assert len(cites) == 2

    def test_the_same_passage_cited_later_is_preserved(self):
        """Only consecutive repeats collapse; citing [1] again in a later
        sentence is correct and must be left alone."""
        answer, _ = validate_and_rewrite(
            "First claim [1]. A second, separate claim also rests on it [1].",
            make_chunks(2),
        )
        assert answer.count("[1]") == 2

    def test_uncited_answer_yields_no_citations(self):
        answer, cites = validate_and_rewrite("No citations here at all.", make_chunks(3))
        assert cites == []
        assert answer == "No citations here at all."

    def test_no_chunks_strips_everything(self):
        answer, cites = validate_and_rewrite("Claim [1].", [])
        assert "[1]" not in answer
        assert cites == []

    def test_punctuation_is_tidied_after_stripping(self):
        answer, _ = validate_and_rewrite("A claim [9] .", make_chunks(2))
        assert "  " not in answer
        assert " ." not in answer

    def test_citation_carries_display_metadata(self):
        chunks = make_chunks(2)
        _, cites = validate_and_rewrite("Claim [1].", chunks)
        c = cites[0]
        assert c.title == "Paper 0"
        assert c.section == "Method"
        assert c.page == 1
        assert c.retriever_path == "dense"
        assert c.snippet.endswith("…")

    def test_snippet_is_truncated(self):
        _, cites = validate_and_rewrite("Claim [1].", make_chunks(1))
        assert len(cites[0].snippet) <= 321


class TestCitationCoverage:
    def test_fully_cited_answer_scores_one(self):
        chunks = make_chunks(2)
        text = (
            "This is a substantive first sentence about the method [1]. "
            "This is a substantive second sentence about results [2]."
        )
        assert citation_coverage(text, chunks) == 1.0

    def test_uncited_answer_scores_zero(self):
        text = (
            "This is a substantive first sentence about the method. "
            "This is a substantive second sentence about the results."
        )
        assert citation_coverage(text, make_chunks(2)) == 0.0

    def test_partial_coverage(self):
        chunks = make_chunks(2)
        text = (
            "This first substantive sentence carries a citation [1]. "
            "This second substantive sentence carries none at all."
        )
        assert citation_coverage(text, chunks) == 0.5

    def test_short_fragments_are_not_counted(self):
        # "Yes." should not drag the score down; only substantive sentences
        # are expected to carry evidence.
        chunks = make_chunks(1)
        text = "Yes. This is a properly substantive sentence with a citation [1]."
        assert citation_coverage(text, chunks) == 1.0

    def test_out_of_range_citation_does_not_count_as_covered(self):
        chunks = make_chunks(2)
        text = "This substantive sentence cites a passage that does not exist [9]."
        assert citation_coverage(text, chunks) == 0.0

    def test_empty_answer(self):
        assert citation_coverage("", make_chunks(2)) == 0.0
