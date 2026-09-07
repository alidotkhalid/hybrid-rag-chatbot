from ragcore.chunking import (
    chunk_document,
    count_tokens,
    detect_heading,
    split_into_sections,
    split_sentences,
)
from ragcore.types import Document


class TestHeadingDetection:
    def test_markdown_headings(self):
        assert detect_heading("## Model Architecture") == "Model Architecture"
        assert detect_heading("# Abstract") == "Abstract"

    def test_numbered_academic_headings(self):
        assert detect_heading("3 Method") == "3 Method"
        assert detect_heading("3.1 Training Data") == "3.1 Training Data"

    def test_known_section_names(self):
        assert detect_heading("Abstract") == "Abstract"
        assert detect_heading("References") == "References"
        assert detect_heading("Related Work") == "Related Work"

    def test_prose_is_not_a_heading(self):
        assert detect_heading("The model was trained on eight GPUs for 12 hours.") is None
        assert detect_heading("") is None
        assert detect_heading("   ") is None

    def test_long_lines_are_not_headings(self):
        assert detect_heading("A " * 100) is None


class TestSentenceSplitting:
    def test_basic(self):
        assert len(split_sentences("First one. Second one. Third one.")) == 3

    def test_does_not_split_on_abbreviations(self):
        text = "We use dropout, e.g. 0.1, in every layer. Results follow."
        assert len(split_sentences(text)) == 2

    def test_does_not_split_on_citation_style_abbrev(self):
        text = "Vaswani et al. introduced the transformer. It works well."
        assert len(split_sentences(text)) == 2

    def test_empty(self):
        assert split_sentences("") == []


class TestSectionSplitting:
    def test_splits_on_headings(self):
        text = "# One\n\nalpha text\n\n# Two\n\nbeta text"
        sections = split_into_sections(text)
        assert [name for name, _, _ in sections] == ["One", "Two"]

    def test_text_before_first_heading_is_kept(self):
        sections = split_into_sections("orphan line\n\n# One\n\nbody")
        assert sections[0][0] == "Preamble"
        assert "orphan" in sections[0][1]

    def test_no_headings_yields_single_section(self):
        sections = split_into_sections("just some prose with no structure at all")
        assert len(sections) == 1
        assert sections[0][0] == "Preamble"


class TestChunking:
    def test_produces_chunks_with_stable_ids(self, docs):
        chunks = chunk_document(docs[0])
        assert chunks
        assert all(c.chunk_id.startswith("attn::") for c in chunks)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_chunk_text_carries_title_and_section(self, docs):
        chunks = chunk_document(docs[0])
        arch = [c for c in chunks if "Architecture" in c.section]
        assert arch, "expected a chunk from the Model Architecture section"
        assert arch[0].text.startswith("Attention Is All You Need — Model Architecture")
        # The body must NOT contain the prefix — it is display text.
        assert not arch[0].body.startswith("Attention Is All You Need —")

    def test_chunks_respect_the_token_budget(self, docs):
        chunks = chunk_document(docs[0], target_tokens=60, overlap_tokens=10)
        # Allow headroom for the prefix and for a single over-budget sentence
        # that cannot be split further.
        assert all(c.token_count < 200 for c in chunks)
        assert len(chunks) > 1

    def test_chunks_never_span_two_sections(self, docs):
        for chunk in chunk_document(docs[0], target_tokens=40):
            assert chunk.section in {"Abstract", "Model Architecture", "Training"}

    def test_overlap_repeats_content_between_neighbours(self):
        paragraphs = [f"Paragraph number {i} contains some filler words here." for i in range(12)]
        doc = Document(doc_id="d", title="T", text="# S\n\n" + "\n\n".join(paragraphs))
        chunks = chunk_document(doc, target_tokens=40, overlap_tokens=20)
        assert len(chunks) > 1
        # Consecutive chunks should share at least one paragraph.
        first_words = set(chunks[0].body.split())
        second_words = set(chunks[1].body.split())
        assert len(first_words & second_words) > 3

    def test_zero_overlap_is_supported(self):
        paragraphs = [f"Paragraph number {i} has filler words." for i in range(10)]
        doc = Document(doc_id="d", title="T", text="# S\n\n" + "\n\n".join(paragraphs))
        chunks = chunk_document(doc, target_tokens=30, overlap_tokens=0)
        assert len(chunks) > 1

    def test_tiny_tail_is_merged_not_emitted(self):
        doc = Document(
            doc_id="d",
            title="T",
            text="# S\n\n" + "word " * 200 + "\n\nshort tail.",
        )
        chunks = chunk_document(doc, target_tokens=100, min_tokens=48)
        assert all(c.token_count >= 20 for c in chunks)

    def test_empty_document_yields_nothing(self):
        assert chunk_document(Document(doc_id="d", title="T", text="")) == []

    def test_oversized_paragraph_is_split_on_sentences(self):
        long_para = " ".join(f"Sentence number {i} is here." for i in range(80))
        doc = Document(doc_id="d", title="T", text=f"# S\n\n{long_para}")
        chunks = chunk_document(doc, target_tokens=60)
        assert len(chunks) > 3

    def test_page_numbers_are_attributed(self):
        doc = Document(
            doc_id="d",
            title="T",
            text="# One\n\nalpha\n\n# Two\n\nbeta",
            page_map=[(0, 1), (14, 2)],
        )
        chunks = chunk_document(doc)
        assert {c.page for c in chunks} <= {1, 2}


class TestTokenCounting:
    def test_scales_with_length(self):
        assert count_tokens("one two three") < count_tokens("one two three four five six")

    def test_empty_is_cheap(self):
        assert count_tokens("") <= 1
