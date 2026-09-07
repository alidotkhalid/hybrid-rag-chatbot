"""Loader tests.

The PDF tests build real PDFs with PyMuPDF rather than committing fixture
binaries, so the repository stays text-only and the fixtures are readable.
"""

from __future__ import annotations

import pytest

from ragcore.loaders import _strip_repeated_lines, load_file, load_text


def make_pdf(path, pages: list[str]) -> None:
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 90), text, fontsize=11)
    doc.save(str(path))
    doc.close()


class TestTextAndMarkdown:
    def test_reads_plain_text(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("Some plain content here.", encoding="utf-8")
        doc = load_file(path)
        assert doc.text == "Some plain content here."
        assert doc.title == "notes"

    def test_takes_the_title_from_a_markdown_heading(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# Real Title\n\nBody text.", encoding="utf-8")
        assert load_file(path).title == "Real Title"

    def test_falls_back_to_the_filename(self, tmp_path):
        path = tmp_path / "untitled.md"
        path.write_text("No heading here at all.", encoding="utf-8")
        assert load_file(path).title == "untitled"

    def test_handles_bad_encoding_without_crashing(self, tmp_path):
        path = tmp_path / "weird.txt"
        path.write_bytes(b"valid text \xff\xfe then more")
        assert "valid text" in load_text(path).text

    def test_doc_ids_are_content_stable(self, tmp_path):
        for name in ("a.md", "b.md"):
            (tmp_path / name).write_text("# T\n\nsame body", encoding="utf-8")
        # Same content, different filename -> different id (source is part of
        # the hash), but the same file re-read must give the same id.
        first = load_file(tmp_path / "a.md").doc_id
        second = load_file(tmp_path / "a.md").doc_id
        assert first == second
        assert first != load_file(tmp_path / "b.md").doc_id


class TestPDF:
    def test_extracts_text(self, tmp_path):
        path = tmp_path / "paper.pdf"
        make_pdf(path, ["The Transformer uses multi-head attention."])
        doc = load_file(path)
        assert "multi-head attention" in doc.text
        assert doc.source == "paper.pdf"

    def test_builds_a_page_map(self, tmp_path):
        path = tmp_path / "multi.pdf"
        make_pdf(path, ["First page content here.", "Second page content here."])
        doc = load_file(path)
        assert len(doc.page_map) == 2
        assert doc.page_map[0][1] == 1
        assert doc.page_map[1][1] == 2

    def test_page_lookup_by_offset(self, tmp_path):
        path = tmp_path / "multi.pdf"
        make_pdf(path, ["First page content here.", "Second page content here."])
        doc = load_file(path)
        assert doc.page_for_offset(0) == 1
        assert doc.page_for_offset(len(doc.text) - 1) == 2

    def test_rejoins_hyphenated_line_breaks(self, tmp_path):
        # The cleanup runs on extracted text; assert on the helper directly
        # since PDF layout controls where breaks actually land.
        from ragcore.loaders import _clean_page

        assert "attention" in _clean_page("atten-\ntion mechanism")

    def test_normalises_ligatures(self, tmp_path):
        from ragcore.loaders import _clean_page

        assert _clean_page("eﬃcient ﬁne-tuning") == "efficient fine-tuning"

    def test_preserves_paragraph_breaks(self, tmp_path):
        from ragcore.loaders import _clean_page

        cleaned = _clean_page("First para line one\nline two\n\nSecond para")
        assert "line one line two" in cleaned
        assert "\n\n" in cleaned


class TestBoilerplateStripping:
    def test_removes_a_running_header(self):
        pages = [f"Preprint under review\nUnique body text for page {i}" for i in range(8)]
        cleaned = _strip_repeated_lines(pages)
        assert all("Preprint under review" not in p for p in cleaned)
        assert all("Unique body text" in p for p in cleaned)

    def test_leaves_short_documents_alone(self):
        pages = ["Repeated line\nbody"] * 3
        assert _strip_repeated_lines(pages) == pages

    def test_keeps_a_line_that_appears_only_sometimes(self):
        pages = [f"body {i}" for i in range(8)]
        pages[0] = "Occasional note\n" + pages[0]
        pages[1] = "Occasional note\n" + pages[1]
        cleaned = _strip_repeated_lines(pages)
        assert "Occasional note" in cleaned[0]


class TestDocx:
    def test_extracts_paragraphs_and_headings(self, tmp_path):
        docx = pytest.importorskip("docx")
        path = tmp_path / "report.docx"
        document = docx.Document()
        document.add_heading("Findings", level=1)
        document.add_paragraph("The results were conclusive.")
        document.save(str(path))

        doc = load_file(path)
        assert "# Findings" in doc.text
        assert "conclusive" in doc.text
        assert doc.title == "Findings"


class TestUnsupported:
    def test_raises_with_a_helpful_message(self, tmp_path):
        path = tmp_path / "x.parquet"
        path.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="Supported"):
            load_file(path)


def make_two_column_pdf(path) -> None:
    """A page laid out like an arXiv paper: full-width title, then two columns."""
    import fitz

    left = [
        "Input/Output Representations",
        "To make BERT handle a variety",
        "of down-stream tasks, our input",
        "representation is able to",
        "unambiguously represent tokens.",
    ]
    right = [
        "In order to train a deep",
        "bidirectional representation,",
        "we simply mask some percentage",
        "of the input tokens at random,",
        "and then predict those tokens.",
    ]
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "BERT: Pre-training of Deep Bidirectional Transformers", fontsize=13)
    for i, line in enumerate(left):
        page.insert_text((72, 120 + i * 16), line, fontsize=10)
    for i, line in enumerate(right):
        page.insert_text((330, 120 + i * 16), line, fontsize=10)
    doc.save(str(path))
    doc.close()


class TestTwoColumnLayout:
    """The failure this guards against is silent: text extracts, chunks build,
    embeddings compute, and every passage is quietly incoherent because the
    two columns were read across the page instead of down each one."""

    def test_columns_are_not_interleaved(self, tmp_path):
        path = tmp_path / "twocol.pdf"
        make_two_column_pdf(path)
        doc = load_file(path)

        # The whole left column must appear before any of the right column.
        last_left = doc.text.index("unambiguously represent tokens")
        first_right = doc.text.index("In order to train a deep")
        assert last_left < first_right, f"columns interleaved:\n{doc.text}"

    def test_left_column_stays_contiguous(self, tmp_path):
        path = tmp_path / "twocol.pdf"
        make_two_column_pdf(path)
        text = load_file(path).text
        start = text.index("Input/Output Representations")
        end = text.index("unambiguously represent tokens")
        # No right-column content may appear inside the left column's span.
        assert "In order to train" not in text[start:end]
        assert "we simply mask" not in text[start:end]

    def test_full_width_title_comes_first(self, tmp_path):
        path = tmp_path / "twocol.pdf"
        make_two_column_pdf(path)
        text = load_file(path).text
        assert text.index("BERT: Pre-training") < text.index("Input/Output Representations")

    def test_single_column_pages_are_unaffected(self, tmp_path):
        """The column logic must not disturb ordinary single-column documents."""
        path = tmp_path / "single.pdf"
        make_pdf(path, ["First line here.\nSecond line here.\nThird line here."])
        text = load_file(path).text
        assert text.index("First line") < text.index("Second line") < text.index("Third line")


class TestParagraphStructureSurvivesExtraction:
    """Reading order is not the only thing extraction has to get right.

    Blocks joined without a blank line leave only single newlines, which
    `_clean_page` folds into spaces — so the page becomes one paragraph, the
    chunker falls back to sentence splitting, and chunks silently exceed the
    configured token budget. Nothing errors; the index is just worse.
    """

    def test_paragraph_breaks_are_preserved(self, tmp_path):
        import fitz

        path = tmp_path / "paras.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        body = (
            "The Transformer architecture relies entirely on attention to draw "
            "global dependencies between input and output positions. "
        )
        y = 100
        for i in range(5):
            page.insert_textbox(fitz.Rect(60, y, 290, y + 95), f"Left {i}. " + body, fontsize=9)
            page.insert_textbox(fitz.Rect(320, y, 550, y + 95), f"Right {i}. " + body, fontsize=9)
            y += 105
        doc.save(str(path))
        doc.close()

        text = load_file(path).text
        assert text.count("\n\n") >= 5, "paragraph boundaries were lost"

    def test_chunks_stay_within_the_token_budget(self, tmp_path):
        from ragcore.chunking import chunk_documents

        path = tmp_path / "budget.pdf"
        doc = fitz_new_two_column(path)  # noqa: F841

        document = load_file(path)
        chunks = chunk_documents([document], target_tokens=320, overlap_tokens=64)
        assert chunks
        oversized = [c for c in chunks if c.token_count > 420]
        assert not oversized, (
            f"{len(oversized)} chunk(s) exceeded the budget; "
            "paragraph structure was probably lost during extraction"
        )


def fitz_new_two_column(path):
    import fitz

    body = (
        "The Transformer architecture relies entirely on attention mechanisms "
        "to draw global dependencies between input and output positions. This "
        "removes the sequential computation inherent to recurrent models. "
    )
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "A Two Column Paper About Attention", fontsize=13)
    y = 110
    for i in range(6):
        page.insert_textbox(fitz.Rect(60, y, 290, y + 95), f"Paragraph {i}. " + body, fontsize=9)
        page.insert_textbox(fitz.Rect(320, y, 550, y + 95), f"Right {i}. " + body, fontsize=9)
        y += 105
    doc.save(str(path))
    doc.close()
    return path


class TestTitleHeuristic:
    """The fallback used for uploaded documents. For the curated corpus,
    ingestion overrides it with the known title from corpus.py — guessing is
    not reliable enough to put in a permanently visible UI list."""

    def _title_of(self, tmp_path, first_page: str) -> str:
        from ragcore.loaders import _title_from_text

        return _title_from_text(first_page)

    def test_picks_a_real_title(self, tmp_path):
        page = "arXiv:2210.03629v3 [cs.CL] 10 Mar 2023\nReAct: Synergizing Reasoning and Acting in Language Models\nShunyu Yao"
        assert self._title_of(tmp_path, page).startswith("ReAct:")

    def test_rejects_a_bare_arxiv_identifier(self, tmp_path):
        page = "2210.03629\nReAct: Synergizing Reasoning and Acting in Language Models"
        assert self._title_of(tmp_path, page).startswith("ReAct:")

    def test_rejects_an_author_list_with_affiliation_markers(self, tmp_path):
        page = "Akari Asai†, Zeqiu Wu†, Yizhong Wang†§, Avirup Sil‡\nSelf-RAG: Learning to Retrieve, Generate and Critique"
        assert self._title_of(tmp_path, page).startswith("Self-RAG")

    def test_rejects_a_comma_heavy_author_list(self, tmp_path):
        page = "Jane Doe, John Roe, Ada Lovelace, Alan Turing\nAttention Is All You Need Revisited"
        assert self._title_of(tmp_path, page) == "Attention Is All You Need Revisited"

    def test_rejects_an_email_line(self, tmp_path):
        page = "correspondence to someone@example.com here\nA Real Paper Title About Retrieval"
        assert self._title_of(tmp_path, page) == "A Real Paper Title About Retrieval"

    def test_rejects_abstract_and_preprint_banners(self, tmp_path):
        page = "Preprint under review at ICLR 2024\nAbstract\nDense Passage Retrieval for Open Domain QA"
        assert self._title_of(tmp_path, page).startswith("Dense Passage Retrieval")

    def test_returns_empty_when_nothing_qualifies(self, tmp_path):
        assert self._title_of(tmp_path, "arXiv:1234.5678\n2024\nx") == ""


class TestCuratedTitlesOverrideGuessing:
    def test_known_arxiv_id_gets_its_real_title(self):
        from ragcore.corpus import BY_ID
        from ragcore.ingest import _apply_known_titles
        from ragcore.types import Document

        docs = [
            Document(doc_id="a", title="2210.03629", text="body", source="2210.03629.pdf"),
            Document(doc_id="b", title="Akari Asai†, Zeqiu Wu†", text="body", source="2310.11511.pdf"),
        ]
        renamed = _apply_known_titles(docs)
        assert renamed == 2
        assert docs[0].title == BY_ID["2210.03629"].title
        assert docs[1].title == BY_ID["2310.11511"].title

    def test_unknown_files_keep_their_scraped_title(self):
        from ragcore.ingest import _apply_known_titles
        from ragcore.types import Document

        docs = [Document(doc_id="x", title="My Own Report", text="b", source="my_report.pdf")]
        assert _apply_known_titles(docs) == 0
        assert docs[0].title == "My Own Report"

    def test_every_corpus_paper_has_a_usable_title(self):
        """Guards the manifest itself: a blank or ID-like title there would
        propagate straight into the UI."""
        from ragcore.corpus import CORPUS

        for paper in CORPUS:
            assert len(paper.title) > 10, paper.arxiv_id
            assert not paper.title[0].isdigit(), paper.arxiv_id
