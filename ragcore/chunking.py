"""Structure-aware chunking.

Naive fixed-width chunking is the most common reason a RAG system retrieves
plausible-looking but useless passages: it slices mid-sentence, it separates a
claim from the heading that gives it meaning, and it produces chunks that are
lexically similar to everything else in the corpus.

This module does three things differently:

1. **Split on document structure first.** Sections are detected from headings
   (Markdown, numbered academic headings, ALL-CAPS lines) and never crossed by
   a chunk boundary.
2. **Pack whole paragraphs** into a token budget, only falling back to sentence
   splitting when a single paragraph exceeds the budget on its own.
3. **Prefix each chunk with its document title and section path.** The chunk
   that gets embedded is therefore self-describing, which measurably improves
   both dense recall and BM25 precision on section-scoped queries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .types import Chunk, Document

# --- Token counting ---------------------------------------------------------
#
# We deliberately avoid a tokenizer dependency here. Chunk sizing only needs to
# be approximately right, and a hard dependency on a downloadable tokenizer
# would make ingestion fail in offline/air-gapped environments. The ratio below
# is calibrated against BERT-family tokenizers on English academic prose, where
# a "word" (whitespace-delimited) averages ~1.3 word-pieces.

_WORD_RE = re.compile(r"\S+")
_TOKENS_PER_WORD = 1.3


def count_tokens(text: str) -> int:
    """Approximate word-piece count for `text`."""
    return int(len(_WORD_RE.findall(text)) * _TOKENS_PER_WORD) + 1


# --- Structure detection ----------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.{1,120})$")
# "3 Method", "3.1 Training data", "IV. Results"
_NUMBERED_HEADING = re.compile(r"^\s*((?:\d+|[IVXLC]+)(?:\.\d+)*)\.?\s+([A-Z][^.!?]{2,80})\s*$")
_CAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z \-&]{3,60})\s*$")

_ABSTRACT_ETC = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "appendix",
    "acknowledgements",
    "acknowledgments",
}


def detect_heading(line: str) -> str | None:
    """Return the heading text if `line` looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 130:
        return None

    m = _MD_HEADING.match(stripped)
    if m:
        return m.group(2).strip()

    m = _NUMBERED_HEADING.match(stripped)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()

    if stripped.lower().rstrip(":") in _ABSTRACT_ETC:
        return stripped.rstrip(":")

    m = _CAPS_HEADING.match(stripped)
    if m and len(stripped.split()) <= 8:
        return m.group(1).strip().title()

    return None


# --- Paragraph and sentence splitting ---------------------------------------

_PARA_SPLIT = re.compile(r"\n\s*\n+")

# Candidate sentence boundary: terminal punctuation, whitespace, then something
# that could start a sentence. Python's `re` does not support variable-width
# lookbehind, so abbreviations are filtered out after matching rather than
# excluded in the pattern.
_SENT_BOUNDARY = re.compile(r"[.!?]\s+(?=[\"“'(\[]?[A-Z0-9])")

# Tokens that end in a period but do not end a sentence. Academic prose is
# dense with these, and splitting on "et al." or "e.g." produces fragments that
# retrieve badly and read worse.
_ABBREVIATIONS = frozenset(
    """
    e.g i.e al cf vs fig eq sec tab ref approx dr mr mrs ms prof et
    no vol pp ch resp etc inc ltd jr sr st
    """.split()
)

_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.?$")


def _is_real_boundary(text: str, dot_index: int) -> bool:
    """Decide whether the terminal punctuation at `dot_index` ends a sentence."""
    if text[dot_index] in "!?":
        return True
    preceding = text[:dot_index]
    match = _TRAILING_WORD.search(preceding)
    if not match:
        return True
    word = match.group(1).rstrip(".").lower()
    if word in _ABBREVIATIONS:
        return False
    # A single capital letter is an initial ("J. Smith"), not a sentence end.
    return not (len(word) == 1 and preceding[-1:].isupper())


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENT_BOUNDARY.finditer(text):
        if not _is_real_boundary(text, match.start()):
            continue
        piece = text[start : match.start() + 1].strip()
        if piece:
            parts.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or ([text.strip()] if text.strip() else [])


def _split_oversized(paragraph: str, budget: int) -> list[str]:
    """Break a single over-budget paragraph on sentence boundaries."""
    sentences = split_sentences(paragraph)
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        st = count_tokens(sent)
        if buf and buf_tokens + st > budget:
            out.append(" ".join(buf))
            buf, buf_tokens = [], 0
        # A single sentence longer than the budget (tables, long equations)
        # is emitted as-is rather than mangled mid-clause.
        buf.append(sent)
        buf_tokens += st
    if buf:
        out.append(" ".join(buf))
    return out


# --- Section extraction -----------------------------------------------------


def split_into_sections(text: str) -> list[tuple[str, str, int]]:
    """Return [(section_name, section_text, char_offset), ...].

    Text appearing before the first heading is attributed to a synthetic
    "Preamble" section so that nothing is silently dropped.
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str, int]] = []
    current_name = "Preamble"
    current_lines: list[str] = []
    current_offset = 0
    offset = 0

    for line in lines:
        heading = detect_heading(line)
        if heading is not None:
            body = "".join(current_lines).strip()
            if body:
                sections.append((current_name, body, current_offset))
            current_name = heading
            current_lines = []
            current_offset = offset + len(line)
        else:
            current_lines.append(line)
        offset += len(line)

    body = "".join(current_lines).strip()
    if body:
        sections.append((current_name, body, current_offset))

    return sections


# --- Chunking ---------------------------------------------------------------


def _context_prefix(title: str, section: str) -> str:
    """The self-describing header prepended to every embedded chunk."""
    section = section.strip()
    if section and section.lower() != "preamble":
        return f"{title} — {section}\n\n"
    return f"{title}\n\n"


def chunk_document(
    doc: Document,
    *,
    target_tokens: int = 320,
    overlap_tokens: int = 64,
    min_tokens: int = 48,
) -> list[Chunk]:
    """Chunk one document into retrieval units."""
    chunks: list[Chunk] = []
    ordinal = 0

    for section_name, section_text, section_offset in split_into_sections(doc.text):
        paragraphs: list[str] = []
        for para in _PARA_SPLIT.split(section_text):
            para = re.sub(r"[ \t]*\n[ \t]*", " ", para).strip()
            if not para:
                continue
            if count_tokens(para) > target_tokens:
                paragraphs.extend(_split_oversized(para, target_tokens))
            else:
                paragraphs.append(para)

        # Pack paragraphs into windows, carrying `overlap_tokens` of trailing
        # text into the next window so a claim split across a boundary is
        # still fully present in at least one chunk.
        window: list[str] = []
        window_tokens = 0
        for para in paragraphs:
            pt = count_tokens(para)
            if window and window_tokens + pt > target_tokens:
                ordinal = _emit(
                    chunks, doc, section_name, section_offset, window, ordinal
                )
                window, window_tokens = _carry_overlap(window, overlap_tokens)
            window.append(para)
            window_tokens += pt

        if window:
            text = " ".join(window)
            if count_tokens(text) < min_tokens and chunks and chunks[-1].section == section_name:
                # Merge a runt tail into the previous chunk instead of
                # indexing a fragment that will match everything weakly.
                prev = chunks[-1]
                prev.body = f"{prev.body} {text}".strip()
                prev.text = _context_prefix(doc.title, section_name) + prev.body
                prev.token_count = count_tokens(prev.text)
            else:
                ordinal = _emit(
                    chunks, doc, section_name, section_offset, window, ordinal
                )

    return chunks


def _carry_overlap(window: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Take the trailing paragraphs of `window` worth ~`overlap_tokens`."""
    if overlap_tokens <= 0:
        return [], 0
    carried: list[str] = []
    total = 0
    for para in reversed(window):
        pt = count_tokens(para)
        if total + pt > overlap_tokens and carried:
            break
        carried.insert(0, para)
        total += pt
        if total >= overlap_tokens:
            break
    return carried, total


def _emit(
    chunks: list[Chunk],
    doc: Document,
    section: str,
    section_offset: int,
    window: list[str],
    ordinal: int,
) -> int:
    body = " ".join(window).strip()
    if not body:
        return ordinal
    prefixed = _context_prefix(doc.title, section) + body
    chunks.append(
        Chunk(
            chunk_id=f"{doc.doc_id}::{ordinal}",
            doc_id=doc.doc_id,
            text=prefixed,
            body=body,
            title=doc.title,
            source=doc.source,
            section=section,
            page=doc.page_for_offset(section_offset),
            ordinal=ordinal,
            token_count=count_tokens(prefixed),
        )
    )
    return ordinal + 1


def chunk_documents(docs: Iterable[Document], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kwargs))
    return out
