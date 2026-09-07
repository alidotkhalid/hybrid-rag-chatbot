"""Document loaders.

PDF extraction is where most of the quality of a paper-based RAG system is won
or lost. Academic PDFs are two-column, header/footer-laden, and full of
equations that extract as line noise. The cleanup below is targeted at exactly
those artefacts and is deliberately conservative: it is better to leave a
little junk in a chunk than to delete a sentence that happened to look like a
header.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from .types import Document

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".docx"}


def _doc_id(source: str, text: str) -> str:
    """Stable ID from source + content, so re-ingesting an unchanged file
    produces the same chunk IDs and the index diff stays readable."""
    h = hashlib.blake2b(f"{source}\0{text[:4096]}".encode(), digest_size=6)
    return h.hexdigest()


# --- PDF --------------------------------------------------------------------

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "—", " ": " ",
}


def _clean_page(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # Rejoin words hyphenated across a line break: "atten-\ntion" -> "attention".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A single newline inside a paragraph is a layout artefact; a blank line is
    # a real paragraph break. Preserve the latter, join the former.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers.

    Detected by frequency rather than position: a short line appearing on more
    than 60% of pages is a header, a page number, or a venue watermark, and
    none of those should be retrievable text. Requires 4+ pages so that a
    short document with a repeated section title is not damaged.
    """
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if 0 < len(ln.strip()) < 80}:
            counts[line] += 1
    threshold = max(3, int(len(pages) * 0.6))
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    if not boilerplate:
        return pages
    return [
        "\n".join(ln for ln in page.splitlines() if ln.strip() not in boilerplate)
        for page in pages
    ]


def _page_text_in_reading_order(page) -> str:
    """Extract a page's text down each column rather than across the page.

    PyMuPDF's `get_text("text", sort=True)` sorts blocks primarily by vertical
    position. On a single-column page that is correct. On the two-column
    layout used by almost every arXiv paper it is actively wrong: blocks from
    the left and right columns sit at the same height, so they interleave, and
    the extracted text reads

        "Input/Output Representations  In order to train a deep
         To make BERT handle a variety  bidirectional representation, ..."

    Every chunk built from that is incoherent, which poisons both embeddings
    and BM25 — and it is invisible unless you actually read the retrieved
    passages, which is why it survived the first round of testing here.

    The fix is to group blocks into columns before ordering them:
    full-width blocks (title, abstract, wide figures) keep their position
    relative to the columns, and column blocks are read top-to-bottom, left
    column fully before the right.
    """
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if not blocks:
        return ""

    rect = page.rect
    midpoint = (rect.x0 + rect.x1) / 2
    width = rect.width or 1.0

    def is_full_width(b) -> bool:
        return (b[2] - b[0]) > width * 0.55

    column_blocks = [b for b in blocks if not is_full_width(b)]
    left = [b for b in column_blocks if (b[0] + b[2]) / 2 < midpoint]
    right = [b for b in column_blocks if (b[0] + b[2]) / 2 >= midpoint]

    # Not a two-column page (or too little evidence to be sure): the ordinary
    # top-to-bottom, left-to-right sort is correct and safer.
    # Blocks are joined with a blank line, not nothing: they are
    # paragraph-sized units, and `_clean_page` folds *single* newlines into
    # spaces. Without the blank line every paragraph boundary on the page is
    # lost, the chunker sees one enormous paragraph, and chunks blow past the
    # token budget (measured: mean 499 vs a 320 target).
    if len(left) < 2 or len(right) < 2:
        return "\n\n".join(b[4] for b in sorted(blocks, key=lambda b: (round(b[1]), b[0])))

    full = [b for b in blocks if is_full_width(b)]
    col_top = min(b[1] for b in column_blocks)
    col_bottom = max(b[3] for b in column_blocks)
    header = [b for b in full if b[3] <= col_top + 2]
    footer = [b for b in full if b[1] >= col_bottom - 2]
    middle = [b for b in full if b not in header and b not in footer]

    def by_y(b):
        return b[1]

    ordered = (
        sorted(header, key=by_y)
        + sorted(left, key=by_y)
        + sorted(right, key=by_y)
        + sorted(middle, key=by_y)
        + sorted(footer, key=by_y)
    )
    return "\n\n".join(b[4] for b in ordered)


def load_pdf(path: Path, title: str = "") -> Document:
    import fitz  # PyMuPDF

    with fitz.open(path) as pdf:
        raw_pages = [_page_text_in_reading_order(page) for page in pdf]
        meta_title = (pdf.metadata or {}).get("title") or ""

    pages = _strip_repeated_lines([_clean_page(p) for p in raw_pages])

    text_parts: list[str] = []
    page_map: list[tuple[int, int]] = []
    offset = 0
    for page_no, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        page_map.append((offset, page_no))
        text_parts.append(page_text)
        offset += len(page_text) + 2
    text = "\n\n".join(text_parts)

    resolved = title or _title_from_text(text) or meta_title.strip() or path.stem
    return Document(
        doc_id=_doc_id(path.name, text),
        title=resolved,
        text=text,
        source=path.name,
        page_map=page_map,
    )


# Lines that look like a title but are not. Author lists and arXiv stamps are
# the two that actually bite: both sit within the first few lines of page 1,
# and which one appears first depends on the PDF's block ordering.
_NOT_A_TITLE = re.compile(
    r"""(
        ^arxiv:                     # arXiv stamp
      | ^\d{4}\.\d{4,5}             # bare arXiv identifier
      | ^abstract\b
      | ^(under\s+)?review\b
      | ^preprint\b
      | @                           # email address
      | [†‡§¶]                       # affiliation markers -> author list
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _title_from_text(text: str) -> str:
    """Best-effort title from a document's first page.

    Only a fallback: for the curated corpus, ingestion overrides this with the
    known title from `corpus.py`, because guessing is not reliable enough to
    put in a UI.
    """
    for line in text.splitlines()[:12]:
        line = line.strip()
        if not (15 < len(line) < 160):
            continue
        if _NOT_A_TITLE.search(line):
            continue
        # A line that is mostly digits is a date, an ID or a table row.
        if sum(c.isdigit() for c in line) >= len(line) * 0.3:
            continue
        # Three or more commas is an author list, not a title.
        if line.count(",") >= 3:
            continue
        if len(line.split()) < 3:
            continue
        return line
    return ""


# --- Plain text / Markdown / DOCX -------------------------------------------


def load_text(path: Path, title: str = "") -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        doc_id=_doc_id(path.name, text),
        title=title or _markdown_title(text) or path.stem,
        text=text,
        source=path.name,
        page_map=[(0, 1)],
    )


def _markdown_title(text: str) -> str:
    for line in text.splitlines()[:20]:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def load_docx(path: Path, title: str = "") -> Document:
    import docx  # python-docx

    document = docx.Document(str(path))
    parts: list[str] = []
    for para in document.paragraphs:
        content = para.text.strip()
        if not content:
            continue
        # Preserve Word heading levels as Markdown so the chunker's structure
        # detection works on .docx exactly as it does on .md.
        style = (para.style.name or "").lower()
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            parts.append(f"{'#' * min(int(level), 6)} {content}")
        else:
            parts.append(content)
    text = "\n\n".join(parts)
    return Document(
        doc_id=_doc_id(path.name, text),
        title=title or _markdown_title(text) or path.stem,
        text=text,
        source=path.name,
        page_map=[(0, 1)],
    )


def load_file(path: Path, title: str = "") -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, title)
    if suffix == ".docx":
        return load_docx(path, title)
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text(path, title)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def load_directory(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                docs.append(load_file(path))
            except Exception as exc:  # noqa: BLE001 - one bad file must not
                # abort a 24-document ingest run.
                print(f"  ! skipped {path.name}: {exc}")
    return docs
