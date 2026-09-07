"""Citation extraction and validation.

A model asked to cite will occasionally cite a passage number that does not
exist, or cite nothing at all. Both are silent failures — the answer still
reads well — so they are handled explicitly rather than trusted.

Two operations:

* `extract_markers` finds every [n] the model wrote.
* `validate_and_rewrite` drops markers pointing outside the retrieved set and
  renumbers the survivors densely, so the UI never shows a citation chip that
  links to nothing, and the numbers a reader sees always start at [1].
"""

from __future__ import annotations

import re

from .types import Citation, ScoredChunk

MARKER_RE = re.compile(r"\[(\d{1,2})\]")

# A model will sometimes cite the same passage twice in a row ("…[1][1].").
# It is not wrong, but it renders as two identical chips and reads as a bug.
# Only *consecutive* repeats are collapsed — the same passage cited again
# later in the answer is legitimate and is left alone.
REPEATED_MARKER_RE = re.compile(r"(\[\d{1,2}\])(?:\s*\1)+")


# Models carry strong trained priors for their provider's own citation syntax
# and will use it regardless of what the prompt asks for. OpenAI's gpt-oss
# models emit "【1†L9-L12】"; others use "[1†source]". Normalising these at the
# boundary is far more robust than trying to win the argument in the prompt:
# swapping the generation model should never silently break citation
# rendering, and this failure is quiet — the answer still reads perfectly,
# it just has no working citations.
ALT_MARKER_RES = (
    re.compile(r"【\s*(\d{1,2})\s*(?:†[^】]*)?】"),
    re.compile(r"〔\s*(\d{1,2})\s*(?:†[^〕]*)?〕"),
    re.compile(r"\[\s*(\d{1,2})\s*†[^\]]*\]"),
)


def normalize_markers(text: str) -> str:
    """Rewrite known provider-specific citation syntaxes as plain [n]."""
    for pattern in ALT_MARKER_RES:
        text = pattern.sub(lambda m: f"[{m.group(1)}]", text)
    return text


def extract_markers(text: str) -> list[int]:
    """Every citation number in `text`, in order of first appearance."""
    seen: list[int] = []
    for m in MARKER_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def validate_and_rewrite(
    answer: str, chunks: list[ScoredChunk], *, snippet_chars: int = 320
) -> tuple[str, list[Citation]]:
    """Strip invalid markers, renumber the rest, and build the citation list.

    Returns the rewritten answer and the citations actually referenced, in
    the order the reader encounters them.
    """
    answer = normalize_markers(answer)
    valid_range = range(1, len(chunks) + 1)
    used = [n for n in extract_markers(answer) if n in valid_range]

    # Old marker -> new dense marker, assigned by order of first appearance.
    remap = {old: new for new, old in enumerate(used, start=1)}

    def _sub(match: re.Match[str]) -> str:
        n = int(match.group(1))
        return f"[{remap[n]}]" if n in remap else ""

    rewritten = MARKER_RE.sub(_sub, answer)
    rewritten = REPEATED_MARKER_RE.sub(r"\1", rewritten)
    # Collapse whitespace left behind by removed markers, without touching
    # paragraph breaks.
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
    rewritten = re.sub(r" ([.,;:)])", r"\1", rewritten)

    citations: list[Citation] = []
    for old in used:
        sc = chunks[old - 1]
        c = sc.chunk
        body = c.body.strip().replace("\n", " ")
        snippet = body[:snippet_chars] + ("…" if len(body) > snippet_chars else "")
        citations.append(
            Citation(
                marker=remap[old],
                chunk_id=c.chunk_id,
                title=c.title,
                source=c.source,
                section=c.section,
                page=c.page,
                snippet=snippet,
                retriever_path=sc.retriever_path,
                rerank_score=sc.rerank_score,
            )
        )
    return rewritten.strip(), citations


def citation_coverage(answer: str, chunks: list[ScoredChunk]) -> float:
    """Fraction of substantive sentences carrying at least one citation.

    A cheap, model-free proxy for groundedness used by the evaluation harness.
    It cannot detect a *wrong* citation, only a missing one — but an uncited
    sentence in a system prompted to cite everything is the strongest
    available signal that the model went outside its evidence.
    """
    answer = normalize_markers(answer)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 25]
    if not sentences:
        return 0.0
    valid = set(range(1, len(chunks) + 1))
    cited = sum(1 for s in sentences if any(n in valid for n in extract_markers(s)))
    return cited / len(sentences)
