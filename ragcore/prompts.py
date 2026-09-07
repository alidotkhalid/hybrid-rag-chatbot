"""Prompt construction.

Kept in its own module because prompts are the part of a RAG system that gets
changed most often and reviewed least carefully. Isolating them makes the
diff of a prompt change readable and lets the evaluation harness pin a prompt
version alongside a metrics run.
"""

from __future__ import annotations

from .types import ScoredChunk

SYSTEM_PROMPT = """\
You are a research assistant answering questions strictly from a set of \
retrieved passages taken from machine-learning papers and user-uploaded documents.

Rules, in priority order:

1. Ground every factual claim in the passages. If the passages do not contain \
the answer, say so plainly and stop — do not fall back on your own knowledge, \
and do not speculate about what the papers "probably" say.
2. Cite with plain ASCII square brackets around a number, exactly like [2] or \
[1][4], referring to the passage numbers given below. Never use any other \
citation syntax - no daggers, no line ranges, no full-width brackets. Place a citation immediately after the claim it supports, \
not bundled at the end of the paragraph. Every substantive sentence needs one.
3. If passages disagree, say so and cite both sides rather than silently \
picking one.
4. Answer in prose. Use a short list only when the question genuinely asks for \
an enumeration. Do not restate the question, do not open with a preamble, and \
do not describe what you are about to do.
5. Be concise. Three or four sentences is usually right; go longer only when \
the question needs it.
6. Never invent a citation number that does not appear in the passages."""


CONDENSE_PROMPT = """\
Rewrite the follow-up question as a standalone question that can be understood \
without the conversation history. Resolve every pronoun and implicit reference \
using the history. Preserve the original wording and technical terms wherever \
possible — do not add detail that is not there, and do not answer the question.

Conversation so far:
{history}

Follow-up question: {question}

Standalone question:"""


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as numbered passages.

    Each passage carries its source and section, which does double duty: it
    gives the model the context needed to interpret the text, and it makes a
    fabricated citation obvious to a reader checking the answer.
    """
    blocks = []
    for i, sc in enumerate(chunks, start=1):
        c = sc.chunk
        header = f"[{i}] {c.title}"
        if c.section and c.section.lower() != "preamble":
            header += f" — {c.section}"
        if c.page:
            header += f" (p. {c.page})"
        blocks.append(f"{header}\n{c.body}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    return (
        f"{format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the passages above, with bracketed citations."
    )


def build_condense_prompt(history: list[tuple[str, str]], question: str) -> str:
    rendered = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}" for role, text in history
    )
    return CONDENSE_PROMPT.format(history=rendered, question=question)


NO_ANSWER_MESSAGE = (
    "I could not find anything in the indexed documents that answers this. "
    "The corpus covers foundational papers on transformers, language models "
    "and retrieval-augmented generation — plus anything you upload. "
    "Try rephrasing, or upload a document that covers the topic."
)
