"""The preloaded corpus: a curated reading list of the papers this system is built on.

The choice of corpus is not incidental. A demo indexed over generic web pages
proves nothing, because a visitor cannot tell a correct answer from a fluent
one. These papers are ones that anyone evaluating a retrieval system already
knows, so the answers are checkable on sight — and the retrieval literature is
over-represented on purpose, because a RAG system whose corpus is the RAG
literature can be asked to explain its own design.

Papers are downloaded from arXiv at ingest time rather than committed to the
repository: they are the authors' work, not mine, and a 60MB blob of PDFs does
not belong in git. The built index is committed instead, so a deployment does
not need to re-download anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Paper:
    arxiv_id: str
    title: str
    year: str
    note: str = ""

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def filename(self) -> str:
        return f"{self.arxiv_id.replace('/', '_')}.pdf"


CORPUS: list[Paper] = [
    # --- Architecture -------------------------------------------------------
    Paper("1706.03762", "Attention Is All You Need", "2017", "The transformer."),
    Paper("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers", "2018"),
    Paper("1910.10683", "Exploring the Limits of Transfer Learning with T5", "2019"),
    Paper("2104.09864", "RoFormer: Enhanced Transformer with Rotary Position Embedding", "2021"),
    Paper("2205.14135", "FlashAttention: Fast and Memory-Efficient Exact Attention", "2022"),

    # --- Scale and training -------------------------------------------------
    Paper("2001.08361", "Scaling Laws for Neural Language Models", "2020"),
    Paper("2005.14165", "Language Models are Few-Shot Learners (GPT-3)", "2020"),
    Paper("2203.15556", "Training Compute-Optimal Large Language Models (Chinchilla)", "2022"),
    Paper("2302.13971", "LLaMA: Open and Efficient Foundation Language Models", "2023"),
    Paper("2307.09288", "Llama 2: Open Foundation and Fine-Tuned Chat Models", "2023"),

    # --- Alignment and adaptation -------------------------------------------
    Paper("2203.02155", "Training Language Models to Follow Instructions (InstructGPT)", "2022"),
    Paper("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", "2021"),
    Paper("2305.18290", "Direct Preference Optimization", "2023"),

    # --- Reasoning and agents -----------------------------------------------
    Paper("2201.11903", "Chain-of-Thought Prompting Elicits Reasoning", "2022"),
    Paper("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", "2022"),

    # --- Retrieval: the part this system is made of -------------------------
    Paper("2004.04906", "Dense Passage Retrieval for Open-Domain QA", "2020"),
    Paper("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP", "2020"),
    Paper("2007.01282", "Leveraging Passage Retrieval with Generative Models (FiD)", "2020"),
    Paper("2112.09118", "Unsupervised Dense Information Retrieval (Contriever)", "2021"),
    Paper("2212.10496", "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)", "2022"),
    Paper("2310.11511", "Self-RAG: Learning to Retrieve, Generate and Critique", "2023"),
    Paper("2401.18059", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", "2024"),
    Paper("2404.16130", "From Local to Global: A Graph RAG Approach", "2024"),
    Paper("2312.10997", "Retrieval-Augmented Generation for LLMs: A Survey", "2023"),
]

BY_ID = {p.arxiv_id: p for p in CORPUS}
