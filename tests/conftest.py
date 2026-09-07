"""Shared fixtures.

The whole suite runs without downloading a model or touching the network:
`HashEmbedder` stands in for the bi-encoder, `IdentityReranker` for the
cross-encoder, and `ScriptedLLM` for generation. That is a deliberate
property — a test suite that needs 500MB of weights and an API key is a test
suite nobody runs.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragcore.config import Settings  # noqa: E402
from ragcore.embedder import HashEmbedder  # noqa: E402
from ragcore.pipeline import RAGPipeline  # noqa: E402
from ragcore.reranker import IdentityReranker  # noqa: E402
from ragcore.retriever import HybridRetriever  # noqa: E402
from ragcore.store import ChunkStore  # noqa: E402
from ragcore.types import Document  # noqa: E402


class ScriptedLLM:
    """Returns a fixed script, one word at a time, and records its prompts."""

    model = "scripted"

    def __init__(self, script: str = "The answer is grounded [1] and checked [2].") -> None:
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        for word in self.script.split(" "):
            yield word + " "


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        dense_top_k=10,
        sparse_top_k=10,
        rerank_candidates=10,
        final_top_k=4,
        llm_provider="echo",
    )


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            doc_id="attn",
            title="Attention Is All You Need",
            source="1706.03762.pdf",
            text=(
                "# Abstract\n\n"
                "The dominant sequence transduction models are based on complex recurrent "
                "or convolutional neural networks. We propose the Transformer, a model "
                "architecture based solely on attention mechanisms, dispensing with "
                "recurrence and convolutions entirely.\n\n"
                "# Model Architecture\n\n"
                "The Transformer follows an encoder-decoder structure using stacked "
                "self-attention and point-wise fully connected layers. Multi-head "
                "attention allows the model to jointly attend to information from "
                "different representation subspaces at different positions. We employ "
                "h equals 8 parallel attention heads in the base configuration.\n\n"
                "# Training\n\n"
                "We trained on the WMT 2014 English-German dataset consisting of about "
                "4.5 million sentence pairs. Training used the Adam optimizer with a "
                "warmup schedule of 4000 steps."
            ),
        ),
        Document(
            doc_id="lora",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            source="2106.09685.pdf",
            text=(
                "# Abstract\n\n"
                "We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained "
                "model weights and injects trainable rank decomposition matrices into "
                "each layer of the Transformer architecture, greatly reducing the number "
                "of trainable parameters for downstream tasks.\n\n"
                "# Method\n\n"
                "For a pretrained weight matrix W0, we constrain its update by "
                "representing it with a low-rank decomposition BA where the rank r is "
                "much smaller than the dimension. LoRA reduces trainable parameters by "
                "up to 10000 times and the GPU memory requirement by 3 times."
            ),
        ),
        Document(
            doc_id="rag",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP",
            source="2005.11401.pdf",
            text=(
                "# Abstract\n\n"
                "We explore a general-purpose fine-tuning recipe for retrieval-augmented "
                "generation, models which combine pretrained parametric and "
                "non-parametric memory for language generation.\n\n"
                "# Method\n\n"
                "Our retriever is a Dense Passage Retriever which uses a bi-encoder "
                "architecture over a Wikipedia dump. The generator is BART. We compare "
                "against extractive baselines and find that RAG models generate more "
                "specific and factual language than a parametric-only baseline."
            ),
        ),
    ]


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dimension=96)


@pytest.fixture
def store(docs, embedder, settings) -> ChunkStore:
    from ragcore.chunking import chunk_documents

    chunks = chunk_documents(
        docs,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    return ChunkStore.build(chunks, embedder, embedding_model="hash-test")


@pytest.fixture
def retriever(store, embedder, settings) -> HybridRetriever:
    return HybridRetriever(store, embedder, IdentityReranker(), settings)


@pytest.fixture
def pipeline(retriever, settings) -> RAGPipeline:
    return RAGPipeline(retriever, ScriptedLLM(), settings)
