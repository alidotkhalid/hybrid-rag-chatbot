"""Full ingest → persist → reload → retrieve → answer, over real files on disk.

The unit tests each verify one component against in-memory fixtures. This one
verifies that the components still fit together when the data has been through
the filesystem — which is where index/metadata mismatches, encoding problems
and path bugs actually show up.
"""

from __future__ import annotations

from conftest import ScriptedLLM

from ragcore.chunking import chunk_documents
from ragcore.loaders import load_directory
from ragcore.pipeline import RAGPipeline
from ragcore.reranker import IdentityReranker
from ragcore.retriever import HybridRetriever
from ragcore.store import ChunkStore

PAPERS = {
    "attention.md": """# Attention Is All You Need

## Abstract

We propose the Transformer, a model architecture based solely on attention
mechanisms, dispensing with recurrence and convolutions entirely. Experiments
show these models are superior in quality while being more parallelizable.

## Model Architecture

The Transformer uses stacked self-attention and point-wise fully connected
layers for both the encoder and decoder. We employ h equals 8 parallel
attention heads. For each head we use a dimension of 64.

## Training

We trained on the WMT 2014 English-German dataset consisting of about 4.5
million sentence pairs. Training took 12 hours on eight NVIDIA P100 GPUs.
""",
    "chinchilla.md": """# Training Compute-Optimal Large Language Models

## Abstract

We investigate the optimal model size and number of tokens for training a
transformer language model under a given compute budget. We find that current
large language models are significantly undertrained.

## Findings

For compute-optimal training, the model size and the number of training tokens
should be scaled equally: for every doubling of model size the number of
training tokens should also be doubled. We test this by training Chinchilla,
a 70 billion parameter model trained on 1.4 trillion tokens.
""",
    "lora.md": """# LoRA: Low-Rank Adaptation of Large Language Models

## Abstract

We propose Low-Rank Adaptation, which freezes the pretrained model weights and
injects trainable rank decomposition matrices into each layer of the
Transformer architecture.

## Method

LoRA reduces the number of trainable parameters by up to 10000 times and the
GPU memory requirement by 3 times compared to full fine-tuning. The rank r is
typically much smaller than the model dimension.
""",
}


def build_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, body in PAPERS.items():
        (corpus / name).write_text(body, encoding="utf-8")
    return corpus


class TestEndToEnd:
    def test_full_cycle(self, tmp_path, embedder, settings):
        corpus = build_corpus(tmp_path)

        # --- ingest ---------------------------------------------------------
        docs = load_directory(corpus)
        assert len(docs) == 3
        assert {d.title for d in docs} == {
            "Attention Is All You Need",
            "Training Compute-Optimal Large Language Models",
            "LoRA: Low-Rank Adaptation of Large Language Models",
        }

        chunks = chunk_documents(docs, target_tokens=120, overlap_tokens=24)
        assert len(chunks) >= 6
        store = ChunkStore.build(chunks, embedder, embedding_model="hash-test")

        # --- persist and reload --------------------------------------------
        index_dir = tmp_path / "index"
        store.save(index_dir)
        assert (index_dir / "manifest.json").exists()
        assert (index_dir / "chunks.jsonl").exists()
        assert (index_dir / "vectors.npy").exists()
        assert (index_dir / "bm25_matrix.npz").exists()

        reloaded = ChunkStore.load(index_dir)
        assert reloaded.size == store.size
        assert reloaded.doc_count == 3

        # --- retrieve -------------------------------------------------------
        retriever = HybridRetriever(reloaded, embedder, IdentityReranker(), settings)
        result = retriever.retrieve("How many attention heads are used?")
        assert result.grounded
        assert any("attention heads" in c.chunk.body for c in result.chunks)

        # --- answer ---------------------------------------------------------
        pipeline = RAGPipeline(retriever, ScriptedLLM("Eight heads are used [1]."), settings)
        answer = pipeline.answer("How many attention heads are used?")
        assert answer.answer
        assert len(answer.citations) == 1
        assert answer.citations[0].title in {p.split("\n")[0].lstrip("# ") for p in PAPERS.values()}
        assert answer.timings_ms["total_ms"] > 0

    def test_exact_term_query_survives_the_round_trip(self, tmp_path, embedder, settings):
        """A literal rare token must still be findable after persistence —
        this is the BM25 half of the system, end to end."""
        corpus = build_corpus(tmp_path)
        store = ChunkStore.build(
            chunk_documents(load_directory(corpus), target_tokens=120), embedder
        )
        index_dir = tmp_path / "index"
        store.save(index_dir)

        retriever = HybridRetriever(
            ChunkStore.load(index_dir), embedder, IdentityReranker(), settings
        )
        result = retriever.retrieve("1.4 trillion tokens Chinchilla 70 billion")
        assert any("Chinchilla" in c.chunk.body for c in result.chunks)

    def test_citations_resolve_to_real_sources(self, tmp_path, embedder, settings):
        corpus = build_corpus(tmp_path)
        store = ChunkStore.build(
            chunk_documents(load_directory(corpus), target_tokens=120), embedder
        )
        retriever = HybridRetriever(store, embedder, IdentityReranker(), settings)
        pipeline = RAGPipeline(retriever, ScriptedLLM("A [1] and B [2]."), settings)

        result = pipeline.answer("What does LoRA do to trainable parameters?")
        valid_sources = {c.source for c in store.chunks}
        for citation in result.citations:
            assert citation.source in valid_sources
            assert citation.snippet
            assert citation.title
