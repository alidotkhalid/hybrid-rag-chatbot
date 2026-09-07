"""A production-shaped retrieval-augmented generation system.

Layout:
    config.py      every tunable knob, in one place
    types.py       Document / Chunk / ScoredChunk / Citation
    loaders.py     PDF, Markdown, text and DOCX parsing
    chunking.py    structure-aware chunking
    embedder.py    bi-encoder (+ a dependency-free stub for tests)
    sparse.py      BM25, implemented directly
    fusion.py      Reciprocal Rank Fusion
    reranker.py    cross-encoder reranking
    store.py       vectors + BM25 + metadata, persisted as one unit
    retriever.py   the hybrid pipeline
    llm.py         provider-agnostic streaming generation
    prompts.py     prompt templates
    citations.py   citation extraction and validation
    pipeline.py    question in, streamed grounded answer out
    ingest.py      the ingestion CLI
"""

__version__ = "1.0.0"
