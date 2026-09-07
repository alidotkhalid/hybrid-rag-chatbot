"""Ingestion CLI: fetch the corpus, chunk it, build the index.

    python -m ragcore.ingest fetch     # download the arXiv corpus
    python -m ragcore.ingest build     # chunk + embed + index everything in data/corpus
    python -m ragcore.ingest all       # both
    python -m ragcore.ingest inspect   # what is in the built index

`build` is idempotent and rebuilds from scratch. Incremental indexing is
deliberately not implemented: at this corpus size a full rebuild takes under a
minute, and an incremental path that can leave the vector index and the BM25
index disagreeing about document identity is a bug factory. The place to add
it is `ChunkStore`, once the rebuild time actually hurts.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import httpx

from .chunking import chunk_documents
from .config import settings
from .corpus import CORPUS
from .loaders import load_directory
from .store import ChunkStore

USER_AGENT = "rag-chatbot-portfolio/1.0 (+https://github.com/; polite bulk download)"


def fetch_corpus(force: bool = False) -> int:
    """Download the curated arXiv PDFs.

    arXiv asks bulk downloaders to identify themselves and to rate-limit; we
    do both. Files already present are skipped, so a partial run resumes.
    """
    target = settings.corpus_dir
    target.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    with httpx.Client(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        for i, paper in enumerate(CORPUS, start=1):
            dest = target / paper.filename
            if dest.exists() and not force:
                print(f"[{i:2}/{len(CORPUS)}] have  {paper.arxiv_id}  {paper.title[:58]}")
                continue
            print(f"[{i:2}/{len(CORPUS)}] get   {paper.arxiv_id}  {paper.title[:58]}", flush=True)
            try:
                resp = client.get(paper.pdf_url)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    print("           ! not a PDF, skipping")
                    continue
                dest.write_bytes(resp.content)
                downloaded += 1
            except Exception as exc:  # noqa: BLE001
                print(f"           ! failed: {exc}")
            time.sleep(1.0)  # be a good citizen

    print(f"\nDownloaded {downloaded} new file(s) into {target}")
    return downloaded


def _apply_known_titles(docs) -> int:
    """Prefer the curated title over whatever was scraped from page 1.

    Guessing a paper's title from its first page is unreliable: arXiv stamps,
    author lists and affiliation footnotes all look plausible, and which one
    lands first depends on the PDF's internal block order. Since the corpus
    files are named by arXiv ID and `corpus.py` already maps those IDs to
    real titles, there is no reason to guess at all. The heuristic stays as
    the fallback for uploaded documents, where no manifest exists.
    """
    from .corpus import BY_ID

    renamed = 0
    for doc in docs:
        paper = BY_ID.get(Path(doc.source).stem)
        if paper and doc.title != paper.title:
            doc.title = paper.title
            renamed += 1
    return renamed


def build_index(rebuild: bool = True) -> ChunkStore:
    from .embedder import build_embedder  # torch import deferred to here

    corpus_dir = settings.corpus_dir
    if not corpus_dir.exists() or not any(corpus_dir.iterdir()):
        print(
            f"No documents in {corpus_dir}.\n"
            "Run `python -m ragcore.ingest fetch` first, or drop your own "
            "PDF/MD/TXT/DOCX files in that folder.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Loading documents from {corpus_dir} …")
    t0 = time.perf_counter()
    docs = load_directory(corpus_dir)
    named = _apply_known_titles(docs)
    print(f"  {len(docs)} documents in {time.perf_counter() - t0:.1f}s")
    if named:
        print(f"  {named} title(s) taken from the curated corpus manifest")
    if not docs:
        raise SystemExit("No parseable documents found.")

    print("Chunking …")
    t0 = time.perf_counter()
    chunks = chunk_documents(
        docs,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    print(f"  {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s")

    print(f"Embedding with {settings.embedding_model} (first run downloads the model) …")
    t0 = time.perf_counter()
    embedder = build_embedder(settings)
    store = ChunkStore.build(chunks, embedder, embedding_model=settings.embedding_model)
    print(f"  embedded in {time.perf_counter() - t0:.1f}s")

    index_dir = settings.index_dir
    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)
    store.save(index_dir)

    print(f"\nIndex written to {index_dir}")
    for key, value in store.stats().items():
        print(f"  {key:20} {value}")
    return store


def inspect_index() -> None:
    store = ChunkStore.load(settings.index_dir)
    for key, value in store.stats().items():
        print(f"{key:20} {value}")
    print("\nDocuments:")
    for doc in store.documents():
        n = sum(1 for c in store.chunks if c.doc_id == doc["doc_id"])
        print(f"  {n:4} chunks  {doc['title'][:70]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch", "build", "all", "inspect"])
    parser.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    args = parser.parse_args(argv)

    if args.command == "fetch":
        fetch_corpus(force=args.force)
    elif args.command == "build":
        build_index()
    elif args.command == "all":
        fetch_corpus(force=args.force)
        build_index()
    elif args.command == "inspect":
        inspect_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
