"""BM25 lexical retrieval, implemented directly.

Why hand-roll BM25 instead of pulling in `rank_bm25`:

* `rank_bm25` scores by looping over documents in Python, which is O(N) per
  query with a large constant. This implementation builds a CSR term-document
  matrix once and answers a query with a single sparse matrix-vector product,
  which is both faster and easier to reason about.
* Persisting a library's internal state across versions is fragile. Here the
  on-disk format is our own and explicit.
* It is the part of a hybrid retriever most likely to be asked about in an
  interview, so it should be legible rather than a black box.

Why BM25 at all, when we already have embeddings: a bi-encoder maps text into a
semantic space where *exact tokens are not preserved*. Queries containing rare
identifiers — "RRF", "k=60", "bge-small", a specific author name — routinely
miss on dense retrieval and hit trivially on lexical retrieval. The two failure
modes are close to independent, which is exactly the condition under which
fusion helps.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# A deliberately short stopword list. Aggressive stopword removal hurts on
# technical corpora where "no", "not" and "all" carry meaning ("Attention Is
# All You Need" is a title, not noise).
STOPWORDS = frozenset(
    """
    a an the and or of to in on for with as at by from is are was were be been
    being this that these those it its we our you your they their he she his
    her i me my us them there here then than so such but if into over under
    about above below can could may might will would shall should do does did
    have has had having also very more most much many just only own same too
    """.split()
)


def normalize(token: str) -> str:
    """Conservative morphological normalisation.

    Full stemming (Porter) over-merges on technical vocabulary — it collapses
    "generative" and "generator", and mangles identifiers. We only fold regular
    plurals, which is where the real recall loss is ("embeddings" vs
    "embedding", "transformers" vs "transformer").
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("es") and token[:-2].endswith(("ch", "sh", "x", "z")):
            return token[:-2]
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [
        normalize(t)
        for t in _TOKEN_RE.findall(text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


class BM25Index:
    """Okapi BM25 over a fixed document collection.

    Parameters follow the standard defaults (k1=1.5, b=0.75). b controls how
    much document length is penalised; our chunks are near-uniform in length by
    construction, so the setting matters less here than it would on raw
    documents.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.matrix: sparse.csr_matrix | None = None
        """Pre-weighted term-document matrix: entry (d, t) already holds the
        full BM25 term saturation term, so query time is one lookup + sum."""
        self.doc_count = 0

    # -- build ---------------------------------------------------------------

    def fit(self, texts: list[str]) -> BM25Index:
        tokenized = [tokenize(t) for t in texts]
        self.doc_count = len(tokenized)
        if self.doc_count == 0:
            self.matrix = sparse.csr_matrix((0, 0), dtype=np.float32)
            return self

        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        n_terms = len(vocab)

        doc_lens = np.array([len(t) for t in tokenized], dtype=np.float32)
        avgdl = float(doc_lens.mean()) if doc_lens.size else 1.0
        avgdl = avgdl or 1.0

        # Document frequency for IDF.
        df = np.zeros(n_terms, dtype=np.float32)
        for tokens in tokenized:
            for tok in set(tokens):
                df[vocab[tok]] += 1.0

        # Lucene/Robertson IDF with the +1 that keeps it non-negative.
        self.idf = np.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5)).astype(np.float32)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for d, tokens in enumerate(tokenized):
            if not tokens:
                continue
            norm = self.k1 * (1.0 - self.b + self.b * (doc_lens[d] / avgdl))
            for tok, tf in Counter(tokens).items():
                t = vocab[tok]
                rows.append(d)
                cols.append(t)
                vals.append((tf * (self.k1 + 1.0)) / (tf + norm))

        self.matrix = sparse.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(self.doc_count, n_terms),
        )
        return self

    # -- query ---------------------------------------------------------------

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """Return [(doc_index, score), ...] sorted by descending score."""
        if self.matrix is None or self.doc_count == 0:
            return []

        q_tokens = tokenize(query)
        term_ids = [self.vocab[t] for t in q_tokens if t in self.vocab]
        if not term_ids:
            return []

        # Repeated query terms legitimately increase weight.
        weights = np.zeros(len(self.vocab), dtype=np.float32)
        for t in term_ids:
            weights[t] += self.idf[t]

        scores = self.matrix.dot(weights)
        if not np.any(scores):
            return []

        k = min(top_k, self.doc_count)
        # argpartition avoids a full sort of a potentially large score vector.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0.0]

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        assert self.matrix is not None, "fit() must be called before save()"
        sparse.save_npz(path / "bm25_matrix.npz", self.matrix)
        np.save(path / "bm25_idf.npy", self.idf)
        (path / "bm25_meta.json").write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "doc_count": self.doc_count,
                    "vocab": self.vocab,
                },
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        meta = json.loads((path / "bm25_meta.json").read_text(encoding="utf-8"))
        idx = cls(k1=meta["k1"], b=meta["b"])
        idx.doc_count = meta["doc_count"]
        idx.vocab = meta["vocab"]
        idx.idf = np.load(path / "bm25_idf.npy")
        idx.matrix = sparse.load_npz(path / "bm25_matrix.npz").tocsr()
        return idx
