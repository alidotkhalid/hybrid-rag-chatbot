import pytest

from ragcore.sparse import BM25Index, normalize, tokenize

CORPUS = [
    "The transformer uses multi-head self attention over token embeddings",
    "LoRA freezes pretrained weights and injects low-rank decomposition matrices",
    "Dense passage retrieval trains a bi-encoder for open domain question answering",
    "BM25 is a lexical ranking function based on term frequency and inverse document frequency",
    "The transformer encoder stacks attention and feedforward layers",
]


class TestTokenization:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("The Transformer, indeed!") == ["transformer", "indeed"]

    def test_removes_stopwords(self):
        assert "the" not in tokenize("the model")
        assert "of" not in tokenize("out of scope")

    def test_keeps_negation_and_technical_terms(self):
        # "not" and "no" carry meaning in technical text and are deliberately
        # not in the stopword list.
        tokens = tokenize("no recurrence and not convolution")
        assert "no" in tokens and "not" in tokens

    def test_preserves_hyphenated_identifiers(self):
        assert "bge-small" in tokenize("we use bge-small for embeddings")
        assert "multi-head" in tokenize("multi-head attention")

    def test_drops_single_characters(self):
        assert tokenize("a b cd") == ["cd"]


class TestNormalization:
    def test_folds_regular_plurals(self):
        assert normalize("embeddings") == "embedding"
        assert normalize("transformers") == "transformer"

    def test_folds_ies_plurals(self):
        assert normalize("queries") == "query"

    def test_leaves_ss_us_is_endings_alone(self):
        assert normalize("loss") == "loss"
        assert normalize("corpus") == "corpus"
        assert normalize("basis") == "basis"

    def test_leaves_short_words_alone(self):
        assert normalize("is") == "is"
        assert normalize("as") == "as"

    def test_query_and_document_forms_agree(self):
        # This is the property that actually matters: a query for the plural
        # must match a document containing the singular.
        assert tokenize("embeddings")[0] == tokenize("embedding")[0]


class TestBM25:
    @pytest.fixture
    def index(self):
        return BM25Index().fit(CORPUS)

    def test_finds_the_exact_term(self, index):
        results = index.search("LoRA low-rank matrices", top_k=3)
        assert results
        assert results[0][0] == 1

    def test_ranks_by_relevance(self, index):
        results = index.search("transformer attention", top_k=5)
        top_ids = [i for i, _ in results]
        assert 0 in top_ids[:2] and 4 in top_ids[:2]

    def test_rare_terms_outweigh_common_ones(self, index):
        # "transformer" appears twice; "BM25" once. The rare term should win.
        results = index.search("bm25 transformer", top_k=5)
        assert results[0][0] == 3

    def test_unknown_terms_return_nothing(self, index):
        assert index.search("zzzz qqqq wwww", top_k=5) == []

    def test_empty_query_returns_nothing(self, index):
        assert index.search("", top_k=5) == []

    def test_respects_top_k(self, index):
        assert len(index.search("attention retrieval transformer and", top_k=2)) <= 2

    def test_all_scores_positive_and_descending(self, index):
        results = index.search("transformer attention encoder", top_k=5)
        scores = [s for _, s in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_is_safe(self):
        assert BM25Index().fit([]).search("anything") == []

    def test_roundtrips_through_disk(self, index, tmp_path):
        before = index.search("transformer attention", top_k=5)
        index.save(tmp_path)
        after = BM25Index.load(tmp_path).search("transformer attention", top_k=5)
        assert [i for i, _ in before] == [i for i, _ in after]
        assert all(
            abs(a - b) < 1e-5 for (_, a), (_, b) in zip(before, after, strict=False)
        )

    def test_load_preserves_parameters(self, index, tmp_path):
        index.save(tmp_path)
        loaded = BM25Index.load(tmp_path)
        assert loaded.k1 == index.k1
        assert loaded.b == index.b
        assert loaded.doc_count == index.doc_count
