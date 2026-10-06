"""Tests for the retrievers: BM25, dense, and hybrid fusion.

All data comes from ``fakes.py``. No model or dataset download happens.
"""

from __future__ import annotations

import pytest
from fakes import (
    CORPUS,
    FakeEmbedder,
    FixedRetriever,
    block_network,  # noqa: F401  (fixture used via pytestmark)
)

from retrieval_eval.retrievers.bm25 import BM25Retriever, tokenize
from retrieval_eval.retrievers.dense import (
    DenseRetriever,
    cache_key,
    cosine_similarity,
    corpus_fingerprint,
)
from retrieval_eval.retrievers.hybrid import HybridRetriever, rrf_fuse

pytestmark = pytest.mark.usefixtures("block_network")


class TestTokenize:
    def test_lowercases_and_splits_on_word_boundaries(self) -> None:
        assert tokenize("The DNA double-helix!") == ["the", "dna", "double", "helix"]

    def test_empty_text(self) -> None:
        assert tokenize("") == []


class TestBm25Retriever:
    def test_ranks_the_matching_document_first(self) -> None:
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        results = retriever.search("bread baking", 5)
        # Only d7 talks about bread baking.
        assert results[0][0] == "d7"

    def test_ranks_both_term_match_above_single_term_match(self) -> None:
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        results = retriever.search("quantum physics", 8)
        # d3 has both "quantum" and "physics", d8 has only "quantum".
        assert [doc_id for doc_id, _ in results[:2]] == ["d3", "d8"]

    def test_returns_scores_in_descending_order(self) -> None:
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        scores = [score for _, score in retriever.search("cat", 8)]
        assert scores == sorted(scores, reverse=True)

    def test_respects_the_k_limit(self) -> None:
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        assert len(retriever.search("cat", 3)) == 3

    def test_empty_query_returns_nothing_for_small_k(self) -> None:
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        # All scores are zero, so nothing beats a k of 0 results.
        assert retriever.search("cat", 0) == []

    def test_search_before_index_raises(self) -> None:
        with pytest.raises(RuntimeError):
            BM25Retriever().search("cat", 1)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        import numpy as np

        matrix = np.array([[1.0, 2.0, 3.0]])
        vector = np.array([1.0, 2.0, 3.0])
        assert float(cosine_similarity(matrix, vector)[0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        import numpy as np

        matrix = np.array([[1.0, 0.0]])
        vector = np.array([0.0, 1.0])
        assert float(cosine_similarity(matrix, vector)[0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        import numpy as np

        matrix = np.array([[1.0, 0.0]])
        vector = np.array([-1.0, 0.0])
        assert float(cosine_similarity(matrix, vector)[0]) == pytest.approx(-1.0)

    def test_zero_vectors_do_not_divide_by_zero(self) -> None:
        import numpy as np

        matrix = np.array([[0.0, 0.0]])
        vector = np.array([0.0, 0.0])
        assert float(cosine_similarity(matrix, vector)[0]) == 0.0


class TestDenseRetriever:
    def test_ranks_the_sharing_document_first(self) -> None:
        retriever = DenseRetriever(embedder=FakeEmbedder())
        retriever.index(CORPUS)
        results = retriever.search("bread baking", 5)
        # d7 shares both vocabulary words with the query.
        assert results[0][0] == "d7"
        assert results[0][1] == pytest.approx(1.0)

    def test_partial_match_ranks_below_full_match(self) -> None:
        retriever = DenseRetriever(embedder=FakeEmbedder())
        retriever.index(CORPUS)
        results = retriever.search("quantum physics", 8)
        # d3 has both words, d8 has only "quantum".
        assert [doc_id for doc_id, _ in results[:2]] == ["d3", "d8"]
        assert results[0][1] > results[1][1]

    def test_respects_the_k_limit(self) -> None:
        retriever = DenseRetriever(embedder=FakeEmbedder())
        retriever.index(CORPUS)
        assert len(retriever.search("cat", 4)) == 4

    def test_empty_corpus_returns_nothing(self) -> None:
        retriever = DenseRetriever(embedder=FakeEmbedder())
        retriever.index({})
        assert retriever.search("cat", 5) == []

    def test_search_before_index_raises(self) -> None:
        with pytest.raises(RuntimeError):
            DenseRetriever(embedder=FakeEmbedder()).search("cat", 1)

    def test_injected_embedder_is_used(self) -> None:
        class CountingEmbedder(FakeEmbedder):
            calls = 0

            def encode(self, sentences, **kwargs):
                type(self).calls += len(sentences)
                return super().encode(sentences, **kwargs)

        embedder = CountingEmbedder()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index({"d1": "bread", "d2": "cat"})
        retriever.search("bread", 1)
        # 2 documents + 1 query were encoded, nothing was downloaded.
        assert CountingEmbedder.calls == 3

    def test_indexing_same_corpus_twice_encodes_only_once(self) -> None:
        # Regression test for the double encoding that made runs take
        # twice as long: hybrid shares the dense retriever and evaluate
        # indexes every entry.
        class CountingEmbedder(FakeEmbedder):
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, sentences, **kwargs):
                self.calls += len(sentences)
                return super().encode(sentences, **kwargs)

        embedder = CountingEmbedder()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(CORPUS)
        retriever.index(CORPUS)
        assert embedder.calls == len(CORPUS)

    def test_indexing_a_changed_corpus_encodes_again(self) -> None:
        class CountingEmbedder(FakeEmbedder):
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, sentences, **kwargs):
                self.calls += len(sentences)
                return super().encode(sentences, **kwargs)

        embedder = CountingEmbedder()
        retriever = DenseRetriever(embedder=embedder)
        retriever.index(CORPUS)
        changed = dict(CORPUS)
        changed["d99"] = "one more document"
        retriever.index(changed)
        assert embedder.calls == len(CORPUS) + len(changed)

    def test_embedding_cache_roundtrip(self, tmp_path) -> None:
        class CountingEmbedder(FakeEmbedder):
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, sentences, **kwargs):
                self.calls += len(sentences)
                return super().encode(sentences, **kwargs)

        first = CountingEmbedder()
        writer = DenseRetriever(embedder=first, cache_dir=tmp_path, use_cache=True)
        writer.index(CORPUS)
        assert first.calls == len(CORPUS)
        assert list(tmp_path.glob("*.npy"))

        # A second retriever loads the .npy file and never encodes.
        second = CountingEmbedder()
        reader = DenseRetriever(embedder=second, cache_dir=tmp_path, use_cache=True)
        reader.index(CORPUS)
        assert second.calls == 0

        query = "bread baking"
        reader_hits = reader.search(query, 5)
        writer_hits = writer.search(query, 5)
        assert [d for d, _ in reader_hits] == [d for d, _ in writer_hits]
        assert [s for _, s in reader_hits] == pytest.approx(
            [s for _, s in writer_hits]
        )


class TestCacheKey:
    def test_same_corpus_and_model_give_the_same_key(self) -> None:
        assert cache_key(CORPUS, "model/a") == cache_key(dict(CORPUS), "model/a")

    def test_different_model_changes_the_key(self) -> None:
        assert cache_key(CORPUS, "model/a") != cache_key(CORPUS, "model/b")

    def test_different_corpus_changes_the_key(self) -> None:
        other = dict(CORPUS)
        other["d99"] = "one more document"
        assert cache_key(CORPUS, "model/a") != cache_key(other, "model/a")

    def test_key_is_file_name_safe(self) -> None:
        key = cache_key(CORPUS, "sentence-transformers/paraphrase-multilingual")
        assert "/" not in key
        assert len(corpus_fingerprint(CORPUS)) == 16


class TestRrfFuse:
    def test_document_in_both_lists_wins(self) -> None:
        # rrf_k = 60:
        #   b = 1 / 62 + 1 / 61 = 0.032522  (rank 2 in A, rank 1 in B)
        #   a = 1 / 61          = 0.016393  (rank 1 in A)
        #   c = 1 / 62          = 0.016129  (rank 2 in B)
        #   z = 1 / 63          = 0.015873  (rank 3 in A)
        ranked_a = [("a", 9.0), ("b", 5.0), ("z", 1.0)]
        ranked_b = [("b", 0.9), ("c", 0.4)]
        fused = rrf_fuse([ranked_a, ranked_b], rrf_k=60)

        assert [doc_id for doc_id, _ in fused] == ["b", "a", "c", "z"]
        assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
        assert round(fused[0][1], 6) == 0.032522

    def test_raw_scores_are_ignored_only_ranks_matter(self) -> None:
        # y wins even though its raw scores are tiny, because it appears
        # first in one list and second in the other.
        ranked_a = [("x", 0.1), ("y", 0.01)]
        ranked_b = [("y", 100.0), ("z", 90.0)]
        fused = rrf_fuse([ranked_a, ranked_b], rrf_k=60)
        assert [doc_id for doc_id, _ in fused] == ["y", "x", "z"]

    def test_top_k_limits_the_output(self) -> None:
        fused = rrf_fuse([[("a", 1.0), ("b", 1.0)]], top_k=1)
        assert len(fused) == 1

    def test_empty_inputs(self) -> None:
        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []


class TestHybridRetriever:
    def test_needs_at_least_one_retriever(self) -> None:
        with pytest.raises(ValueError):
            HybridRetriever([])

    def test_index_forwards_to_all_components(self) -> None:
        first = FixedRetriever({})
        second = FixedRetriever({})
        hybrid = HybridRetriever([first, second])
        hybrid.index(CORPUS)
        assert first.indexed and second.indexed
        assert first.corpus == CORPUS
        assert second.corpus == CORPUS

    def test_fuses_component_rankings(self) -> None:
        first = FixedRetriever({"q": [("a", 5.0), ("b", 4.0)]})
        second = FixedRetriever({"q": [("b", 0.9), ("c", 0.4)]})
        hybrid = HybridRetriever([first, second])
        hybrid.index(CORPUS)

        results = hybrid.search("q", 3)
        # b appears in both lists, so it must come first.
        assert [doc_id for doc_id, _ in results] == ["b", "a", "c"]

    def test_asks_components_for_the_candidate_pool(self) -> None:
        component = FixedRetriever({"q": [("a", 1.0)]})
        hybrid = HybridRetriever([component], candidate_pool=3)
        hybrid.index(CORPUS)
        hybrid.search("q", 2)
        # pool = max(candidate_pool, k) = max(3, 2)
        assert component.last_k == 3

    def test_respects_the_k_limit(self) -> None:
        first = FixedRetriever({"q": [("a", 5.0), ("b", 4.0), ("c", 3.0)]})
        second = FixedRetriever({"q": [("c", 5.0), ("b", 4.0), ("a", 3.0)]})
        hybrid = HybridRetriever([first, second])
        hybrid.index(CORPUS)
        assert len(hybrid.search("q", 2)) == 2

    def test_hybrid_of_real_bm25_and_dense_finds_the_shared_top_doc(self) -> None:
        bm25 = BM25Retriever()
        dense = DenseRetriever(embedder=FakeEmbedder())
        hybrid = HybridRetriever([bm25, dense])
        hybrid.index(CORPUS)

        results = hybrid.search("bread baking", 5)
        # Both retrievers put d7 first, so fusion must keep it first.
        assert results[0][0] == "d7"
        assert len(results) == 5
