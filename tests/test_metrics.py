"""Tests for the hand written retrieval metrics.

Every expected value is calculated by hand in the test comment.
"""

from __future__ import annotations

from fakes import block_network  # noqa: F401  (fixture used via pytestmark)

import pytest

from retrieval_eval.metrics import (
    average_precision,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.usefixtures("block_network")


class TestRecallAtK:
    def test_finds_half_of_the_relevant_docs(self) -> None:
        # Relevant: d1, d3. Top 2 contains only d1. 1 hit / 2 relevant.
        assert recall_at_k(["d2", "d1", "d4"], {"d1", "d3"}, k=2) == 0.5

    def test_finds_all_relevant_docs(self) -> None:
        # Relevant: d1, d3. Top 2 contains both. 2 / 2.
        assert recall_at_k(["d1", "d3", "d2"], {"d1", "d3"}, k=2) == 1.0

    def test_zero_when_no_relevant_doc_is_found(self) -> None:
        assert recall_at_k(["d2", "d3"], {"d1"}, k=2) == 0.0

    def test_zero_when_query_has_no_relevant_docs(self) -> None:
        assert recall_at_k(["d1"], set(), k=1) == 0.0

    def test_k_larger_than_result_list(self) -> None:
        # Only one retrieved doc, one relevant doc found: 1 / 1.
        assert recall_at_k(["d1"], {"d1"}, k=10) == 1.0


class TestMrrAtK:
    def test_first_result_is_relevant(self) -> None:
        # First relevant doc at rank 1: 1 / 1.
        assert mrr_at_k(["d1", "d2"], {"d1"}, k=5) == 1.0

    def test_second_result_is_relevant(self) -> None:
        # First relevant doc at rank 2: 1 / 2.
        assert mrr_at_k(["d2", "d1"], {"d1"}, k=5) == 0.5

    def test_third_result_is_relevant(self) -> None:
        # First relevant doc at rank 3: 1 / 3.
        assert mrr_at_k(["d2", "d3", "d1"], {"d1"}, k=5) == 1.0 / 3.0

    def test_zero_when_no_relevant_doc_in_top_k(self) -> None:
        assert mrr_at_k(["d2", "d3"], {"d1"}, k=2) == 0.0

    def test_relevant_doc_beyond_k_is_ignored(self) -> None:
        # d1 is at rank 3, but k is 2, so the reciprocal rank is 0.
        assert mrr_at_k(["d2", "d3", "d1"], {"d1"}, k=2) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self) -> None:
        # d1 has the highest grade and is ranked first.
        assert ndcg_at_k(["d1", "d2"], {"d1": 2, "d2": 1}, k=2) == 1.0

    def test_swapped_ranking_matches_hand_calculation(self) -> None:
        # Hand calculation with grades d1 = 2 and d2 = 1, k = 2:
        #   DCG  = (2^1 - 1) / log2(2) + (2^2 - 1) / log2(3)
        #        = 1 + 3 / 1.58496 = 2.89279
        #   IDCG = (2^2 - 1) / log2(2) + (2^1 - 1) / log2(3)
        #        = 3 + 1 / 1.58496 = 3.63093
        #   nDCG = 2.89279 / 3.63093 = 0.79671
        score = ndcg_at_k(["d2", "d1"], {"d1": 2, "d2": 1}, k=2)
        assert round(score, 4) == 0.7967

    def test_zero_when_no_document_is_relevant(self) -> None:
        assert ndcg_at_k(["d1", "d2"], {}, k=2) == 0.0
        assert ndcg_at_k(["d1"], {"d1": 0}, k=1) == 0.0

    def test_unretrieved_relevant_doc_lowers_the_score(self) -> None:
        # d2 (grade 2) is never retrieved, so its gain is missing.
        # DCG  = (2^1 - 1) / log2(2) = 1
        # IDCG = (2^2 - 1) / log2(2) = 3
        # nDCG = 1 / 3
        score = ndcg_at_k(["d1"], {"d1": 1, "d2": 2}, k=1)
        assert abs(score - 1.0 / 3.0) < 1e-9

    def test_k_limits_both_dcg_and_ideal(self) -> None:
        # With k = 1 only the first position counts:
        #   DCG  = (2^1 - 1) / log2(2) = 1
        #   IDCG = (2^3 - 1) / log2(2) = 7
        #   nDCG = 1 / 7
        score = ndcg_at_k(["d2", "d3", "d1"], {"d1": 3, "d2": 1, "d3": 1}, k=1)
        assert abs(score - 1.0 / 7.0) < 1e-9


class TestAveragePrecision:
    def test_hand_calculated_ap(self) -> None:
        # Relevant: d1, d3.
        #   rank 1 is relevant: precision = 1 / 1
        #   rank 3 is relevant: precision = 2 / 3
        #   AP = (1 + 0.6667) / 2 = 0.8333
        score = average_precision(["d1", "d2", "d3"], {"d1", "d3"})
        assert round(score, 4) == 0.8333

    def test_ap_with_relevant_docs_swapped(self) -> None:
        # Relevant: d1, d3.
        #   rank 1 is relevant: precision = 1 / 1
        #   rank 3 is relevant: precision = 2 / 3
        #   AP = (1 + 0.6667) / 2 = 0.8333
        score = average_precision(["d3", "d2", "d1"], {"d1", "d3"})
        assert round(score, 4) == 0.8333

    def test_ap_cut_off_at_k(self) -> None:
        # Only rank 1 counts and it is relevant: 1 / 1.
        # Divided by 2 relevant docs: 1 / 2 = 0.5
        score = average_precision(["d1", "d2"], {"d1", "d3"}, k=1)
        assert score == 0.5

    def test_ap_zero_without_relevant_docs(self) -> None:
        assert average_precision(["d1"], set()) == 0.0

    def test_ap_is_zero_when_nothing_relevant_is_found(self) -> None:
        assert average_precision(["d2", "d3"], {"d1"}, k=2) == 0.0


class TestMapAtK:
    def test_map_matches_ap(self) -> None:
        # AP over the full list is (1 + 2/3) / 2 = 0.8333
        score = map_at_k(["d1", "d2", "d3"], {"d1", "d3"}, k=3)
        assert round(score, 4) == 0.8333

    def test_map_at_smaller_k(self) -> None:
        # Only d1 in the top 1: precision 1 / 1, over 2 relevant docs.
        assert map_at_k(["d1", "d2", "d3"], {"d1", "d3"}, k=1) == 0.5

    def test_map_zero_without_relevant_docs(self) -> None:
        assert map_at_k(["d1"], set(), k=1) == 0.0
