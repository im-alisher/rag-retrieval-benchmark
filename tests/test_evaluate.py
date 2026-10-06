"""Tests for the evaluation runner, plus offline guard tests.

The guard tests make the whole suite fail if a dataset download, a model
download, or any other internet access is attempted.
"""

from __future__ import annotations

import socket
import sys
import urllib.request

import pytest
from fakes import (
    FakeEmbedder,
    NetworkAccessBlocked,
    block_network,  # noqa: F401  (fixture used via pytestmark)
    make_dataset,
)

from retrieval_eval.evaluate import evaluate
from retrieval_eval.retrievers.bm25 import BM25Retriever
from retrieval_eval.retrievers.dense import DenseRetriever
from retrieval_eval.retrievers.hybrid import HybridRetriever

pytestmark = pytest.mark.usefixtures("block_network")

KS = (1, 5, 10)


def build_retrievers() -> dict:
    """Build the three retrievers the same way the CLI does."""
    bm25 = BM25Retriever()
    dense = DenseRetriever(embedder=FakeEmbedder())
    return {
        "bm25": bm25,
        "dense": dense,
        "hybrid": HybridRetriever([bm25, dense]),
    }


class TestFullEvaluation:
    def test_one_full_offline_evaluation(self) -> None:
        dataset = make_dataset()
        result = evaluate(
            dataset,
            build_retrievers(),
            ks=KS,
            dataset_name="fakes",
            model_name="fake-embedder",
        )

        assert result.dataset_name == "fakes"
        assert result.model_name == "fake-embedder"
        assert result.ks == [1, 5, 10]
        assert [entry.name for entry in result.results] == [
            "bm25",
            "dense",
            "hybrid",
        ]
        for entry in result.results:
            assert entry.num_queries == 3
            assert len(entry.query_results) == 3
            assert entry.avg_latency_ms >= 0.0

    def test_every_metric_is_present_and_bounded(self) -> None:
        dataset = make_dataset()
        result = evaluate(dataset, build_retrievers(), ks=KS)

        expected_keys = {
            f"{metric}@{k}"
            for k in KS
            for metric in ("recall", "mrr", "ndcg", "map")
        }
        for entry in result.results:
            assert set(entry.metrics) == expected_keys
            for value in entry.metrics.values():
                assert 0.0 <= value <= 1.0

    def test_recall_at_10_covers_the_whole_tiny_corpus(self) -> None:
        # The corpus has 8 documents, so k = 10 retrieves everything.
        dataset = make_dataset()
        result = evaluate(dataset, build_retrievers(), ks=KS)
        for entry in result.results:
            assert entry.metrics["recall@10"] == 1.0

    def test_bm25_finds_a_relevant_document_first_for_every_query(self) -> None:
        dataset = make_dataset()
        result = evaluate(dataset, build_retrievers(), ks=KS)
        assert result.get("bm25").metrics["mrr@10"] == 1.0

    def test_metrics_do_not_drop_when_k_grows(self) -> None:
        # recall, mrr and map are cumulative, so they cannot shrink.
        dataset = make_dataset()
        result = evaluate(dataset, build_retrievers(), ks=KS)
        for entry in result.results:
            for metric in ("recall", "mrr", "map"):
                values = [entry.metrics[f"{metric}@{k}"] for k in KS]
                assert values == sorted(values), f"{entry.name} {metric}"

    def test_query_results_keep_retrieved_ids(self) -> None:
        dataset = make_dataset()
        result = evaluate(dataset, build_retrievers(), ks=(10,))
        for entry in result.results:
            ids = {q.query_id for q in entry.query_results}
            assert ids == set(dataset.queries)

    def test_get_returns_result_by_name(self) -> None:
        dataset = make_dataset()
        result = evaluate(dataset, build_retrievers(), ks=(5,))
        assert result.get("dense").name == "dense"
        with pytest.raises(KeyError):
            result.get("nope")

    def test_rejects_empty_inputs(self) -> None:
        dataset = make_dataset()
        with pytest.raises(ValueError):
            evaluate(dataset, {}, ks=KS)
        with pytest.raises(ValueError):
            evaluate(dataset, build_retrievers(), ks=())


class TestOfflineGuards:
    def test_dataset_download_would_be_blocked(self) -> None:
        # If the loader ever tried to reach the network, this raises.
        with pytest.raises(NetworkAccessBlocked):
            socket.create_connection(("example.com", 443))

    def test_model_download_would_be_blocked(self) -> None:
        with pytest.raises(NetworkAccessBlocked):
            urllib.request.urlopen("https://example.com/model.bin")

    def test_dns_lookup_would_be_blocked(self) -> None:
        with pytest.raises(NetworkAccessBlocked):
            socket.getaddrinfo("example.com", 443)

    def test_heavy_libraries_are_never_imported(self) -> None:
        for name in ("sentence_transformers", "ir_datasets", "beir"):
            assert name not in sys.modules, f"{name} must stay lazy"
