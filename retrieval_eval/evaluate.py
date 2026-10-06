"""Evaluation runner.

The runner indexes every retriever once, then asks each retriever the
same queries, scores the answers with the metrics module, and collects
everything into structured result objects that the report generator can
format.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from retrieval_eval.data import Dataset
from retrieval_eval.metrics import map_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from retrieval_eval.retrievers.base import Retriever

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


@dataclass
class QueryResult:
    """What one retriever answered for one query.

    Attributes:
        query_id: The query id.
        retrieved: Retrieved document ids, best first. Truncated to the
            largest k that was requested.
    """

    query_id: str
    retrieved: list[str]


@dataclass
class RetrieverResult:
    """Aggregated results for one retriever.

    Attributes:
        name: Retriever name, for example ``bm25``.
        metrics: Metric name to average score, for example
            ``recall@10``.
        avg_latency_ms: Mean time of one search call in milliseconds.
        num_queries: Number of queries that were evaluated.
        query_results: Per query retrieved ids, used for qualitative
            comparison in the report.
    """

    name: str
    metrics: dict[str, float]
    avg_latency_ms: float
    num_queries: int
    query_results: list[QueryResult] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Full output of one evaluation run.

    Attributes:
        dataset_name: Name of the evaluated dataset.
        model_name: Embedding model name, empty for lexical only runs.
        ks: Cut-off ranks that were scored.
        results: One entry per retriever, in the order requested.
    """

    dataset_name: str
    model_name: str
    ks: list[int]
    results: list[RetrieverResult]

    def get(self, name: str) -> RetrieverResult:
        """Return the result of one retriever by name.

        Raises:
            KeyError: If no retriever with that name was evaluated.
        """
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)


def _score_query(
    retrieved: list[str],
    qrels: dict[str, int],
    ks: Sequence[int],
) -> dict[str, float]:
    """Compute every metric for one query at every cut-off rank."""
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    grades = dict(qrels)
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"mrr@{k}"] = mrr_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, grades, k)
        scores[f"map@{k}"] = map_at_k(retrieved, relevant, k)
    return scores


def evaluate(
    dataset: Dataset,
    retrievers: dict[str, Retriever],
    ks: Sequence[int] = DEFAULT_KS,
    dataset_name: str = "",
    model_name: str = "",
) -> EvaluationResult:
    """Run every retriever over every query and score the results.

    Each retriever is indexed once with the full corpus. Search time is
    measured per query with a high resolution timer, and the mean is
    reported as query latency.

    Args:
        dataset: Corpus, queries and qrels to evaluate against.
        retrievers: Retriever name to retriever object. For example
            ``{"bm25": BM25Retriever()}``.
        ks: Cut-off ranks to score at.
        dataset_name: Stored in the result for reporting.
        model_name: Stored in the result for reporting.

    Returns:
        An :class:`EvaluationResult` with one :class:`RetrieverResult`
        per retriever.

    Raises:
        ValueError: If ``retrievers`` is empty or ``ks`` is empty.
    """
    if not retrievers:
        raise ValueError("evaluate() needs at least one retriever.")
    if not ks:
        raise ValueError("evaluate() needs at least one k.")

    max_k = max(ks)
    results: list[RetrieverResult] = []

    for name, retriever in retrievers.items():
        retriever.index(dataset.corpus)

        sums: dict[str, float] = {}
        query_results: list[QueryResult] = []
        latencies: list[float] = []

        for query_id, query_text in dataset.queries.items():
            start = time.perf_counter()
            hits = retriever.search(query_text, max_k)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            retrieved = [doc_id for doc_id, _ in hits]
            query_results.append(QueryResult(query_id=query_id, retrieved=retrieved))

            query_scores = _score_query(retrieved, dataset.qrels.get(query_id, {}), ks)
            for key, value in query_scores.items():
                sums[key] = sums.get(key, 0.0) + value

        count = max(len(dataset.queries), 1)
        metrics = {key: value / count for key, value in sums.items()}
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        results.append(
            RetrieverResult(
                name=name,
                metrics=metrics,
                avg_latency_ms=avg_latency,
                num_queries=len(dataset.queries),
                query_results=query_results,
            )
        )

    return EvaluationResult(
        dataset_name=dataset_name,
        model_name=model_name,
        ks=list(ks),
        results=results,
    )
