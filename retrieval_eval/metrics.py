"""Information retrieval metrics implemented from scratch.

All functions are per query. The evaluation runner averages them over
every query to get the final benchmark numbers.

Functions:

- :func:`recall_at_k` - how many relevant documents were found
- :func:`mrr_at_k` - how high the first relevant document appears
- :func:`ndcg_at_k` - rank quality with graded relevance
- :func:`average_precision` / :func:`map_at_k` - precision across the
  whole relevant list

Relevance is taken from qrels. Grades are treated as gains: grade 0 or a
missing document means not relevant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Recall@k: fraction of relevant documents found in the top k.

    Args:
        retrieved: Retrieved document ids, best first.
        relevant: The set of relevant document ids.
        k: Cut-off rank.

    Returns:
        Hits divided by the number of relevant documents. Returns 0.0
        when the query has no relevant documents.

    Example:
        >>> recall_at_k(["d2", "d1", "d4"], {"d1", "d3"}, k=2)
        0.5
        >>> recall_at_k(["d2", "d1", "d4"], {"d1", "d3"}, k=3)
        0.5
        >>> recall_at_k(["d2"], set(), k=1)
        0.0
    """
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """MRR@k: reciprocal rank of the first relevant document.

    MRR is Mean Reciprocal Rank. This function returns the reciprocal
    rank for one query, and the evaluation runner takes the mean.

    Args:
        retrieved: Retrieved document ids, best first.
        relevant: The set of relevant document ids.
        k: Cut-off rank.

    Returns:
        ``1 / rank`` of the first relevant document inside the top k,
        or 0.0 if none is found.

    Example:
        >>> mrr_at_k(["d2", "d1"], {"d1"}, k=5)
        0.5
        >>> mrr_at_k(["d1", "d2"], {"d1"}, k=5)
        1.0
        >>> mrr_at_k(["d2", "d3"], {"d1"}, k=5)
        0.0
    """
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain for a list of gains, best first.

    Position 1 is not discounted, position 2 divides by log2(3), and so
    on.
    """
    total = 0.0
    for position, gain in enumerate(gains, start=1):
        total += gain / math.log2(position + 1)
    return total


def ndcg_at_k(
    retrieved: Sequence[str],
    grades: Mapping[str, int],
    k: int,
) -> float:
    """nDCG@k: normalized discounted cumulative gain at rank k.

    Gain of a document is ``2 ** grade - 1``, so grade 1 gives 1, grade
    2 gives 3, grade 4 gives 15. The score is DCG divided by the ideal
    DCG, where the ideal ranking puts the highest grades first.

    Args:
        retrieved: Retrieved document ids, best first.
        grades: Relevance grade per document for this query. Documents
            missing from the mapping get grade 0.
        k: Cut-off rank.

    Returns:
        A value in ``[0, 1]``. Returns 0.0 when no document has a grade
        above 0.

    Example:
        Two documents retrieved in the wrong order: d2 (grade 1) first,
        d1 (grade 2) second.

        >>> round(ndcg_at_k(["d2", "d1"], {"d1": 2, "d2": 1}, k=2), 4)
        0.7967
        >>> ndcg_at_k(["d1", "d2"], {"d1": 2, "d2": 1}, k=2)
        1.0
    """
    gains = [float(grades.get(doc_id, 0)) for doc_id in retrieved[:k]]
    gains = [2**gain - 1 for gain in gains]

    ideal = sorted((float(g) for g in grades.values() if g > 0), reverse=True)
    ideal_gains = [2**gain - 1 for gain in ideal[:k]]

    ideal_dcg = _dcg(ideal_gains)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def average_precision(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int | None = None,
) -> float:
    """Average precision for one query.

    Precision is checked at every position where a relevant document
    appears, then averaged over the number of relevant documents.

    Args:
        retrieved: Retrieved document ids, best first.
        relevant: The set of relevant document ids.
        k: Optional cut-off rank. When given, only the top k counts.

    Returns:
        A value in ``[0, 1]``. Returns 0.0 when the query has no
        relevant documents.

    Example:
        Relevant documents are d1 and d3. d1 is at rank 1 and d3 at
        rank 3, so precision is 1/1 and 2/3.

        >>> round(average_precision(["d1", "d2", "d3"], {"d1", "d3"}), 4)
        0.8333
        >>> average_precision(["d1", "d2"], {"d1", "d3"}, k=1)
        0.5
        >>> average_precision(["d2"], set())
        0.0
    """
    if not relevant:
        return 0.0

    cutoff = retrieved if k is None else retrieved[:k]
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(cutoff, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def map_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """MAP@k: average precision cut off at rank k.

    MAP is Mean Average Precision: the mean of average precision over
    all queries. This function returns the per-query value at cut-off
    k, and the evaluation runner takes the mean over queries.

    Args:
        retrieved: Retrieved document ids, best first.
        relevant: The set of relevant document ids.
        k: Cut-off rank.

    Returns:
        A value in ``[0, 1]``.

    Example:
        >>> round(map_at_k(["d1", "d2", "d3"], {"d1", "d3"}, k=3), 4)
        0.8333
        >>> map_at_k(["d1", "d2", "d3"], {"d1", "d3"}, k=1)
        0.5
    """
    return average_precision(retrieved, relevant, k)
