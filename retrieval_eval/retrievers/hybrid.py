"""Hybrid retriever based on Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF) combines ranked lists without caring about
the original score scales. BM25 scores and cosine similarities live on
completely different scales, but ranks are comparable. Each retriever
gives a document at rank r a contribution of ``1 / (rrf_k + r)``, and the
final score is the sum over all retrievers.

The default constant is 60, which is the value suggested by the original
RRF paper.
"""

from __future__ import annotations

from retrieval_eval.retrievers.base import Retriever

DEFAULT_RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists with Reciprocal Rank Fusion.

    Ranks start at 1, so the best document of each list contributes
    ``1 / (rrf_k + 1)``.

    Args:
        ranked_lists: One ranked list per retriever. Each list is a
            sequence of ``(doc_id, score)`` pairs sorted by descending
            score. Scores are ignored, only ranks matter.
        rrf_k: RRF smoothing constant. Larger values flatten the
            influence of the top ranks.
        top_k: If given, return only this many results.

    Returns:
        Fused ``(doc_id, score)`` pairs sorted by descending score.

    Example:
        >>> lists = [[("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.4)]]
        >>> fused = rrf_fuse(lists, rrf_k=1)
        >>> fused[0][0]
        'b'
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused


class HybridRetriever(Retriever):
    """Combine any number of retrievers with Reciprocal Rank Fusion.

    Args:
        retrievers: The retrievers to fuse. Any object that implements
            the :class:`~retrieval_eval.retrievers.base.Retriever`
            interface works.
        rrf_k: RRF smoothing constant.
        candidate_pool: How many candidates each sub-retriever contributes.
            Bigger pools give fusion more to work with. Clamped to at
            least the requested ``k`` at search time.

    Example:
        >>> from retrieval_eval.retrievers.bm25 import BM25Retriever
        >>> corpus = {"d1": "cat sat", "d2": "dog ran", "d3": "cat ran"}
        >>> retrievers = [BM25Retriever(), BM25Retriever()]
        >>> hybrid = HybridRetriever(retrievers)
        >>> _ = hybrid.index(corpus)
        >>> len(hybrid.search("cat", 2))
        2
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool: int = 100,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever.")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def index(self, corpus: dict[str, str]) -> None:
        """Build an index in every sub-retriever.

        Args:
            corpus: Maps document id to document text.
        """
        for retriever in self.retrievers:
            retriever.index(corpus)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Collect candidates from every sub-retriever and fuse them.

        Each sub-retriever is asked for ``max(candidate_pool, k)``
        results so that fusion has a wide candidate pool.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            Top k ``(doc_id, score)`` pairs sorted by descending fused
            RRF score.
        """
        pool = max(self.candidate_pool, k)
        ranked_lists = [retriever.search(query, pool) for retriever in self.retrievers]
        return rrf_fuse(ranked_lists, rrf_k=self.rrf_k, top_k=k)
