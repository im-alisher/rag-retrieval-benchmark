"""Retriever interface shared by all retrieval methods."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Retriever(ABC):
    """Abstract base class for a retriever.

    A retriever has two phases:

    1. ``index(corpus)`` prepares the corpus. It can build an inverted
       index, embed documents, or do nothing at all.
    2. ``search(query, k)`` returns the top k documents for one query.

    Results are always returned as a list of ``(doc_id, score)`` tuples
    sorted by descending score. Higher score means better match.
    """

    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        """Prepare the retriever for a corpus.

        Args:
            corpus: Maps document id to document text.
        """

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top k documents for a query.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            A list of ``(doc_id, score)`` tuples sorted by descending
            score. The list has at most ``k`` entries.
        """
