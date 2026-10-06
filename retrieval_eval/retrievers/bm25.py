"""BM25 lexical retriever.

BM25 is a classic sparse retrieval function. It scores a document by how
often the query terms appear in it, corrected for document length and for
how rare the term is across the corpus. It only matches exact tokens, so
it needs no model download and no GPU.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from retrieval_eval.retrievers.base import Retriever

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Args:
        text: Input text.

    Returns:
        A list of lowercase tokens. Punctuation and symbols are dropped.

    Example:
        >>> tokenize("The DNA double-helix!")
        ['the', 'dna', 'double', 'helix']
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(Retriever):
    """Lexical retrieval with the Okapi BM25 ranking function.

    Example:
        >>> retriever = BM25Retriever()
        >>> _ = retriever.index({"d1": "cat sat", "d2": "dog ran"})
        >>> doc_ids = [doc_id for doc_id, _ in retriever.search("cat", 1)]
        >>> doc_ids
        ['d1']
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Store BM25 hyperparameters.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def index(self, corpus: dict[str, str]) -> None:
        """Tokenize the corpus and build the BM25 index.

        Documents keep their insertion order, so scores always map back
        to the right doc ids.

        Args:
            corpus: Maps document id to document text.
        """
        self._doc_ids = list(corpus.keys())
        tokenized = [tokenize(text) for text in corpus.values()]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Score the query against the corpus and return the top k.

        A score of 0 means the document shares no term with the query.
        Such documents are still returned when the corpus is too small
        to fill k, which keeps the ranking complete and predictable.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            Top k ``(doc_id, score)`` pairs sorted by descending score.

        Raises:
            RuntimeError: If :meth:`index` was not called first.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.index() must be called before search().")

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._doc_ids, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(doc_id, float(score)) for doc_id, score in ranked[:k]]
