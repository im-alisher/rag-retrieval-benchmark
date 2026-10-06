"""Dense embedding retriever.

Documents and queries are turned into vectors by a sentence-transformers
model. Search is plain cosine similarity computed with numpy, so there is
no vector database involved.

The model is only imported when it is actually needed, so this module can
be imported without ``sentence-transformers`` installed. Tests inject a
fake embedder instead of downloading a real model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from retrieval_eval.retrievers.base import Retriever

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@runtime_checkable
class Embedder(Protocol):
    """Anything with an ``encode`` method can be used as an embedder."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any:
        """Return one vector per input sentence."""


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of a matrix and a vector.

    Zero vectors are handled safely and yield a similarity of 0.

    Args:
        matrix: Shape ``(n, d)`` with one document per row.
        vector: Shape ``(d,)``, the query.

    Returns:
        Shape ``(n,)`` with one similarity per row. Values are in
        ``[-1, 1]``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)

    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    vector_norm = np.linalg.norm(vector)

    safe_matrix = np.divide(
        matrix,
        matrix_norms,
        out=np.zeros_like(matrix),
        where=matrix_norms > 0,
    )
    if vector_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    return safe_matrix @ (vector / vector_norm)


def corpus_fingerprint(corpus: dict[str, str]) -> str:
    """Build a short stable fingerprint of a corpus.

    The fingerprint depends on the document ids and their text, so a
    different dataset (or an edited one) produces a different cache key.

    Args:
        corpus: Maps document id to document text.

    Returns:
        A 16 character hex string.
    """
    digest = hashlib.sha256()
    for doc_id in sorted(corpus):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(corpus[doc_id].encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def cache_key(corpus: dict[str, str], model_name: str) -> str:
    """Build the cache file stem for a corpus and model pair.

    Args:
        corpus: Maps document id to document text.
        model_name: Name of the embedding model.

    Returns:
        A string like ``scifact_<modelhash>_<corpushash>`` safe to use
        as a file name.
    """
    model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    safe_model = (
        model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    )[:40]
    return f"{safe_model}_{model_hash}_{corpus_fingerprint(corpus)}"


class DenseRetriever(Retriever):
    """Embedding retrieval with cosine similarity.

    Args:
        model_name: Sentence Transformers model to use when no embedder
            is injected.
        embedder: Any object with an ``encode(sentences)`` method. When
            ``None``, the model is loaded lazily from
            ``sentence-transformers`` on first use.
        cache_dir: Directory for embedding cache files. Used only when
            ``use_cache`` is True.
        use_cache: If True, document embeddings are saved to and loaded
            from a ``.npy`` file keyed by corpus and model.

    Example:
        >>> class TinyEmbedder:
        ...     def encode(self, sentences, **kwargs):
        ...         return [[float("a" in s), 1.0] for s in sentences]
        >>> retriever = DenseRetriever(embedder=TinyEmbedder())
        >>> _ = retriever.index({"d1": "alpha", "d2": "beta"})
        >>> len(retriever.search("alpha", 1))
        1
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedder: Embedder | None = None,
        cache_dir: str | Path = ".cache/embeddings",
        use_cache: bool = False,
    ) -> None:
        self.model_name = model_name
        self._embedder = embedder
        self._cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self._doc_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def _get_embedder(self) -> Embedder:
        """Return the injected embedder or lazily load the real model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _cache_path(self, corpus: dict[str, str]) -> Path:
        """Return the ``.npy`` path for this corpus and model."""
        return self._cache_dir / f"{cache_key(corpus, self.model_name)}.npy"

    def index(self, corpus: dict[str, str]) -> None:
        """Embed every document and keep the vectors in memory.

        When ``use_cache`` is on, an existing cache file is loaded
        instead of re-encoding. Otherwise the corpus is encoded and the
        result is written to the cache file.

        Args:
            corpus: Maps document id to document text.
        """
        self._doc_ids = list(corpus.keys())
        texts = list(corpus.values())

        cache_path = self._cache_path(corpus)
        if self.use_cache and cache_path.exists():
            self._matrix = np.load(cache_path)
            return

        embedder = self._get_embedder()
        matrix = np.asarray(embedder.encode(texts), dtype=np.float64)
        self._matrix = matrix

        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, matrix)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Embed the query and return the k most similar documents.

        Args:
            query: Query text.
            k: Number of results to return.

        Returns:
            Top k ``(doc_id, score)`` pairs sorted by descending cosine
            similarity.

        Raises:
            RuntimeError: If :meth:`index` was not called first.
        """
        if self._matrix is None:
            raise RuntimeError("DenseRetriever.index() must be called before search().")
        if len(self._doc_ids) == 0:
            return []

        embedder = self._get_embedder()
        query_vector = np.asarray(embedder.encode([query]), dtype=np.float64)[0]
        scores = cosine_similarity(self._matrix, query_vector)

        top = np.argsort(-scores)[:k]
        return [(self._doc_ids[int(i)], float(scores[int(i)])) for i in top]
