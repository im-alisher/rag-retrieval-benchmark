"""Shared offline test data and fixtures.

Everything the test suite needs lives here:

- a tiny synthetic corpus, queries and qrels
- a fake embedder, so no model download is ever needed
- a fake retriever with fixed rankings, for hybrid fusion tests
- a guard that blocks all network access during tests

The tests never download anything and never import heavy libraries.
"""

from __future__ import annotations

import re
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

# Make the project root importable no matter how pytest is started.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from retrieval_eval.data import Dataset  # noqa: E402
from retrieval_eval.retrievers.base import Retriever  # noqa: E402

CORPUS: dict[str, str] = {
    "d1": "the cat sat on the mat",
    "d2": "dogs are loyal pets",
    "d3": "quantum physics explained",
    "d4": "the dog chased the cat",
    "d5": "machine learning models need data",
    "d6": "cats sleep most of the day",
    "d7": "bread baking with yeast",
    "d8": "the quantum entanglement puzzle",
}

QUERIES: dict[str, str] = {
    "q1": "quantum physics",
    "q2": "cat and dog",
    "q3": "bread baking",
}

QRELS: dict[str, dict[str, int]] = {
    "q1": {"d3": 1, "d8": 1},
    "q2": {"d1": 1, "d4": 2, "d6": 1},
    "q3": {"d7": 2},
}

VOCAB: list[str] = [
    "cat",
    "dog",
    "quantum",
    "physics",
    "bread",
    "baking",
    "machine",
    "learning",
    "data",
    "pet",
]

_TOKEN = re.compile(r"\w+")


def make_dataset() -> Dataset:
    """Build the tiny test dataset from the fakes above."""
    return Dataset(corpus=dict(CORPUS), queries=dict(QUERIES), qrels=dict(QRELS))


class FakeEmbedder:
    """Deterministic embedder that counts vocabulary words.

    A sentence becomes a vector with one dimension per word in
    :data:`VOCAB`. Two texts that share vocabulary get a high cosine
    similarity, so ranking behaves like a very simple semantic model
    without downloading anything.
    """

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        """Return one count vector per sentence."""
        return [self._vector(text) for text in sentences]

    @staticmethod
    def _vector(text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        return [float(tokens.count(word)) for word in VOCAB]


class FixedRetriever(Retriever):
    """Retriever that returns pre-programmed rankings.

    Useful for testing hybrid fusion without caring about how the
    ranked lists were produced.

    Args:
        results: Maps query text to a ranked list of
            ``(doc_id, score)`` pairs.
    """

    def __init__(self, results: dict[str, list[tuple[str, float]]]) -> None:
        self.results = results
        self.indexed = False
        self.corpus: dict[str, str] = {}
        self.last_k: int | None = None

    def index(self, corpus: dict[str, str]) -> None:
        """Record that indexing happened."""
        self.indexed = True
        self.corpus = corpus

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the pre-programmed list truncated to k."""
        self.last_k = k
        return self.results.get(query, [])[:k]


# --- Fakes that mimic the ir_datasets API -------------------------------
# The real package is never imported. These classes mirror just enough
# of the API to test the loader fallback for BEIR datasets that expose
# qrels only on split subsets, such as beir/scifact.


class DocWithTitle:
    """Document whose ``default_text`` prepends the title, like BEIR."""

    def __init__(self, doc_id: str, title: str, text: str) -> None:
        self.doc_id = doc_id
        self.title = title
        self.text = text

    def default_text(self) -> str:
        return f"{self.title} {self.text}"


class PlainDoc:
    """Document without ``default_text``, like GenericDoc fallbacks."""

    def __init__(self, doc_id: str, text: str) -> None:
        self.doc_id = doc_id
        self.text = text


class FakeQuery:
    """Query with the fields the loader reads."""

    def __init__(self, query_id: str, text: str) -> None:
        self.query_id = query_id
        self.text = text


class FakeQrel:
    """Relevance judgment with the fields the loader reads."""

    def __init__(self, query_id: str, doc_id: str, relevance: int) -> None:
        self.query_id = query_id
        self.doc_id = doc_id
        self.relevance = relevance


class BaseFakeDataset:
    """Dataset with docs and queries but no qrels, like ``beir/scifact``."""

    def __init__(self, docs: list, queries: list) -> None:
        self._docs = list(docs)
        self._queries = list(queries)

    def docs_iter(self):
        return iter(self._docs)

    def queries_iter(self):
        return iter(self._queries)


class SplitFakeDataset(BaseFakeDataset):
    """Dataset that also exposes ``qrels_iter``, like ``.../test``."""

    def __init__(self, docs: list, queries: list, qrels: list) -> None:
        super().__init__(docs, queries)
        self._qrels = list(qrels)

    def qrels_iter(self):
        return iter(self._qrels)


class FakeIrDatasets:
    """Stand-in for the ``ir_datasets`` module with a fixed registry.

    Args:
        datasets: Maps dataset id to a fake dataset object.
    """

    def __init__(self, datasets: dict[str, BaseFakeDataset]) -> None:
        self._datasets = datasets
        self.requested: list[str] = []

    def load(self, name: str) -> BaseFakeDataset:
        """Return a registered dataset or raise KeyError, like ir_datasets."""
        self.requested.append(name)
        if name not in self._datasets:
            raise KeyError(name)
        return self._datasets[name]


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test tries to reach the network."""


def _blocked(*args: object, **kwargs: object) -> None:
    raise NetworkAccessBlocked(
        "network access is blocked during tests. "
        "Tests must run fully offline with fakes."
    )


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a network connection.

    This catches accidental dataset downloads, model downloads, and any
    other internet access.
    """
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
