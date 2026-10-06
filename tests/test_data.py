"""Regression tests for the ir_datasets loader.

The reported runtime bug was::

    AttributeError: qrels_iter

``beir/scifact`` is registered in ir_datasets without qrels on the base
id. The qrels only exist on split subsets such as ``beir/scifact/test``.
The loader must detect this and fall back to the test split.

Everything here runs offline against the fakes in ``fakes.py``. The real
``ir_datasets`` package is never imported.
"""

from __future__ import annotations

import pytest
from fakes import (
    BaseFakeDataset,
    DocWithTitle,
    FakeIrDatasets,
    FakeQrel,
    FakeQuery,
    PlainDoc,
    SplitFakeDataset,
    block_network,  # noqa: F401  (fixture used via pytestmark)
)

from retrieval_eval.data import _doc_text, _load_from_ir_datasets

pytestmark = pytest.mark.usefixtures("block_network")


def make_base() -> BaseFakeDataset:
    """Build a dataset shaped like ``beir/scifact``: no qrels."""
    docs = [DocWithTitle("d1", "A claim", "Evidence text")]
    queries = [FakeQuery("q1", "what is the evidence")]
    return BaseFakeDataset(docs, queries)


def make_split() -> SplitFakeDataset:
    """Build a dataset shaped like ``beir/scifact/test``: with qrels."""
    docs = [DocWithTitle("d1", "A claim", "Evidence text")]
    queries = [FakeQuery("q1", "what is the evidence")]
    qrels = [FakeQrel("q1", "d1", 1)]
    return SplitFakeDataset(docs, queries, qrels)


class TestQrelsFallback:
    def test_base_id_without_qrels_falls_back_to_test_split(self) -> None:
        # Regression test for AttributeError: qrels_iter
        fake = FakeIrDatasets(
            {
                "beir/scifact": make_base(),
                "beir/scifact/test": make_split(),
            }
        )

        dataset = _load_from_ir_datasets(fake, "beir/scifact")

        assert dataset.qrels == {"q1": {"d1": 1}}
        assert dataset.corpus == {"d1": "A claim Evidence text"}
        assert dataset.queries == {"q1": "what is the evidence"}
        # The base id was tried first, then the test split.
        assert fake.requested == ["beir/scifact", "beir/scifact/test"]

    def test_explicit_split_name_loads_without_fallback(self) -> None:
        fake = FakeIrDatasets({"beir/scifact/test": make_split()})

        dataset = _load_from_ir_datasets(fake, "beir/scifact/test")

        assert dataset.qrels == {"q1": {"d1": 1}}
        assert fake.requested == ["beir/scifact/test"]

    def test_raises_value_error_when_no_qrels_exist_at_all(self) -> None:
        fake = FakeIrDatasets({"beir/scifact": make_base()})

        with pytest.raises(ValueError, match="provides no qrels"):
            _load_from_ir_datasets(fake, "beir/scifact")

        # Both the base id and the test split were tried.
        assert fake.requested == ["beir/scifact", "beir/scifact/test"]

    def test_unknown_name_raises_key_error_so_beir_can_take_over(self) -> None:
        # _load_remote catches KeyError and falls back to the beir
        # package, so this error must not be swallowed here.
        fake = FakeIrDatasets({})

        with pytest.raises(KeyError):
            _load_from_ir_datasets(fake, "not-a-dataset")


class TestDocText:
    def test_uses_default_text_so_beir_titles_are_indexed(self) -> None:
        assert _doc_text(DocWithTitle("d1", "A claim", "Evidence")) == "A claim Evidence"

    def test_falls_back_to_plain_text(self) -> None:
        assert _doc_text(PlainDoc("d1", "Evidence")) == "Evidence"

    def test_empty_text_becomes_empty_string(self) -> None:
        assert _doc_text(PlainDoc("d1", "")) == ""
