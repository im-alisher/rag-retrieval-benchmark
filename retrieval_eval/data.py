"""Dataset model and loaders.

This module defines the shared data structure used everywhere else in the
project, plus a loader that can pull a dataset from three sources:

1. A local JSON file with keys ``corpus``, ``queries`` and ``qrels``.
2. A local directory in BEIR layout (``corpus.jsonl``, ``queries.jsonl``,
   ``qrels.tsv``).
3. A remote public benchmark name such as ``beir/scifact`` loaded lazily
   through ``ir_datasets`` (and ``beir`` as a fallback).

Heavy libraries are imported inside the functions that need them, so this
module can be imported on a machine without those packages installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    """A standard retrieval benchmark triple.

    Attributes:
        corpus: Maps document id to document text.
        queries: Maps query id to query text.
        qrels: Maps query id to a dict of document id and its relevance
            grade. Grade 0 usually means not relevant, grade 1 and above
            mean relevant.
    """

    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of queries in the dataset."""
        return len(self.queries)


def load_dataset(
    name: str,
    max_queries: int | None = None,
) -> Dataset:
    """Load a dataset by name or path.

    The ``name`` argument is resolved in this order:

    1. If it points to an existing ``.json`` file, the file is read as a
       single JSON object with the keys ``corpus``, ``queries``, ``qrels``.
    2. If it points to an existing directory that contains
       ``corpus.jsonl`` / ``queries.jsonl`` / ``qrels.tsv``, the BEIR file
       layout is read.
    3. Otherwise it is treated as a public benchmark name and loaded
       lazily through ``ir_datasets`` (for example ``beir/scifact``).
       If ``ir_datasets`` is not available, the ``beir`` package is tried.

    Args:
        name: Dataset name or local path.
        max_queries: If given, keep only the first N queries (and drop
            qrels that do not match). Useful for quick experiments.

    Returns:
        A :class:`Dataset` with corpus, queries and qrels.

    Raises:
        FileNotFoundError: If ``name`` looks like a local path but the
            file or directory does not exist.
        ValueError: If the benchmark cannot be loaded by any backend.
    """
    path = Path(name)

    if path.suffix.lower() == ".json" and path.exists():
        dataset = _load_local_json(path)
    elif path.is_dir():
        dataset = _load_local_beir_dir(path)
    elif name.lower().endswith((".json", ".jsonl", ".tsv")):
        raise FileNotFoundError(f"Local dataset path not found: {name}")
    else:
        dataset = _load_remote(name)

    if max_queries is not None:
        dataset = _truncate_queries(dataset, max_queries)
    return dataset


def _truncate_queries(dataset: Dataset, max_queries: int) -> Dataset:
    """Keep only the first ``max_queries`` queries and their qrels."""
    kept = dict(list(dataset.queries.items())[:max_queries])
    kept_qrels = {qid: rel for qid, rel in dataset.qrels.items() if qid in kept}
    return Dataset(corpus=dataset.corpus, queries=kept, qrels=kept_qrels)


def _load_local_json(path: Path) -> Dataset:
    """Load a dataset from a single JSON file.

    Expected shape::

        {
          "corpus": {"d1": "text", ...},
          "queries": {"q1": "text", ...},
          "qrels": {"q1": {"d1": 1}, ...}
        }
    """
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return Dataset(
        corpus={str(k): str(v) for k, v in raw.get("corpus", {}).items()},
        queries={str(k): str(v) for k, v in raw.get("queries", {}).items()},
        qrels={
            str(qid): {str(did): int(score) for did, score in rel.items()}
            for qid, rel in raw.get("qrels", {}).items()
        },
    )


def _load_local_beir_dir(path: Path) -> Dataset:
    """Load a dataset from a directory in BEIR file layout.

    Reads ``corpus.jsonl``, ``queries.jsonl`` and ``qrels.tsv``. This is
    the layout that the ``beir`` package downloads, so a downloaded
    dataset folder can be reused offline.
    """
    corpus_path = path / "corpus.jsonl"
    queries_path = path / "queries.jsonl"
    qrels_path = path / "qrels.tsv"

    corpus: dict[str, str] = {}
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                corpus[str(row["_id"])] = str(row.get("text", ""))

    queries: dict[str, str] = {}
    if queries_path.exists():
        with queries_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                queries[str(row["_id"])] = str(row.get("text", ""))

    qrels: dict[str, dict[str, int]] = {}
    if qrels_path.exists():
        with qrels_path.open("r", encoding="utf-8") as handle:
            header = handle.readline()
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                qid, did, score = parts[0], parts[1], parts[2]
                qrels.setdefault(qid, {})[did] = int(float(score))

    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


def _load_remote(name: str) -> Dataset:
    """Load a public benchmark by name using ir_datasets or beir.

    Both packages are imported lazily so that unit tests and offline
    environments never need them installed.
    """
    try:
        import ir_datasets  # noqa: PLC0415 - lazy on purpose
    except ImportError:
        ir_datasets = None

    if ir_datasets is not None:
        try:
            return _load_from_ir_datasets(ir_datasets, name)
        except (KeyError, ValueError):
            pass

    try:
        import beir  # noqa: F401, PLC0415 - lazy on purpose
    except ImportError as exc:
        raise ValueError(
            f"Cannot load dataset '{name}'. Install ir_datasets or beir, "
            "or pass a local JSON file or BEIR directory."
        ) from exc

    return _load_from_beir(name)


def _load_from_ir_datasets(ir_datasets_module, name: str) -> Dataset:
    """Convert an ir_datasets benchmark into our :class:`Dataset`."""
    dataset = ir_datasets_module.load(name)
    corpus = {
        doc.doc_id: (doc.text or "")
        for doc in dataset.docs_iter()
    }
    queries = {
        query.query_id: query.text
        for query in dataset.queries_iter()
    }
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


def _load_from_beir(name: str) -> Dataset:
    """Convert a BEIR benchmark into our :class:`Dataset``.

    The dataset name is split into ``domain`` and ``subset`` on the last
    slash, for example ``beir/scifact`` becomes domain ``beir`` and
    subset ``scifact``.
    """
    from beir import util as beir_util  # noqa: PLC0415 - lazy on purpose
    from beir.datasets.data_loader import GenericDataLoader  # noqa: PLC0415

    if "/" not in name:
        raise ValueError(
            f"BEIR dataset names look like 'beir/scifact', got: {name}"
        )
    domain, subset = name.split("/", 1)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{subset}.zip"
    data_path = beir_util.download_and_unzip(url, str(Path("data") / subset))
    corpus, queries, qrels = GenericDataLoader(data_path).load(data_path)
    return Dataset(corpus=dict(corpus), queries=dict(queries), qrels=dict(qrels))
