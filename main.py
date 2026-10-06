"""Command line interface for the retrieval evaluation.

Usage example::

    python main.py --dataset beir/scifact --max-queries 200

The CLI loads the dataset, builds the requested retrievers, runs the
evaluation, prints a summary table, and writes markdown and CSV reports.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from retrieval_eval.data import load_dataset
from retrieval_eval.evaluate import DEFAULT_KS, evaluate
from retrieval_eval.report import generate_reports, summary_table
from retrieval_eval.retrievers.base import Retriever
from retrieval_eval.retrievers.bm25 import BM25Retriever
from retrieval_eval.retrievers.dense import DEFAULT_MODEL, DenseRetriever
from retrieval_eval.retrievers.hybrid import HybridRetriever

SUPPORTED_RETRIEVERS = ("bm25", "dense", "hybrid")


def parse_ks(value: str) -> list[int]:
    """Parse a comma separated list of cut-off ranks.

    Args:
        value: For example ``"1,5,10"``.

    Returns:
        A list of integers.

    Example:
        >>> parse_ks("1,5,10")
        [1, 5, 10]
    """
    ks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ks:
        raise argparse.ArgumentTypeError("need at least one k value")
    if any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("k values must be >= 1")
    return ks


def parse_retrievers(value: str) -> list[str]:
    """Parse a comma separated list of retriever names.

    Args:
        value: For example ``"bm25,hybrid"``.

    Returns:
        A list of known retriever names, in the order given.

    Example:
        >>> parse_retrievers("bm25,dense")
        ['bm25', 'dense']
    """
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("need at least one retriever")
    unknown = [name for name in names if name not in SUPPORTED_RETRIEVERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown retriever(s): {', '.join(unknown)}. "
            f"choose from: {', '.join(SUPPORTED_RETRIEVERS)}"
        )
    return names


def build_retrievers(
    names: Sequence[str],
    model_name: str,
    use_cache: bool = False,
) -> dict[str, Retriever]:
    """Create the retriever objects for the requested names.

    The hybrid retriever fuses a fresh BM25 retriever with a dense
    retriever. Dense retriever objects are shared between the ``dense``
    entry and the ``hybrid`` entry so the model is only loaded once
    and the corpus is only encoded once.

    Args:
        names: Retriever names, any of ``bm25``, ``dense``, ``hybrid``.
        model_name: Sentence Transformers model for dense retrieval.
        use_cache: If True, document embeddings are stored on disk and
            reused on later runs instead of being recomputed.

    Returns:
        Retriever name to retriever object.
    """
    wanted = set(names)
    dense: DenseRetriever | None = None
    bm25: BM25Retriever | None = None

    if "dense" in wanted or "hybrid" in wanted:
        dense = DenseRetriever(model_name=model_name, use_cache=use_cache)
    if "bm25" in wanted or "hybrid" in wanted:
        bm25 = BM25Retriever()

    built: dict[str, Retriever] = {}
    for name in names:
        if name == "bm25":
            built[name] = bm25  # type: ignore[assignment]
        elif name == "dense":
            built[name] = dense  # type: ignore[assignment]
        elif name == "hybrid":
            components = [r for r in (bm25, dense) if r is not None]
            built[name] = HybridRetriever(components)
    return built


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="retrieval-eval",
        description="Compare BM25, dense and hybrid retrieval with IR metrics.",
    )
    parser.add_argument(
        "--dataset",
        default="beir/scifact",
        help="Dataset name or local path (default: beir/scifact).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries (default: all).",
    )
    parser.add_argument(
        "--k",
        type=parse_ks,
        default=list(DEFAULT_KS),
        help="Comma separated cut-off ranks (default: 1,5,10).",
    )
    parser.add_argument(
        "--retrievers",
        type=parse_retrievers,
        default=list(SUPPORTED_RETRIEVERS),
        help="Comma separated retrievers: bm25,dense,hybrid (default: all).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "Store document embeddings in .cache/embeddings and reuse "
            "them on later runs. First run pays the embedding cost, "
            "later runs skip it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Folder for the generated reports (default: reports).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full evaluation pipeline.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        0 on success.
    """
    args = build_parser().parse_args(argv)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, max_queries=args.max_queries)
    print(
        f"Loaded {len(dataset)} queries and "
        f"{len(dataset.corpus)} documents."
    )

    retrievers = build_retrievers(args.retrievers, args.model, use_cache=args.use_cache)
    print(f"Running retrievers: {', '.join(retrievers)}")

    result = evaluate(
        dataset,
        retrievers,
        ks=args.k,
        dataset_name=args.dataset,
        model_name=args.model,
    )

    print()
    print(summary_table(result, plain=True))
    print()

    paths = generate_reports(result, output_dir=args.output_dir, queries=dataset.queries)
    print(f"Markdown report: {paths['markdown']}")
    print(f"CSV report: {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
