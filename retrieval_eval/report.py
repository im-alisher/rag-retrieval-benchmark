"""Report generator.

Turns an :class:`~retrieval_eval.evaluate.EvaluationResult` into:

1. A markdown report with a summary table and a qualitative section.
2. A CSV file with one row per retriever and one column per metric.

The qualitative section lists the queries where BM25 and Dense retrieval
disagree the most. It shows that lexical and semantic search often find
different documents for the same question.

Reports are written to ``reports/`` only when this module is called.
Nothing is written on import.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

from retrieval_eval.evaluate import EvaluationResult, RetrieverResult

DEFAULT_OUTPUT_DIR = "reports"
TOP_DISAGREEMENTS = 5
COMPARED_RETRIEVERS = ("bm25", "dense")


def metric_columns(result: EvaluationResult) -> list[str]:
    """Return ordered metric column names for the requested cut-offs.

    Args:
        result: The evaluation output.

    Returns:
        Names like ``recall@1``, ``mrr@1``, ... ordered by k.
    """
    columns: list[str] = []
    for k in result.ks:
        columns.extend([f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", f"map@{k}"])
    return columns


def _format_row(result: EvaluationResult, entry: RetrieverResult) -> list[str]:
    """Format one retriever row: name, metrics, latency, query count."""
    columns = metric_columns(result)
    row = [entry.name]
    row.extend(f"{entry.metrics.get(col, 0.0):.4f}" for col in columns)
    row.append(f"{entry.avg_latency_ms:.3f}")
    row.append(str(entry.num_queries))
    return row


def table_header(result: EvaluationResult) -> list[str]:
    """Return the header row used by both the CSV and the tables."""
    return ["retriever", *metric_columns(result), "avg_latency_ms", "queries"]


def summary_table(result: EvaluationResult, plain: bool = False) -> str:
    """Render the results as a markdown table.

    Args:
        result: The evaluation output.
        plain: If True, pad cells with spaces instead of using markdown
            pipes. Useful for printing in a terminal.

    Returns:
        A table as a string.
    """
    header = table_header(result)
    rows = [_format_row(result, entry) for entry in result.results]

    if plain:
        widths = [
            max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
            for i in range(len(header))
        ]
        lines = [
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)),
            "  ".join("-" * widths[i] for i in range(len(header))),
        ]
        lines.extend(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in rows
        )
        return "\n".join(lines)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def disagreement_score(top_a: Sequence[str], top_b: Sequence[str]) -> float:
    """Jaccard distance between two top-k lists.

    0.0 means both retrievers returned exactly the same documents.
    1.0 means they share no document at all.

    Args:
        top_a: Document ids from the first retriever, best first.
        top_b: Document ids from the second retriever, best first.

    Returns:
        A value in ``[0, 1]``.

    Example:
        >>> round(disagreement_score(["d1", "d2"], ["d3", "d4"]), 2)
        1.0
        >>> disagreement_score(["d1", "d2"], ["d2", "d1"])
        0.0
    """
    set_a, set_b = set(top_a), set(top_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - len(set_a & set_b) / len(union)


def top_disagreements(
    result: EvaluationResult,
    top_n: int = TOP_DISAGREEMENTS,
) -> list[tuple[str, float, list[str], list[str]]]:
    """Find the queries where two retrievers disagree the most.

    Args:
        result: The evaluation output.
        top_n: How many queries to return.

    Returns:
        Tuples of ``(query_id, score, first_list, second_list)`` sorted
        by descending disagreement score. Empty when either retriever
        is missing from the run.
    """
    try:
        first = result.get(COMPARED_RETRIEVERS[0])
        second = result.get(COMPARED_RETRIEVERS[1])
    except KeyError:
        return []

    by_id_second = {q.query_id: q.retrieved for q in second.query_results}
    scored: list[tuple[str, float, list[str], list[str]]] = []
    for query in first.query_results:
        other = by_id_second.get(query.query_id, [])
        score = disagreement_score(query.retrieved, other)
        scored.append((query.query_id, score, query.retrieved, other))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_n]


def to_csv(result: EvaluationResult) -> str:
    """Render the summary table as CSV text.

    Args:
        result: The evaluation output.

    Returns:
        CSV content including the header row.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table_header(result))
    for entry in result.results:
        writer.writerow(_format_row(result, entry))
    return buffer.getvalue()


def to_markdown(
    result: EvaluationResult,
    queries: Mapping[str, str] | None = None,
    top_n: int = TOP_DISAGREEMENTS,
) -> str:
    """Render the full markdown report.

    Args:
        result: The evaluation output.
        queries: Optional query id to query text map, used to show the
            text of disagreeing queries.
        top_n: How many disagreeing queries to show.

    Returns:
        Markdown text.
    """
    lines: list[str] = []
    lines.append("# Retrieval Evaluation Report")
    lines.append("")
    if result.dataset_name:
        lines.append(f"- Dataset: `{result.dataset_name}`")
    if result.model_name:
        lines.append(f"- Model: `{result.model_name}`")
    lines.append(f"- Queries: {result.results[0].num_queries if result.results else 0}")
    lines.append(f"- Cut-offs (k): {', '.join(str(k) for k in result.ks)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(summary_table(result))
    lines.append("")

    disagreements = top_disagreements(result, top_n=top_n)
    if disagreements:
        lines.append(f"## Top {len(disagreements)} queries where BM25 and Dense disagree")
        lines.append("")
        lines.append(
            "Disagreement is the Jaccard distance between the two top-k "
            "lists: 0 means identical results, 1 means no shared document."
        )
        lines.append("")
        for query_id, score, bm25_hits, dense_hits in disagreements:
            text = queries.get(query_id, "") if queries else ""
            lines.append(f"### {query_id} (disagreement {score:.2f})")
            lines.append("")
            if text:
                lines.append(f"Query: {text}")
                lines.append("")
            lines.append(f"- BM25 top 5: {', '.join(bm25_hits[:5]) or '(none)'}")
            lines.append(f"- Dense top 5: {', '.join(dense_hits[:5]) or '(none)'}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_reports(
    result: EvaluationResult,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    queries: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Write the markdown and CSV reports to ``output_dir``.

    Args:
        result: The evaluation output.
        output_dir: Folder for the reports. Created if missing.
        queries: Optional query id to query text map.

    Returns:
        Mapping of report kind (``markdown`` or ``csv``) to file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "report.md"
    md_path.write_text(to_markdown(result, queries=queries), encoding="utf-8")

    csv_path = out / "report.csv"
    csv_path.write_text(to_csv(result), encoding="utf-8")

    return {"markdown": str(md_path), "csv": str(csv_path)}
