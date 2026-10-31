"""Experiment runner (Â§5.8, Stages 45â€“52).

Reads a golden ``dataset.jsonl`` (one line per evaluated query), runs
each query through the retrieval layer, writes the metrics to a
reproducible CSV and a Markdown summary, and never calls a model: the
runner evaluates evidence quality, not answer quality, because the
answer-quality evaluation (grounding check, abstention rate) lives in
`tests/unit/test_citations.py` and `tests/unit/test_answer_pipeline.py`.

The CSV carries one row per query with the machine-readable floats from
`metrics.retrieval_metrics_for_query()`; the Markdown renders the same
numbers with explanations so a reviewer can see which query failed and
why, without opening Python.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def load_dataset(path: Path) -> Sequence[dict[str, Any]]:
    """Load a newline-delimited JSON dataset.

    Each line must carry at minimum ``query`` (str) and ``relevant_chunks``
    (list of chunk ids); ``graded_relevance`` is optional (defaults to
    binary 1.0 for every relevant chunk). The loader rejects missing
    fields with an actionable message rather than a bare ``KeyError``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Dataset line {line_no} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Dataset line {line_no}: expected JSON object, got {type(record).__name__}"
                )
            for key in ("query", "relevant_chunks"):
                if key not in record:
                    raise ValueError(f"Dataset line {line_no}: missing required key '{key}'")
            records.append(record)
    return records


def write_csv(results: Iterable[dict[str, Any]], path: Path) -> None:
    """Write results to a reproducible CSV (no formatting, no rounding)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        for row in rows:
            writer.writerow([row.get(key, "") for key in keys])


def write_markdown(
    results: Iterable[dict[str, Any]],
    path: Path,
    dataset_name: str = "golden",
) -> None:
    """Write a human-readable Markdown report with per-query explanations.

    The report never includes chunk text or query text (redacted by the
    metric contract), only the query identifier, the metric names, their
    float values, and a note on whether the score meets a minimum threshold.
    Because the module does not claim any measured results exist, the
    report says exactly that: "Metric values computed; benchmark results
    pending a reproducible evaluation run."
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"# Evaluation Report: {dataset_name}\n\n")
        stream.write(
            "> Metric values computed by `app/evaluation/metrics.py`; "
            "benchmark results, thresholds and regression gates are "
            "pending a reproducible evaluation run with a live model. "
            "No performance numbers exist yet.\n\n"
        )
        stream.write("| Query | P@5 | R@5 | F1@5 | MRR | nDCG@5 | Note |\n")
        stream.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            query_id = str(row.get("query", "unknown"))[:40]
            p = float(row.get("precision_at_k", 0.0))
            r_ = float(row.get("recall_at_k", 0.0))
            f1 = float(row.get("f1_at_k", 0.0))
            mrr = float(row.get("mrr", 0.0))
            ndcg = float(row.get("ndcg_at_k", 0.0))
            note = "good" if p > 0.5 and r_ > 0.5 else "needs review"
            stream.write(
                f"| `{query_id}` | {p:.2f} | {r_:.2f} | {f1:.2f} | {mrr:.2f} "
                f"| {ndcg:.2f} | {note} |\n"
            )
        stream.write(
            f"\n*{len(rows)} query(s) evaluated. Scores are pure retrieval metrics; "
            "no model-generated answer text is included in this report.*\n"
        )
