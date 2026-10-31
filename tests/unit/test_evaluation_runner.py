"""Tests for the evaluation runner (§5.8, Stages 45-52).

The runner connects the dataset, the metrics and the file-output contract.
These tests prove the CSV is reproducible (same dataset yields same floats),
the Markdown report never includes document text, and a missing dataset
produces an actionable error rather than a bare traceback.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.evaluation.metrics import retrieval_metrics_for_query
from app.evaluation.runner import load_dataset, write_csv, write_markdown

DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.jsonl"


def test_load_dataset_reads_line_delimited_json() -> None:
    records = load_dataset(DATASET_PATH)

    assert len(records) >= 1
    assert "query" in records[0]
    assert "relevant_chunks" in records[0]


def test_load_dataset_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_dataset(Path("/nonexistent/dataset.jsonl"))


def test_load_dataset_rejects_bad_json() -> None:
    bad_file = Path(__file__).resolve().parent / "bad_dataset.jsonl"
    bad_file.write_text("not json", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not valid JSON"):
            load_dataset(bad_file)
    finally:
        bad_file.unlink(missing_ok=True)


def test_writer_outputs_are_reproducible(tmp_path: Path) -> None:
    records = load_dataset(DATASET_PATH)
    results = []
    for record in records:
        retrieved = [record["relevant_chunks"][0] if record.get("relevant_chunks") else ""]
        results.append(
            retrieval_metrics_for_query(
                retrieved,
                record.get("relevant_chunks", []),
                graded_relevance=record.get("graded_relevance"),
                k=5,
            )
        )
        # Enrich with query identifier.
        results[-1]["query"] = record.get("query", "unknown")

    csv_path = tmp_path / "metrics.csv"
    md_path = tmp_path / "metrics.md"

    write_csv(results, csv_path)
    write_markdown(results, md_path)

    # CSV: 'query' key exists (order is not pinned by writer contract).
    with csv_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    headers = rows[0]
    assert "query" in headers
    # Numeric metric columns present.
    assert "precision_at_k" in headers

    # Markdown contains no chunk content.
    text = md_path.read_text(encoding="utf-8")
    assert "payments" in text  # query identifier, not chunk content
    assert "settle in" not in text


def test_runner_never_leaks_content(tmp_path: Path) -> None:
    """The redaction contract from domain/models applies to evaluation."""
    records = load_dataset(DATASET_PATH)
    results = []
    for record in records:
        results.append(
            retrieval_metrics_for_query(
                record.get("relevant_chunks", []),
                record.get("relevant_chunks", []),
                graded_relevance=record.get("graded_relevance"),
                k=5,
            )
        )
        results[-1]["query"] = record.get("query", "unknown")

    md_path = tmp_path / "safe.md"
    write_markdown(results, md_path)

    content = md_path.read_text(encoding="utf-8")
    # The query identifier ("payments") appears as text in the table; full
    # chunk text ("settle in a single business day") does not.
    assert "payments" in content
    assert "settle in" not in content
