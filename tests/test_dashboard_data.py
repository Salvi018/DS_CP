from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dashboard_data import (
    available_filter_columns,
    filter_workloads,
    latest_benchmark_dir,
    list_report_artifacts,
    load_report_artifact,
    report_source_freshness,
)


def test_available_filter_columns_only_returns_present_fields() -> None:
    categorical, numeric = available_filter_columns(
        pd.DataFrame({"role": ["HN"], "cpu_request": [2], "custom": [1]})
    )

    assert categorical == ["role"]
    assert numeric == ["cpu_request"]


def test_filter_workloads_applies_categories_and_inclusive_ranges() -> None:
    frame = pd.DataFrame(
        {
            "role": ["HN", "HN", "CN", "CN"],
            "cpu_request": [2.0, 4.0, "8", "invalid"],
        }
    )

    selected = filter_workloads(
        frame,
        categories={"role": ["HN"]},
        numeric_ranges={"cpu_request": (2.0, 2.0)},
    )

    assert selected.index.tolist() == [0]
    assert len(frame) == 4


def test_filter_workloads_ignores_unavailable_columns_and_rejects_bad_ranges() -> None:
    frame = pd.DataFrame({"role": ["HN", "CN"]})

    selected = filter_workloads(
        frame,
        categories={"missing": ["x"]},
        numeric_ranges={"missing": (0.0, 1.0)},
    )

    assert selected.equals(frame)


def test_report_artifacts_handle_missing_and_malformed_files(tmp_path: Path) -> None:
    assert list_report_artifacts(tmp_path / "not-created") == []
    missing = load_report_artifact(tmp_path / "missing.json")
    assert missing.kind == "missing"
    assert missing.error

    malformed = tmp_path / "broken.json"
    malformed.write_text("{not valid json", encoding="utf-8")
    result = load_report_artifact(malformed)
    assert result.kind == "error"
    assert result.error


def test_latest_benchmark_directory_uses_newest_comparison_report(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    old_report = older / "scheduler_comparison.csv"
    new_report = newer / "scheduler_comparison.csv"
    old_report.write_text("policy\nfirst_fit\n", encoding="utf-8")
    new_report.write_text("policy\nbest_fit\n", encoding="utf-8")
    old_report.touch()
    new_report.touch()

    assert latest_benchmark_dir(tmp_path) == newer
    assert latest_benchmark_dir(tmp_path / "missing") is None


def test_cleaning_report_source_fingerprint_detects_stale_and_unknown_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.csv"
    source.write_text("id\none\n", encoding="utf-8")
    stat = source.stat()
    report = {
        "input_path": str(source.resolve()),
        "input_size_bytes": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
    }

    assert report_source_freshness(report, source) == "current"
    source.write_text("id\ntwo\nthree\n", encoding="utf-8")
    assert report_source_freshness(report, source) == "stale"
    assert report_source_freshness({}, source) == "unknown"
    assert report_source_freshness(report, tmp_path / "missing.csv") == "source_unavailable"


def test_report_artifacts_load_actual_json_csv_and_markdown_formats(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps({"row_count": 2}), encoding="utf-8")
    csv_path = tmp_path / "report.csv"
    csv_path.write_text("column,count\ncpu_request,2\n", encoding="utf-8")
    md_path = tmp_path / "report.md"
    md_path.write_text("# Report", encoding="utf-8")

    assert load_report_artifact(json_path).content == {"row_count": 2}
    assert load_report_artifact(csv_path).content.loc[0, "count"] == 2
    assert load_report_artifact(md_path).content == "# Report"
    assert list_report_artifacts(tmp_path) == [csv_path, json_path, md_path]


def test_malformed_csv_is_reported_without_raising(tmp_path: Path) -> None:
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text('"unfinished\n"unterminated', encoding="utf-8")

    result = load_report_artifact(csv_path)

    assert result.kind == "error"
    assert result.error
