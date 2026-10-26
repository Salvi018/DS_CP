"""Small, testable data helpers used by the Streamlit dashboard."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CATEGORICAL_FILTER_COLUMNS = ("role", "app_name", "role_category")


@dataclass
class ReportArtifact:
    """A safely loaded saved report or a non-fatal explanation."""

    kind: str
    content: Any = None
    error: str | None = None


def report_source_freshness(
    report: dict[str, Any], source_path: str | Path
) -> str:
    """Compare a cleaning report's recorded source fingerprint to the current file."""
    source = Path(source_path)
    try:
        stat = source.stat()
    except OSError:
        return "source_unavailable"
    recorded_path = report.get("input_path")
    recorded_size = report.get("input_size_bytes")
    recorded_mtime = report.get("input_mtime_ns")
    if (
        not isinstance(recorded_path, str)
        or not recorded_path
        or not isinstance(recorded_size, int)
        or not isinstance(recorded_mtime, int)
    ):
        return "unknown"
    try:
        recorded_source = Path(recorded_path).resolve()
        current_source = source.resolve()
    except OSError:
        return "unknown"
    if (
        recorded_source != current_source
        or recorded_size != stat.st_size
        or recorded_mtime != stat.st_mtime_ns
    ):
        return "stale"
    return "current"


def available_filter_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return useful categorical and numeric filter fields present in a frame."""
    categorical = [
        column for column in CATEGORICAL_FILTER_COLUMNS if column in frame.columns
    ]
    numeric = [
        column
        for column in frame.columns
        if column.endswith(("_request", "_limit"))
        or column
        in {
            "creation_time",
            "scheduled_time",
            "deletion_time",
            "scheduling_delay_raw_units",
            "observed_lifetime_raw_units",
            "max_instance_per_node",
        }
    ]
    return categorical, numeric


def filter_workloads(
    frame: pd.DataFrame,
    *,
    categories: dict[str, list[str]] | None = None,
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Apply inclusive filters only to existing columns and return a copy."""
    selected = frame.copy()
    for column, values in (categories or {}).items():
        if column not in selected.columns or not values:
            continue
        selected = selected.loc[selected[column].isin(values)]
    for column, bounds in (numeric_ranges or {}).items():
        if column not in selected.columns:
            continue
        if len(bounds) != 2:
            raise ValueError(f"Numeric range for {column!r} must have two bounds")
        try:
            lower, upper = (float(value) for value in bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric range for {column!r}: {bounds!r}"
            ) from exc
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
        ):
            raise ValueError(f"Invalid numeric range for {column!r}: {bounds!r}")
        values = pd.to_numeric(selected[column], errors="coerce")
        selected = selected.loc[values.notna() & values.between(lower, upper)]
    return selected.copy()


def list_report_artifacts(report_root: str | Path) -> list[Path]:
    """Return existing report files in stable order; missing folders are empty."""
    root = Path(report_root)
    if not root.is_dir():
        return []
    try:
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") for part in path.relative_to(root).parts
            )
        )
    except OSError:
        return []


def latest_benchmark_dir(report_root: str | Path) -> Path | None:
    """Return the newest benchmark folder containing a comparison CSV."""
    candidates = [
        path
        for path in list_report_artifacts(report_root)
        if path.name == "scheduler_comparison.csv"
    ]
    try:
        newest = max(
            candidates,
            key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
            default=None,
        )
    except OSError:
        return None
    return newest.parent if newest is not None else None


def load_report_artifact(path: str | Path) -> ReportArtifact:
    """Load JSON, CSV, and text reports without letting bad artifacts crash UI."""
    source = Path(path)
    if not source.is_file():
        return ReportArtifact("missing", error=f"Artifact not found: {source}")
    suffix = source.suffix.lower()
    try:
        if suffix == ".json":
            return ReportArtifact("json", json.loads(source.read_text(encoding="utf-8")))
        if suffix == ".csv":
            with source.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, None)
                if header is None:
                    return ReportArtifact("csv", pd.DataFrame())
                for row_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"row {row_number} has {len(row)} fields; "
                            f"expected {len(header)}"
                        )
            return ReportArtifact("csv", pd.read_csv(source, low_memory=False))
        if suffix in {".md", ".txt"}:
            return ReportArtifact("text", source.read_text(encoding="utf-8"))
        if suffix in {".html", ".htm"}:
            return ReportArtifact(
                "link",
                f"HTML report saved at {source}. It is not embedded in the dashboard.",
            )
        return ReportArtifact(
            "metadata",
            f"Saved artifact ({source.stat().st_size:,} bytes): {source}",
        )
    except (
        OSError,
        UnicodeDecodeError,
        csv.Error,
        json.JSONDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
    ) as exc:
        return ReportArtifact("error", error=f"Could not read {source.name}: {exc}")
