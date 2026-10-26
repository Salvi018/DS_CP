"""Read-only, chunked inspection and validation for workload trace CSV files."""

from __future__ import annotations

import csv
import difflib
import json
import logging
import math
import sqlite3
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from config.settings import DATASET_FILENAME, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    "instance_sn",
    "role",
    "app_name",
    "cpu_request",
    "cpu_limit",
    "gpu_request",
    "gpu_limit",
    "rdma_request",
    "rdma_limit",
    "memory_request",
    "memory_limit",
    "disk_request",
    "disk_limit",
    "max_instance_per_node",
    "creation_time",
    "scheduled_time",
    "deletion_time",
)
RESOURCE_NAMES = ("cpu", "gpu", "memory", "disk", "rdma")
TIMESTAMP_COLUMNS = ("creation_time", "scheduled_time", "deletion_time")
NUMERIC_COLUMNS = frozenset(
    f"{resource}_{kind}"
    for resource in RESOURCE_NAMES
    for kind in ("request", "limit")
) | {"max_instance_per_node", *TIMESTAMP_COLUMNS}
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_SAMPLE_SIZE = 10_000


class DatasetInspectionError(Exception):
    """An expected, actionable error while locating or reading a trace CSV."""


@dataclass
class DatasetInspection:
    """Inspection outputs kept small enough to pass between CLI and dashboard."""

    report: dict[str, Any]
    schema: pd.DataFrame
    missing_values: pd.DataFrame
    numerical_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    data_quality: pd.DataFrame
    sample: pd.DataFrame


def resolve_dataset_path(
    input_path: str | Path | None = None,
    raw_data_dir: str | Path = RAW_DATA_DIR,
) -> Path:
    """Resolve an explicit CSV, the expected name, or one unambiguous raw CSV."""
    if input_path is not None:
        path = Path(input_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise DatasetInspectionError(f"Dataset file does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise DatasetInspectionError(f"Input must be a CSV file: {path}")
        return path.resolve()

    raw_dir = Path(raw_data_dir)
    expected = raw_dir / DATASET_FILENAME
    if expected.is_file():
        return expected.resolve()

    candidates = sorted(path.resolve() for path in raw_dir.glob("*.csv") if path.is_file())
    if len(candidates) == 1:
        LOGGER.info("Expected filename not found; using the only raw CSV: %s", candidates[0].name)
        return candidates[0]
    if len(candidates) > 1:
        choices = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise DatasetInspectionError(
            "Multiple CSV files were found in the raw data directory. Select one with "
            f"--input:\n{choices}"
        )
    raise DatasetInspectionError(
        "No dataset CSV was found. Download the Alibaba GPU Cluster Dataset 2025 from "
        "https://www.kaggle.com/datasets/mdsultanulislamovi/alibaba-gpu-cluster-dataset-2025 "
        f"and place {DATASET_FILENAME} in {expected}."
    )


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """Read one CSV for small interactive use, translating parser errors clearly."""
    csv_path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                csv_path, low_memory=False, index_col=False, on_bad_lines="error"
            )
    except EmptyDataError as exc:
        raise DatasetInspectionError(f"CSV is empty and has no header: {csv_path}") from exc
    except (ParserError, pd.errors.ParserWarning, UnicodeDecodeError, OSError, csv.Error) as exc:
        raise DatasetInspectionError(f"Could not read CSV {csv_path}: {exc}") from exc
    if frame.empty and len(frame.columns) == 0:
        raise DatasetInspectionError(f"CSV is empty and has no header: {csv_path}")
    return frame


def _iter_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            yield from pd.read_csv(
                path,
                chunksize=chunksize,
                low_memory=False,
                index_col=False,
                on_bad_lines="error",
            )
    except EmptyDataError as exc:
        raise DatasetInspectionError(f"CSV is empty and has no header: {path}") from exc
    except (
        ParserError,
        pd.errors.ParserWarning,
        UnicodeDecodeError,
        OSError,
        csv.Error,
        ValueError,
    ) as exc:
        raise DatasetInspectionError(f"Could not parse CSV {path}: {exc}") from exc


def _format_dtype(values: set[str]) -> str:
    return " / ".join(sorted(values))


def _numeric_candidates(frame: pd.DataFrame) -> set[str]:
    return {
        column
        for column in frame.columns
        if column in NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(frame[column].dtype)
    }


def inspect_dataset(
    path: str | Path,
    *,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> DatasetInspection:
    """Inspect and validate a CSV in chunks without modifying its source."""
    csv_path = Path(path).resolve()
    if not csv_path.is_file():
        raise DatasetInspectionError(f"Dataset file does not exist: {csv_path}")
    if chunksize < 1 or sample_size < 1:
        raise ValueError("chunksize and sample_size must be positive integers")

    dtypes: dict[str, set[str]] = {}
    missing_counts: Counter[str] = Counter()
    categorical_counts: dict[str, Counter[str]] = {}
    gpu_request_counts: Counter[str] = Counter()
    numeric_state: dict[str, dict[str, Any]] = {}
    quality_counts: Counter[tuple[str, str]] = Counter()
    seen_values = sqlite3.connect("")
    seen_values.execute("CREATE TABLE seen_rows (value TEXT PRIMARY KEY)")
    seen_values.execute("CREATE TABLE seen_identifiers (value TEXT PRIMARY KEY)")
    observed_domains: dict[str, set[str]] = {"role": set(), "max_instance_per_node": set()}
    duplicate_rows = 0
    duplicate_identifiers = 0
    zero_gpu_requests = 0
    timestamp_unit = "unverified; dataset documentation was not available to confirm units"
    timestamp_values: dict[str, list[float]] = {column: [] for column in TIMESTAMP_COLUMNS}
    delay_samples: list[float] = []
    lifetime_samples: list[float] = []
    samples: list[pd.DataFrame] = []
    sampled_rows = 0
    row_count = 0
    column_names: list[str] | None = None

    for chunk in _iter_chunks(csv_path, chunksize):
        if column_names is None:
            column_names = list(chunk.columns)
        elif list(chunk.columns) != column_names:
            raise DatasetInspectionError("CSV columns changed while reading the file.")
        row_count += len(chunk)

        for column in chunk.columns:
            dtypes.setdefault(column, set()).add(str(chunk[column].dtype))
            missing_counts[column] += int(chunk[column].isna().sum())
            if column not in NUMERIC_COLUMNS:
                counts = chunk[column].dropna().astype(str).value_counts()
                category_counter = categorical_counts.setdefault(column, Counter())
                category_counter.update(counts.to_dict())

        chunk_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        previous_changes = seen_values.total_changes
        seen_values.executemany(
            "INSERT OR IGNORE INTO seen_rows(value) VALUES (?)",
            ((str(int(value)),) for value in chunk_hashes),
        )
        duplicate_rows += len(chunk_hashes) - (seen_values.total_changes - previous_changes)

        if "instance_sn" in chunk.columns:
            ids = chunk["instance_sn"].dropna().astype(str)
            previous_changes = seen_values.total_changes
            seen_values.executemany(
                "INSERT OR IGNORE INTO seen_identifiers(value) VALUES (?)",
                ((identifier,) for identifier in ids),
            )
            duplicate_identifiers += len(ids) - (seen_values.total_changes - previous_changes)

        numeric_columns = _numeric_candidates(chunk)
        converted: dict[str, pd.Series] = {}
        for column in numeric_columns:
            values = pd.to_numeric(chunk[column], errors="coerce")
            converted[column] = values
            invalid_mask = chunk[column].notna() & values.isna()
            quality_counts[("invalid_numeric", column)] += int(invalid_mask.sum())
            non_finite_mask = values.notna() & ~np.isfinite(values)
            quality_counts[("non_finite_numeric", column)] += int(non_finite_mask.sum())
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            state = numeric_state.setdefault(
                column,
                {"count": 0, "sum": 0.0, "sum_squares": 0.0, "min": None, "max": None, "sample": []},
            )
            if len(values):
                array = values.to_numpy(dtype=float)
                state["count"] += len(array)
                state["sum"] += float(array.sum())
                state["sum_squares"] += float(np.square(array).sum())
                minimum, maximum = float(array.min()), float(array.max())
                state["min"] = minimum if state["min"] is None else min(state["min"], minimum)
                state["max"] = maximum if state["max"] is None else max(state["max"], maximum)
                sample = state["sample"]
                available = max(0, sample_size - len(sample))
                if available:
                    sample.extend(array[:available].tolist())

        if "gpu_request" in converted:
            gpu_values = converted["gpu_request"].dropna().astype(str).value_counts()
            gpu_request_counts.update(gpu_values.to_dict())

        for resource in RESOURCE_NAMES:
            request_column = f"{resource}_request"
            limit_column = f"{resource}_limit"
            request = converted.get(request_column)
            limit = converted.get(limit_column)
            if request is not None:
                quality_counts[("negative_request", request_column)] += int((request < 0).sum())
                if resource == "gpu":
                    zero_gpu_requests += int((request == 0).sum())
            if limit is not None:
                quality_counts[("negative_limit", limit_column)] += int((limit < 0).sum())
            if request is not None and limit is not None:
                finite_limit = np.isfinite(limit) & limit.notna()
                exceeds = request.notna() & finite_limit & (request > limit)
                quality_counts[("request_exceeds_limit", request_column)] += int(exceeds.sum())
            for column in (request_column, limit_column):
                if column in chunk.columns:
                    quality_counts[("missing_resource_value", column)] += int(chunk[column].isna().sum())

        if "role" in chunk.columns:
            observed_domains["role"].update(chunk["role"].dropna().astype(str).unique())
        if "max_instance_per_node" in converted:
            values = converted["max_instance_per_node"]
            quality_counts[("negative_max_instance_per_node", "max_instance_per_node")] += int(
                (values < 0).sum()
            )
            observed_domains["max_instance_per_node"].update(
                values.dropna().astype(str).unique()
            )

        for column in TIMESTAMP_COLUMNS:
            if column not in chunk.columns:
                continue
            raw = chunk[column]
            values = pd.to_numeric(raw, errors="coerce")
            quality_counts[("missing_timestamp", column)] += int(raw.isna().sum())
            invalid_timestamp = raw.notna() & (values.isna() | ~np.isfinite(values))
            quality_counts[("invalid_timestamp", column)] += int(invalid_timestamp.sum())
            valid = values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid):
                remaining = max(0, sample_size - len(timestamp_values[column]))
                timestamp_values[column].extend(valid.iloc[:remaining].astype(float).tolist())

        if "creation_time" in converted and "scheduled_time" in converted:
            delay = converted["scheduled_time"] - converted["creation_time"]
            valid_delay = delay.replace([np.inf, -np.inf], np.nan).dropna()
            quality_counts[("negative_scheduling_delay", "scheduled_time - creation_time")] += int(
                (valid_delay < 0).sum()
            )
            delay_samples.extend(valid_delay.iloc[: max(0, sample_size - len(delay_samples))].astype(float))
        if "scheduled_time" in converted and "deletion_time" in converted:
            lifetime = converted["deletion_time"] - converted["scheduled_time"]
            valid_lifetime = lifetime.replace([np.inf, -np.inf], np.nan).dropna()
            quality_counts[("negative_observed_lifetime", "deletion_time - scheduled_time")] += int(
                (valid_lifetime < 0).sum()
            )
            lifetime_samples.extend(
                valid_lifetime.iloc[: max(0, sample_size - len(lifetime_samples))].astype(float)
            )

        if sampled_rows < sample_size:
            take = min(sample_size - sampled_rows, len(chunk))
            if take:
                samples.append(chunk.head(take).copy())
                sampled_rows += take

    seen_values.close()
    if column_names is None:
        raise DatasetInspectionError(f"CSV is empty and has no header: {csv_path}")

    expected = set(EXPECTED_COLUMNS)
    observed = set(column_names)
    if "role" in observed:
        quality_counts[("role_value_domain_unverified", "role")] = len(observed_domains["role"])
    if "max_instance_per_node" in observed:
        quality_counts[("max_instance_domain_unverified", "max_instance_per_node")] = len(
            observed_domains["max_instance_per_node"]
        )
    missing_expected = sorted(expected - observed)
    extra_columns = sorted(observed - expected)
    quality_rows: list[dict[str, Any]] = []

    def add_quality(issue: str, column: str, count: int, details: str) -> None:
        if count:
            quality_rows.append(
                {"issue": issue, "column": column, "count": int(count), "details": details}
            )

    add_quality("duplicate_rows", "", duplicate_rows, "Repeated full-row values detected.")
    add_quality(
        "duplicate_instance_identifiers",
        "instance_sn",
        duplicate_identifiers,
        "Repeated non-missing instance_sn values detected.",
    )
    for (issue, column), count in sorted(quality_counts.items()):
        descriptions = {
            "invalid_numeric": "Non-empty values could not be parsed as numeric.",
            "non_finite_numeric": "Numeric values are positive or negative infinity.",
            "negative_request": "Negative resource requests require validation.",
            "negative_limit": "Negative limits are reported as observed; no sentinel meaning is assumed.",
            "request_exceeds_limit": "Request exceeds the corresponding finite numeric limit.",
            "missing_resource_value": "Resource value is missing.",
            "role_value_domain_unverified": (
                "Distinct observed role values; the available schema documentation does not define an allowed domain."
            ),
            "negative_max_instance_per_node": (
                "Negative value requires validation; no special sentinel meaning is assumed."
            ),
            "max_instance_domain_unverified": (
                "Distinct observed values; the available schema documentation does not define an allowed domain."
            ),
            "missing_timestamp": "May indicate incomplete observation; not automatically classified as corruption.",
            "invalid_timestamp": "Non-empty timestamp could not be parsed as a numeric value.",
            "negative_scheduling_delay": "scheduled_time precedes creation_time.",
            "negative_observed_lifetime": "deletion_time precedes scheduled_time.",
        }
        detail = descriptions.get(issue, issue.replace("_", " "))
        if issue == "role_value_domain_unverified":
            detail += f" Observed values: {sorted(observed_domains['role'])}."
        elif issue == "max_instance_domain_unverified":
            detail += (
                " Observed values: "
                f"{sorted(observed_domains['max_instance_per_node'], key=lambda value: float(value))}."
            )
        add_quality(issue, column, count, detail)

    for column in missing_expected:
        quality_rows.append(
            {
                "issue": "missing_expected_column",
                "column": column,
                "count": 0,
                "details": "Expected documented schema field is absent from the CSV.",
            }
        )
    for column in extra_columns:
        quality_rows.append(
            {
                "issue": "extra_column",
                "column": column,
                "count": 0,
                "details": "Observed field is not in the documented expected-column list.",
            }
        )

    schema_rows = []
    for column in column_names:
        closest = difflib.get_close_matches(column, missing_expected, n=1, cutoff=0.75)
        schema_rows.append(
            {
                "column": column,
                "dtype": _format_dtype(dtypes[column]),
                "schema_status": "expected" if column in expected else "extra",
                "possible_rename_of": closest[0] if closest else "",
                "missing_values": int(missing_counts[column]),
                "missing_percent": (100.0 * missing_counts[column] / row_count) if row_count else 0.0,
            }
        )
    for column in missing_expected:
        schema_rows.append(
            {
                "column": column,
                "dtype": "",
                "schema_status": "missing_expected",
                "possible_rename_of": "",
                "missing_values": 0,
                "missing_percent": 0.0,
            }
        )

    numeric_rows = []
    for column, state in sorted(numeric_state.items()):
        count = state["count"]
        mean = state["sum"] / count if count else math.nan
        variance = max(0.0, state["sum_squares"] / count - mean**2) if count else math.nan
        sample = pd.Series(state["sample"], dtype=float)
        numeric_rows.append(
            {
                "column": column,
                "count": count,
                "mean": mean,
                "std": math.sqrt(variance) if count else math.nan,
                "min": state["min"],
                "25%": sample.quantile(0.25) if count else math.nan,
                "50%": sample.median() if count else math.nan,
                "75%": sample.quantile(0.75) if count else math.nan,
                "max": state["max"],
                "invalid_values": int(quality_counts[("invalid_numeric", column)]),
                "quantiles_are_sampled": count > len(sample),
            }
        )

    categorical_rows: list[dict[str, Any]] = []
    for column, counts in sorted(categorical_counts.items()):
        categorical_rows.extend(
            {
                "column": column,
                "category": category,
                "count": int(count),
                "distinct_values": len(counts),
            }
            for category, count in counts.most_common()
        )
    categorical_summary = pd.DataFrame(
        categorical_rows,
        columns=["column", "category", "count", "distinct_values"],
    )
    if gpu_request_counts:
        categorical_summary = pd.concat(
            [
                categorical_summary,
                pd.DataFrame(
                    [
                        {
                            "column": "gpu_request",
                            "category": category,
                            "count": int(count),
                            "distinct_values": len(gpu_request_counts),
                        }
                        for category, count in gpu_request_counts.most_common()
                    ]
                ),
            ],
            ignore_index=True,
        )

    constant_columns = [
        column
        for column in column_names
        if row_count > 0 and len(categorical_counts.get(column, {})) == 1
        and column not in numeric_state
    ]
    for column, state in numeric_state.items():
        if state["count"] and state["min"] == state["max"]:
            constant_columns.append(column)

    sample_frame = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=column_names)
    if "gpu_request" in sample_frame.columns:
        gpu_values = pd.to_numeric(sample_frame["gpu_request"], errors="coerce")
        zero_gpu_requests = max(zero_gpu_requests, int((gpu_values == 0).sum()))

    numeric_summary = pd.DataFrame(
        numeric_rows,
        columns=[
            "column",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "invalid_values",
            "quantiles_are_sampled",
        ],
    )
    representative_rows = json.loads(sample_frame.head(5).to_json(orient="records"))
    missing_values = pd.DataFrame(
        [
            {
                "column": column,
                "missing_count": int(missing_counts[column]),
                "missing_percent": (100.0 * missing_counts[column] / row_count) if row_count else 0.0,
            }
            for column in column_names
        ]
    )
    quality = pd.DataFrame(
        quality_rows, columns=["issue", "column", "count", "details"]
    )
    timestamp_summary = {
        "unit_assumption": timestamp_unit,
        "timestamp_columns": {
            column: {
                "sample_min": min(values) if values else None,
                "sample_max": max(values) if values else None,
                "sample_size": len(values),
            }
            for column, values in timestamp_values.items()
            if column in column_names
        },
        "scheduling_delay": _sample_statistics(delay_samples),
        "observed_scheduled_lifetime": _sample_statistics(lifetime_samples),
    }
    summary: dict[str, Any] = {
        "dataset_filename": csv_path.name,
        "dataset_path": str(csv_path),
        "inspection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "column_count": len(column_names),
        "columns": [
            {"name": column, "dtype": _format_dtype(dtypes[column])}
            for column in column_names
        ],
        "representative_rows": representative_rows,
        "expected_columns": list(EXPECTED_COLUMNS),
        "missing_expected_columns": missing_expected,
        "extra_columns": extra_columns,
        "possible_renames": {
            row["column"]: row["possible_rename_of"]
            for row in schema_rows
            if row["possible_rename_of"]
        },
        "duplicate_rows": duplicate_rows,
        "duplicate_instance_identifiers": duplicate_identifiers,
        "constant_value_columns": sorted(set(constant_columns)),
        "observed_unverified_domains": {
            column: sorted(values)
            for column, values in observed_domains.items()
            if column in observed
        },
        "zero_gpu_request_count": zero_gpu_requests,
        "timestamp_analysis": timestamp_summary,
        "sample_rows": len(sample_frame),
        "statistics_note": (
            f"Numerical quantiles and chart data use at most {sample_size:,} observed values per "
            "column; count, mean, standard deviation, minimum, and maximum are streamed."
        ),
    }
    return DatasetInspection(
        report=summary,
        schema=pd.DataFrame(schema_rows),
        missing_values=missing_values,
        numerical_summary=numeric_summary,
        categorical_summary=categorical_summary,
        data_quality=quality,
        sample=sample_frame,
    )


def _sample_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def inspection_to_json(inspection: DatasetInspection) -> str:
    """Serialize the concise report with non-finite values represented as null."""
    return json.dumps(inspection.report, indent=2, allow_nan=False)
