"""Audited workload cleaning and feature engineering.

Raw data is never modified. Timestamp-derived values retain the source units
until authoritative documentation verifies a conversion to seconds.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    PROCESSED_DATA_DIR,
    REPORT_OUTPUT_DIR,
)
from src.dataset_inspector import (
    EXPECTED_COLUMNS,
    RESOURCE_NAMES,
    TIMESTAMP_COLUMNS,
    DatasetInspectionError,
    read_csv_safely,
    resolve_dataset_path,
)

LOGGER = logging.getLogger(__name__)

BASIC_IDENTIFIER_COLUMNS = ("instance_sn",)
OBSERVED_ROLE_VALUES = frozenset({"CN", "HN"})
RESOURCE_NUMERIC_COLUMNS = tuple(
    f"{resource}_{kind}"
    for resource in RESOURCE_NAMES
    for kind in ("request", "limit")
)
NUMERIC_COLUMNS = frozenset(
    {*RESOURCE_NUMERIC_COLUMNS, "max_instance_per_node", *TIMESTAMP_COLUMNS}
)
TIMESTAMP_UNIT_ASSUMPTION = (
    "Unverified. Source values and derived deltas are preserved in original trace units; "
    "no conversion to seconds is performed."
)


class PreprocessingError(Exception):
    """Actionable validation or execution error from dataset preparation."""


@dataclass
class CleaningResult:
    """Prepared workload data and row-level audit records."""

    workloads: pd.DataFrame
    validation_issues: pd.DataFrame
    rejected_rows: pd.DataFrame
    report: dict[str, Any]


def _normalize_name(name: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return normalized or "unnamed_column"


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers to snake_case, rejecting ambiguous name collisions."""
    names = [_normalize_name(column) for column in frame.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        mappings: dict[str, list[str]] = {}
        for original, normalized in zip(frame.columns, names):
            mappings.setdefault(normalized, []).append(str(original))
        collisions = {name: mappings[name] for name in duplicates}
        raise PreprocessingError(
            "Column-name normalization is ambiguous; refusing to guess between "
            f"these source columns: {collisions}"
        )
    normalized_frame = frame.copy()
    normalized_frame.columns = names
    return normalized_frame


def _check_source_header(path: Path) -> None:
    """Catch duplicate headers before pandas can automatically rename them."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PreprocessingError(f"Cannot inspect CSV header {path}: {exc}") from exc
    if not header:
        raise PreprocessingError(f"CSV is empty and has no header: {path}")
    normalized = [_normalize_name(column) for column in header]
    collisions = sorted({name for name in normalized if normalized.count(name) > 1})
    if collisions:
        raise PreprocessingError(
            "Source CSV contains duplicate or ambiguously normalized headers: "
            f"{collisions}"
        )


def load_raw_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Resolve and read a raw CSV without modifying it."""
    try:
        resolved = resolve_dataset_path(path)
        _check_source_header(resolved)
        raw = read_csv_safely(resolved)
    except DatasetInspectionError as exc:
        raise PreprocessingError(str(exc)) from exc
    return raw, resolved


def validate_schema(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Classify observed fields and return schema issues and field categories."""
    observed = set(frame.columns)
    fields: dict[str, str] = {}
    for column in frame.columns:
        if column in BASIC_IDENTIFIER_COLUMNS:
            fields[column] = "required_basic_identification"
        elif column == "role":
            fields[column] = "required_for_role_analysis"
        elif column == "app_name":
            fields[column] = "required_for_application_analysis"
        elif column in RESOURCE_NUMERIC_COLUMNS:
            fields[column] = "required_for_resource_analysis"
        elif column in TIMESTAMP_COLUMNS:
            fields[column] = "required_for_timing_analysis"
        elif column == "max_instance_per_node":
            fields[column] = "optional_for_scheduler_analysis"
        elif column in EXPECTED_COLUMNS:
            fields[column] = "optional"
        else:
            fields[column] = "unknown_or_unexpected"

    issues: list[dict[str, Any]] = []
    for column in BASIC_IDENTIFIER_COLUMNS:
        if column not in observed:
            issues.append(
                _issue(
                    "error",
                    "missing_required_identifier_column",
                    column,
                    None,
                    f"Required basic workload identifier column {column!r} is missing.",
                )
            )
    for column in EXPECTED_COLUMNS:
        if column not in observed and column not in BASIC_IDENTIFIER_COLUMNS:
            issues.append(
                _issue(
                    "warning",
                    "missing_optional_column",
                    column,
                    None,
                    "Optional analysis field is absent; dependent checks/features are skipped.",
                )
            )
    for column in frame.columns:
        if column not in EXPECTED_COLUMNS:
            issues.append(
                _issue(
                    "warning",
                    "unexpected_column",
                    column,
                    None,
                    "Unrecognized column is preserved without interpretation.",
                )
            )
    return issues, fields


def _issue(
    severity: str,
    category: str,
    column: str,
    source_record_number: int | None,
    detail: str,
    observed_value: Any = None,
) -> dict[str, Any]:
    if pd.isna(observed_value):
        observed_value = None
    elif isinstance(observed_value, np.generic):
        observed_value = observed_value.item()
    return {
        "severity": severity,
        "category": category,
        "column": column,
        "source_record_number": source_record_number,
        "detail": detail,
        "observed_value": observed_value,
    }


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def validate_resource_requests(
    frame: pd.DataFrame, source_records: pd.Series
) -> list[dict[str, Any]]:
    """Report invalid, missing, negative, and request-over-limit resource values."""
    issues: list[dict[str, Any]] = []
    for resource in RESOURCE_NAMES:
        request_column = f"{resource}_request"
        limit_column = f"{resource}_limit"
        request = frame[request_column] if request_column in frame else None
        limit = frame[limit_column] if limit_column in frame else None
        for column, values, value_kind in (
            (request_column, request, "request"),
            (limit_column, limit, "limit"),
        ):
            if values is None:
                continue
            for row_position in np.flatnonzero(values.isna().to_numpy()):
                issues.append(
                    _issue(
                        "warning",
                        f"missing_resource_{value_kind}",
                        column,
                        int(source_records.iloc[row_position]),
                        "Missing resource value is preserved; no value is inferred.",
                    )
                )
            for row_position in np.flatnonzero((values < 0).fillna(False).to_numpy()):
                issues.append(
                    _issue(
                        "warning",
                        f"negative_resource_{value_kind}",
                        column,
                        int(source_records.iloc[row_position]),
                        "Negative resource value requires domain validation; no sentinel meaning is assumed.",
                        values.iloc[row_position],
                    )
                )
        if request is not None and limit is not None:
            finite_limit = limit.notna() & np.isfinite(limit)
            exceeds = request.notna() & finite_limit & (request > limit)
            for row_position in np.flatnonzero(exceeds.to_numpy()):
                issues.append(
                    _issue(
                        "warning",
                        "request_exceeds_finite_limit",
                        request_column,
                        int(source_records.iloc[row_position]),
                        f"{request_column} exceeds the finite {limit_column}; values are preserved.",
                        request.iloc[row_position],
                    )
                )
    return issues


def validate_timestamps(
    frame: pd.DataFrame, source_records: pd.Series
) -> list[dict[str, Any]]:
    """Validate timestamp values and order without assuming their units."""
    issues: list[dict[str, Any]] = []
    for column in TIMESTAMP_COLUMNS:
        if column not in frame:
            continue
        values = frame[column]
        invalid_raw = frame.get(f"{column}_raw", pd.Series(pd.NA, index=frame.index))
        missing = values.isna() & invalid_raw.isna()
        for row_position in np.flatnonzero(missing.to_numpy()):
            issues.append(
                _issue(
                    "warning",
                    "missing_timestamp",
                    column,
                    int(source_records.iloc[row_position]),
                    "Timestamp is missing and is retained; incomplete observation may be legitimate.",
                )
            )
        invalid = ~np.isfinite(values.fillna(0).to_numpy(dtype=float)) & values.notna().to_numpy()
        for row_position in np.flatnonzero(invalid):
            issues.append(
                _issue(
                    "error",
                    "invalid_timestamp",
                    column,
                    int(source_records.iloc[row_position]),
                    "Timestamp is non-finite and cannot be used in temporal comparisons.",
                    values.iloc[row_position],
                )
            )
    if {"creation_time", "scheduled_time"}.issubset(frame.columns):
        delay = frame["scheduled_time"] - frame["creation_time"]
        for row_position in np.flatnonzero((delay < 0).fillna(False).to_numpy()):
            issues.append(
                _issue(
                    "warning",
                    "negative_scheduling_delay",
                    "scheduled_time",
                    int(source_records.iloc[row_position]),
                    "scheduled_time precedes creation_time; row is retained.",
                    delay.iloc[row_position],
                )
            )
    if {"scheduled_time", "deletion_time"}.issubset(frame.columns):
        lifetime = frame["deletion_time"] - frame["scheduled_time"]
        for row_position in np.flatnonzero((lifetime < 0).fillna(False).to_numpy()):
            issues.append(
                _issue(
                    "warning",
                    "negative_observed_lifetime",
                    "deletion_time",
                    int(source_records.iloc[row_position]),
                    "deletion_time precedes scheduled_time; row is retained.",
                    lifetime.iloc[row_position],
                )
            )
    return issues


def _append_row_flags(
    frame: pd.DataFrame, issues: list[dict[str, Any]], source_records: pd.Series
) -> pd.DataFrame:
    by_record: dict[int, set[str]] = {}
    counts: Counter[int] = Counter()
    for issue in issues:
        record = issue["source_record_number"]
        if record is not None:
            by_record.setdefault(record, set()).add(issue["category"])
            counts[record] += 1
    flags = [
        ";".join(sorted(by_record.get(int(record), set())))
        for record in source_records
    ]
    frame["validation_issue_count"] = [int(counts.get(int(record), 0)) for record in source_records]
    frame["data_quality_flags"] = flags
    return frame


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create auditable resource/context features and separate historical outcomes.

    Creation-time features contain request/context information only. Scheduled
    and deletion timestamps and derived outcomes are explicitly excluded from
    that feature list. Full-dataset app frequencies and relative arrival
    offsets are descriptive only because they use whole-trace information.
    """
    result = frame.copy()
    generated: list[str] = []
    for resource in RESOURCE_NAMES:
        column = f"{resource}_request"
        if column not in result:
            continue
        feature = f"has_{resource}_request"
        result[feature] = result[column].gt(0).astype("boolean").where(result[column].notna())
        generated.append(feature)

    request_columns = [f"{resource}_request" for resource in RESOURCE_NAMES if f"{resource}_request" in result]
    if request_columns:
        request_values = result[request_columns]
        result["resource_request_observed_count"] = request_values.notna().sum(axis=1).astype("int64")
        positive = request_values.gt(0).where(request_values.notna())
        result["positive_resource_request_count"] = positive.sum(axis=1).astype("int64")
        generated.extend(["resource_request_observed_count", "positive_resource_request_count"])

    normalized_columns: list[str] = []
    for resource in ("cpu", "gpu", "memory"):
        request_column, limit_column = f"{resource}_request", f"{resource}_limit"
        if request_column not in result or limit_column not in result:
            continue
        feature = f"{resource}_request_to_limit_ratio"
        request = result[request_column]
        limit = result[limit_column]
        valid_limit = limit.notna() & np.isfinite(limit) & (limit > 0)
        result[feature] = (request / limit).where(request.notna() & valid_limit)
        generated.append(feature)
        normalized_columns.append(feature)
    if len(normalized_columns) >= 2:
        ratios = result[normalized_columns]
        result["normalized_resource_imbalance"] = ratios.max(axis=1, skipna=True) - ratios.min(
            axis=1, skipna=True
        )
        result.loc[ratios.notna().sum(axis=1) < 2, "normalized_resource_imbalance"] = np.nan
        generated.append("normalized_resource_imbalance")

    if "role" in result:
        result["role_category"] = result["role"].astype("string")
        generated.append("role_category")
    if "app_name" in result:
        result["application_id"] = result["app_name"].astype("string")
        generated.append("application_id")
        result["application_workload_frequency"] = result.groupby(
            "app_name", dropna=False
        )["app_name"].transform("size").astype("int64")
        generated.append("application_workload_frequency")

    if "gpu_request" in result:
        result["has_gpu_request"] = result["gpu_request"].gt(0).astype("boolean").where(
            result["gpu_request"].notna()
        )
        if "has_gpu_request" not in generated:
            generated.append("has_gpu_request")

    historical_outcomes: list[str] = []
    if {"creation_time", "scheduled_time"}.issubset(result.columns):
        result["scheduling_delay_raw_units"] = result["scheduled_time"] - result["creation_time"]
        historical_outcomes.append("scheduling_delay_raw_units")
        generated.append("scheduling_delay_raw_units")
    if {"scheduled_time", "deletion_time"}.issubset(result.columns):
        result["observed_lifetime_raw_units"] = result["deletion_time"] - result["scheduled_time"]
        historical_outcomes.append("observed_lifetime_raw_units")
        generated.append("observed_lifetime_raw_units")
    if "creation_time" in result:
        creation = result["creation_time"]
        baseline = creation.min(skipna=True)
        if pd.notna(baseline):
            result["relative_arrival_time_raw_units"] = creation - baseline
            generated.append("relative_arrival_time_raw_units")

    creation_time_features = [
        column
        for column in (
            "cpu_request",
            "gpu_request",
            "memory_request",
            "rdma_request",
            "disk_request",
            "has_cpu_request",
            "has_gpu_request",
            "has_memory_request",
            "has_rdma_request",
            "has_disk_request",
            "resource_request_observed_count",
            "positive_resource_request_count",
            "cpu_request_to_limit_ratio",
            "gpu_request_to_limit_ratio",
            "memory_request_to_limit_ratio",
            "normalized_resource_imbalance",
            "role_category",
            "application_id",
            "creation_time",
        )
        if column in result.columns
    ]
    return result, {
        "generated_features": generated,
        "creation_time_input_features": creation_time_features,
        "historical_outcome_fields": [
            column
            for column in ("scheduled_time", "deletion_time", *historical_outcomes)
            if column in result
        ],
        "targets": {
            "scheduling_delay": {
                "column": "scheduling_delay_raw_units",
                "unit": "unverified source timestamp units; not seconds",
            }
            if "scheduling_delay_raw_units" in result
            else None
        },
        "excluded_from_creation_time_inputs": [
            column
            for column in (
                "scheduled_time",
                "deletion_time",
                "scheduling_delay_raw_units",
                "observed_lifetime_raw_units",
                "application_workload_frequency",
                "relative_arrival_time_raw_units",
            )
            if column in result
        ],
        "descriptive_only_features": [
            column
            for column in ("application_workload_frequency", "relative_arrival_time_raw_units")
            if column in result
        ],
        "feature_notes": [
            "No resource requests are imputed; unknown requests remain missing.",
            "has_gpu_request is missing when gpu_request is missing and false for observed zero requests.",
            "Request-to-limit ratios are missing for missing requests or non-positive/non-finite limits.",
            "normalized_resource_imbalance is the range of available CPU/GPU/memory request-to-limit ratios; it is missing unless at least two ratios exist.",
            "application_workload_frequency and relative_arrival_time_raw_units use whole-trace statistics and are descriptive-only, not creation-time prediction inputs.",
            "GPU utilization, power, and temperature are not inferred from requested resources.",
            "Observed lifetime may not equal the complete workload lifetime when the observation window is incomplete.",
            "No *_sec features are created while timestamp units remain unverified.",
        ],
    }


def clean_workloads(
    raw: pd.DataFrame,
    *,
    strict: bool = False,
    input_filename: str = "<dataframe>",
    input_size_bytes: int | None = None,
) -> CleaningResult:
    """Validate, clean, and feature-engineer a copy of the input frame."""
    try:
        frame = normalize_column_names(raw).reset_index(drop=True)
    except PreprocessingError:
        raise
    if frame.columns.duplicated().any():
        raise PreprocessingError("Normalized input has duplicate column names.")
    reserved = {
        "validation_issue_count",
        "data_quality_flags",
        "scheduling_delay_raw_units",
        "observed_lifetime_raw_units",
        "relative_arrival_time_raw_units",
        "application_workload_frequency",
        "normalized_resource_imbalance",
        *(f"{resource}_request_to_limit_ratio" for resource in ("cpu", "gpu", "memory")),
        *(f"has_{resource}_request" for resource in RESOURCE_NAMES),
        "resource_request_observed_count",
        "positive_resource_request_count",
        "role_category",
        "application_id",
    }
    collisions = sorted(reserved.intersection(frame.columns))
    if collisions:
        raise PreprocessingError(
            "Input already contains generated feature names; refusing to overwrite "
            f"possibly meaningful source fields: {collisions}"
        )

    source_records = pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
    original = frame.copy()
    schema_issues, field_classification = validate_schema(frame)
    issues = list(schema_issues)

    for column in (name for name in frame.columns if name in NUMERIC_COLUMNS):
        original_values = frame[column].copy()
        converted = pd.to_numeric(original_values, errors="coerce")
        invalid = original_values.notna() & converted.isna()
        non_finite = converted.notna() & ~np.isfinite(converted)
        if invalid.any() or non_finite.any():
            raw_column = f"{column}_raw"
            if raw_column in frame.columns:
                raise PreprocessingError(
                    f"Cannot preserve invalid {column} values because {raw_column} already exists."
                )
            frame[raw_column] = original_values.where(invalid | non_finite)
            for position in np.flatnonzero(invalid.to_numpy()):
                issues.append(
                    _issue(
                        "error",
                        "invalid_timestamp" if column in TIMESTAMP_COLUMNS else "invalid_numeric",
                        column,
                        int(source_records.iloc[position]),
                        f"Value could not be parsed numerically; original value is preserved in {raw_column}.",
                        original_values.iloc[position],
                    )
                )
        for position in np.flatnonzero(non_finite.to_numpy()):
            issues.append(
                _issue(
                    "error",
                    "invalid_timestamp" if column in TIMESTAMP_COLUMNS else "non_finite_numeric",
                    column,
                    int(source_records.iloc[position]),
                    "Infinite numeric value is set missing in the numeric column; raw value is preserved.",
                    converted.iloc[position],
                )
            )
        frame[column] = converted.replace([np.inf, -np.inf], np.nan)

    issues.extend(validate_resource_requests(frame, source_records))
    issues.extend(validate_timestamps(frame, source_records))
    if "role" in frame:
        unknown_roles = frame["role"].notna() & ~frame["role"].astype(str).isin(OBSERVED_ROLE_VALUES)
        for position in np.flatnonzero(unknown_roles.to_numpy()):
            issues.append(
                _issue(
                    "warning",
                    "unknown_role_value",
                    "role",
                    int(source_records.iloc[position]),
                    "Role value was not observed in the inspected Alibaba trace; preserved for review.",
                    frame["role"].iloc[position],
                )
            )
    if "max_instance_per_node" in frame:
        negative = frame["max_instance_per_node"].notna() & (
            frame["max_instance_per_node"] < 0
        )
        for position in np.flatnonzero(negative.to_numpy()):
            issues.append(
                _issue(
                    "warning",
                    "negative_max_instance_per_node",
                    "max_instance_per_node",
                    int(source_records.iloc[position]),
                    "Negative value is preserved; no special sentinel meaning is assumed.",
                    frame["max_instance_per_node"].iloc[position],
                )
            )

    rejection_reasons: dict[int, str] = {}
    duplicate_rows_count = int(frame.duplicated(keep="first").sum())
    duplicate_identifier_count = 0
    if "instance_sn" not in frame:
        # Preserve records in non-strict mode, but make the missing identifier a
        # critical schema error and prevent strict-mode publication.
        pass
    else:
        identifiers = frame["instance_sn"].astype("string").str.strip()
        invalid_identifiers = identifiers.isna() | identifiers.eq("")
        for position in np.flatnonzero(invalid_identifiers.to_numpy(dtype=bool)):
            record = int(source_records.iloc[position])
            rejection_reasons[position] = "missing_or_blank_instance_sn"
            issues.append(
                _issue(
                    "error",
                    "invalid_essential_identifier",
                    "instance_sn",
                    record,
                    "Workload has no usable instance identifier; excluded from cleaned workloads.",
                    frame["instance_sn"].iloc[position],
                )
            )
        eligible = ~invalid_identifiers
        duplicate_rows = frame.loc[eligible].duplicated(keep="first")
        for position in np.flatnonzero(duplicate_rows.to_numpy()):
            record = int(source_records.loc[frame.loc[eligible].index[position]])
            rejection_reasons[position] = "duplicate_full_row"
            issues.append(
                _issue(
                    "warning",
                    "duplicate_full_row",
                    "",
                    record,
                    "Exact repeated record excluded; first occurrence is retained.",
                )
            )
        duplicate_ids = identifiers.duplicated(keep="first") & eligible
        duplicate_identifier_count = int(duplicate_ids.sum())
        for position in np.flatnonzero(duplicate_ids.to_numpy()):
            record = int(source_records.iloc[position])
            rejection_reasons.setdefault(position, "duplicate_instance_identifier")
            issues.append(
                _issue(
                    "error",
                    "duplicate_essential_identifier",
                    "instance_sn",
                    record,
                    "Duplicate essential identifier; later occurrence is excluded.",
                    frame["instance_sn"].iloc[position],
                )
            )

    if strict:
        fatal = [
            issue
            for issue in issues
            if issue["severity"] == "error"
            and issue["category"]
            in {
                "missing_required_identifier_column",
                "invalid_essential_identifier",
                "duplicate_essential_identifier",
                "invalid_numeric",
                "non_finite_numeric",
                "invalid_timestamp",
            }
        ]
        if fatal:
            counts = Counter(issue["category"] for issue in fatal)
            raise PreprocessingError(
                "Strict validation failed: "
                + ", ".join(f"{category}={count}" for category, count in sorted(counts.items()))
            )

    rejected_positions = sorted(rejection_reasons)
    rejected = frame.iloc[rejected_positions].copy() if rejected_positions else pd.DataFrame()
    if not rejected.empty:
        rejected.insert(0, "source_record_number", source_records.iloc[rejected_positions].to_numpy())
        rejected.insert(
            1,
            "exclusion_reason",
            [rejection_reasons[position] for position in rejected_positions],
        )
    keep_mask = np.ones(len(frame), dtype=bool)
    if rejected_positions:
        keep_mask[rejected_positions] = False
    cleaned = frame.iloc[np.flatnonzero(keep_mask)].copy()
    clean_source_records = source_records.iloc[np.flatnonzero(keep_mask)].reset_index(drop=True)
    cleaned = cleaned.reset_index(drop=True)
    cleaned = _append_row_flags(cleaned, issues, clean_source_records)
    cleaned, feature_metadata = engineer_features(cleaned)

    missing_by_column = {
        str(column): int(count)
        for column, count in original.isna().sum().items()
    }
    issue_counts = Counter(issue["category"] for issue in issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    report: dict[str, Any] = {
        "status": "completed_with_errors" if severity_counts["error"] else "completed",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": input_filename,
        "input_size_bytes": input_size_bytes,
        "input_row_count": len(raw),
        "output_row_count": len(cleaned),
        "input_columns": list(frame.columns),
        "field_classification": field_classification,
        "missing_expected_columns": sorted(set(EXPECTED_COLUMNS) - set(frame.columns)),
        "extra_columns": sorted(set(frame.columns) - set(EXPECTED_COLUMNS)),
        "duplicate_rows": duplicate_rows_count,
        "duplicate_instance_identifiers": duplicate_identifier_count,
        "missing_values_by_column": missing_by_column,
        "issue_counts": dict(sorted(issue_counts.items())),
        "warning_count": int(severity_counts["warning"]),
        "error_count": int(severity_counts["error"]),
        "excluded_rows": len(rejected),
        "exclusions_by_reason": dict(sorted(Counter(rejection_reasons.values()).items())),
        "timestamp_unit_assumption": TIMESTAMP_UNIT_ASSUMPTION,
        "missing_value_policy": {
            "timestamps": "Preserved as missing; never replaced with zero.",
            "resource_values": "No imputation; missing values remain missing.",
            "invalid_numeric_values": "Numeric value is missing in normalized field; source text is retained in a *_raw column.",
            "incomplete_rows": "Rows are retained unless a required instance identifier is missing/blank or the row is a duplicate.",
        },
        "transformations_applied": [
            "Column labels normalized to lowercase snake_case only when collision-free.",
            "Known numeric resource, density, and timestamp columns parsed numerically.",
            "Non-numeric and non-finite raw values retained in *_raw columns when applicable.",
            "Rows missing/blank instance_sn, exact duplicate rows, and later non-identical duplicate IDs are excluded with row-level reasons.",
            "Resource request/limit ratios, request-presence flags, workload identity context, and raw-unit timing outcomes generated where supported.",
            "Per-row validation_issue_count and data_quality_flags added.",
        ],
        "feature_metadata": feature_metadata,
        "limitations": [
            "Timestamp units and semantic domain conventions are unverified; raw-unit deltas are not seconds.",
            "Observed lifetime may be incomplete when a workload extends beyond the observation window.",
            "Full-trace application frequencies and relative arrival offsets are descriptive only and must not be used as creation-time prediction inputs.",
            "Scheduling delay is a historical target candidate, not a creation-time input feature.",
            "No GPU utilization, power, or temperature is derived from resource requests.",
            "workload_age_at_scheduling is not added because it is identical to scheduling delay in the observed schema.",
        ],
        "outputs": {},
    }
    issues_frame = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "category",
            "column",
            "source_record_number",
            "detail",
            "observed_value",
        ],
    )
    return CleaningResult(cleaned, issues_frame, rejected, report)


def validate_schema_strict(result: CleaningResult) -> None:
    """Raise for critical schema/identifier/numeric errors in strict mode."""
    fatal_categories = {
        "missing_required_identifier_column",
        "invalid_essential_identifier",
        "duplicate_essential_identifier",
        "invalid_numeric",
        "non_finite_numeric",
        "invalid_timestamp",
    }
    if not result.validation_issues.empty:
        fatal = result.validation_issues[
            (result.validation_issues["severity"] == "error")
            & result.validation_issues["category"].isin(fatal_categories)
        ]
        if not fatal.empty:
            counts = fatal["category"].value_counts().to_dict()
            raise PreprocessingError(
                "Strict validation failed: "
                + ", ".join(f"{category}={count}" for category, count in sorted(counts.items()))
            )


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def save_processed_dataset(
    frame: pd.DataFrame, output_path: str | Path
) -> Path:
    """Atomically save the cleaned CSV, preserving any prior output on failure."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def generate_cleaning_report(
    result: CleaningResult,
    *,
    input_path: Path,
    output_path: Path,
    report_dir: Path,
) -> None:
    """Write JSON, Markdown, and applicable row-level issue/rejection reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    result.report["outputs"] = {
        "cleaned_dataset": str(output_path),
        "cleaning_report_json": str(report_dir / "cleaning_report.json"),
        "cleaning_summary_markdown": str(report_dir / "cleaning_summary.md"),
        "validation_issues_csv": (
            str(report_dir / "validation_issues.csv")
            if not result.validation_issues.empty
            else None
        ),
        "rejected_rows_csv": (
            str(report_dir / "rejected_rows.csv") if not result.rejected_rows.empty else None
        ),
    }
    result.report["input_filename"] = input_path.name
    result.report["input_path"] = str(input_path.resolve())
    result.report["input_mtime_ns"] = input_path.stat().st_mtime_ns
    result.report["input_modified_at_utc"] = datetime.fromtimestamp(
        input_path.stat().st_mtime, timezone.utc
    ).isoformat()
    report_text = json.dumps(_json_safe(result.report), indent=2, allow_nan=False)
    issues_path = report_dir / "validation_issues.csv"
    rejected_path = report_dir / "rejected_rows.csv"
    report_path = report_dir / "cleaning_report.json"
    summary_path = report_dir / "cleaning_summary.md"

    # Stage files first; the processed dataset is committed only after report
    # generation succeeds, preserving an existing valid output on failures.
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, writer in (
            (report_path, lambda stream: stream.write(report_text)),
            (summary_path, lambda stream: stream.write(_summary_markdown(result))),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=report_dir, delete=False
            ) as stream:
                staged_path = Path(stream.name)
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((staged_path, destination))
        if not result.validation_issues.empty:
            staged_path = _stage_csv(result.validation_issues, report_dir)
            staged.append((staged_path, issues_path))
        if not result.rejected_rows.empty:
            staged_path = _stage_csv(result.rejected_rows, report_dir)
            staged.append((staged_path, rejected_path))
        for staged_path, destination in staged:
            os.replace(staged_path, destination)
        if result.validation_issues.empty:
            issues_path.unlink(missing_ok=True)
        if result.rejected_rows.empty:
            rejected_path.unlink(missing_ok=True)
    except OSError:
        for staged_path, _ in staged:
            staged_path.unlink(missing_ok=True)
        raise


def _stage_csv(frame: pd.DataFrame, directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=directory, delete=False
    ) as stream:
        path = Path(stream.name)
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _summary_markdown(result: CleaningResult) -> str:
    report = result.report
    issue_counts = report["issue_counts"]
    issue_lines = (
        "\n".join(f"- `{category}`: {count}" for category, count in sorted(issue_counts.items()))
        if issue_counts
        else "- No validation issues recorded."
    )
    feature_metadata = report["feature_metadata"]
    generated = ", ".join(f"`{feature}`" for feature in feature_metadata["generated_features"]) or "None"
    excluded = (
        "\n".join(
            f"- `{reason}`: {count}"
            for reason, count in sorted(report["exclusions_by_reason"].items())
        )
        if report["exclusions_by_reason"]
        else "- No rows excluded."
    )
    return f"""# GPUlytics cleaning summary

- **Input:** `{report['input_filename']}`
- **Input size:** {report['input_size_bytes']} bytes
- **Input rows:** {report['input_row_count']}
- **Output rows:** {report['output_row_count']}
- **Status:** {report['status']}
- **Warnings:** {report['warning_count']}
- **Errors:** {report['error_count']}
- **Timestamp interpretation:** {report['timestamp_unit_assumption']}

## Validation findings

{issue_lines}

## Excluded records

{excluded}

Every excluded row is written to `rejected_rows.csv` with its source record number and exclusion reason.

## Generated features

{generated}

## Prediction boundary

- Creation-time input features: {', '.join(f'`{name}`' for name in feature_metadata['creation_time_input_features']) or 'None'}
- Historical outcomes: {', '.join(f'`{name}`' for name in feature_metadata['historical_outcome_fields']) or 'None'}
- Excluded from creation-time inputs: {', '.join(f'`{name}`' for name in feature_metadata['excluded_from_creation_time_inputs']) or 'None'}
- Scheduling delay is a historical target candidate, not an input feature. The current derived column is raw timestamp units, not seconds.
- Workload age at scheduling is omitted because it is identical to scheduling delay in the observed schema.
- Observed lifetime may be shorter than the true workload lifetime when the trace observation window is incomplete.

## Missing-value policy

- Missing timestamps and resource requests remain missing.
- No resource values are imputed; invalid numeric strings are preserved in corresponding `*_raw` columns.
- Rows are excluded only for missing/blank essential identifiers or duplicate records/identifiers, with reasons recorded.

## Limitations and next steps

- Confirm timestamp units and role/density semantics from authoritative dataset documentation before conversion or modeling.
- Review warnings before use in scheduling or ML; no unusual resource value was clipped or removed.
- Application frequencies and relative arrival offsets use full-trace context and are descriptive-only, not prediction-safe features.
"""


def prepare_dataset(
    input_path: str | Path | None = None,
    output_path: str | Path = PROCESSED_DATA_DIR / "cleaned_workloads.csv",
    report_dir: str | Path = REPORT_OUTPUT_DIR / "cleaning",
    *,
    strict: bool = False,
) -> CleaningResult:
    """Run the full audited preparation flow and atomically publish its CSV."""
    raw, resolved_input = load_raw_dataset(input_path)
    input_size = resolved_input.stat().st_size
    result = clean_workloads(
        raw,
        strict=False,
        input_filename=resolved_input.name,
        input_size_bytes=input_size,
    )
    if strict:
        validate_schema_strict(result)

    output = Path(output_path).expanduser().resolve()
    reports = Path(report_dir).expanduser().resolve()
    result.report["outputs"] = {
        "cleaned_dataset": str(output),
        "cleaning_report_json": str(reports / "cleaning_report.json"),
        "cleaning_summary_markdown": str(reports / "cleaning_summary.md"),
        "validation_issues_csv": (
            str(reports / "validation_issues.csv")
            if not result.validation_issues.empty
            else None
        ),
        "rejected_rows_csv": (
            str(reports / "rejected_rows.csv") if not result.rejected_rows.empty else None
        ),
    }
    # Build reports in memory and stage their writes before publishing output.
    generate_cleaning_report(
        result,
        input_path=resolved_input,
        output_path=output,
        report_dir=reports,
    )
    save_processed_dataset(result.workloads, output)
    LOGGER.info(
        "Prepared %s: %d input rows, %d output rows, %d warnings, %d errors",
        resolved_input.name,
        result.report["input_row_count"],
        result.report["output_row_count"],
        result.report["warning_count"],
        result.report["error_count"],
    )
    return result
