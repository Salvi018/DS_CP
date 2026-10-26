"""Leak-safe creation-time feature and scheduling-delay target preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "scheduling_delay"
TARGET_UNIT = "unverified source timestamp units; not seconds"
PERMITTED_NUMERIC_FEATURES = (
    "cpu_request",
    "gpu_request",
    "memory_request",
    "rdma_request",
    "disk_request",
    "creation_time",
)
PERMITTED_CATEGORICAL_FEATURES = ("role", "app_name")
FORBIDDEN_FEATURES = (
    "scheduled_time",
    "deletion_time",
    "scheduling_delay",
    "scheduling_delay_raw_units",
    "scheduling_delay_sec",
    "observed_lifetime_raw_units",
    "observed_lifetime_sec",
    "relative_arrival_time_raw_units",
    "application_workload_frequency",
)
ESSENTIAL_PREDICTION_FEATURES = ("creation_time", "role")


class FeaturePreparationError(ValueError):
    """Raised for unusable feature or target schemas."""


@dataclass
class SupervisedWorkloadData:
    """Creation-time features, raw-unit targets, and row-level exclusions."""

    features: pd.DataFrame
    target: pd.Series
    creation_time: pd.Series
    source_positions: list[int]
    exclusions: pd.DataFrame
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    unavailable_features: list[str]


def _finite_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric.astype(float))
    return numeric.where(finite), series.notna() & ~finite


def validate_feature_names(feature_names: list[str] | tuple[str, ...]) -> None:
    """Reject target/outcome fields from any model input schema."""
    forbidden = set(feature_names).intersection(FORBIDDEN_FEATURES)
    if forbidden:
        raise FeaturePreparationError(
            "Forbidden target-derived or future-outcome features: "
            + ", ".join(sorted(forbidden))
        )
    allowed = set(PERMITTED_NUMERIC_FEATURES) | set(PERMITTED_CATEGORICAL_FEATURES)
    unsupported = set(feature_names) - allowed
    if unsupported:
        raise FeaturePreparationError(
            "Features are not approved as available at workload creation: "
            + ", ".join(sorted(unsupported))
        )


def build_supervised_data(frame: pd.DataFrame) -> SupervisedWorkloadData:
    """Recompute delay from source timestamps and report every excluded row.

    Negative delays are excluded as invalid observations; missing or non-finite
    timestamps are never imputed. Feature columns are selected only from the
    explicit creation-time allowlist. Density semantics are unverified and that
    field is intentionally absent.
    """
    if "creation_time" not in frame.columns or "scheduled_time" not in frame.columns:
        missing = [
            name
            for name in ("creation_time", "scheduled_time")
            if name not in frame.columns
        ]
        raise FeaturePreparationError(
            "Cannot construct scheduling-delay target; missing timestamp columns: "
            + ", ".join(missing)
        )
    creation, invalid_creation = _finite_numeric(frame["creation_time"])
    scheduled, invalid_scheduled = _finite_numeric(frame["scheduled_time"])
    delay = scheduled - creation
    invalid_delay = delay.notna() & ~np.isfinite(delay.astype(float))

    exclusions: list[dict[str, Any]] = []
    eligible_positions: list[int] = []
    for position in range(len(frame)):
        reasons: list[str] = []
        if pd.isna(frame.iloc[position]["creation_time"]):
            reasons.append("missing_creation_time")
        elif bool(invalid_creation.iloc[position]):
            reasons.append("invalid_creation_time")
        if pd.isna(frame.iloc[position]["scheduled_time"]):
            reasons.append("missing_scheduled_time")
        elif bool(invalid_scheduled.iloc[position]):
            reasons.append("invalid_scheduled_time")
        if bool(invalid_delay.iloc[position]):
            reasons.append("invalid_scheduling_delay")
        elif pd.notna(delay.iloc[position]) and delay.iloc[position] < 0:
            reasons.append("negative_scheduling_delay")
        if reasons:
            identifier = frame.iloc[position].get("instance_sn", f"row-{position}")
            exclusions.append(
                {
                    "source_row": position,
                    "workload_id": str(identifier),
                    "reasons": ";".join(reasons),
                }
            )
        else:
            eligible_positions.append(position)

    numeric_features = [
        name for name in PERMITTED_NUMERIC_FEATURES if name in frame.columns
    ]
    categorical_features = [
        name for name in PERMITTED_CATEGORICAL_FEATURES if name in frame.columns
    ]
    feature_names = numeric_features + categorical_features
    validate_feature_names(feature_names)
    if not feature_names:
        raise FeaturePreparationError(
            "No approved creation-time features are available in the dataset"
        )
    features = frame.iloc[eligible_positions][feature_names].copy()
    for name in numeric_features:
        features[name], _ = _finite_numeric(features[name])
    for name in categorical_features:
        features[name] = features[name].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    return SupervisedWorkloadData(
        features=features.reset_index(drop=True),
        target=delay.iloc[eligible_positions].astype(float).reset_index(drop=True),
        creation_time=creation.iloc[eligible_positions].astype(float).reset_index(drop=True),
        source_positions=eligible_positions,
        exclusions=pd.DataFrame(
            exclusions, columns=["source_row", "workload_id", "reasons"]
        ),
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        unavailable_features=[
            name
            for name in (*PERMITTED_NUMERIC_FEATURES, *PERMITTED_CATEGORICAL_FEATURES)
            if name not in frame.columns
        ],
    )


def prediction_features(
    records: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    feature_schema: dict[str, Any],
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Validate inference records and select the persisted feature schema."""
    frame = (
        records.copy()
        if isinstance(records, pd.DataFrame)
        else pd.DataFrame([records])
        if isinstance(records, dict)
        else pd.DataFrame(records)
    )
    forbidden_present = set(frame.columns).intersection(FORBIDDEN_FEATURES)
    if forbidden_present:
        raise FeaturePreparationError(
            "Prediction input contains forbidden future/outcome fields: "
            + ", ".join(sorted(forbidden_present))
        )
    feature_names = list(feature_schema["feature_names"])
    validate_feature_names(feature_names)
    missing_essential = [
        name
        for name in feature_schema.get(
            "essential_prediction_features", ESSENTIAL_PREDICTION_FEATURES
        )
        if name not in frame.columns or frame[name].isna().any()
    ]
    if missing_essential:
        raise FeaturePreparationError(
            "Prediction input is missing essential fields: "
            + ", ".join(missing_essential)
        )
    missing_features = [name for name in feature_names if name not in frame.columns]
    required_schema = set(feature_schema.get("required_prediction_features", []))
    missing_required = sorted(required_schema.intersection(missing_features))
    if missing_required:
        raise FeaturePreparationError(
            "Prediction input is missing required model features: "
            + ", ".join(missing_required)
        )
    warnings: list[list[str]] = []
    for _, row in frame.iterrows():
        row_warnings: list[str] = []
        for name in missing_features:
            row_warnings.append(f"optional feature {name!r} is missing; fitted imputer will be used")
        for name in feature_names:
            if name in row and pd.isna(row[name]) and name not in feature_schema.get(
                "essential_prediction_features", ESSENTIAL_PREDICTION_FEATURES
            ):
                row_warnings.append(f"optional feature {name!r} is missing; fitted imputer will be used")
        warnings.append(row_warnings)
    selected = pd.DataFrame(index=frame.index)
    for name in feature_names:
        selected[name] = frame[name] if name in frame else np.nan
        if name in feature_schema.get("numeric_features", []):
            selected[name], _ = _finite_numeric(selected[name])
        elif name in feature_schema.get("categorical_features", []):
            selected[name] = selected[name].map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )
    return selected.reset_index(drop=True), warnings
