"""Inference helpers for serialized historical scheduling-delay models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml_features import (
    FeaturePreparationError,
    TARGET_UNIT,
    prediction_features,
)


class PredictionError(ValueError):
    """Raised when a stored model or prediction input is invalid."""


def _validate_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise PredictionError("Model file does not contain a valid GPUlytics delay-model bundle")
    pipeline = bundle.get("pipeline")
    schema = bundle.get("feature_schema")
    if not callable(getattr(pipeline, "predict", None)):
        raise PredictionError("Model bundle does not contain a usable prediction pipeline")
    if not isinstance(schema, dict):
        raise PredictionError("Model bundle does not contain a valid feature schema")
    feature_names = schema.get("feature_names")
    numeric_features = schema.get("numeric_features")
    categorical_features = schema.get("categorical_features")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) for name in feature_names)
        or not isinstance(numeric_features, list)
        or not isinstance(categorical_features, list)
        or any(not isinstance(name, str) for name in numeric_features)
        or any(not isinstance(name, str) for name in categorical_features)
    ):
        raise PredictionError("Model bundle feature schema is incomplete or invalid")
    if set(numeric_features).union(categorical_features) != set(feature_names):
        raise PredictionError("Model bundle feature types do not match its feature schema")
    for key in ("essential_prediction_features", "required_prediction_features"):
        values = schema.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(name, str) for name in values
        ):
            raise PredictionError(f"Model bundle field {key!r} must be a list of feature names")
        if not set(values).issubset(feature_names):
            raise PredictionError(
                f"Model bundle field {key!r} references an unknown feature"
            )
    return bundle


def load_delay_model(path: str | Path) -> dict[str, Any]:
    """Load the full preprocessing-plus-estimator bundle."""
    source = Path(path)
    if not source.is_file():
        raise PredictionError(f"Trained delay model not found: {source}")
    try:
        bundle = joblib.load(source)
    except (
        OSError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        ImportError,
        KeyError,
        TypeError,
    ) as exc:
        raise PredictionError(f"Could not load trained model {source}: {exc}") from exc
    return _validate_bundle(bundle)


def predict_scheduling_delay(
    records: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    model: dict[str, Any] | str | Path,
) -> pd.DataFrame:
    """Predict historical delay with warnings, model ID, and source-unit label."""
    bundle = (
        load_delay_model(model)
        if isinstance(model, (str, Path))
        else _validate_bundle(model)
    )
    try:
        features, row_warnings = prediction_features(
            records, bundle["feature_schema"]
        )
        predictions = np.asarray(bundle["pipeline"].predict(features), dtype=float)
    except (FeaturePreparationError, AttributeError, TypeError, ValueError) as exc:
        raise PredictionError(str(exc)) from exc
    if not np.isfinite(predictions).all():
        raise PredictionError("The model produced a non-finite delay prediction")
    identifiers: list[str | None]
    if isinstance(records, dict):
        identifiers = [str(records["instance_sn"]) if records.get("instance_sn") else None]
    elif isinstance(records, pd.DataFrame) and "instance_sn" in records:
        identifiers = [
            str(value) if pd.notna(value) else None
            for value in records["instance_sn"].tolist()
        ]
    elif isinstance(records, list):
        identifiers = [
            str(item["instance_sn"]) if item.get("instance_sn") else None
            for item in records
        ]
    else:
        identifiers = [None] * len(predictions)
    return pd.DataFrame(
        {
            "workload_id": identifiers,
            "predicted_scheduling_delay": predictions,
            "target_unit": bundle.get("target_unit", TARGET_UNIT),
            "model_id": bundle.get("model_id", "unknown"),
            "input_warnings": ["; ".join(items) for items in row_warnings],
            "interpretation": (
                "Historical scheduling-delay estimate; not a guaranteed outcome "
                "or scheduler-specific performance prediction."
            ),
        }
    )
