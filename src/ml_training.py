"""Reproducible chronological training pipeline for historical delay prediction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import MODEL_OUTPUT_DIR, PROJECT_VERSION, REPORT_OUTPUT_DIR
from src.ml_evaluation import (
    grouped_metrics,
    regression_metrics,
    save_diagnostic_charts,
)
from src.ml_features import (
    ESSENTIAL_PREDICTION_FEATURES,
    FORBIDDEN_FEATURES,
    TARGET_COLUMN,
    TARGET_UNIT,
    SupervisedWorkloadData,
    build_supervised_data,
    validate_feature_names,
)


class TrainingDataError(ValueError):
    """Raised when data cannot support a leakage-safe temporal experiment."""


@dataclass
class TemporalSplit:
    """Ordered train, validation, and test row positions and boundaries."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    boundaries: dict[str, Any]


def chronological_split(
    creation_time: pd.Series,
    *,
    test_size: float = 0.2,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Partition by distinct creation times with strict chronological order."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    values = pd.to_numeric(creation_time, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise TrainingDataError("Chronological split requires finite creation timestamps")
    unique_times = np.unique(values)
    if len(unique_times) < 5:
        raise TrainingDataError(
            "At least five distinct valid creation times are required for chronological "
            "train/validation/test evaluation."
        )
    test_start_index = int(math.floor(len(unique_times) * (1 - test_size)))
    test_start_index = min(max(test_start_index, 2), len(unique_times) - 1)
    train_validation_count = test_start_index
    train_end_index = int(
        math.floor(train_validation_count * (1 - validation_fraction))
    )
    train_end_index = min(max(train_end_index, 1), test_start_index - 1)
    train_cut = unique_times[train_end_index]
    test_cut = unique_times[test_start_index]
    train = np.flatnonzero(values < train_cut)
    validation = np.flatnonzero((values >= train_cut) & (values < test_cut))
    test = np.flatnonzero(values >= test_cut)
    if min(len(train), len(validation), len(test)) == 0:
        raise TrainingDataError("Chronological split produced an empty partition")
    if not (
        np.max(values[train]) < np.min(values[validation])
        and np.max(values[validation]) < np.min(values[test])
    ):
        raise TrainingDataError("Chronological partitions overlap in creation time")
    boundaries = {
        "train_creation_time_min": float(np.min(values[train])),
        "train_creation_time_max": float(np.max(values[train])),
        "validation_creation_time_min": float(np.min(values[validation])),
        "validation_creation_time_max": float(np.max(values[validation])),
        "test_creation_time_min": float(np.min(values[test])),
        "test_creation_time_max": float(np.max(values[test])),
        "unique_creation_times": int(len(unique_times)),
    }
    return TemporalSplit(train, validation, test, boundaries)


def _preprocessor(data: SupervisedWorkloadData) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if data.numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                data.numeric_features,
            )
        )
    if data.categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                data.categorical_features,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _models(data: SupervisedWorkloadData, seed: int) -> dict[str, Pipeline]:
    return {
        "median_baseline": Pipeline(
            [
                ("preprocessor", _preprocessor(data)),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "dummy_median": Pipeline(
            [
                ("preprocessor", _preprocessor(data)),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "ridge": Pipeline(
            [
                ("preprocessor", _preprocessor(data)),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _preprocessor(data)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=14,
                        min_samples_leaf=3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _candidate_metrics(
    models: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    train_positions: np.ndarray,
    evaluation_positions: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(features.iloc[train_positions], target.iloc[train_positions])
        predictions = model.predict(features.iloc[evaluation_positions])
        result[name] = regression_metrics(
            target.iloc[evaluation_positions].to_numpy(), predictions
        )
    return result


def _feature_importances(
    pipeline: Pipeline,
) -> pd.DataFrame:
    estimator = pipeline.named_steps["regressor"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            columns=["feature", "importance", "model", "interpretation"]
        )
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    values = np.asarray(estimator.feature_importances_, dtype=float)
    rows = sorted(
        zip(names, values),
        key=lambda pair: (-float(pair[1]), str(pair[0])),
    )
    return pd.DataFrame(
        [
            {
                "feature": str(name),
                "importance": float(value),
                "model": "random_forest",
                "interpretation": "Predictive association only; not causal.",
            }
            for name, value in rows
        ],
        columns=["feature", "importance", "model", "interpretation"],
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (Path,)):
        return str(value)
    return value


def _gpu_request_categories(series: pd.Series) -> pd.Series:
    """Keep observed zero requests distinct from missing or invalid requests."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    finite = np.isfinite(values)
    categories = np.full(len(series), "invalid_gpu_request", dtype=object)
    categories[~series.notna().to_numpy()] = "missing_gpu_request"
    categories[finite & (values == 0)] = "zero_gpu_request"
    categories[finite & (values > 0)] = "gpu_requesting"
    categories[finite & (values < 0)] = "negative_gpu_request"
    return pd.Series(categories, name="resource_demand_category")


def train_delay_model(
    frame: pd.DataFrame,
    *,
    dataset_filename: str,
    model_output_dir: str | Path = MODEL_OUTPUT_DIR,
    report_output_dir: str | Path = REPORT_OUTPUT_DIR / "ml",
    seed: int = 42,
    test_size: float = 0.2,
    model: str = "auto",
) -> dict[str, Any]:
    """Train, select on validation MAE, then evaluate candidates on held-out test."""
    data = build_supervised_data(frame)
    if len(data.target) < 30:
        raise TrainingDataError(
            f"At least 30 valid non-negative scheduling-delay rows are required; "
            f"found {len(data.target)}."
        )
    model_options = {"auto", "median_baseline", "dummy_median", "ridge", "random_forest"}
    if model not in model_options:
        raise ValueError(
            "model must be one of auto, median_baseline, dummy_median, ridge, random_forest"
        )
    validate_feature_names(data.feature_names)
    split = chronological_split(data.creation_time, test_size=test_size)
    features, target = data.features, data.target
    models = _models(data, seed)
    validation_metrics = _candidate_metrics(
        models, features, target, split.train, split.validation
    )
    selected_model = (
        min(
            validation_metrics,
            key=lambda name: (
                validation_metrics[name]["mae"],
                0 if name in {"median_baseline", "dummy_median"} else 1,
                name,
            ),
        )
        if model == "auto"
        else model
    )

    fitting_positions = np.concatenate([split.train, split.validation])
    test_predictions: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    final_models: dict[str, Pipeline] = {}
    for name in models:
        final_model = _models(data, seed)[name]
        final_model.fit(features.iloc[fitting_positions], target.iloc[fitting_positions])
        prediction = np.asarray(final_model.predict(features.iloc[split.test]), dtype=float)
        final_models[name] = final_model
        test_predictions[name] = prediction
        test_metrics[name] = regression_metrics(
            target.iloc[split.test].to_numpy(), prediction
        )

    selected_pipeline = final_models[selected_model]
    test_actual = target.iloc[split.test].to_numpy()
    test_feature_frame = features.iloc[split.test].reset_index(drop=True)
    test_target_frame = target.iloc[split.test].reset_index(drop=True)
    test_positions = split.test
    actual_positions = [data.source_positions[index] for index in test_positions]
    identifiers = (
        frame.iloc[actual_positions]["instance_sn"].astype(str).tolist()
        if "instance_sn" in frame
        else [f"row-{position}" for position in actual_positions]
    )

    output_models = Path(model_output_dir)
    output_reports = Path(report_output_dir)
    output_models.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)
    model_path = output_models / "scheduling_delay_pipeline.joblib"
    generated_at = datetime.now(timezone.utc).isoformat()
    feature_schema = {
        "feature_names": data.feature_names,
        "numeric_features": data.numeric_features,
        "categorical_features": data.categorical_features,
        "essential_prediction_features": [
            name
            for name in ESSENTIAL_PREDICTION_FEATURES
            if name in data.feature_names
        ],
        "required_prediction_features": [
            name
            for name in ESSENTIAL_PREDICTION_FEATURES
            if name in data.feature_names
        ],
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "target": TARGET_COLUMN,
        "target_unit": TARGET_UNIT,
        "unavailable_optional_features": data.unavailable_features,
        "feature_source_policy": (
            "Explicit creation-time allowlist only; max_instance_per_node is excluded "
            "because density-limit semantics are unverified."
        ),
    }
    bundle = {
        "model_id": f"gpulytics-delay-{selected_model}-v1",
        "created_at_utc": generated_at,
        "pipeline": selected_pipeline,
        "feature_schema": feature_schema,
        "target_unit": TARGET_UNIT,
        "selected_model": selected_model,
        "selection_mode": "validation_mae" if model == "auto" else "user_specified_model",
    }
    joblib.dump(bundle, model_path)

    model_rows: list[dict[str, Any]] = []
    for name in models:
        model_rows.append(
            {
                "model": name,
                "selected_on_validation": name == selected_model,
                "validation_sample_count": validation_metrics[name]["sample_count"],
                **{
                    f"validation_{metric}": value
                    for metric, value in validation_metrics[name].items()
                    if metric != "sample_count"
                },
                "test_sample_count": test_metrics[name]["sample_count"],
                **{
                    f"test_{metric}": value
                    for metric, value in test_metrics[name].items()
                    if metric != "sample_count"
                },
            }
        )
    model_comparison = pd.DataFrame(model_rows)
    comparison_path = output_reports / "model_comparison.csv"
    model_comparison.to_csv(comparison_path, index=False)

    predictions_path = output_reports / "prediction_examples.csv"
    prediction_frame = pd.DataFrame(
        {
            "workload_id": identifiers,
            "creation_time": data.creation_time.iloc[split.test].to_numpy(),
            "actual_scheduling_delay": test_actual,
            "predicted_scheduling_delay": test_predictions[selected_model],
            "absolute_error": np.abs(test_actual - test_predictions[selected_model]),
            "target_unit": TARGET_UNIT,
        }
    )
    prediction_frame.to_csv(predictions_path, index=False)

    role_metrics: dict[str, list[dict[str, Any]]] = {}
    workload_size_metrics: dict[str, list[dict[str, Any]]] = {}
    if "role" in test_feature_frame:
        role_metrics = {
            name: grouped_metrics(
                test_actual,
                test_predictions[name],
                test_feature_frame["role"],
            )
            for name in models
        }
    if "gpu_request" in test_feature_frame:
        size_group = _gpu_request_categories(test_feature_frame["gpu_request"])
        workload_size_metrics = {
            name: grouped_metrics(
                test_actual,
                test_predictions[name],
                size_group,
            )
            for name in models
        }

    diagnostic_paths = save_diagnostic_charts(
        test_actual,
        test_predictions[selected_model],
        output_reports / "diagnostics",
    )
    importance = _feature_importances(final_models["random_forest"])
    importance_path = output_reports / "feature_importance.csv"
    if not importance.empty:
        importance.to_csv(importance_path, index=False)
    feature_schema_path = output_reports / "feature_schema.json"
    feature_schema_path.write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )
    exclusion_counts = (
        data.exclusions["reasons"].value_counts().to_dict()
        if not data.exclusions.empty
        else {}
    )
    exclusions_path = output_reports / "training_exclusions.csv"
    data.exclusions.to_csv(exclusions_path, index=False)

    training_config = {
        "dataset_filename": dataset_filename,
        "project_version": PROJECT_VERSION,
        "created_at_utc": generated_at,
        "random_seed": seed,
        "requested_test_size": test_size,
        "validation_fraction_of_pre_test_period": 0.15,
        "split_method": "chronological by distinct creation_time values; train < validation < test",
        "split_boundaries": split.boundaries,
        "split_row_counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "valid_target_count": len(data.target),
        "excluded_target_count": len(data.exclusions),
        "exclusion_counts": exclusion_counts,
        "target_formula": "scheduled_time - creation_time",
        "target_unit": TARGET_UNIT,
        "target_negative_policy": "Negative delay rows are excluded and recorded; no target imputation.",
        "feature_names": data.feature_names,
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "unavailable_optional_features": data.unavailable_features,
        "model_parameters": {
            "median_baseline": {"estimator": "DummyRegressor", "strategy": "median"},
            "dummy_median": {"estimator": "DummyRegressor", "strategy": "median"},
            "ridge": {"alpha": 1.0},
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 14,
                "min_samples_leaf": 3,
                "random_state": seed,
                "n_jobs": -1,
            },
        },
        "preprocessing": (
            "Numeric median imputation with missingness indicators and scaling; "
            "categorical constant imputation and OneHotEncoder(handle_unknown='ignore'). "
            "All fitted on train data for validation model selection and train+validation "
            "for final test evaluation."
        ),
        "model_selection": "Lowest validation MAE, then baseline preference and model name for ties; test set is not used for selection.",
        "selected_model": selected_model,
        "model_artifact": str(model_path),
        "limitations": [
            "Timestamp unit and timestamp convention remain unverified; targets and predictions are raw source-unit deltas, not seconds.",
            "This estimates historical scheduling delay and does not estimate actual GPU utilization, power, temperature, or scheduler-specific improvement.",
            "Application name may encode workload-family identity; no group-based unseen-application evaluation was performed.",
            "Observed trace relationships are predictive, not causal.",
        ],
    }
    config_path = output_reports / "training_config.json"
    config_path.write_text(
        json.dumps(_json_safe(training_config), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    evaluation_metrics = {
        "target_unit": TARGET_UNIT,
        "selected_model": selected_model,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_role": role_metrics,
        "test_metrics_by_resource_category": workload_size_metrics,
        "baseline_comparison": {
            name: {
                "test_mae_difference_vs_median_baseline": (
                    test_metrics[name]["mae"]
                    - test_metrics["median_baseline"]["mae"]
                ),
                "test_mae_improvement_over_baseline": (
                    test_metrics["median_baseline"]["mae"]
                    - test_metrics[name]["mae"]
                ),
            }
            for name in models
        },
        "test_row_count": len(split.test),
        "diagnostic_charts": [str(path) for path in diagnostic_paths],
        "feature_importance_available": not importance.empty,
    }
    evaluation_path = output_reports / "evaluation_metrics.json"
    evaluation_path.write_text(
        json.dumps(_json_safe(evaluation_metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report_path = output_reports / "model_report.md"
    report_lines = [
        "# Historical scheduling-delay model",
        "",
        f"- Dataset: `{dataset_filename}`",
        f"- Analysis timestamp (UTC): `{generated_at}`",
        f"- Target: `scheduled_time - creation_time` ({TARGET_UNIT})",
        f"- Valid target rows: {len(data.target)}",
        f"- Excluded target rows: {len(data.exclusions)}",
        f"- Chronological split: train {len(split.train)}, validation {len(split.validation)}, test {len(split.test)}",
        f"- Selected model by validation MAE: `{selected_model}`",
        f"- Serialized pipeline: `{model_path}`",
        "",
        "## Held-out test metrics",
        "",
        "| Model | MAE | Median absolute error | RMSE | R-squared |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in test_metrics.items():
        report_lines.append(
            f"| {name} | {metrics['mae']:.6g} | "
            f"{metrics['median_absolute_error']:.6g} | {metrics['rmse']:.6g} | "
            f"{metrics['r_squared'] if metrics['r_squared'] is not None else 'N/A (test n < 2)'} |"
        )
    report_lines.extend(
        [
            "",
            "Models were selected using validation MAE only. The test results above are held out and were not used for model selection. `median_baseline` and `dummy_median` are intentionally equivalent constant-median references.",
            "",
            "## Creation-time features and leakage controls",
            "",
            *[f"- `{name}`" for name in data.feature_names],
            "",
            "Forbidden outcome/future columns include: "
            + ", ".join(f"`{name}`" for name in FORBIDDEN_FEATURES)
            + ".",
            "Density limit is excluded because its semantics are not verified. Numeric imputers and categorical encoders are fitted only on the training partition for validation, and on train plus validation for final test scoring.",
            "",
            "## Exclusions",
            "",
            *[
                f"- `{reason}`: {count}"
                for reason, count in exclusion_counts.items()
            ],
            "- Negative delays are excluded as invalid; missing timestamps are not imputed.",
            "",
            "## Limitations",
            "",
            "- Source timestamp units and conventions remain unverified; output values are not seconds.",
            "- The model predicts historical scheduling delay only. It does not predict real GPU utilization or scheduler-specific gains.",
            "- Application names may identify workload families; no group-based unseen-application test was run.",
            "- Feature importance is predictive association, not causal evidence.",
            "",
            "## Artifacts",
            "",
            "- `model_comparison.csv`",
            "- `evaluation_metrics.json`",
            "- `feature_schema.json`",
            "- `training_config.json`",
            "- `training_exclusions.csv`",
            "- `prediction_examples.csv`",
            "- `feature_importance.csv` for the fitted random-forest candidate, when supported; importance is not attributed to another selected model.",
            "- `diagnostics/` contains charts only when test sample size permits.",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "selected_model": selected_model,
        "pipeline": selected_pipeline,
        "model_path": model_path,
        "report_dir": output_reports,
        "model_comparison": model_comparison,
        "evaluation_metrics": evaluation_metrics,
        "feature_schema": feature_schema,
        "training_config": training_config,
        "exclusions": data.exclusions,
        "split": split,
        "test_predictions": test_predictions,
        "test_target": test_target_frame,
        "test_features": test_feature_frame,
        "artifacts": [
            model_path,
            comparison_path,
            evaluation_path,
            feature_schema_path,
            config_path,
            predictions_path,
            exclusions_path,
            report_path,
            *diagnostic_paths,
            *([importance_path] if not importance.empty else []),
        ],
    }
