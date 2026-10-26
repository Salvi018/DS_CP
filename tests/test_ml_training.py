from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_evaluation import regression_metrics
from src.ml_features import (
    FeaturePreparationError,
    build_supervised_data,
    prediction_features,
    validate_feature_names,
)
from src.ml_prediction import PredictionError, load_delay_model, predict_scheduling_delay
from src.ml_training import (
    TrainingDataError,
    _gpu_request_categories,
    chronological_split,
    train_delay_model,
)


def _frame(count: int = 100) -> pd.DataFrame:
    creation = np.arange(count, dtype=float)
    frame = pd.DataFrame(
        {
            "instance_sn": [f"workload-{index}" for index in range(count)],
            "creation_time": creation,
            "scheduled_time": creation + (creation % 9),
            "cpu_request": np.where(creation >= count * 0.8, 100_000.0, 2.0 + creation % 5),
            "gpu_request": np.where(creation % 3 == 0, 1.0, 0.0),
            "memory_request": 4.0 + creation % 7,
            "rdma_request": 1.0,
            "disk_request": 10.0,
            "role": np.where(creation % 2 == 0, "HN", "CN"),
            "app_name": np.where(creation % 4 == 0, "app-a", "app-b"),
            "scheduled_delay_raw_units": 777.0,
            "scheduling_delay_raw_units": 777.0,
            "deletion_time": creation + 999,
            "observed_lifetime_raw_units": 555.0,
            "relative_arrival_time_raw_units": creation,
            "application_workload_frequency": 1234,
            "max_instance_per_node": -1,
        }
    )
    frame.loc[4, "cpu_request"] = np.nan
    frame.loc[5, "app_name"] = None
    return frame


def test_target_is_recomputed_from_source_timestamps_and_exclusions_recorded() -> None:
    frame = _frame(15)
    frame["scheduled_time"] = frame["scheduled_time"].astype(object)
    frame.loc[0, "scheduled_time"] = float(frame.loc[0, "creation_time"]) - 1.0
    frame.loc[1, "creation_time"] = np.nan
    frame.loc[2, "scheduled_time"] = np.nan
    frame.loc[3, "scheduled_time"] = "not-a-timestamp"

    data = build_supervised_data(frame)

    assert len(data.target) == 11
    assert data.target.iloc[0] == 4
    reasons = set(data.exclusions["reasons"])
    assert "negative_scheduling_delay" in reasons
    assert "missing_creation_time" in reasons
    assert "missing_scheduled_time" in reasons
    assert "invalid_scheduled_time" in reasons
    assert "scheduling_delay_raw_units" not in data.feature_names
    assert "deletion_time" not in data.feature_names
    assert "max_instance_per_node" not in data.feature_names


def test_missing_timestamp_columns_and_forbidden_feature_schema_fail_clearly() -> None:
    with pytest.raises(FeaturePreparationError, match="missing timestamp columns"):
        build_supervised_data(pd.DataFrame({"creation_time": [1]}))
    with pytest.raises(FeaturePreparationError, match="Forbidden"):
        validate_feature_names(["cpu_request", "scheduled_time"])


def test_chronological_split_has_strict_non_overlapping_time_order() -> None:
    split = chronological_split(pd.Series(np.arange(100, dtype=float)), test_size=0.2)
    values = np.arange(100, dtype=float)

    assert max(values[split.train]) < min(values[split.validation])
    assert max(values[split.validation]) < min(values[split.test])
    assert len(split.train) > len(split.validation) > 0
    assert len(split.test) > 0


def test_chronological_split_rejects_insufficient_unique_timestamps() -> None:
    with pytest.raises(TrainingDataError, match="five distinct"):
        chronological_split(pd.Series([1, 1, 1, 2]))


def test_gpu_request_grouping_distinguishes_missing_zero_and_invalid_values() -> None:
    categories = _gpu_request_categories(
        pd.Series([1, 0, np.nan, "invalid", -1, np.inf])
    )

    assert categories.tolist() == [
        "gpu_requesting",
        "zero_gpu_request",
        "missing_gpu_request",
        "invalid_gpu_request",
        "negative_gpu_request",
        "invalid_gpu_request",
    ]


def test_feature_pipeline_imputer_fits_without_looking_at_test_rows(tmp_path: Path) -> None:
    frame = _frame()
    trained = train_delay_model(
        frame,
        dataset_filename="synthetic.csv",
        model_output_dir=tmp_path / "models",
        report_output_dir=tmp_path / "reports",
        seed=17,
    )
    bundle = load_delay_model(trained["model_path"])
    numeric_pipeline = bundle["pipeline"].named_steps["preprocessor"].named_transformers_[
        "numeric"
    ]
    imputer = numeric_pipeline.named_steps["imputer"]
    fitted_cpu_median = imputer.statistics_[
        bundle["feature_schema"]["numeric_features"].index("cpu_request")
    ]
    fitting_creation_max = trained["training_config"]["split_boundaries"][
        "validation_creation_time_max"
    ]
    fitting_rows = frame[frame["creation_time"] <= fitting_creation_max]
    expected = pd.to_numeric(fitting_rows["cpu_request"], errors="coerce").median()

    assert fitted_cpu_median == expected
    assert fitted_cpu_median < 100_000


def test_unseen_category_prediction_and_optional_features_are_supported(tmp_path: Path) -> None:
    trained = train_delay_model(
        _frame(),
        dataset_filename="synthetic.csv",
        model_output_dir=tmp_path / "models",
        report_output_dir=tmp_path / "reports",
        seed=11,
    )
    prediction = predict_scheduling_delay(
        {
            "instance_sn": "unseen-workload",
            "creation_time": 101,
            "role": "NEW_ROLE",
            "app_name": "unseen-application",
            "cpu_request": 4,
            "gpu_request": 0,
        },
        trained["model_path"],
    )

    assert len(prediction) == 1
    assert prediction.loc[0, "workload_id"] == "unseen-workload"
    assert prediction.loc[0, "model_id"] == (
        f"gpulytics-delay-{trained['selected_model']}-v1"
    )
    assert prediction.loc[0, "target_unit"] == (
        "unverified source timestamp units; not seconds"
    )
    assert np.isfinite(prediction.loc[0, "predicted_scheduling_delay"])
    assert "memory_request" in prediction.loc[0, "input_warnings"]


def test_prediction_rejects_missing_essential_and_forbidden_inputs(tmp_path: Path) -> None:
    trained = train_delay_model(
        _frame(60),
        dataset_filename="synthetic.csv",
        model_output_dir=tmp_path / "models",
        report_output_dir=tmp_path / "reports",
    )

    with pytest.raises(PredictionError, match="essential fields"):
        predict_scheduling_delay({"creation_time": 101}, trained["model_path"])
    with pytest.raises(PredictionError, match="forbidden"):
        predict_scheduling_delay(
            {
                "creation_time": 101,
                "role": "HN",
                "scheduled_time": 102,
            },
            trained["model_path"],
        )


def test_missing_model_artifact_has_a_clear_prediction_error(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="Trained delay model not found"):
        load_delay_model(tmp_path / "missing_pipeline.joblib")


def test_corrupt_and_incompatible_model_artifacts_fail_with_prediction_error(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a joblib artifact")
    with pytest.raises(PredictionError, match="Could not load trained model"):
        load_delay_model(corrupt)

    incompatible = tmp_path / "incompatible.joblib"
    incompatible.write_bytes(b"")
    import joblib

    joblib.dump({"pipeline": object(), "feature_schema": {}}, incompatible)
    with pytest.raises(PredictionError, match="usable prediction pipeline"):
        load_delay_model(incompatible)

    with pytest.raises(PredictionError, match="feature schema is incomplete"):
        predict_scheduling_delay(
            {"creation_time": 1, "role": "HN"},
            {
                "pipeline": type("PipelineStub", (), {"predict": lambda self, _: [0]})(),
                "feature_schema": {"feature_names": []},
            },
        )


def test_training_saves_held_out_metrics_baseline_and_diagnostics(tmp_path: Path) -> None:
    trained = train_delay_model(
        _frame(),
        dataset_filename="synthetic.csv",
        model_output_dir=tmp_path / "models",
        report_output_dir=tmp_path / "reports",
        seed=42,
    )
    metrics = trained["evaluation_metrics"]
    reports = tmp_path / "reports"

    assert set(metrics["test_metrics"]) == {
        "median_baseline",
        "dummy_median",
        "ridge",
        "random_forest",
    }
    assert metrics["test_metrics"]["median_baseline"]["mae"] == metrics[
        "test_metrics"
    ]["dummy_median"]["mae"]
    assert metrics["test_metrics"]["random_forest"]["sample_count"] > 1
    assert trained["selected_model"] in metrics["validation_metrics"]
    assert trained["training_config"]["split_row_counts"]["test"] > 0
    assert (reports / "model_comparison.csv").is_file()
    assert (reports / "evaluation_metrics.json").is_file()
    assert (reports / "feature_schema.json").is_file()
    assert (reports / "training_config.json").is_file()
    assert (reports / "prediction_examples.csv").is_file()
    assert (reports / "training_exclusions.csv").is_file()
    assert (reports / "model_report.md").is_file()
    assert (reports / "diagnostics" / "actual_vs_predicted.html").is_file()


def test_reproducible_training_and_explicit_model_selection(tmp_path: Path) -> None:
    frame = _frame(60)
    first = train_delay_model(
        frame,
        dataset_filename="synthetic.csv",
        model_output_dir=tmp_path / "models-a",
        report_output_dir=tmp_path / "reports-a",
        seed=7,
        model="ridge",
    )
    second = train_delay_model(
        frame,
        dataset_filename="synthetic.csv",
        model_output_dir=tmp_path / "models-b",
        report_output_dir=tmp_path / "reports-b",
        seed=7,
        model="ridge",
    )

    assert first["selected_model"] == second["selected_model"] == "ridge"
    assert first["evaluation_metrics"]["test_metrics"]["ridge"] == second[
        "evaluation_metrics"
    ]["test_metrics"]["ridge"]


def test_training_rejects_insufficient_valid_target_data(tmp_path: Path) -> None:
    frame = _frame(40)
    frame.loc[29:, "scheduled_time"] = np.nan

    with pytest.raises(TrainingDataError, match="At least 30"):
        train_delay_model(
            frame,
            dataset_filename="small.csv",
            model_output_dir=tmp_path / "models",
            report_output_dir=tmp_path / "reports",
        )


def test_regression_metrics_report_r_squared_only_when_meaningful() -> None:
    one = regression_metrics(np.array([1.0]), np.array([2.0]))
    multiple = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    assert one["mae"] == 1
    assert one["r_squared"] is None
    assert multiple["sample_count"] == 2
    assert multiple["rmse"] > 0


def test_prediction_features_persist_schema_and_unknown_categories() -> None:
    schema = {
        "feature_names": ["cpu_request", "creation_time", "role", "app_name"],
        "numeric_features": ["cpu_request", "creation_time"],
        "categorical_features": ["role", "app_name"],
        "essential_prediction_features": ["creation_time", "role"],
    }

    features, warnings = prediction_features(
        {"cpu_request": 0, "creation_time": 10, "role": "new", "app_name": "new-app"},
        schema,
    )

    assert features.loc[0, "cpu_request"] == 0
    assert features.loc[0, "role"] == "new"
    assert warnings == [[]]
