from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics import (
    AnalyticsError,
    analyze_workloads,
    classify_workload_size,
    generate_analytics,
    load_cleaned_dataset,
    numeric_summary,
    plot_resource_distributions,
)


@pytest.fixture
def workloads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_sn": ["i1", "i2", "i3", "i4"],
            "role": ["HN", "CN", "HN", "CN"],
            "app_name": ["app-a", "app-a", "app-b", "app-c"],
            "cpu_request": [2.0, 16.0, None, 48.0],
            "cpu_limit": [4.0, 16.0, 8.0, 64.0],
            "gpu_request": [1.0, 0.0, 0.0, 2.0],
            "gpu_limit": [1.0, 0.0, 0.0, 2.0],
            "memory_request": [8.0, 128.0, 32.0, 512.0],
            "memory_limit": [16.0, 256.0, 64.0, 512.0],
            "disk_request": [10.0, 20.0, 30.0, 40.0],
            "rdma_request": [0.0, 1.0, 0.0, 2.0],
            "scheduling_delay_raw_units": [2.0, 4.0, -1.0, None],
            "observed_lifetime_raw_units": [10.0, 20.0, 30.0, -5.0],
            "validation_issue_count": [0, 0, 1, 1],
            "data_quality_flags": ["", "", "missing_timestamp", "negative_delay"],
        }
    )


def test_empty_dataset_without_columns_is_rejected() -> None:
    with pytest.raises(AnalyticsError, match="no columns"):
        analyze_workloads(pd.DataFrame())


def test_empty_dataset_with_schema_has_no_fabricated_statistics() -> None:
    frame = pd.DataFrame(
        {"instance_sn": pd.Series(dtype=str), "cpu_request": pd.Series(dtype=float)}
    )

    result = analyze_workloads(frame)

    assert result.overview["row_count"] == 0
    cpu_summary = result.resource_summary.query("column == 'cpu_request'").iloc[0]
    assert cpu_summary["count"] == 0
    assert pd.isna(cpu_summary["mean"])
    assert result.overview["gpu_requesting_workload_count"] is None


def test_missing_optional_columns_are_reported_not_fatal() -> None:
    result = analyze_workloads(pd.DataFrame({"cpu_request": [1, 2]}))

    assert result.overview["row_count"] == 2
    assert result.role_summary.empty
    assert result.application_summary.empty
    assert "role_summary" in result.unavailable_analyses
    assert "scheduling_delay" in result.unavailable_analyses


def test_numeric_summary_reports_missing_and_invalid_observations() -> None:
    frame = pd.DataFrame({"cpu_request": [1.0, None, "invalid", float("inf")]})

    summary = numeric_summary(frame, "cpu_request")

    assert summary is not None
    assert summary["count"] == 1
    assert summary["mean"] == 1.0
    assert summary["missing_count"] == 3
    assert summary["invalid_count"] == 2


def test_cpu_only_workloads_and_gpu_comparison() -> None:
    frame = pd.DataFrame({"gpu_request": [1, 0, 0, None], "cpu_request": [4, 8, 2, 4]})

    result = analyze_workloads(frame)
    comparison = result.gpu_comparison.set_index("workload_group")

    assert result.overview["gpu_requesting_workload_count"] == 1
    assert result.overview["cpu_only_workload_count"] == 2
    assert comparison.loc["cpu_only_zero_gpu_request", "workload_count"] == 2
    assert comparison.loc["gpu_requesting", "cpu_request_mean"] == 4


def test_workload_classification_handles_thresholds_and_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "cpu_request": [4, 16, 48, None, None],
            "gpu_request": [0, 0, 0, 1, None],
            "memory_request": [32, 128, 512, None, None],
        }
    )

    classified = classify_workload_size(frame)

    assert classified["workload_size"].tolist() == [
        "small",
        "medium",
        "large",
        "medium",
        "unclassified",
    ]
    assert classified["workload_size_resource_count"].tolist() == [3, 3, 3, 1, 0]
    assert classified.loc[3, "workload_size_basis"] == "gpu_request"


def test_scheduling_delay_and_lifetime_report_negatives_and_missing_separately(
    workloads: pd.DataFrame,
) -> None:
    result = analyze_workloads(workloads)

    delay = result.scheduling_delay_summary
    lifetime = result.lifetime_summary
    assert delay is not None and lifetime is not None
    assert delay["mean"] == 3
    assert delay["median"] == 3
    assert delay["negative_count"] == 1
    assert delay["missing_count"] == 1
    assert lifetime["mean"] == 20
    assert lifetime["negative_count"] == 1
    assert lifetime["unit"].startswith("unverified")


def test_application_summary_includes_counts_and_resource_aggregates(
    workloads: pd.DataFrame,
) -> None:
    result = analyze_workloads(workloads)

    app_a = result.application_summary.set_index("app_name").loc["app-a"]
    assert app_a["workload_count"] == 2
    assert app_a["gpu_requested_total"] == 1
    assert app_a["cpu_requested_total"] == 18
    assert result.overview["application_group_count"] == 3


def test_correlations_are_calculated_for_available_request_columns(
    workloads: pd.DataFrame,
) -> None:
    result = analyze_workloads(workloads)

    assert "cpu_request" in result.correlation_matrix
    assert result.correlation_matrix.loc["gpu_request", "gpu_request"] == 1.0
    assert result.overview["definitions"]["correlation"].endswith("not causation.")


def test_plot_functions_skip_unavailable_fields() -> None:
    assert plot_resource_distributions(pd.DataFrame({"role": ["HN"]})) is None
    figure = plot_resource_distributions(pd.DataFrame({"gpu_request": [0, 1]}))
    assert figure is not None
    assert len(figure.data) == 1


def test_report_generation_and_repeat_run_statistics_are_deterministic(
    workloads: pd.DataFrame, tmp_path: Path
) -> None:
    first = generate_analytics(workloads, tmp_path / "analytics", dataset_filename="fixture.csv")
    first_numeric = first.resource_summary.to_csv(index=False)
    first_app = first.application_summary.to_csv(index=False)
    second = generate_analytics(workloads, tmp_path / "analytics", dataset_filename="fixture.csv")

    assert second.resource_summary.to_csv(index=False) == first_numeric
    assert second.application_summary.to_csv(index=False) == first_app
    report_dir = tmp_path / "analytics"
    expected = {
        "overview.json",
        "resource_summary.csv",
        "role_summary.csv",
        "application_summary.csv",
        "scheduling_delay_summary.json",
        "lifetime_summary.json",
        "correlation_matrix.csv",
        "analytics_report.md",
        "gpu_workload_comparison.csv",
        "workload_size_summary.csv",
        "missing_summary.csv",
        "data_quality_summary.csv",
    }
    assert expected.issubset({path.name for path in report_dir.iterdir()})
    overview = json.loads((report_dir / "overview.json").read_text())
    assert overview["dataset_filename"] == "fixture.csv"
    assert overview["row_count"] == 4
    assert overview["generated_charts"]
    assert (report_dir / overview["generated_charts"][0]).is_file()


def test_load_missing_cleaned_dataset_has_preprocessing_guidance(tmp_path: Path) -> None:
    with pytest.raises(AnalyticsError, match="scripts/prepare_dataset.py"):
        load_cleaned_dataset(tmp_path / "missing.csv")

