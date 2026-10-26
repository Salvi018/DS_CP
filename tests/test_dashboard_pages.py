from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize(
    "section",
    (
        "Overview",
        "Dataset Inspection",
        "Workload Analytics",
        "Data Quality & Reports",
        "Data Preparation",
        "Cluster Simulator",
        "Scheduler Comparison",
        "AI Predictions",
        "Algorithm Comparison",
    ),
)
def test_dashboard_sections_render_without_implicit_jobs(section: str) -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value(section)
    app.run()

    assert not app.exception
    assert app.session_state.get("gpulytics_simulation") is None
    assert app.session_state.get("gpulytics_simulation_comparison") is None
    assert app.session_state.get("gpulytics_ml_training_result") is None
    assert app.session_state.get("gpulytics_benchmark") is None
    assert app.session_state.get("gpulytics_algorithm_comparison") is None


def test_overview_uses_actual_prepared_dataset_and_shows_artifact_status() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(metric.label == "Cleaned records" for metric in app.metric)
    assert any(metric.label == "Delay model artifact" for metric in app.metric)
    assert any(table.value is not None for table in app.dataframe)
    assert any(
        "Creation-time trend" in heading.value for heading in app.subheader
    )
    assert any(
        "Resource-request breakdown" in heading.value
        for heading in app.subheader
    )


def test_report_selector_offers_supported_artifact_download() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Data Quality & Reports")
    app.run()

    assert not app.exception
    assert any(
        button.label == "Download selected artifact"
        for button in app.get("download_button")
    )


def test_scheduler_comparison_page_uses_explicit_run_action() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Scheduler Comparison")
    app.run()

    assert not app.exception
    assert any("Scheduler Comparison" in header.value for header in app.header)
    assert any("Run scheduler benchmark" in button.label for button in app.button)
    assert app.session_state.get("gpulytics_benchmark") is None
    assert any(
        {"policy", "resource_capacity_violations"}.issubset(table.value.columns)
        for table in app.dataframe
        if table.value is not None
    )


def test_dataset_inspection_shows_sample_and_summaries() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Dataset Inspection")
    app.run()

    assert not app.exception
    assert any("Detected columns and types" in heading.value for heading in app.subheader)
    assert any("Dataset sample preview" in heading.value for heading in app.subheader)
    assert any("Numerical column statistics" in heading.value for heading in app.subheader)
    assert any("Categorical column summary" in heading.value for heading in app.subheader)


def test_algorithm_comparison_page_renders_and_has_action() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Algorithm Comparison")
    app.run()

    assert not app.exception
    assert any("Algorithm Comparison" in header.value for header in app.header)
    assert any("Run algorithm comparison" in button.label for button in app.button)
    assert app.session_state.get("gpulytics_algorithm_comparison") is None
    assert any(
        {"policy", "resource_capacity_violations"}.issubset(table.value.columns)
        for table in app.dataframe
        if table.value is not None
    )


def test_algorithm_comparison_explicit_run() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Algorithm Comparison")
    app.run()

    assert not app.exception
    for num_inp in app.number_input:
        if "Maximum earliest-arriving workloads" in num_inp.label:
            num_inp.set_value(30)
            break
    run_btn = [b for b in app.button if "Run algorithm comparison" in b.label][0]
    run_btn.click()
    app.run()

    assert not app.exception
    saved = app.session_state.get("gpulytics_algorithm_comparison")
    assert isinstance(saved, dict)
    metrics = saved.get("metrics", {})
    assert "first_fit" in metrics and "best_fit" in metrics and "ifas" in metrics
    for pol, res in metrics.items():
        assert res.get("resource_capacity_violations") == 0


def test_workload_analytics_handles_empty_filtered_results() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Workload Analytics")
    app.run()

    app.session_state["gpulytics_analytics_filters"] = {
        "categories": {"role": ["NONEXISTENT_ROLE_XYZZY"]},
        "numeric_ranges": {},
    }
    app.run()

    assert not app.exception
    assert any("No workloads match the selected filter criteria" in warn.value for warn in app.warning)


def test_workload_analytics_numeric_filter_selection() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Workload Analytics")
    app.run()

    num_selector = [m for m in app.multiselect if "Numeric columns to constrain" in m.label][0]
    num_selector.select("cpu_request")
    app.run()

    assert not app.exception
    assert any("Cpu Request range" in s.label for s in app.slider)


def test_data_preparation_renders_and_offers_downloads() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.sidebar.radio[0].set_value("Data Preparation")
    app.run()

    assert not app.exception
    assert any("Data Preparation" in header.value for header in app.header)
    assert any("Prepare dataset" in button.label for button in app.button)
    assert any("Download" in button.label for button in app.get("download_button"))


def test_load_report_artifact_empty_csv(tmp_path: Path) -> None:
    from src.dashboard_data import load_report_artifact

    empty_csv = tmp_path / "empty_report.csv"
    empty_csv.write_text("", encoding="utf-8")
    artifact = load_report_artifact(empty_csv)
    assert artifact.kind == "csv"
    assert artifact.content.empty
