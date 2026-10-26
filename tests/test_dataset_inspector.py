from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.dataset_inspector import (
    DatasetInspectionError,
    inspect_dataset,
    read_csv_safely,
    resolve_dataset_path,
)
from src.eda import run_eda

VALID_CSV = """instance_sn,role,app_name,cpu_request,cpu_limit,gpu_request,gpu_limit,creation_time,scheduled_time,deletion_time
id-1,HN,app-a,2,4,1,1,10,12,20
id-2,worker,app-b,1,2,0,0,11,14,30
"""

ANOMALOUS_CSV = """instance_sn,role,app_name,cpu_request,cpu_limit,gpu_request,gpu_limit,creation_time,scheduled_time,deletion_time,extra_field
id-1,HN,app-a,-2,1,0,0,10,5,3,alpha
id-1,unknown,app-b,invalid,2,1,1,,,,beta
id-2,worker,app-c,5,2,1,1,,,,gamma
"""


def write_csv(tmp_path: Path, content: str, name: str = "trace.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_inspects_valid_csv_and_uses_only_observed_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_CSV)

    inspection = inspect_dataset(path, chunksize=1)

    assert inspection.report["row_count"] == 2
    assert inspection.report["column_count"] == 10
    assert inspection.report["dataset_filename"] == "trace.csv"
    assert inspection.report["duplicate_rows"] == 0
    assert inspection.report["duplicate_instance_identifiers"] == 0
    assert "rdma_request" in inspection.report["missing_expected_columns"]
    assert not inspection.data_quality.query("issue == 'invalid_numeric'").shape[0]
    assert inspection.report["timestamp_analysis"]["unit_assumption"].startswith("unverified")


def test_reports_duplicates_missing_values_negative_times_and_invalid_numbers(
    tmp_path: Path,
) -> None:
    path = write_csv(tmp_path, ANOMALOUS_CSV)

    inspection = inspect_dataset(path, chunksize=1)
    counts = inspection.data_quality.set_index(["issue", "column"])["count"].to_dict()

    assert inspection.report["duplicate_instance_identifiers"] == 1
    assert counts[("invalid_numeric", "cpu_request")] == 1
    assert counts[("negative_request", "cpu_request")] == 1
    assert counts[("request_exceeds_limit", "cpu_request")] == 1
    assert counts[("negative_scheduling_delay", "scheduled_time - creation_time")] == 1
    assert counts[("negative_observed_lifetime", "deletion_time - scheduled_time")] == 1
    assert counts[("missing_timestamp", "creation_time")] == 2
    assert inspection.report["zero_gpu_request_count"] == 1
    assert "extra_field" in inspection.report["extra_columns"]
    assert "role_value_domain_unverified" in set(inspection.data_quality["issue"])
    assert inspection.report["observed_unverified_domains"]["role"] == ["HN", "unknown", "worker"]


def test_duplicate_rows_and_negative_limits_are_reported(tmp_path: Path) -> None:
    repeated_row = "id-1,HN,app-a,2,4,1,-1,10,12,20"
    path = write_csv(
        tmp_path,
        "instance_sn,role,app_name,cpu_request,cpu_limit,gpu_request,gpu_limit,creation_time,scheduled_time,deletion_time\n"
        + repeated_row
        + "\n"
        + repeated_row
        + "\n",
    )

    inspection = inspect_dataset(path)

    assert inspection.report["duplicate_rows"] == 1
    assert inspection.data_quality.query(
        "issue == 'negative_limit' and column == 'gpu_limit'"
    )["count"].iloc[0] == 2


def test_missing_expected_columns_do_not_break_eda(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "instance_sn,app_name,custom\nx,app-a,1\ny,app-a,2\n")

    inspection = run_eda(path, tmp_path / "reports", skip_plots=True)

    assert inspection.report["row_count"] == 2
    assert "gpu_request" in inspection.report["missing_expected_columns"]
    assert (tmp_path / "reports" / "dataset_summary.json").is_file()
    assert "resource_distributions.png" in inspection.report["skipped_charts"]
    assert not (tmp_path / "reports" / "resource_distributions.png").exists()


def test_complete_frequency_tables_and_report_files_are_written(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_CSV)
    output = tmp_path / "eda"

    run_eda(path, output, skip_plots=True)

    expected = {
        "dataset_summary.json",
        "schema_report.csv",
        "missing_values.csv",
        "numerical_summary.csv",
        "categorical_summary.csv",
        "data_quality_report.csv",
        "resource_correlation.csv",
        "eda_report.md",
    }
    assert expected.issubset({file.name for file in output.iterdir()})
    categories = pd.read_csv(output / "categorical_summary.csv")
    app_categories = categories.query("column == 'app_name'")["category"].tolist()
    assert set(app_categories) == {"app-a", "app-b"}
    gpu_request_counts = categories[categories["column"] == "gpu_request"]
    assert gpu_request_counts["count"].sum() == 2
    report_text = (output / "eda_report.md").read_text(encoding="utf-8")
    assert "timestamp unit assumption" in report_text.lower()
    assert "No charts generated" in report_text


def test_applicable_plots_are_generated_from_observed_fields(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_CSV)
    output = tmp_path / "plots"

    inspection = run_eda(path, output, sample_size=20)

    assert "role_distribution.png" in inspection.report["generated_charts"]
    assert "top_applications.png" in inspection.report["generated_charts"]
    assert "resource_request_vs_limit.png" in inspection.report["generated_charts"]
    assert (output / "role_distribution.png").is_file()
    assert "resource_distributions.png" in inspection.report["generated_charts"]


def test_raw_csv_is_never_modified(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_CSV)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    inspect_dataset(path)
    run_eda(path, tmp_path / "reports", skip_plots=True)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_dataset_resolution_prefers_expected_then_single_candidate(tmp_path: Path) -> None:
    expected = tmp_path / "disaggregated_DLRM_trace.csv"
    expected.write_text(VALID_CSV, encoding="utf-8")
    other = tmp_path / "other.csv"
    other.write_text(VALID_CSV, encoding="utf-8")

    assert resolve_dataset_path(raw_data_dir=tmp_path) == expected.resolve()
    expected.unlink()
    assert resolve_dataset_path(raw_data_dir=tmp_path) == other.resolve()


def test_multiple_candidates_require_explicit_selection(tmp_path: Path) -> None:
    (tmp_path / "one.csv").write_text(VALID_CSV, encoding="utf-8")
    (tmp_path / "two.csv").write_text(VALID_CSV, encoding="utf-8")

    with pytest.raises(DatasetInspectionError, match="Multiple CSV files"):
        resolve_dataset_path(raw_data_dir=tmp_path)


def test_missing_and_empty_files_have_actionable_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetInspectionError, match="does not exist"):
        inspect_dataset(tmp_path / "missing.csv")
    with pytest.raises(DatasetInspectionError, match="Download the Alibaba GPU Cluster Dataset"):
        resolve_dataset_path(raw_data_dir=tmp_path)
    empty = write_csv(tmp_path, "", "empty.csv")
    with pytest.raises(DatasetInspectionError, match="empty"):
        read_csv_safely(empty)


def test_header_only_csv_reports_zero_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "instance_sn,role\n")

    inspection = inspect_dataset(path)

    assert inspection.report["row_count"] == 0
    assert inspection.report["column_count"] == 2


def test_missing_resource_values_and_malformed_csv_are_reported(tmp_path: Path) -> None:
    missing_resource = write_csv(
        tmp_path,
        "instance_sn,gpu_request,gpu_limit\nid-1,,1\n",
    )
    inspection = inspect_dataset(missing_resource)
    assert (
        inspection.data_quality.query(
            "issue == 'missing_resource_value' and column == 'gpu_request'"
        )["count"].iloc[0]
        == 1
    )

    malformed = write_csv(tmp_path, "first,second\n1,2,3\n", "malformed.csv")
    with pytest.raises(DatasetInspectionError, match="Could not parse CSV"):
        inspect_dataset(malformed)


def test_invalid_and_non_finite_timestamps_are_counted(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "creation_time,scheduled_time,deletion_time\n"
        "not-a-time,10,20\n"
        "inf,11,inf\n"
        ",12,13\n",
    )

    inspection = inspect_dataset(path)
    counts = inspection.data_quality.set_index(["issue", "column"])["count"].to_dict()

    assert counts[("invalid_timestamp", "creation_time")] == 2
    assert counts[("invalid_timestamp", "deletion_time")] == 1
    assert counts[("missing_timestamp", "creation_time")] == 1


def test_cli_exits_with_download_instructions_when_dataset_is_missing(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "inspect_dataset.py"
    project = script.parents[1]
    isolated = tmp_path / "isolated_project"
    (isolated / "scripts").mkdir(parents=True)
    shutil.copy2(script, isolated / "scripts" / script.name)
    for package in ("config", "src"):
        shutil.copytree(project / package, isolated / package)
    result = subprocess.run(
        [sys.executable, str(isolated / "scripts" / script.name), "--skip-plots"],
        cwd=isolated,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Download the Alibaba GPU Cluster Dataset 2025" in result.stderr
