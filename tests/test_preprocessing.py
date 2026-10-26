from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import (
    PreprocessingError,
    clean_workloads,
    normalize_column_names,
    prepare_dataset,
)


def workload_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_sn": ["i-1", "i-2", "i-3"],
            "role": ["HN", "CN", "HN"],
            "app_name": ["app-a", "app-a", "app-b"],
            "cpu_request": [2, 1, 4],
            "cpu_limit": [4, 2, 4],
            "gpu_request": [1, 0, 0],
            "gpu_limit": [1, 0, 0],
            "memory_request": [8.0, 4.0, 10.0],
            "memory_limit": [16.0, 8.0, 20.0],
            "creation_time": [10.0, 11.0, None],
            "scheduled_time": [12.0, 14.0, None],
            "deletion_time": [20.0, None, None],
        }
    )


def test_valid_input_features_and_prediction_target_separation() -> None:
    result = clean_workloads(workload_frame())

    assert len(result.workloads) == 3
    assert result.workloads["has_gpu_request"].tolist() == [True, False, False]
    assert result.workloads["scheduling_delay_raw_units"].tolist()[:2] == [2.0, 3.0]
    assert result.workloads["observed_lifetime_raw_units"].iloc[0] == 8.0
    assert "scheduling_delay_sec" not in result.workloads
    assert "gpu_request" in result.report["feature_metadata"]["creation_time_input_features"]
    assert "scheduled_time" in result.report["feature_metadata"]["excluded_from_creation_time_inputs"]
    assert "deletion_time" in result.report["feature_metadata"]["excluded_from_creation_time_inputs"]
    assert result.report["timestamp_unit_assumption"].startswith("Unverified")


def test_missing_optional_columns_are_supported() -> None:
    result = clean_workloads(pd.DataFrame({"instance_sn": ["i-1"], "cpu_request": [3]}))

    assert result.workloads.loc[0, "cpu_request"] == 3
    assert "gpu_request" not in result.workloads
    assert "missing_optional_column" in set(result.validation_issues["category"])


def test_missing_required_identifier_is_a_critical_schema_error() -> None:
    result = clean_workloads(pd.DataFrame({"cpu_request": [1]}))

    assert result.report["error_count"] >= 1
    assert "missing_required_identifier_column" in set(result.validation_issues["category"])
    with pytest.raises(PreprocessingError, match="Strict validation failed"):
        prepare_dataset_from_frame(pd.DataFrame({"cpu_request": [1]}), strict=True)


def test_rows_with_missing_identifiers_are_traceably_rejected(tmp_path: Path) -> None:
    frame = workload_frame()
    frame.loc[1, "instance_sn"] = None
    result = clean_workloads(frame)

    assert len(result.workloads) == 2
    assert result.rejected_rows["source_record_number"].tolist() == [2]
    assert result.rejected_rows["exclusion_reason"].tolist() == [
        "missing_or_blank_instance_sn"
    ]
    assert "invalid_essential_identifier" in set(result.validation_issues["category"])

    raw = tmp_path / "missing_id.csv"
    frame.to_csv(raw, index=False)
    output = tmp_path / "cleaned.csv"
    report_dir = tmp_path / "reports"
    prepared = prepare_dataset(raw, output, report_dir)
    assert prepared.report["outputs"]["rejected_rows_csv"]
    assert (report_dir / "rejected_rows.csv").is_file()


def prepare_dataset_from_frame(frame: pd.DataFrame, strict: bool = False):
    from src.preprocessing import validate_schema_strict

    result = clean_workloads(frame)
    if strict:
        validate_schema_strict(result)
    return result


def test_duplicate_rows_and_identifiers_are_rejected_with_reasons() -> None:
    frame = pd.concat([workload_frame().iloc[[0]], workload_frame().iloc[[0]], workload_frame().iloc[[0]].assign(cpu_request=3)])

    result = clean_workloads(frame)

    assert len(result.workloads) == 1
    assert result.report["duplicate_rows"] == 1
    assert result.report["duplicate_instance_identifiers"] == 2
    assert set(result.rejected_rows["exclusion_reason"]) == {
        "duplicate_full_row",
        "duplicate_instance_identifier",
    }
    assert result.rejected_rows["source_record_number"].tolist() == [2, 3]


def test_missing_numeric_values_and_invalid_numeric_values_are_explicit() -> None:
    frame = workload_frame().iloc[:2].copy()
    frame["gpu_request"] = frame["gpu_request"].astype(object)
    frame.loc[0, "cpu_request"] = None
    frame.loc[1, "gpu_request"] = "not-a-number"

    result = clean_workloads(frame)

    assert pd.isna(result.workloads.loc[0, "cpu_request"])
    assert result.workloads.loc[0, "has_cpu_request"] is pd.NA or pd.isna(
        result.workloads.loc[0, "has_cpu_request"]
    )
    assert result.workloads.loc[1, "gpu_request_raw"] == "not-a-number"
    assert {"missing_resource_request", "invalid_numeric"}.issubset(
        set(result.validation_issues["category"])
    )


def test_negative_resources_and_request_limit_violations_are_warnings() -> None:
    frame = workload_frame().iloc[:1].copy()
    frame.loc[0, "cpu_request"] = -1
    frame.loc[0, "cpu_limit"] = -2
    frame.loc[0, "memory_request"] = 20
    frame.loc[0, "memory_limit"] = 10

    result = clean_workloads(frame)

    categories = set(result.validation_issues["category"])
    assert "negative_resource_request" in categories
    assert "negative_resource_limit" in categories
    assert "request_exceeds_finite_limit" in categories
    assert len(result.workloads) == 1


def test_missing_timestamps_and_negative_timing_deltas_are_preserved() -> None:
    frame = workload_frame().iloc[[0, 2]].copy().reset_index(drop=True)
    frame.loc[0, ["creation_time", "scheduled_time", "deletion_time"]] = [10, 5, 4]

    result = clean_workloads(frame)
    categories = set(result.validation_issues["category"])

    assert "missing_timestamp" in categories
    assert "negative_scheduling_delay" in categories
    assert "negative_observed_lifetime" in categories
    assert result.workloads.loc[0, "scheduling_delay_raw_units"] == -5
    assert result.workloads.loc[0, "observed_lifetime_raw_units"] == -1
    assert pd.isna(result.workloads.loc[1, "creation_time"])


def test_cpu_only_workloads_and_unrecognized_roles_are_not_dropped() -> None:
    frame = pd.DataFrame(
        {
            "instance_sn": ["cpu-only"],
            "role": ["NEW_ROLE"],
            "cpu_request": [2],
            "gpu_request": [0],
        }
    )

    result = clean_workloads(frame)

    assert len(result.workloads) == 1
    assert bool(result.workloads.loc[0, "has_gpu_request"]) is False
    assert "unknown_role_value" in set(result.validation_issues["category"])


def test_extra_columns_are_preserved_and_classified() -> None:
    frame = workload_frame().assign(custom_metric=[1, 2, 3])

    result = clean_workloads(frame)

    assert "custom_metric" in result.workloads
    assert result.report["field_classification"]["custom_metric"] == "unknown_or_unexpected"
    assert "unexpected_column" in set(result.validation_issues["category"])


def test_ambiguous_normalized_columns_fail_without_guessing() -> None:
    frame = pd.DataFrame({"instance-sn": [1], "instance sn": [2]})

    with pytest.raises(PreprocessingError, match="ambiguous"):
        normalize_column_names(frame)


def test_duplicate_source_header_is_rejected_before_pandas_mangles_names(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "duplicate_header.csv"
    raw.write_text("instance_sn,instance_sn\nfirst,second\n", encoding="utf-8")

    with pytest.raises(PreprocessingError, match="duplicate or ambiguously normalized"):
        prepare_dataset(raw, tmp_path / "cleaned.csv", tmp_path / "reports")


def test_deterministic_output_and_audit_report_creation(tmp_path: Path) -> None:
    raw = tmp_path / "trace.csv"
    workload_frame().to_csv(raw, index=False)
    output = tmp_path / "processed" / "cleaned_workloads.csv"
    reports = tmp_path / "reports" / "cleaning"

    first = prepare_dataset(raw, output, reports)
    first_bytes = output.read_bytes()
    second = prepare_dataset(raw, output, reports)

    assert output.read_bytes() == first_bytes
    assert not second.workloads.columns.duplicated().any()
    report = json.loads((reports / "cleaning_report.json").read_text())
    assert report["output_row_count"] == len(pd.read_csv(output))
    assert (reports / "cleaning_summary.md").is_file()
    assert (reports / "validation_issues.csv").is_file()
    assert first.report["outputs"]["cleaned_dataset"] == str(output.resolve())


def test_raw_data_is_not_modified_and_output_is_compatible_with_eda(tmp_path: Path) -> None:
    raw = tmp_path / "trace.csv"
    workload_frame().to_csv(raw, index=False)
    checksum_before = hashlib.sha256(raw.read_bytes()).hexdigest()
    output = tmp_path / "cleaned.csv"

    prepare_dataset(raw, output, tmp_path / "reports")
    reopened = pd.read_csv(output)

    assert hashlib.sha256(raw.read_bytes()).hexdigest() == checksum_before
    assert len(reopened) == 3
    assert "cpu_request" in reopened
    assert "gpu_request" in reopened
    assert "scheduling_delay_raw_units" in reopened


def test_strict_failure_preserves_existing_processed_output(tmp_path: Path) -> None:
    raw = tmp_path / "trace.csv"
    pd.DataFrame({"instance_sn": ["i-1", "i-1"], "cpu_request": [1, 2]}).to_csv(
        raw, index=False
    )
    output = tmp_path / "cleaned.csv"
    output.write_text("previous valid output\n", encoding="utf-8")

    with pytest.raises(PreprocessingError, match="Strict validation failed"):
        prepare_dataset(raw, output, tmp_path / "reports", strict=True)

    assert output.read_text(encoding="utf-8") == "previous valid output\n"


def test_strict_invalid_numeric_failure_preserves_existing_output(tmp_path: Path) -> None:
    raw = tmp_path / "trace.csv"
    pd.DataFrame(
        {"instance_sn": ["i-1"], "cpu_request": ["not-numeric"]}
    ).to_csv(raw, index=False)
    output = tmp_path / "cleaned.csv"
    output.write_text("previous valid output\n", encoding="utf-8")

    with pytest.raises(PreprocessingError, match="Strict validation failed"):
        prepare_dataset(raw, output, tmp_path / "reports", strict=True)

    assert output.read_text(encoding="utf-8") == "previous valid output\n"
