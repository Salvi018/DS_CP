from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.cluster_models import (
    ClusterState,
    ResourceCapacity,
    ResourceDemand,
    SimulatedNode,
    SimulatedWorkload,
)
from src.evaluation import (
    benchmark_metrics,
    fragmentation_metric_rows,
    resource_metric_rows,
    run_experiment,
    run_sensitivity_analysis,
    save_benchmark_outputs,
    select_evaluation_rows,
    workload_outcomes,
)
from src.fragmentation import fragmentation_metrics
from src.schedulers import IFASConfig


def _cluster(
    cpu: float = 10,
    gpu: int = 2,
    memory: float = 20,
) -> ClusterState:
    return ClusterState(
        cluster_name="evaluation fixture",
        nodes={
            "gpu-a": SimulatedNode(
                "gpu-a",
                "HN",
                ResourceCapacity(cpu, gpu, memory, 10, 100),
            ),
            "gpu-b": SimulatedNode(
                "gpu-b",
                "HN",
                ResourceCapacity(cpu, gpu, memory, 10, 100),
            ),
        },
        role_compatibility={"HN": ["HN"]},
        assumptions=["Synthetic test cluster."],
    )


def _workload(
    name: str,
    *,
    cpu: float = 2,
    gpu: int = 1,
    memory: float = 4,
    arrival: float = 0,
    duration: float | None = 2,
    density: int | None = None,
) -> SimulatedWorkload:
    return SimulatedWorkload(
        name,
        ResourceDemand(cpu, gpu, memory, 1, 10),
        role="HN",
        app_name="test-app",
        max_instance_per_node=density,
        arrival_time=arrival,
        duration=duration,
    )


def test_run_experiment_uses_fresh_cluster_and_identical_workload_sequence() -> None:
    cluster = _cluster()
    workloads = [_workload("a"), _workload("b", arrival=1)]

    results = run_experiment(cluster, workloads)

    assert tuple(results) == ("first_fit", "best_fit", "ifas")
    assert all(result.summary["accepted_workload_count"] == 2 for result in results.values())
    assert all(result.summary["resource_capacity_violation_count"] == 0 for result in results.values())
    assert all(
        result.summary["allocated_workload_count"] == 2
        for result in results.values()
    )
    assert all(not node.allocations for node in cluster.nodes.values())


def test_run_experiment_rejects_unknown_duplicate_and_empty_policy_lists() -> None:
    with pytest.raises(ValueError, match="At least one"):
        run_experiment(_cluster(), [_workload("one")], policies=[])
    with pytest.raises(ValueError, match="must not be repeated"):
        run_experiment(
            _cluster(), [_workload("one")], policies=["first_fit", "first_fit"]
        )
    with pytest.raises(ValueError, match="Unsupported"):
        run_experiment(_cluster(), [_workload("one")], policies=["random"])


def test_deterministic_runs_have_identical_outcomes_and_reports(tmp_path: Path) -> None:
    workloads = [_workload("a"), _workload("b", arrival=1)]
    first = run_experiment(_cluster(), workloads)
    second = run_experiment(_cluster(), workloads)

    for policy in first:
        assert workload_outcomes(first[policy], workloads) == workload_outcomes(
            second[policy], workloads
        )
        assert first[policy].summary["placement_attempt_count"] == second[policy].summary[
            "placement_attempt_count"
        ]

    sensitivity = run_sensitivity_analysis(
        _cluster(),
        workloads,
        policies=["first_fit"],
        ifas_config=IFASConfig(),
    )
    files = save_benchmark_outputs(
        first,
        _cluster(),
        workloads,
        tmp_path / "run",
        input_filename="synthetic.csv",
        source_row_count=2,
        selected_row_count=2,
        seed=42,
        time_unit="test_units",
        ifas_config=IFASConfig(),
        repeats_requested=4,
        sensitivity_results=sensitivity,
    )
    assert (tmp_path / "run" / "experiment_config.json").is_file()
    config = json.loads((tmp_path / "run" / "experiment_config.json").read_text())
    assert config["repeats_requested"] == 4
    assert config["repeats_executed"] == 1
    assert "deterministic" in config["repeat_note"]
    assert {"scheduler_comparison.csv", "workload_outcomes.csv", "fragmentation_metrics.csv"} <= {
        path.name for path in files
    }
    assert (tmp_path / "run" / "charts" / "workload_outcomes.html").is_file()
    assert json.loads((tmp_path / "run" / "experiment_log.json").read_text())
    saved_sensitivity = json.loads(
        (tmp_path / "run" / "sensitivity" / "sensitivity_configurations.json").read_text()
    )
    assert saved_sensitivity


def test_queue_delay_uses_simulated_arrival_and_successful_allocation() -> None:
    cluster = _cluster(cpu=6, gpu=0)
    cluster.nodes.pop("gpu-b")
    workloads = [
        _workload("first", cpu=6, gpu=0, arrival=0, duration=2),
        _workload("waiter", cpu=6, gpu=0, arrival=0, duration=1),
    ]
    result = run_experiment(cluster, workloads, policies=["first_fit"])["first_fit"]
    metrics = benchmark_metrics(result, workloads)

    assert metrics["queue_delay_observation_count"] == 2
    assert metrics["queue_delay_mean"] == pytest.approx(1.0)
    assert metrics["queue_delay_median"] == pytest.approx(1.0)
    assert metrics["queue_delay_maximum"] == pytest.approx(2.0)
    assert result.summary["queue_retry_count"] == 1


def test_pending_and_rejected_are_unique_workloads_not_retry_attempts() -> None:
    cluster = _cluster(cpu=1, gpu=0, memory=1)
    workloads = [
        _workload("blocked", cpu=2, gpu=0, memory=1, duration=None),
        _workload("feasible", cpu=1, gpu=0, memory=1, duration=None),
    ]
    rejected = [{"workload_id": "bad", "reasons": ["invalid_arrival_time"]}]
    result = run_experiment(
        cluster,
        workloads,
        policies=["first_fit"],
        rejected_inputs=rejected,
    )["first_fit"]
    metrics = benchmark_metrics(result, workloads)

    assert metrics["unique_unplaced_workloads"] == 1
    assert metrics["unique_rejected_workloads"] == 1
    assert metrics["failed_placement_attempts"] == 1
    assert metrics["pending_queue_count"] == 1
    assert metrics["queue_delay_observation_count"] == 1
    assert result.summary["simulated_duration"] == 0
    assert result.summary["final_active_workload_count"] == 1


def test_resource_metrics_report_zero_capacity_as_not_applicable() -> None:
    cluster = _cluster(gpu=0)
    results = run_experiment(
        cluster, [_workload("cpu", gpu=0)], policies=["first_fit"]
    )
    rows = resource_metric_rows(results)
    gpu_row = next(row for row in rows if row["resource"] == "gpu_devices")

    assert gpu_row["configured_capacity"] == 0
    assert gpu_row["peak_allocation_ratio"] is None
    assert gpu_row["time_weighted_allocation_ratio"] is None
    assert gpu_row["metric_status"] == "not_applicable_zero_capacity"


def test_fragmentation_separates_single_node_blocking_from_aggregate_insufficiency() -> None:
    cluster = _cluster(cpu=10, gpu=2, memory=20)
    blocked = _workload("blocked", cpu=2, gpu=3, memory=4)
    insufficient = _workload("insufficient", cpu=30, gpu=5, memory=60)

    metrics = fragmentation_metrics(cluster, [blocked, insufficient])

    assert metrics["gpu_workloads_with_compatible_nodes"] == 2
    assert metrics["gpu_workloads_without_single_node_fit"] == 2
    assert metrics["gpu_workloads_fragmentation_blocked"] == 1
    assert metrics["gpu_workloads_aggregate_capacity_insufficient"] == 1
    assert metrics["gpu_residual_fit_fragmentation_fraction"] == 1.0
    assert metrics["gpu_fragmentation_blocked_fraction"] == 0.5
    assert metrics["aggregate_capacity_sufficient_but_no_single_node_count"] == 1


def test_fragmentation_excludes_role_incompatible_jobs_from_denominator() -> None:
    cluster = _cluster()
    cluster.role_compatibility = {"OTHER": ["OTHER"]}
    incompatible = SimulatedWorkload(
        "incompatible",
        ResourceDemand(1, 1, 1, 1, 1),
        role="HN",
        arrival_time=0,
    )

    metrics = fragmentation_metrics(cluster, [incompatible])

    assert metrics["gpu_workloads_with_compatible_nodes"] == 0
    assert metrics["gpu_residual_fit_fragmentation_fraction"] is None
    assert metrics["role_compatible_pending_workload_count"] == 0


def test_time_weighted_fragmentation_and_resource_metrics_are_available() -> None:
    workloads = [
        _workload("first", arrival=0, duration=2),
        _workload("second", arrival=1, duration=2),
    ]
    result = run_experiment(
        _cluster(), workloads, policies=["first_fit"]
    )["first_fit"]
    metric = benchmark_metrics(result, workloads)
    rows = fragmentation_metric_rows({"first_fit": result})

    assert "gpu_residual_fit_fragmentation_fraction" in metric[
        "time_weighted_fragmentation_metrics"
    ]
    assert any(row["snapshot"] == "time_weighted" for row in rows)
    assert metric["scheduler_decision_count"] == result.summary["placement_attempt_count"]
    assert metric["feasible_candidates_considered"] >= metric["scheduler_decision_count"]


def test_sensitivity_scenarios_record_single_factor_configurations() -> None:
    cluster = _cluster()
    workloads = [_workload("one", density=1)]

    scenarios = run_sensitivity_analysis(
        cluster,
        workloads,
        policies=["first_fit", "ifas"],
        ifas_config=IFASConfig(),
    )

    names = {scenario["scenario"] for scenario in scenarios}
    assert "gpu_devices_capacity_0.75x" in names
    assert "gpu_node_count_plus_1" in names
    assert "ifas_balance_weight_plus_0.1" in names
    assert "positive_density_caps_enforced_assumption" in names
    assert all(scenario["configuration"]["workload_ids"] == ["one"] for scenario in scenarios)
    assert all(scenario["resource_capacity_violations"] == 0 for scenario in scenarios)
    density_scenario = next(
        scenario
        for scenario in scenarios
        if scenario["scenario"] == "positive_density_caps_enforced_assumption"
    )
    assert (
        density_scenario["configuration"]["cluster_configuration"][
            "density_limit_semantics_verified"
        ]
        is True
    )


def test_sensitivity_reverses_only_equal_arrival_source_order() -> None:
    workloads = [_workload("first"), _workload("second")]

    scenarios = run_sensitivity_analysis(
        _cluster(), workloads, policies=["first_fit"]
    )

    tied_order = next(
        scenario
        for scenario in scenarios
        if scenario["scenario"] == "arrival_tie_order_reversed"
    )
    assert tied_order["configuration"]["workload_ids"] == ["second", "first"]


def test_evaluation_row_selection_is_seeded_time_ordered_and_auditable() -> None:
    frame = pd.DataFrame(
        {
            "instance_sn": ["late", "early", "missing", "negative"],
            "relative_arrival_time_raw_units": [3.0, 1.0, float("nan"), -1.0],
        }
    )

    selected, exclusions = select_evaluation_rows(frame, max_workloads=1, seed=7)
    sample_a, _ = select_evaluation_rows(frame, sample_size=1, seed=7)
    sample_b, _ = select_evaluation_rows(frame, sample_size=1, seed=7)

    assert selected["instance_sn"].tolist() == ["early"]
    assert selected["__benchmark_source_row"].tolist() == [1]
    assert sample_a["instance_sn"].tolist() == sample_b["instance_sn"].tolist()
    assert {item["reason"] for item in exclusions} == {
        "missing_or_invalid_arrival_time",
        "negative_arrival_time",
        "beyond_max_workloads_limit",
    }


def test_experiment_waiting_for_unplaceable_workload_stops_after_events_drain() -> None:
    workload = _workload("too-large", cpu=100, duration=None)

    result = run_experiment(
        _cluster(), [workload], policies=["best_fit"]
    )["best_fit"]

    assert result.summary["pending_workload_count"] == 1
    assert result.summary["placement_attempt_count"] == 1
    assert result.summary["placement_attempt_failure_count"] == 1
    assert result.summary["simulated_duration"] == 0
