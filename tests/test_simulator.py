from __future__ import annotations

import json
import heapq
from pathlib import Path

import pandas as pd
import pytest

from src.cluster_config import ClusterConfigError, load_cluster_config
from src.cluster_models import (
    ClusterState,
    ResourceCapacity,
    ResourceDemand,
    SimulatedNode,
    SimulatedWorkload,
)
from src.resource_accounting import (
    ResourceAccountingError,
    allocate,
    can_allocate,
    check_invariants,
    release,
)
from src.simulation_reports import save_simulation_report
from src.simulator import (
    PolicyDecision,
    UniqueFeasibleNodePolicy,
    adapt_workload_frame,
    run_simulation,
    select_earliest_arrival_rows,
)


def _cluster(
    *,
    cpu_capacity: float = 10,
    gpu_capacity: int = 2,
    role_compatibility: dict[str, list[str]] | None = None,
    node_count: int = 1,
) -> ClusterState:
    nodes = {}
    for index in range(node_count):
        node_id = f"node-{index}"
        nodes[node_id] = SimulatedNode(
            node_id=node_id,
            role="HN",
            capacity=ResourceCapacity(cpu_capacity, gpu_capacity, 100, 100, 100),
        )
    return ClusterState(
        cluster_name="test cluster",
        nodes=nodes,
        role_compatibility=role_compatibility or {"HN": ["HN"]},
    )


def _workload(
    workload_id: str = "w1",
    *,
    cpu: float = 2,
    gpu: int = 1,
    arrival: float | None = 0,
    duration: float | None = 1,
    role: str = "HN",
    app: str = "app",
    density: int | None = None,
) -> SimulatedWorkload:
    return SimulatedWorkload(
        workload_id=workload_id,
        demand=ResourceDemand(cpu, gpu, 10, 2, 5),
        role=role,
        app_name=app,
        max_instance_per_node=density,
        arrival_time=arrival,
        duration=duration,
    )


def _frame(**overrides: list[object]) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "instance_sn": ["w1", "w2"],
        "role": ["HN", "HN"],
        "app_name": ["app", "app"],
        "cpu_request": [2.0, 4.0],
        "gpu_request": [1.0, 0.0],
        "memory_request": [10.0, 20.0],
        "rdma_request": [2.0, 1.0],
        "disk_request": [5.0, 10.0],
        "max_instance_per_node": [4, 4],
        "relative_arrival_time_raw_units": [0.0, 3.0],
        "observed_lifetime_raw_units": [2.0, None],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_config_loads_explicit_hypothetical_capacities() -> None:
    cluster = load_cluster_config()

    assert len(cluster.nodes) == 3
    assert cluster.nodes["gpu-node-01"].capacity.gpu_devices == 4
    assert cluster.density_unlimited_values == []
    assert any("hypothetical" in item for item in cluster.assumptions)


def test_config_rejects_invalid_capacity(tmp_path: Path) -> None:
    path = tmp_path / "cluster.json"
    path.write_text(
        json.dumps(
            {
                "cluster_name": "bad",
                "nodes": [
                    {
                        "node_id": "n",
                        "role": "HN",
                        "capacity": {
                            "cpu_units": -1,
                            "gpu_devices": 0,
                            "memory_units": 1,
                            "rdma_units": 1,
                            "disk_units": 1,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ClusterConfigError, match="non-negative"):
        load_cluster_config(path)


def test_atomic_allocation_and_release_preserve_invariants() -> None:
    cluster = _cluster(cpu_capacity=4, gpu_capacity=1)
    workload = _workload(cpu=3, gpu=1)

    assert can_allocate(cluster, "node-0", workload).feasible
    allocate(cluster, "node-0", workload, 0)
    assert cluster.nodes["node-0"].allocated.cpu_units == 3
    check_invariants(cluster)
    assert release(cluster, "w1") == "node-0"
    assert cluster.nodes["node-0"].allocated.gpu_devices == 0


def test_failed_allocation_does_not_partially_mutate_cluster() -> None:
    cluster = _cluster(cpu_capacity=2, gpu_capacity=0)
    workload = _workload(cpu=3, gpu=1)
    before = cluster.nodes["node-0"].allocated.as_dict()

    assert not can_allocate(cluster, "node-0", workload).feasible
    with pytest.raises(ResourceAccountingError, match="insufficient_capacity"):
        allocate(cluster, "node-0", workload, 0)

    assert cluster.nodes["node-0"].allocated.as_dict() == before
    assert cluster.nodes["node-0"].allocations == {}


def test_role_compatibility_is_configured_and_enforced() -> None:
    cluster = _cluster(role_compatibility={"CN": ["CN"]})

    check = can_allocate(cluster, "node-0", _workload(role="HN"))

    assert not check.feasible
    assert check.reason == "role_incompatible"


def test_density_limit_and_explicit_sentinel_behavior() -> None:
    cluster = _cluster()
    cluster.density_limit_semantics_verified = True
    first = _workload("first", density=1)
    second = _workload("second", density=1)
    allocate(cluster, "node-0", first, 0)
    check = can_allocate(cluster, "node-0", second)
    assert not check.feasible
    assert check.reason == "application_density_limit"

    sentinel_cluster = _cluster()
    negative_limit = _workload("negative", density=-1)
    assert can_allocate(sentinel_cluster, "node-0", negative_limit).reason == "invalid_density_limit"
    sentinel_cluster.density_unlimited_values = [-1]
    assert can_allocate(sentinel_cluster, "node-0", negative_limit).feasible


def test_zero_gpu_workload_is_valid_on_zero_gpu_node() -> None:
    cluster = _cluster(gpu_capacity=0)
    check = can_allocate(cluster, "node-0", _workload(gpu=0))

    assert check.feasible


def test_simulation_sorts_events_and_completes_before_equal_time_arrival() -> None:
    cluster = _cluster(cpu_capacity=4, gpu_capacity=2)
    workloads = [
        _workload("first", cpu=4, gpu=1, arrival=0, duration=2),
        _workload("second", cpu=2, gpu=1, arrival=2, duration=1),
    ]

    result = run_simulation(cluster, workloads)
    at_time_two = [event for event in result.events if event["time"] == 2]

    assert [event["event"] for event in at_time_two] == ["completed", "allocated"]
    assert result.summary["completed_workload_count"] == 2
    assert result.summary["pending_workload_count"] == 0
    assert result.summary["time_weighted_allocation_ratios"]["cpu_units"] == pytest.approx(5 / 6)


def test_pending_workload_is_retried_after_release() -> None:
    cluster = _cluster(cpu_capacity=4, gpu_capacity=2)
    workloads = [
        _workload("holder", cpu=4, gpu=1, arrival=0, duration=2),
        _workload("waiting", cpu=2, gpu=1, arrival=1, duration=1),
    ]

    result = run_simulation(cluster, workloads)

    assert result.summary["allocated_workload_count"] == 2
    assert result.summary["completed_workload_count"] == 2
    assert result.summary["queued_workload_count"] == 1
    assert result.summary["pending_workload_count"] == 0
    assert sum(event["event"] == "queued" for event in result.events) == 1
    assert result.summary["queue_retry_count"] == 1


def test_multiple_feasible_nodes_remain_pending_without_a_scheduler() -> None:
    cluster = _cluster(node_count=2)

    result = run_simulation(cluster, [_workload(duration=None)])

    assert result.summary["allocated_workload_count"] == 0
    assert result.summary["pending_workload_count"] == 1
    assert result.events[0]["reason"] == "policy_ambiguous_multiple_feasible_nodes"


def test_custom_policy_can_select_from_feasible_nodes_without_builtin_algorithm() -> None:
    class ExplicitChoicePolicy:
        name = "test_explicit_choice"

        def choose_node(
            self,
            workload: SimulatedWorkload,
            feasible_node_ids: list[str],
            cluster: ClusterState,
        ) -> PolicyDecision:
            if "node-1" in feasible_node_ids:
                return PolicyDecision("node-1", "test_policy_choice")
            return PolicyDecision(None, "test_policy_no_match")

    cluster = _cluster(node_count=2)
    result = run_simulation(
        cluster,
        [_workload()],
        policy=ExplicitChoicePolicy(),
    )

    assert result.policy_name == "test_explicit_choice"
    assert result.events[0]["node_id"] == "node-1"


def test_missing_lifetime_is_censored_and_zero_capacity_ratio_is_not_applicable() -> None:
    cluster = _cluster(cpu_capacity=10, gpu_capacity=0)
    result = run_simulation(cluster, [_workload(gpu=0, duration=None)])

    assert result.summary["final_active_workload_count"] == 1
    assert result.summary["time_weighted_allocation_ratios"]["gpu_devices"] is None


def test_adapter_reports_missing_or_invalid_inputs_without_fabrication() -> None:
    frame = _frame(
        instance_sn=["missing-time", "bad-resource"],
        relative_arrival_time_raw_units=[None, 1.0],
        cpu_request=[1.0, "bad"],
        observed_lifetime_raw_units=[None, 2.0],
    )
    adapted = adapt_workload_frame(frame, _cluster())

    assert adapted.workloads == []
    assert "missing_or_invalid_arrival_time" in adapted.rejected[0]["reasons"]
    assert "missing_or_invalid_resource:cpu_request" in adapted.rejected[1]["reasons"]


def test_missing_resource_columns_are_rejected_as_unavailable() -> None:
    frame = _frame().drop(columns=["memory_request", "disk_request"])

    adapted = adapt_workload_frame(frame, _cluster())

    assert adapted.workloads == []
    assert "missing_resource_column:memory_request" in adapted.rejected[0]["reasons"]
    assert "missing_resource_column:disk_request" in adapted.rejected[0]["reasons"]


def test_adapter_rejects_negative_timing_and_unverified_density_sentinel() -> None:
    frame = _frame(
        instance_sn=["negative-life", "negative-density"],
        observed_lifetime_raw_units=[-2.0, 2.0],
        max_instance_per_node=[4, -1],
    )
    adapted = adapt_workload_frame(frame, _cluster())

    assert not adapted.workloads
    assert "negative_observed_lifetime" in adapted.rejected[0]["reasons"]
    assert "unverified_negative_density_limit" in adapted.rejected[1]["reasons"]


def test_adapter_accepts_explicitly_configured_density_sentinel() -> None:
    cluster = _cluster()
    cluster.density_unlimited_values = [-1]
    frame = _frame(max_instance_per_node=[-1, -1])

    adapted = adapt_workload_frame(frame, cluster)

    assert len(adapted.workloads) == 2
    assert adapted.workloads[0].max_instance_per_node == -1


def test_missing_timestamp_is_not_replaced_and_creation_time_is_normalized() -> None:
    frame = _frame().drop(columns=["relative_arrival_time_raw_units"])
    frame["creation_time"] = [100.0, None]
    adapted = adapt_workload_frame(frame, _cluster())

    assert len(adapted.workloads) == 1
    assert adapted.workloads[0].arrival_time == 0
    assert "missing_or_invalid_arrival_time" in adapted.rejected[0]["reasons"]
    assert "normalized" in str(adapted.arrival_source)


def test_empty_dataset_has_empty_deterministic_result() -> None:
    cluster = _cluster()
    adapted = adapt_workload_frame(_frame().iloc[0:0], cluster)
    result = run_simulation(cluster, adapted.workloads, rejected_inputs=adapted.rejected)

    assert result.summary["source_workload_count"] == 0
    assert result.summary["allocated_workload_count"] == 0
    assert result.events == []


def test_earliest_arrival_selection_is_stable_and_limited() -> None:
    frame = pd.DataFrame(
        {
            "instance_sn": ["late", "first", "middle"],
            "relative_arrival_time_raw_units": [5, 0, 3],
        }
    )

    selected = select_earliest_arrival_rows(frame, 2)

    assert selected["instance_sn"].tolist() == ["first", "middle"]


def test_simulation_report_files_capture_assumptions_and_metrics(tmp_path: Path) -> None:
    cluster = _cluster()
    result = run_simulation(cluster, [_workload()])

    paths = save_simulation_report(
        result, cluster, tmp_path, input_filename="fixture.csv", seed=42, source_row_count=1
    )

    assert len(paths) == 5
    summary = json.loads((tmp_path / "simulation_summary.json").read_text())
    assert summary["input_filename"] == "fixture.csv"
    assert summary["policy"] == UniqueFeasibleNodePolicy.name
    report = (tmp_path / "simulation_report.md").read_text()
    assert "not measured hardware utilization" in report
    assert (tmp_path / "simulation_events.csv").is_file()
    assert (tmp_path / "resource_history.csv").is_file()


def test_repeated_simulations_are_deterministic() -> None:
    first = run_simulation(_cluster(), [_workload("a"), _workload("b", arrival=1)])
    second = run_simulation(_cluster(), [_workload("a"), _workload("b", arrival=1)])

    assert first.events == second.events
    first.summary.pop("scheduler_runtime_seconds")
    second.summary.pop("scheduler_runtime_seconds")
    assert first.summary == second.summary


def test_duplicate_completion_event_releases_allocation_only_once(monkeypatch) -> None:
    import src.simulator as simulator_module

    original_push = heapq.heappush

    def push_duplicate_completion(heap, item) -> None:
        original_push(heap, item)
        if item[3] == "completion":
            original_push(heap, (*item[:2], item[2] + 1000, *item[3:]))

    monkeypatch.setattr(simulator_module.heapq, "heappush", push_duplicate_completion)
    result = run_simulation(_cluster(), [_workload("once")])

    assert result.summary["completed_workload_count"] == 1
    assert sum(event["event"] == "completed" for event in result.events) == 1
