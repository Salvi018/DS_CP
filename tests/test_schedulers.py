from __future__ import annotations

import json
import subprocess
import sys
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
from src.resource_accounting import allocate, can_allocate, check_invariants
from src.fragmentation import fragmentation_metrics
from src.schedulers import (
    BestFitScheduler,
    FirstFitScheduler,
    IFASConfig,
    IntelligentFragmentationAwareScheduler,
    create_scheduler,
    load_ifas_config,
)
from src.simulation_comparison import (
    compare_schedulers,
    comparison_metrics,
    save_comparison_report,
)
from src.simulation_reports import save_simulation_report
from src.simulator import run_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _workload(
    workload_id: str = "workload",
    *,
    demand: ResourceDemand | None = None,
    role: str = "HN",
    app: str = "app",
    density: int | None = None,
    arrival: float = 0,
    duration: float | None = 2,
) -> SimulatedWorkload:
    return SimulatedWorkload(
        workload_id=workload_id,
        demand=demand or ResourceDemand(4, 1, 8, 2, 10),
        role=role,
        app_name=app,
        max_instance_per_node=density,
        arrival_time=arrival,
        duration=duration,
    )


def _cluster(
    specs: list[tuple[str, str, ResourceCapacity]],
    *,
    compatibility: dict[str, list[str]] | None = None,
    density_verified: bool = False,
) -> ClusterState:
    return ClusterState(
        cluster_name="synthetic test cluster",
        nodes={
            node_id: SimulatedNode(node_id, role, capacity)
            for node_id, role, capacity in specs
        },
        role_compatibility=compatibility or {"HN": ["HN"]},
        density_limit_semantics_verified=density_verified,
    )


def _capacity(
    cpu: float = 20,
    gpu: int = 4,
    memory: float = 40,
    rdma: float = 20,
    disk: float = 100,
) -> ResourceCapacity:
    return ResourceCapacity(cpu, gpu, memory, rdma, disk)


def test_first_fit_selects_first_node_id_with_multiple_feasible_nodes() -> None:
    cluster = _cluster(
        [
            ("z-node", "HN", _capacity()),
            ("a-node", "HN", _capacity()),
        ]
    )
    before = {node_id: node.allocated.as_dict() for node_id, node in cluster.nodes.items()}

    decision = FirstFitScheduler().select_node(_workload(), cluster)

    assert decision.node_id == "a-node"
    assert decision.reason == "first_feasible_node_by_node_id"
    assert {node_id: node.allocated.as_dict() for node_id, node in cluster.nodes.items()} == before


def test_configured_order_tie_breaking_is_deterministic() -> None:
    cluster = _cluster(
        [
            ("z-node", "HN", _capacity()),
            ("a-node", "HN", _capacity()),
        ]
    )
    config = IFASConfig(tie_breaking="configured_order")

    decision = IntelligentFragmentationAwareScheduler(config).choose_node(
        _workload(), ["z-node", "a-node"], cluster
    )

    assert decision.node_id == "z-node"


def test_first_fit_returns_no_node_when_every_node_is_infeasible() -> None:
    cluster = _cluster([("node-a", "CN", _capacity())])

    decision = FirstFitScheduler().select_node(_workload(), cluster)

    assert decision.node_id is None
    assert "role_incompatible" in decision.reason


def test_best_fit_chooses_minimum_normalized_residual_score() -> None:
    cluster = _cluster(
        [
            ("large", "HN", _capacity(cpu=40, gpu=4, memory=80, rdma=40, disk=200)),
            ("tight", "HN", _capacity(cpu=10, gpu=2, memory=20, rdma=10, disk=50)),
        ]
    )
    workload = _workload(
        demand=ResourceDemand(8, 1, 10, 4, 20)
    )
    scheduler = BestFitScheduler()

    assert scheduler.score(workload, cluster, "tight") < scheduler.score(
        workload, cluster, "large"
    )
    assert scheduler.select_node(workload, cluster).node_id == "tight"


def test_best_fit_breaks_equal_scores_by_node_id() -> None:
    cluster = _cluster(
        [
            ("z-node", "HN", _capacity()),
            ("a-node", "HN", _capacity()),
        ]
    )

    assert BestFitScheduler().select_node(_workload(), cluster).node_id == "a-node"


def test_best_fit_handles_zero_capacity_and_unrequested_extra_dimensions() -> None:
    capacity = ResourceCapacity(
        20, 0, 40, 20, 100, extra={"custom_resource": 5}
    )
    cluster = _cluster([("node-a", "HN", capacity)])
    workload = _workload(demand=ResourceDemand(4, 0, 8, 2, 10))

    decision = BestFitScheduler().select_node(workload, cluster)

    assert decision.node_id == "node-a"
    assert BestFitScheduler.score(workload, cluster, "node-a") > 0


@pytest.mark.parametrize(
    ("capacity", "demand", "resource_name"),
    [
        (_capacity(cpu=3), ResourceDemand(4, 1, 8, 2, 10), "cpu_units"),
        (_capacity(gpu=0), ResourceDemand(4, 1, 8, 2, 10), "gpu_devices"),
        (_capacity(memory=4), ResourceDemand(4, 1, 8, 2, 10), "memory_units"),
        (_capacity(rdma=1), ResourceDemand(4, 1, 8, 2, 10), "rdma_units"),
        (_capacity(disk=9), ResourceDemand(4, 1, 8, 2, 10), "disk_units"),
    ],
)
def test_both_schedulers_enforce_each_resource_constraint(
    capacity: ResourceCapacity, demand: ResourceDemand, resource_name: str
) -> None:
    cluster = _cluster([("node-a", "HN", capacity)])
    workload = _workload(demand=demand)

    assert not can_allocate(cluster, "node-a", workload).feasible
    for scheduler in (FirstFitScheduler(), BestFitScheduler()):
        assert scheduler.select_node(workload, cluster).node_id is None
        check_invariants(cluster)
        assert cluster.nodes["node-a"].allocations == {}


def test_density_limit_is_applied_only_when_semantics_are_verified() -> None:
    capacity = _capacity()
    workload_a = _workload("a", density=1)
    workload_b = _workload("b", density=1)
    unverified = _cluster([("node-a", "HN", capacity)])
    allocate(unverified, "node-a", workload_a, 0)
    assert can_allocate(unverified, "node-a", workload_b).feasible

    verified = _cluster([("node-a", "HN", capacity)], density_verified=True)
    allocate(verified, "node-a", workload_a, 0)
    assert can_allocate(verified, "node-a", workload_b).reason == "application_density_limit"


def test_unconfigured_negative_density_value_is_not_unlimited() -> None:
    cluster = _cluster([("node-a", "HN", _capacity())])

    assert can_allocate(cluster, "node-a", _workload(density=-1)).reason == "invalid_density_limit"
    cluster.density_unlimited_values = [-1]
    assert can_allocate(cluster, "node-a", _workload(density=-1)).feasible


def test_invalid_direct_workload_is_rejected_instead_of_queued() -> None:
    cluster = _cluster([("node-a", "HN", _capacity())])
    invalid = _workload(demand=ResourceDemand(-1, 1, 8, 2, 10))

    result = run_simulation(cluster, [invalid], policy=FirstFitScheduler())

    assert result.summary["pending_workload_count"] == 0
    assert result.summary["rejected_workload_count"] == 1
    assert result.events[0]["event"] == "rejected"
    assert "invalid_demand:cpu_units" in result.events[0]["reason"]


def test_simulator_allocates_multiple_feasible_workloads_and_releases_once() -> None:
    cluster = _cluster([("node-a", "HN", _capacity())])
    result = run_simulation(
        cluster,
        [_workload("w1"), _workload("w2", arrival=3)],
        policy=FirstFitScheduler(),
    )

    assert result.summary["allocated_workload_count"] == 2
    assert result.summary["completed_workload_count"] == 2
    assert result.summary["final_active_workload_count"] == 0
    assert sum(event["event"] == "completed" for event in result.events) == 2
    assert result.summary["resource_capacity_violation_count"] == 0
    check_invariants(cluster)


def test_failed_placement_attempts_are_distinct_from_unique_pending_workloads() -> None:
    cluster = _cluster([("node-a", "HN", _capacity(cpu=4))])
    workloads = [
        _workload("holder", demand=ResourceDemand(4, 1, 8, 2, 10), duration=2),
        _workload("waiter", demand=ResourceDemand(4, 1, 8, 2, 10), arrival=1, duration=None),
    ]

    result = run_simulation(cluster, workloads, policy=FirstFitScheduler())

    assert result.summary["unique_placement_failure_count"] == 0
    assert result.summary["queued_workload_count"] == 1
    assert result.summary["placement_attempt_failure_count"] == 1
    assert result.summary["queue_retry_count"] == 1
    assert result.summary["placement_failures_by_reason"] == {}
    assert sum(event["event"] == "queued" for event in result.events) == 1


def test_same_inputs_compare_from_fresh_cluster_states_and_are_reproducible() -> None:
    cluster = _cluster(
        [
            ("a-node", "HN", _capacity(cpu=10)),
            ("b-node", "HN", _capacity(cpu=20)),
        ]
    )
    workloads = [_workload("w1"), _workload("w2", arrival=1)]

    first = compare_schedulers(cluster, workloads)
    second = compare_schedulers(cluster, workloads)

    assert first["first_fit"].events == second["first_fit"].events
    assert first["best_fit"].events == second["best_fit"].events
    assert first["first_fit"].summary["allocated_workload_count"] == 2
    assert first["best_fit"].summary["allocated_workload_count"] == 2
    assert first["ifas"].summary["allocated_workload_count"] == 2
    assert all(not node.allocations for node in cluster.nodes.values())


def test_scheduler_factory_rejects_unsupported_policy() -> None:
    with pytest.raises(ValueError, match="choose first_fit, best_fit, or ifas"):
        create_scheduler("round_robin")


def test_cli_policy_selection_and_unsupported_policy(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_simulation.py"
    valid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-demo",
            "--policy",
            "best_fit",
            "--output-dir",
            str(tmp_path / "valid"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stderr
    report = json.loads(
        (tmp_path / "valid" / "best_fit" / "simulation_summary.json").read_text()
    )
    assert report["policy"] == "best_fit"
    assert report["policy_configuration"]["tie_break"] == "ascending node_id"

    invalid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-demo",
            "--policy",
            "round_robin",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "invalid choice" in invalid.stderr

    ifas = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-demo",
            "--policy",
            "ifas",
            "--ifas-config",
            str(PROJECT_ROOT / "config" / "ifas.json"),
            "--output-dir",
            str(tmp_path / "ifas"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ifas.returncode == 0, ifas.stderr
    ifas_report = json.loads(
        (tmp_path / "ifas" / "ifas" / "simulation_summary.json").read_text()
    )
    assert ifas_report["policy"] == "ifas"
    assert (tmp_path / "ifas" / "ifas" / "scheduler_decisions.csv").is_file()


def test_cli_compare_policies_writes_side_by_side_report(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "run_simulation.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-demo",
            "--compare-policies",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    comparison = tmp_path / "reports" / "comparison"
    report = json.loads((comparison / "comparison_summary.json").read_text())
    assert report["results"]["first_fit"]["allocated_workloads"] > 0
    assert report["results"]["best_fit"]["allocated_workloads"] > 0
    assert report["results"]["ifas"]["allocated_workloads"] > 0
    assert (comparison / "comparison_report.md").is_file()


def test_comparison_report_generation(tmp_path: Path) -> None:
    cluster = _cluster([("node-a", "HN", _capacity())])
    results = compare_schedulers(cluster, [_workload()])
    paths = save_comparison_report(
        results,
        cluster,
        tmp_path / "comparison",
        input_filename="synthetic.csv",
        seed=42,
        source_row_count=1,
        time_unit="test_units",
    )

    assert len(paths) == 3
    payload = json.loads(paths[0].read_text())
    assert set(payload["results"]) == {"first_fit", "best_fit", "ifas"}
    assert payload["time_unit"] == "test_units"
    assert payload["policy_configurations"]["first_fit"]["node_order"] == "ascending node_id"
    assert (tmp_path / "comparison" / "first_fit" / "simulation_summary.json").is_file()
    report = paths[2].read_text()
    assert "All runs used the same workload ordering" in report
    assert "Scheduler runtime" in report


def test_ifas_filters_infeasible_candidates_and_records_score_diagnostics() -> None:
    cluster = _cluster(
        [
            ("a-incompatible", "CN", _capacity()),
            ("b-infeasible", "HN", _capacity(cpu=2)),
            ("c-feasible", "HN", _capacity()),
        ],
        compatibility={"HN": ["HN"]},
    )
    scheduler = IntelligentFragmentationAwareScheduler()

    decision = scheduler.choose_node(_workload(), ["c-feasible"], cluster)

    assert decision.node_id == "c-feasible"
    assert decision.diagnostics["feasible_candidate_count"] == 1
    assert decision.diagnostics["rejected_candidate_reasons"] == {
        "a-incompatible": "role_incompatible",
        "b-infeasible": "insufficient_capacity:cpu_units",
    }
    assert decision.diagnostics["score_components"]["balance_penalty"] >= 0
    assert decision.diagnostics["final_score"] >= 0


def test_ifas_reports_authoritative_failure_reasons_when_no_candidate_fits() -> None:
    cluster = _cluster(
        [
            ("a-role-mismatch", "CN", _capacity()),
            ("b-no-capacity", "HN", _capacity(cpu=2)),
        ],
        compatibility={"HN": ["HN"]},
    )

    decision = IntelligentFragmentationAwareScheduler().choose_node(
        _workload(), [], cluster
    )

    assert decision.node_id is None
    assert decision.reason == (
        "no_feasible_node:insufficient_capacity:cpu_units|role_incompatible"
    )
    assert decision.diagnostics["rejected_candidate_reasons"] == {
        "a-role-mismatch": "role_incompatible",
        "b-no-capacity": "insufficient_capacity:cpu_units",
    }


def test_ifas_zero_capacity_zero_demand_is_omitted_from_score() -> None:
    cluster = _cluster(
        [
            (
                "node-a",
                "HN",
                ResourceCapacity(20, 0, 40, 20, 100),
            )
        ]
    )
    workload = _workload(demand=ResourceDemand(4, 0, 8, 2, 10))

    decision = IntelligentFragmentationAwareScheduler().choose_node(
        workload, ["node-a"], cluster
    )

    assert decision.node_id == "node-a"
    assert decision.diagnostics["score_components"]["residual_fit_penalty"] >= 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"balance_weight": -0.1},
        {"fragmentation_weight": float("nan")},
        {"spread_weight": float("inf")},
        {"balance_weight": 0, "fragmentation_weight": 0, "spread_weight": 0, "residual_fit_weight": 0},
        {"enable_future_fit": True, "future_fit_weight": 0},
        {"tie_breaking": "random"},
        {
            "balance_weight": 1e308,
            "fragmentation_weight": 1e308,
            "spread_weight": 0,
            "residual_fit_weight": 0,
        },
    ],
)
def test_ifas_rejects_invalid_weight_or_tie_break_config(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IFASConfig(**overrides)


def test_ifas_loads_documented_defaults_and_rejects_unknown_keys(tmp_path: Path) -> None:
    assert load_ifas_config().balance_weight == 0.25
    path = tmp_path / "ifas.json"
    path.write_text(json.dumps({"balance_weight": 1, "mystery": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown IFAS configuration keys"):
        load_ifas_config(path)


def test_ifas_diagnostics_can_omit_verbose_candidate_breakdown() -> None:
    scheduler = IntelligentFragmentationAwareScheduler(
        IFASConfig(diagnostics_enabled=False)
    )
    cluster = _cluster(
        [("node-a", "HN", _capacity()), ("node-b", "HN", _capacity())]
    )

    decision = scheduler.choose_node(_workload(), [], cluster)

    assert decision.diagnostics["workload_id"] == "workload"
    assert decision.diagnostics["selected_node_id"] == "node-a"
    assert "score_components" in decision.diagnostics
    assert "candidate_scores" not in decision.diagnostics


def test_ifas_prefers_tighter_fit_when_other_terms_are_disabled() -> None:
    cluster = _cluster(
        [
            ("large", "HN", _capacity(cpu=40, gpu=4, memory=80, rdma=40, disk=200)),
            ("tight", "HN", _capacity(cpu=10, gpu=2, memory=20, rdma=10, disk=50)),
        ]
    )
    config = IFASConfig(
        balance_weight=0,
        fragmentation_weight=0,
        spread_weight=0,
        residual_fit_weight=1,
    )

    decision = IntelligentFragmentationAwareScheduler(config).choose_node(
        _workload(demand=ResourceDemand(8, 1, 10, 4, 20)), [], cluster
    )

    assert decision.node_id == "tight"


def test_ifas_spread_penalty_prefers_active_feasible_node() -> None:
    cluster = _cluster(
        [
            ("a-active", "HN", _capacity()),
            ("b-idle", "HN", _capacity()),
        ]
    )
    from src.resource_accounting import allocate

    allocate(
        cluster,
        "a-active",
        _workload("occupant", demand=ResourceDemand(1, 0, 1, 1, 1)),
        0,
    )
    config = IFASConfig(
        balance_weight=0,
        fragmentation_weight=0,
        spread_weight=1,
        residual_fit_weight=0,
    )
    decision = IntelligentFragmentationAwareScheduler(config).choose_node(
        _workload("new"), ["a-active", "b-idle"], cluster
    )

    assert decision.node_id == "a-active"
    assert decision.diagnostics["score_components"]["spread_penalty"] == 0


def test_ifas_fragmentation_penalty_preserves_fit_for_known_queued_job() -> None:
    cluster = _cluster(
        [
            ("a-space", "HN", _capacity(cpu=10, gpu=2, memory=20, rdma=10, disk=50)),
            ("b-occupied", "HN", _capacity(cpu=10, gpu=2, memory=20, rdma=10, disk=50)),
        ]
    )
    from src.resource_accounting import allocate

    allocate(
        cluster,
        "b-occupied",
        _workload("occupant", demand=ResourceDemand(2, 0, 2, 1, 1)),
        0,
    )
    incoming = _workload("incoming", demand=ResourceDemand(4, 0, 4, 2, 5))
    queued = _workload("queued", demand=ResourceDemand(6, 0, 6, 3, 10), arrival=0)
    scheduler = IntelligentFragmentationAwareScheduler(
        IFASConfig(
            balance_weight=0,
            fragmentation_weight=1,
            spread_weight=0,
            residual_fit_weight=0,
        )
    )
    scheduler.set_pending_workloads([queued])

    decision = scheduler.choose_node(incoming, ["a-space", "b-occupied"], cluster)

    assert decision.node_id == "a-space"
    assert decision.diagnostics["candidate_scores"]["a-space"]["score_components"][
        "fragmentation_penalty"
    ] == 0
    assert decision.diagnostics["candidate_scores"]["b-occupied"]["score_components"][
        "fragmentation_penalty"
    ] > 0


def test_ifas_future_fit_uses_only_explicitly_supplied_current_queue() -> None:
    cluster = _cluster(
        [
            ("a-space", "HN", _capacity(cpu=10, gpu=2, memory=20, rdma=10, disk=50)),
            ("b-occupied", "HN-B", _capacity(cpu=10, gpu=2, memory=20, rdma=10, disk=50)),
        ],
        compatibility={"HN": ["HN", "HN-B"], "CN": ["HN-B"]},
    )
    from src.resource_accounting import allocate

    allocate(
        cluster,
        "b-occupied",
        _workload("occupant", demand=ResourceDemand(2, 0, 2, 1, 1)),
        0,
    )
    incoming = _workload("incoming", demand=ResourceDemand(4, 0, 4, 2, 5))
    already_queued = _workload(
        "known", demand=ResourceDemand(6, 0, 6, 3, 10), role="CN"
    )
    scheduler = IntelligentFragmentationAwareScheduler(
        IFASConfig(
            balance_weight=0,
            fragmentation_weight=0,
            spread_weight=0,
            residual_fit_weight=1,
            enable_future_fit=True,
            future_fit_weight=10,
        )
    )
    scheduler.set_pending_workloads([already_queued])

    decision = scheduler.choose_node(incoming, ["a-space", "b-occupied"], cluster)

    assert decision.diagnostics["future_fit_queue_order_assumption"].startswith(
        "only workloads already queued"
    )
    assert decision.diagnostics["candidate_scores"]["b-occupied"]["score_components"][
        "future_fit_penalty"
    ] == 1
    assert decision.diagnostics["candidate_scores"]["a-space"]["score_components"][
        "future_fit_penalty"
    ] == 0


def test_fragmentation_metrics_report_residual_gpu_slots_and_multiresource_stranding() -> None:
    cluster = _cluster(
        [
            ("gpu-a", "HN", _capacity(cpu=8, gpu=2, memory=8, rdma=8, disk=8)),
            ("gpu-b", "HN", _capacity(cpu=8, gpu=2, memory=8, rdma=8, disk=8)),
        ]
    )
    from src.resource_accounting import allocate

    allocate(
        cluster,
        "gpu-a",
        _workload("small", demand=ResourceDemand(4, 1, 4, 4, 4)),
        0,
    )
    waiting = _workload(
        "large", demand=ResourceDemand(10, 2, 10, 10, 10), duration=None
    )
    metrics = fragmentation_metrics(cluster, [waiting])

    assert metrics["residual_gpu_slots"] == 3
    assert metrics["stranded_gpu_slots_for_representative"] == 3
    assert metrics["stranded_gpu_slot_fraction"] == 1
    assert metrics["aggregate_capacity_sufficient_but_no_single_node_count"] == 1
    assert metrics["aggregate_fragmentation_fraction"] == 1


def test_ifas_simulator_writes_diagnostics_to_report(tmp_path: Path) -> None:
    cluster = _cluster([("node-a", "HN", _capacity())])
    result = run_simulation(
        cluster, [_workload("ifas-workload")], policy=IntelligentFragmentationAwareScheduler()
    )
    paths = save_simulation_report(
        result,
        cluster,
        tmp_path,
        input_filename="fixture.csv",
        seed=7,
        source_row_count=1,
    )

    assert len(paths) == 5
    decisions = pd.read_csv(tmp_path / "scheduler_decisions.csv")
    diagnostics = json.loads(decisions.loc[0, "decision_diagnostics"])
    assert diagnostics["workload_id"] == "ifas-workload"
    assert diagnostics["selected_node_id"] == "node-a"
    assert diagnostics["feasible_candidate_count"] == 1
