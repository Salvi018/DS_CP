"""Deterministic discrete-event cluster simulation without placement algorithms."""

from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.cluster_models import (
    ClusterState,
    PolicyDecision,
    ResourceDemand,
    SimulatedWorkload,
)
from src.fragmentation import fragmentation_metrics
from src.resource_accounting import (
    allocate,
    can_allocate,
    check_invariants,
    release,
)
from src.schedulers import Scheduler as PlacementPolicy

RESOURCE_COLUMNS = {
    "cpu_units": "cpu_request",
    "gpu_devices": "gpu_request",
    "memory_units": "memory_request",
    "rdma_units": "rdma_request",
    "disk_units": "disk_request",
}
ARRIVAL_COLUMN = "relative_arrival_time_raw_units"
DURATION_COLUMN = "observed_lifetime_raw_units"


class UniqueFeasibleNodePolicy:
    """Neutral guard: place only when exactly one node is feasible."""

    name = "unique_feasible_node"

    def choose_node(
        self,
        workload: SimulatedWorkload,
        feasible_node_ids: list[str],
        cluster: ClusterState,
    ) -> PolicyDecision:
        if len(feasible_node_ids) == 1:
            return PolicyDecision(feasible_node_ids[0], "unique_feasible_node")
        if len(feasible_node_ids) > 1:
            return PolicyDecision(None, "policy_ambiguous_multiple_feasible_nodes")
        return PolicyDecision(None, "no_feasible_node")


@dataclass
class WorkloadAdapterResult:
    workloads: list[SimulatedWorkload]
    rejected: list[dict[str, object]]
    source_row_count: int
    time_unit: str
    arrival_source: str | None
    duration_source: str | None


@dataclass
class SimulationResult:
    summary: dict[str, object]
    events: list[dict[str, object]]
    resource_history: list[dict[str, object]]
    assumptions: list[str]
    policy_name: str
    time_unit: str
    policy_configuration: dict[str, object] = field(default_factory=dict)


def _finite_value(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def adapt_workload_frame(
    frame: pd.DataFrame,
    cluster: ClusterState,
    *,
    time_unit: str = "trace_units_unverified",
) -> WorkloadAdapterResult:
    """Convert cleaned trace rows without inventing missing demand or timing."""
    rejected: list[dict[str, object]] = []
    workloads: list[SimulatedWorkload] = []
    if ARRIVAL_COLUMN in frame.columns:
        arrival_source: str | None = ARRIVAL_COLUMN
        arrivals = pd.to_numeric(frame[ARRIVAL_COLUMN], errors="coerce")
    elif "creation_time" in frame.columns:
        arrival_source = "creation_time normalized to minimum observed creation_time"
        creation = pd.to_numeric(frame["creation_time"], errors="coerce")
        baseline = creation.min()
        arrivals = creation - baseline if pd.notna(baseline) else pd.Series(np.nan, index=frame.index)
    else:
        arrival_source = None
        arrivals = pd.Series(np.nan, index=frame.index)
    duration_source = DURATION_COLUMN if DURATION_COLUMN in frame.columns else None

    seen_ids: set[str] = set()
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        source_row_value = row.get("__benchmark_source_row", ordinal)
        try:
            source_row = int(source_row_value)
        except (TypeError, ValueError):
            source_row = ordinal
        identifier_value = row.get("instance_sn")
        workload_id = (
            str(identifier_value)
            if identifier_value is not None and not pd.isna(identifier_value) and str(identifier_value).strip()
            else f"row-{source_row}"
        )
        reasons: list[str] = []
        if workload_id in seen_ids:
            reasons.append("duplicate_workload_identifier")
        seen_ids.add(workload_id)

        raw_role = row.get("role")
        role = str(raw_role) if raw_role is not None and not pd.isna(raw_role) else ""
        if not role:
            reasons.append("missing_role")

        demand_values: dict[str, float] = {}
        for resource, column in RESOURCE_COLUMNS.items():
            if column not in frame.columns:
                reasons.append(f"missing_resource_column:{column}")
                continue
            value = _finite_value(row[column])
            if value is None:
                reasons.append(f"missing_or_invalid_resource:{column}")
            elif value < 0:
                reasons.append(f"negative_resource:{column}")
            elif resource == "gpu_devices" and not value.is_integer():
                reasons.append(f"non_integer_gpu_request:{column}")
            else:
                demand_values[resource] = value

        raw_limit = row.get("max_instance_per_node")
        density_limit: int | None = None
        if raw_limit is not None and not pd.isna(raw_limit):
            limit_value = _finite_value(raw_limit)
            if limit_value is None or not limit_value.is_integer():
                reasons.append("invalid_density_limit")
            else:
                density_limit = int(limit_value)
                if density_limit < 0 and density_limit not in cluster.density_unlimited_values:
                    reasons.append("unverified_negative_density_limit")

        arrival = _finite_value(arrivals.iloc[ordinal])
        if arrival is None:
            reasons.append("missing_or_invalid_arrival_time")
        elif arrival < 0:
            reasons.append("negative_arrival_time")
        duration = _finite_value(row.get(DURATION_COLUMN)) if duration_source else None
        if duration is not None and duration < 0:
            reasons.append("negative_observed_lifetime")

        if reasons:
            rejected.append(
                {"workload_id": workload_id, "source_row": source_row, "reasons": reasons}
            )
            continue
        workloads.append(
            SimulatedWorkload(
                workload_id=workload_id,
                demand=ResourceDemand(
                    cpu_units=demand_values["cpu_units"],
                    gpu_devices=int(demand_values["gpu_devices"]),
                    memory_units=demand_values["memory_units"],
                    rdma_units=demand_values["rdma_units"],
                    disk_units=demand_values["disk_units"],
                ),
                role=role,
                app_name=(
                    str(row["app_name"])
                    if "app_name" in frame.columns
                    and pd.notna(row.get("app_name"))
                    and str(row.get("app_name")).strip()
                    else None
                ),
                max_instance_per_node=density_limit,
                arrival_time=arrival,
                duration=duration,
                attributes={"source_row": source_row},
            )
        )
    return WorkloadAdapterResult(
        workloads=workloads,
        rejected=rejected,
        source_row_count=len(frame),
        time_unit=time_unit,
        arrival_source=arrival_source,
        duration_source=duration_source,
    )


def select_earliest_arrival_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select at most ``limit`` rows with valid non-negative arrivals, stably."""
    if limit < 1:
        raise ValueError("workload limit must be a positive integer")
    if ARRIVAL_COLUMN in frame.columns:
        arrivals = pd.to_numeric(frame[ARRIVAL_COLUMN], errors="coerce")
    elif "creation_time" in frame.columns:
        creation = pd.to_numeric(frame["creation_time"], errors="coerce")
        arrivals = creation - creation.min()
    else:
        return frame.iloc[0:0]
    eligible = arrivals.notna() & np.isfinite(arrivals) & (arrivals >= 0)
    selected = (
        pd.DataFrame({"position": np.arange(len(frame)), "arrival": arrivals.to_numpy()})
        .loc[eligible.to_numpy()]
        .sort_values(["arrival", "position"], kind="stable")
        .head(limit)["position"]
    )
    return frame.iloc[selected.to_numpy()]


def _resource_snapshot(
    cluster: ClusterState,
    at_time: float,
    pending_workloads: list[SimulatedWorkload] | None = None,
) -> dict[str, object]:
    totals: dict[str, float] = {}
    allocated: dict[str, float] = {}
    for node in cluster.nodes.values():
        for name, capacity in node.capacity.as_dict().items():
            totals[name] = totals.get(name, 0.0) + float(capacity)
        for name, amount in node.allocated.as_dict().items():
            allocated[name] = allocated.get(name, 0.0) + float(amount)
    ratios: dict[str, float | None] = {
        name: allocated.get(name, 0.0) / capacity if capacity > 0 else None
        for name, capacity in totals.items()
    }
    return {
        "time": at_time,
        "active_workload_count": sum(len(node.allocations) for node in cluster.nodes.values()),
        "allocated_requests": allocated,
        "configured_capacity": totals,
        "allocation_ratios": ratios,
        "fragmentation": fragmentation_metrics(cluster, pending_workloads or []),
    }


def run_simulation(
    cluster: ClusterState,
    workloads: list[SimulatedWorkload],
    *,
    rejected_inputs: list[dict[str, object]] | None = None,
    policy: PlacementPolicy | None = None,
    time_unit: str = "trace_units_unverified",
) -> SimulationResult:
    """Run stable arrival/completion events and account for queue and allocations.

    Completion events precede arrivals at equal timestamps. Pending workloads
    are retried once after each resource-release event, in original queue order.
    """
    placement_policy = policy or UniqueFeasibleNodePolicy()
    input_rejections = list(rejected_inputs or [])
    source_workload_count = len(workloads) + len(input_rejections)
    runtime_rejections: list[dict[str, object]] = []
    pending: dict[str, SimulatedWorkload] = {}
    ever_queued_ids: set[str] = set()
    events: list[dict[str, object]] = []
    history: list[dict[str, object]] = [
        _resource_snapshot(cluster, 0.0, list(pending.values()))
    ]
    completed_ids: set[str] = set()
    submitted_ids: set[str] = set()
    accepted_count = 0
    placement_attempt_count = 0
    placement_attempt_failure_count = 0
    queue_retry_count = 0
    scheduler_runtime_seconds = 0.0
    scheduler_decision_count = 0
    feasible_candidate_total = 0
    pending_failure_reason: dict[str, str] = {}
    queue_delays: list[float] = []
    heap: list[tuple[float, int, int, str, SimulatedWorkload]] = []
    sequence = 0

    for workload in workloads:
        if workload.workload_id in submitted_ids:
            input_rejections.append(
                {"workload_id": workload.workload_id, "reasons": ["duplicate_workload_identifier"]}
            )
            continue
        submitted_ids.add(workload.workload_id)
        if (
            workload.arrival_time is None
            or not math.isfinite(workload.arrival_time)
            or workload.arrival_time < 0
        ):
            input_rejections.append(
                {"workload_id": workload.workload_id, "reasons": ["invalid_arrival_time"]}
            )
            continue
        if workload.duration is not None and (
            not math.isfinite(workload.duration) or workload.duration < 0
        ):
            input_rejections.append(
                {"workload_id": workload.workload_id, "reasons": ["invalid_duration"]}
            )
            continue
        heapq.heappush(heap, (workload.arrival_time, 1, sequence, "arrival", workload))
        sequence += 1
        accepted_count += 1

    def record(
        event: str,
        workload: SimulatedWorkload,
        at_time: float,
        node_id: str | None = None,
        reason: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        events.append(
            {
                "time": at_time,
                "event": event,
                "workload_id": workload.workload_id,
                "node_id": node_id,
                "reason": reason,
                "queue_length": len(pending),
                "decision_diagnostics": diagnostics or {},
            }
        )
        history.append(_resource_snapshot(cluster, at_time, list(pending.values())))

    def attempt(workload: SimulatedWorkload, at_time: float) -> bool:
        nonlocal placement_attempt_count, placement_attempt_failure_count
        nonlocal queue_retry_count, scheduler_runtime_seconds
        nonlocal scheduler_decision_count, feasible_candidate_total
        was_pending = workload.workload_id in pending
        if was_pending:
            queue_retry_count += 1

        validation_checks = [
            can_allocate(cluster, node_id, workload)
            for node_id in cluster.nodes
        ]
        invalid_prefixes = (
            "invalid_demand:",
            "gpu_demand_not_integer",
            "invalid_density_limit",
            "missing_application_for_density_limit",
            "missing_workload_id",
        )
        invalid_reasons = sorted(
            {
                check.reason
                for check in validation_checks
                if check.reason.startswith(invalid_prefixes)
            }
        )
        if invalid_reasons:
            runtime_rejections.append(
                {
                    "workload_id": workload.workload_id,
                    "reasons": invalid_reasons,
                }
            )
            record("rejected", workload, at_time, reason=";".join(invalid_reasons))
            pending.pop(workload.workload_id, None)
            pending_failure_reason.pop(workload.workload_id, None)
            return False

        scheduler_started = time.perf_counter()
        set_queue = getattr(placement_policy, "set_pending_workloads", None)
        if callable(set_queue):
            set_queue(
                [
                    queued
                    for queued in pending.values()
                    if queued.workload_id != workload.workload_id
                ]
            )
        feasible = [
            node_id
            for node_id, check in zip(cluster.nodes, validation_checks)
            if check.feasible
        ]
        decision = placement_policy.choose_node(workload, feasible, cluster)
        decision_seconds = time.perf_counter() - scheduler_started
        scheduler_runtime_seconds += decision_seconds
        scheduler_decision_count += 1
        feasible_candidate_total += len(feasible)
        placement_attempt_count += 1
        diagnostics = dict(decision.diagnostics)

        if decision.node_id is None:
            placement_attempt_failure_count += 1
            pending[workload.workload_id] = workload
            ever_queued_ids.add(workload.workload_id)
            prior_reason = pending_failure_reason.get(workload.workload_id)
            pending_failure_reason[workload.workload_id] = decision.reason
            if not was_pending:
                record(
                    "queued",
                    workload,
                    at_time,
                    reason=decision.reason,
                    diagnostics=diagnostics,
                )
            elif prior_reason != decision.reason:
                record(
                    "retry_failed",
                    workload,
                    at_time,
                    reason=decision.reason,
                    diagnostics=diagnostics,
                )
            return False
        if decision.node_id not in cluster.nodes:
            raise ValueError(
                f"Placement policy {placement_policy.name} selected unknown node "
                f"{decision.node_id!r} for workload {workload.workload_id!r}"
            )
        feasibility = can_allocate(cluster, decision.node_id, workload)
        if not feasibility.feasible:
            raise ValueError(
                f"Placement policy {placement_policy.name} selected infeasible node "
                f"{decision.node_id!r} for workload {workload.workload_id!r}: "
                f"{feasibility.reason}"
            )
        allocate(cluster, decision.node_id, workload, at_time)
        check_invariants(cluster)
        pending.pop(workload.workload_id, None)
        pending_failure_reason.pop(workload.workload_id, None)
        if workload.arrival_time is not None:
            queue_delays.append(max(0.0, at_time - workload.arrival_time))
        record(
            "allocated",
            workload,
            at_time,
            decision.node_id,
            decision.reason,
            diagnostics,
        )
        if workload.duration is not None:
            heapq.heappush(
                heap,
                (at_time + workload.duration, 0, len(events), "completion", workload),
            )
        return True

    previous_time = 0.0
    resource_time: dict[str, float] = {}
    fragmentation_time: dict[str, float] = {}
    fragmentation_time_denominator: dict[str, float] = {}
    active_node_time = 0.0
    max_active_node_count = 0
    total_capacity: dict[str, float] = {}
    for node in cluster.nodes.values():
        for name, capacity in node.capacity.as_dict().items():
            total_capacity[name] = total_capacity.get(name, 0.0) + float(capacity)
    final_time = 0.0
    while heap:
        at_time, _, _, event_type, workload = heapq.heappop(heap)
        elapsed = at_time - previous_time
        if elapsed < -1e-9:
            raise ValueError("Simulation event queue is not time ordered")
        if elapsed > 0:
            snapshot = _resource_snapshot(cluster, previous_time, list(pending.values()))
            current = snapshot["allocated_requests"]
            if not isinstance(current, dict):
                raise RuntimeError("Resource snapshot did not contain allocated request totals")
            for name, amount in current.items():
                resource_time[name] = resource_time.get(name, 0.0) + float(amount) * elapsed
            current_fragmentation = snapshot["fragmentation"]
            if not isinstance(current_fragmentation, dict):
                raise RuntimeError("Resource snapshot did not contain fragmentation metrics")
            for name in (
                "stranded_gpu_slot_fraction",
                "aggregate_fragmentation_fraction",
                "gpu_residual_fit_fragmentation_fraction",
                "gpu_fragmentation_blocked_fraction",
            ):
                value = current_fragmentation[name]
                if value is not None:
                    fragmentation_time[name] = (
                        fragmentation_time.get(name, 0.0) + float(value) * elapsed
                    )
                    fragmentation_time_denominator[name] = (
                        fragmentation_time_denominator.get(name, 0.0) + elapsed
                    )
            current_active_nodes = int(current_fragmentation["active_node_count"])
            active_node_time += current_active_nodes * elapsed
            max_active_node_count = max(max_active_node_count, current_active_nodes)
        previous_time = at_time
        final_time = at_time
        if event_type == "arrival":
            attempt(workload, at_time)
        else:
            if workload.workload_id in completed_ids:
                continue
            release(cluster, workload.workload_id)
            check_invariants(cluster)
            completed_ids.add(workload.workload_id)
            record("completed", workload, at_time)
            for waiting in list(pending.values()):
                attempt(waiting, at_time)

    ratios: dict[str, float | None] = {
        name: resource_time.get(name, 0.0) / (capacity * final_time)
        if capacity > 0 and final_time > 0
        else None
        for name, capacity in total_capacity.items()
    }
    time_weighted_fragmentation = {
        name: (
            fragmentation_time[name] / fragmentation_time_denominator[name]
            if fragmentation_time_denominator.get(name, 0.0) > 0
            else None
        )
        for name in (
            "stranded_gpu_slot_fraction",
            "aggregate_fragmentation_fraction",
            "gpu_residual_fit_fragmentation_fraction",
            "gpu_fragmentation_blocked_fraction",
        )
    }
    failure_counts = Counter(pending_failure_reason.values())
    queue_delay_summary = {
        "count": len(queue_delays),
        "mean": float(np.mean(queue_delays)) if queue_delays else None,
        "median": float(np.median(queue_delays)) if queue_delays else None,
        "p95": float(np.quantile(queue_delays, 0.95)) if queue_delays else None,
        "maximum": float(max(queue_delays)) if queue_delays else None,
        "pending_workload_count": len(pending),
        "definition": "arrival time to first successful allocation; excludes workloads never allocated",
    }
    peak_allocation_ratios: dict[str, float | None] = {}
    for name in total_capacity:
        observed_ratios = [
            float(snapshot["allocation_ratios"][name])
            for snapshot in history
            if isinstance(snapshot.get("allocation_ratios"), dict)
            and snapshot["allocation_ratios"].get(name) is not None
        ]
        peak_allocation_ratios[name] = max(observed_ratios) if observed_ratios else None
    queued_unique = len(ever_queued_ids)
    all_rejections = input_rejections + runtime_rejections
    summary: dict[str, object] = {
        "cluster_name": cluster.cluster_name,
        "source_workload_count": source_workload_count,
        "accepted_workload_count": accepted_count,
        "input_rejected_workload_count": len(rejected_inputs or []),
        "allocated_workload_count": len(
            {event["workload_id"] for event in events if event["event"] == "allocated"}
        ),
        "completed_workload_count": len(completed_ids),
        "pending_workload_count": len(pending),
        "rejected_workload_count": len(all_rejections),
        "queued_workload_count": queued_unique,
        "unique_placement_failure_count": len(pending),
        "placement_failures_by_reason": dict(sorted(failure_counts.items())),
        "placement_attempt_count": placement_attempt_count,
        "placement_attempt_failure_count": placement_attempt_failure_count,
        "queue_retry_count": queue_retry_count,
        "queue_delay": queue_delay_summary,
        "scheduler_runtime_seconds": scheduler_runtime_seconds,
        "scheduler_decision_count": scheduler_decision_count,
        "feasible_candidate_count_total": feasible_candidate_total,
        "resource_capacity_violation_count": 0,
        "time_weighted_fragmentation_metrics": time_weighted_fragmentation,
        "average_active_node_count": (
            active_node_time / final_time if final_time > 0 else None
        ),
        "max_active_node_count": max(
            (
                int(snapshot["fragmentation"]["active_node_count"])
                for snapshot in history
                if isinstance(snapshot.get("fragmentation"), dict)
            ),
            default=max_active_node_count,
        ),
        "rejected_inputs": all_rejections,
        "simulated_duration": final_time,
        "time_unit": time_unit,
        "time_weighted_allocation_ratios": ratios,
        "peak_allocation_ratios": peak_allocation_ratios,
        "final_active_workload_count": sum(len(node.allocations) for node in cluster.nodes.values()),
        "policy": placement_policy.name,
        "metric_definition": (
            "Time-weighted allocation ratio = integral of requested resources reserved "
            "across nodes divided by configured capacity multiplied by simulated duration. "
            "This is not measured hardware utilization."
        ),
        "placement_failure_definition": (
            "Unique unresolved workloads by their latest placement failure reason; "
            "placement_attempt_failure_count separately counts all failed attempts."
        ),
    }
    return SimulationResult(
        summary=summary,
        events=events,
        resource_history=history,
        assumptions=list(cluster.assumptions),
        policy_name=placement_policy.name,
        time_unit=time_unit,
        policy_configuration=dict(
            getattr(placement_policy, "configuration", {})
        ),
    )
