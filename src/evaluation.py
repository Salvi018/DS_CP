"""Reproducible scheduler experiments, metrics, and benchmark artifacts."""

from __future__ import annotations

import copy
import json
import math
import platform
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.express as px

from config.settings import PROJECT_VERSION
from src.cluster_models import (
    ClusterState,
    ResourceCapacity,
    SimulatedNode,
    SimulatedWorkload,
)
from src.schedulers import IFASConfig, create_scheduler
from src.simulation_reports import save_simulation_report
from src.simulator import SimulationResult, run_simulation

SUPPORTED_POLICIES = ("first_fit", "best_fit", "ifas")
RESOURCE_NAMES = ("cpu_units", "gpu_devices", "memory_units", "rdma_units", "disk_units")


def run_experiment(
    cluster: ClusterState,
    workloads: list[SimulatedWorkload],
    *,
    policies: Iterable[str] = SUPPORTED_POLICIES,
    rejected_inputs: list[dict[str, object]] | None = None,
    time_unit: str = "trace_units_unverified",
    ifas_config: IFASConfig | None = None,
) -> dict[str, SimulationResult]:
    """Run selected policies with identical ordered inputs and fresh clusters."""
    selected_policies = tuple(policies)
    if not selected_policies:
        raise ValueError("At least one scheduler policy must be selected")
    if len(set(selected_policies)) != len(selected_policies):
        raise ValueError("Scheduler policies must not be repeated")
    unsupported = set(selected_policies) - set(SUPPORTED_POLICIES)
    if unsupported:
        raise ValueError(f"Unsupported scheduler policies: {', '.join(sorted(unsupported))}")

    results: dict[str, SimulationResult] = {}
    for policy_name in selected_policies:
        run_cluster = copy.deepcopy(cluster)
        scheduler = create_scheduler(
            policy_name, ifas_config if policy_name == "ifas" else None
        )
        result = run_simulation(
            run_cluster,
            copy.deepcopy(workloads),
            rejected_inputs=copy.deepcopy(rejected_inputs or []),
            policy=scheduler,
            time_unit=time_unit,
        )
        if int(result.summary["resource_capacity_violation_count"]) != 0:
            raise RuntimeError(f"{policy_name} reported a resource-capacity violation")
        results[policy_name] = result
    validate_experiment_results(results, workloads)
    return results


def validate_experiment_results(
    results: dict[str, SimulationResult],
    workloads: list[SimulatedWorkload],
) -> None:
    """Check comparable submissions and invariant counters across policy runs."""
    if not results:
        raise ValueError("No scheduler results were supplied")
    expected_ids = [workload.workload_id for workload in workloads]
    for policy_name, result in results.items():
        if result.policy_name != policy_name:
            raise ValueError(
                f"Result key {policy_name!r} does not match policy "
                f"{result.policy_name!r}"
            )
        if int(result.summary["accepted_workload_count"]) > len(expected_ids):
            raise ValueError(f"{policy_name} accepted more workloads than supplied")
        if int(result.summary["resource_capacity_violation_count"]) != 0:
            raise ValueError(f"{policy_name} has resource-capacity violations")
    accepted_counts = {
        int(result.summary["accepted_workload_count"]) for result in results.values()
    }
    if len(accepted_counts) != 1:
        raise ValueError("Scheduler runs did not submit an identical workload count")


def _queue_delays(
    result: SimulationResult, workloads: list[SimulatedWorkload]
) -> list[float]:
    arrivals = {
        workload.workload_id: workload.arrival_time for workload in workloads
    }
    first_allocations: dict[str, float] = {}
    for event in result.events:
        if event.get("event") == "allocated":
            workload_id = str(event["workload_id"])
            first_allocations.setdefault(workload_id, float(event["time"]))
    return [
        max(0.0, allocation_time - float(arrivals[workload_id]))
        for workload_id, allocation_time in first_allocations.items()
        if arrivals.get(workload_id) is not None
    ]


def _peak_active_nodes(result: SimulationResult) -> int:
    return max(
        (
            int(snapshot.get("fragmentation", {}).get("active_node_count", 0))
            for snapshot in result.resource_history
            if isinstance(snapshot.get("fragmentation"), dict)
        ),
        default=0,
    )


def benchmark_metrics(
    result: SimulationResult, workloads: list[SimulatedWorkload]
) -> dict[str, Any]:
    """Calculate comparable outcome, queue, resource, and overhead metrics."""
    summary = result.summary
    delays = _queue_delays(result, workloads)
    rejections = summary.get("rejected_inputs", [])
    rejected_ids = {
        (
            "source_row",
            int(rejection["source_row"]),
        )
        if rejection.get("source_row") is not None
        else ("workload_id", str(rejection["workload_id"]))
        for rejection in rejections
        if isinstance(rejection, dict) and rejection.get("workload_id") is not None
    }
    peak_ratios = summary.get("peak_allocation_ratios", {})
    weighted_ratios = summary.get("time_weighted_allocation_ratios", {})
    resource_metrics: dict[str, dict[str, float | None]] = {}
    for resource in sorted(set(peak_ratios) | set(weighted_ratios)):
        resource_metrics[resource] = {
            "peak_allocation_ratio": peak_ratios.get(resource),
            "time_weighted_allocation_ratio": weighted_ratios.get(resource),
        }
    decision_count = int(summary.get("scheduler_decision_count", 0))
    return {
        "policy": result.policy_name,
        "submitted_workloads": int(summary["accepted_workload_count"]),
        "allocated_workloads": int(summary["allocated_workload_count"]),
        "completed_workloads": int(summary["completed_workload_count"]),
        "pending_workloads": int(summary["pending_workload_count"]),
        "unique_unplaced_workloads": int(summary["unique_placement_failure_count"]),
        "unique_rejected_workloads": len(rejected_ids),
        "placement_attempts": int(summary["placement_attempt_count"]),
        "failed_placement_attempts": int(
            summary["placement_attempt_failure_count"]
        ),
        "placement_failures_by_reason": summary["placement_failures_by_reason"],
        "queue_delay_mean": statistics.fmean(delays) if delays else None,
        "queue_delay_median": statistics.median(delays) if delays else None,
        "queue_delay_p95": (
            float(pd.Series(delays).quantile(0.95)) if delays else None
        ),
        "queue_delay_maximum": max(delays) if delays else None,
        "queue_delay_observation_count": len(delays),
        "pending_queue_count": int(summary["pending_workload_count"]),
        "resource_metrics": resource_metrics,
        "active_node_count_peak": _peak_active_nodes(result),
        "active_node_count_time_weighted_mean": summary[
            "average_active_node_count"
        ],
        "scheduler_runtime_seconds": float(summary["scheduler_runtime_seconds"]),
        "scheduler_decision_count": decision_count,
        "average_decision_time_seconds": (
            float(summary["scheduler_runtime_seconds"]) / decision_count
            if decision_count
            else None
        ),
        "feasible_candidates_considered": int(
            summary.get("feasible_candidate_count_total", 0)
        ),
        "resource_capacity_violations": int(
            summary["resource_capacity_violation_count"]
        ),
        "time_weighted_fragmentation_metrics": summary[
            "time_weighted_fragmentation_metrics"
        ],
        "simulated_duration": summary["simulated_duration"],
        "time_unit": result.time_unit,
    }


def workload_outcomes(
    result: SimulationResult,
    workloads: list[SimulatedWorkload],
) -> list[dict[str, Any]]:
    """Return one auditable status row per submitted or rejected workload."""
    allocation_events: dict[str, dict[str, Any]] = {}
    completion_events: dict[str, dict[str, Any]] = {}
    for event in result.events:
        workload_id = str(event["workload_id"])
        if event.get("event") == "allocated":
            allocation_events.setdefault(workload_id, event)
        elif event.get("event") == "completed":
            completion_events[workload_id] = event

    rejections = result.summary.get("rejected_inputs", [])
    rejection_by_source: dict[tuple[str, int], list[str]] = {}
    rejection_by_id: dict[str, list[str]] = {}
    for rejection in rejections:
        if isinstance(rejection, dict) and rejection.get("workload_id") is not None:
            reasons = [
                str(reason) for reason in rejection.get("reasons", [])
            ]
            workload_id = str(rejection["workload_id"])
            if rejection.get("source_row") is not None:
                rejection_by_source[(workload_id, int(rejection["source_row"]))] = reasons
            else:
                rejection_by_id[workload_id] = reasons
    rows: list[dict[str, Any]] = []
    represented_rejection_keys: set[tuple[str, int]] = set()
    for workload in workloads:
        workload_id = workload.workload_id
        allocated = allocation_events.get(workload_id)
        completed = completion_events.get(workload_id)
        source_row = workload.attributes.get("source_row")
        rejection_key = (
            (workload_id, int(source_row))
            if source_row is not None
            else None
        )
        rejection_reasons = (
            rejection_by_source.get(rejection_key, [])
            if rejection_key is not None
            else []
        )
        rejection_reasons = rejection_reasons or rejection_by_id.get(workload_id, [])
        if rejection_key is not None and rejection_reasons:
            represented_rejection_keys.add(rejection_key)
        if completed:
            status = "completed"
        elif allocated:
            status = "allocated_active_at_end"
        elif rejection_reasons:
            status = "rejected"
        else:
            status = "pending"
        rows.append(
            {
                "workload_id": workload_id,
                "status": status,
                "role": workload.role,
                "app_name": workload.app_name,
                "arrival_time": workload.arrival_time,
                "duration": workload.duration,
                "allocated_at": allocated.get("time") if allocated else None,
                "node_id": allocated.get("node_id") if allocated else None,
                "completed_at": completed.get("time") if completed else None,
                "source_row": source_row,
                "rejection_reasons": ";".join(rejection_reasons),
            }
        )
    for (workload_id, source_row), reasons in rejection_by_source.items():
        if (workload_id, source_row) not in represented_rejection_keys:
            rows.append(
                {
                    "workload_id": workload_id,
                    "source_row": source_row,
                    "status": "rejected",
                    "role": None,
                    "app_name": None,
                    "arrival_time": None,
                    "duration": None,
                    "allocated_at": None,
                    "node_id": None,
                    "completed_at": None,
                    "rejection_reasons": ";".join(reasons),
                }
            )
    workload_ids = {workload.workload_id for workload in workloads}
    for workload_id, reasons in rejection_by_id.items():
        if workload_id not in workload_ids:
            rows.append(
                {
                    "workload_id": workload_id,
                    "source_row": None,
                    "status": "rejected",
                    "role": None,
                    "app_name": None,
                    "arrival_time": None,
                    "duration": None,
                    "allocated_at": None,
                    "node_id": None,
                    "completed_at": None,
                    "rejection_reasons": ";".join(reasons),
                }
            )
    return rows


def resource_metric_rows(
    results: dict[str, SimulationResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, result in results.items():
        summary = result.summary
        capacities: dict[str, float] = {}
        for snapshot in result.resource_history:
            configured = snapshot.get("configured_capacity", {})
            if isinstance(configured, dict):
                capacities = {name: float(value) for name, value in configured.items()}
                break
        peak = summary.get("peak_allocation_ratios", {})
        weighted = summary.get("time_weighted_allocation_ratios", {})
        for resource in sorted(set(capacities) | set(peak) | set(weighted)):
            rows.append(
                {
                    "policy": policy_name,
                    "resource": resource,
                    "configured_capacity": capacities.get(resource),
                    "peak_allocation_ratio": peak.get(resource),
                    "time_weighted_allocation_ratio": weighted.get(resource),
                    "metric_status": (
                        "not_applicable_zero_capacity"
                        if capacities.get(resource) == 0
                        else "available"
                    ),
                }
            )
    return rows


def fragmentation_metric_rows(
    results: dict[str, SimulationResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, result in results.items():
        snapshots = result.resource_history
        for ordinal, snapshot in enumerate(snapshots):
            metrics = snapshot.get("fragmentation")
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "policy": policy_name,
                        "snapshot": ordinal,
                        "time": snapshot.get("time"),
                        **metrics,
                    }
                )
        weighted = result.summary["time_weighted_fragmentation_metrics"]
        for name, value in weighted.items():
            rows.append(
                {
                    "policy": policy_name,
                    "snapshot": "time_weighted",
                    "time": None,
                    "metric": name,
                    "value": value,
                }
            )
    return rows


def _cluster_configuration(cluster: ClusterState) -> dict[str, Any]:
    return {
        "cluster_name": cluster.cluster_name,
        "nodes": {
            node_id: {"role": node.role, "capacity": node.capacity.as_dict()}
            for node_id, node in cluster.nodes.items()
        },
        "role_compatibility": cluster.role_compatibility,
        "density_limit_semantics_verified": cluster.density_limit_semantics_verified,
        "density_unlimited_values": cluster.density_unlimited_values,
        "extra_resource_units": cluster.extra_resource_units,
        "assumptions": cluster.assumptions,
    }


def _plotly_charts(
    comparison: pd.DataFrame,
    resource_metrics: pd.DataFrame,
    fragmentation_metrics: pd.DataFrame,
    failure_metrics: pd.DataFrame,
) -> dict[str, Any]:
    charts: dict[str, Any] = {}
    if not comparison.empty:
        outcomes = comparison.melt(
            id_vars="policy",
            value_vars=[
                "allocated_workloads",
                "pending_workloads",
                "completed_workloads",
                "unique_rejected_workloads",
            ],
            var_name="outcome",
            value_name="workload_count",
        )
        charts["workload_outcomes"] = px.bar(
            outcomes,
            x="policy",
            y="workload_count",
            color="outcome",
            barmode="group",
            title="Workload outcomes by scheduler",
            labels={"workload_count": "Workloads", "policy": "Scheduler"},
        )
        queue_columns = [
            column
            for column in (
                "queue_delay_mean",
                "queue_delay_median",
                "queue_delay_p95",
                "queue_delay_maximum",
            )
            if comparison[column].notna().any()
        ]
        if queue_columns:
            charts["queue_delay"] = px.bar(
                comparison.melt(
                    id_vars="policy",
                    value_vars=queue_columns,
                    var_name="statistic",
                    value_name="delay",
                ),
                x="policy",
                y="delay",
                color="statistic",
                barmode="group",
                title="Simulated queue waiting time",
                labels={"delay": "Queue delay (configured simulation units)"},
            )
        charts["scheduler_runtime"] = px.bar(
            comparison,
            x="policy",
            y="scheduler_runtime_seconds",
            title="Scheduler execution time",
            labels={
                "scheduler_runtime_seconds": "Wall-clock seconds",
                "policy": "Scheduler",
            },
        )
    if not resource_metrics.empty:
        available = resource_metrics[
            resource_metrics["time_weighted_allocation_ratio"].notna()
        ]
        if not available.empty:
            charts["resource_allocation"] = px.bar(
                available,
                x="resource",
                y="time_weighted_allocation_ratio",
                color="policy",
                barmode="group",
                title="Time-weighted requested-resource reservations",
                labels={
                    "time_weighted_allocation_ratio": "Reservation ratio (not utilization)",
                    "resource": "Resource",
                },
            )
    if not fragmentation_metrics.empty:
        snapshots = fragmentation_metrics[
            (fragmentation_metrics["snapshot"] != "time_weighted")
            & fragmentation_metrics["gpu_fragmentation_blocked_fraction"].notna()
        ]
        if not snapshots.empty:
            aggregate = (
                snapshots.groupby("policy", as_index=False)[
                    "gpu_fragmentation_blocked_fraction"
                ]
                .mean()
            )
            charts["fragmentation"] = px.bar(
                aggregate,
                x="policy",
                y="gpu_fragmentation_blocked_fraction",
                title="Mean snapshot GPU fragmentation-blocked fraction",
                labels={
                    "gpu_fragmentation_blocked_fraction": "Affected queued GPU workloads / compatible queued GPU workloads"
                },
            )
    if not failure_metrics.empty:
        charts["placement_failures"] = px.bar(
            failure_metrics,
            x="reason",
            y="unique_pending_workloads",
            color="policy",
            barmode="group",
            title="Unique pending workloads by latest failure reason",
            labels={"unique_pending_workloads": "Unique workloads", "reason": "Reason"},
        )
    return charts


def save_benchmark_outputs(
    results: dict[str, SimulationResult],
    cluster: ClusterState,
    workloads: list[SimulatedWorkload],
    output_dir: str | Path,
    *,
    input_filename: str,
    source_row_count: int,
    selected_row_count: int,
    seed: int,
    time_unit: str,
    ifas_config: IFASConfig | None = None,
    selection_exclusions: list[dict[str, Any]] | None = None,
    repeats_requested: int = 1,
    sensitivity_results: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Write all tabular, diagnostic, configuration, and chart artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        policy: benchmark_metrics(result, workloads)
        for policy, result in results.items()
    }
    comparison = pd.DataFrame(metrics.values())
    outcome_rows = [
        {"policy": policy, **row}
        for policy, result in results.items()
        for row in workload_outcomes(result, workloads)
    ]
    outcomes = pd.DataFrame(outcome_rows)
    resource_rows = resource_metric_rows(results)
    resources = pd.DataFrame(resource_rows)
    fragmentation_rows = fragmentation_metric_rows(results)
    fragmentation = pd.DataFrame(fragmentation_rows)
    failure_rows = [
        {
            "policy": policy,
            "reason": reason,
            "unique_pending_workloads": count,
        }
        for policy, result in results.items()
        for reason, count in result.summary["placement_failures_by_reason"].items()
    ]
    failures = pd.DataFrame(
        failure_rows,
        columns=["policy", "reason", "unique_pending_workloads"],
    )
    exclusions = pd.DataFrame(selection_exclusions or [])
    selection_reasons = {
        "beyond_max_workloads_limit",
        "not_selected_by_fixed_seed_sample",
        "missing_or_invalid_arrival_time",
        "negative_arrival_time",
        "missing_arrival_time_column",
    }
    exclusion_reason_counts = (
        exclusions["reason"].value_counts().to_dict()
        if not exclusions.empty and "reason" in exclusions
        else {}
    )
    selection_exclusion_counts = {
        reason: count
        for reason, count in exclusion_reason_counts.items()
        if reason in selection_reasons
    }
    validation_exclusion_counts = {
        reason: count
        for reason, count in exclusion_reason_counts.items()
        if reason not in selection_reasons
    }

    experiment_config = {
        "experiment_id": output.name,
        "generated_at_utc": generated_at,
        "project_version": PROJECT_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_dataset": input_filename,
        "source_row_count": source_row_count,
        "selected_row_count": selected_row_count,
        "valid_workload_count": len(workloads),
        "selected_workload_ids": [item.workload_id for item in workloads],
        "excluded_record_count": len(selection_exclusions or []),
        "selection_exclusion_count": sum(selection_exclusion_counts.values()),
        "selection_exclusions_by_reason": selection_exclusion_counts,
        "input_validation_exclusions_by_reason": validation_exclusion_counts,
        "all_exclusions_by_reason": exclusion_reason_counts,
        "timestamp_unit_assumption": time_unit,
        "timestamp_policy": "Only valid non-negative observed arrivals are simulated; no missing time values are invented.",
        "cluster_configuration": _cluster_configuration(cluster),
        "role_compatibility": cluster.role_compatibility,
        "density_semantics": {
            "verified": cluster.density_limit_semantics_verified,
            "unlimited_values": cluster.density_unlimited_values,
        },
        "policies": list(results),
        "policy_configurations": {
            policy: result.policy_configuration
            for policy, result in results.items()
        },
        "ifas_configuration": asdict(ifas_config) if ifas_config else None,
        "random_seed": seed,
        "repeats_requested": repeats_requested,
        "repeats_executed": 1,
        "repeat_note": (
            "All included schedulers and the simulator are deterministic; redundant repetitions were skipped."
            if repeats_requested > 1
            else "One deterministic run was requested."
        ),
        "stopping_condition": (
            "Stop when the finite arrival/completion event heap is empty. Permanently unplaceable queued workloads "
            "remain pending and do not trigger retries without a resource-release event."
        ),
        "queue_retry_policy": (
            "Retry pending workloads once after each completion/release event in original pending-queue order."
        ),
        "metric_definitions": {
            "queue_delay": "Simulated arrival to first successful allocation; excludes workloads never allocated and is not historical creation-to-scheduling delay.",
            "peak_allocation_ratio": "Maximum event-snapshot sum of allocated requested resources divided by configured total capacity.",
            "time_weighted_allocation_ratio": "Integrated reserved requests divided by capacity times simulated duration; not hardware utilization.",
            "unique_unplaced_workloads": "Workloads still pending after the event heap is empty.",
            "failed_placement_attempts": "Number of scheduler decisions returning no node, including retries.",
            "scheduler_execution_time": "Monotonic wall-clock time spent inside policy selection, separate from simulated trace time.",
            "fragmentation": "Immediate queue-fit diagnostics at event snapshots; aggregate capacity is resource-wise and does not prove placeability.",
        },
        "limitations": list(cluster.assumptions),
        "sensitivity_results": sensitivity_results or [],
    }
    config_path = output / "experiment_config.json"
    config_path.write_text(
        json.dumps(experiment_config, indent=2, allow_nan=False), encoding="utf-8"
    )
    comparison_path = output / "scheduler_comparison.csv"
    comparison_csv = comparison.copy()
    for column in comparison_csv.columns:
        comparison_csv[column] = comparison_csv[column].map(
            lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
            if isinstance(value, (dict, list))
            else value
        )
    comparison_csv.to_csv(comparison_path, index=False)
    outcomes_path = output / "workload_outcomes.csv"
    outcomes.to_csv(outcomes_path, index=False)
    selected_path = output / "selected_workloads.csv"
    pd.DataFrame(
        [
            {
                "workload_id": workload.workload_id,
                "role": workload.role,
                "app_name": workload.app_name,
                "arrival_time": workload.arrival_time,
                "duration": workload.duration,
                **workload.demand.as_dict(),
            }
            for workload in workloads
        ]
    ).to_csv(selected_path, index=False)
    resource_path = output / "resource_metrics.csv"
    resources.to_csv(resource_path, index=False)
    queue_path = output / "queue_delay_metrics.csv"
    pd.DataFrame(
        [
            {
                "policy": policy,
                **{
                    key: value
                    for key, value in metric.items()
                    if key.startswith("queue_delay_") or key == "pending_queue_count"
                },
            }
            for policy, metric in metrics.items()
        ]
    ).to_csv(queue_path, index=False)
    fragmentation_path = output / "fragmentation_metrics.csv"
    fragmentation.to_csv(fragmentation_path, index=False)
    failures_path = output / "placement_failure_breakdown.csv"
    failures.to_csv(failures_path, index=False)
    exclusions_path = output / "excluded_workloads.csv"
    exclusions.to_csv(exclusions_path, index=False)

    policy_report_paths: list[Path] = []
    for policy, result in results.items():
        policy_report_paths.extend(save_simulation_report(
            result,
            copy.deepcopy(cluster),
            output / policy,
            input_filename=input_filename,
            seed=seed,
            source_row_count=source_row_count,
        ))

    charts_dir = output / "charts"
    charts_dir.mkdir(exist_ok=True)
    charts = _plotly_charts(comparison, resources, fragmentation, failures)
    chart_paths: list[Path] = []
    for name, figure in charts.items():
        chart_path = charts_dir / f"{name}.html"
        figure.write_html(chart_path, include_plotlyjs="cdn", full_html=True)
        chart_paths.append(chart_path)

    sensitivity_paths: list[Path] = []
    if sensitivity_results:
        sensitivity_dir = output / "sensitivity"
        sensitivity_dir.mkdir(exist_ok=True)
        sensitivity_frame = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in scenario.items()
                    if key != "configuration"
                }
                for scenario in sensitivity_results
            ]
        )
        sensitivity_csv = sensitivity_dir / "sensitivity_results.csv"
        sensitivity_frame.to_csv(sensitivity_csv, index=False)
        sensitivity_config = sensitivity_dir / "sensitivity_configurations.json"
        sensitivity_config.write_text(
            json.dumps(sensitivity_results, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        if not sensitivity_frame.empty:
            chart = px.line(
                sensitivity_frame,
                x="scenario",
                y="allocated_workloads",
                color="policy",
                markers=True,
                title="One-factor sensitivity: allocated workloads",
                labels={"allocated_workloads": "Allocated workloads"},
            )
            sensitivity_chart = sensitivity_dir / "sensitivity_allocations.html"
            chart.write_html(
                sensitivity_chart, include_plotlyjs="cdn", full_html=True
            )
            sensitivity_paths.append(sensitivity_chart)
        sensitivity_paths.extend([sensitivity_csv, sensitivity_config])

    log_path = output / "experiment_log.json"
    log_path.write_text(
        json.dumps(
            {
                "experiment_id": output.name,
                "analysis_timestamp_utc": generated_at,
                "policy_metrics": metrics,
                "policy_summaries": {
                    policy: result.summary for policy, result in results.items()
                },
                "outputs": [
                    str(path.relative_to(output))
                    for path in (
                        config_path,
                        comparison_path,
                        outcomes_path,
                        selected_path,
                        resource_path,
                        queue_path,
                        fragmentation_path,
                        failures_path,
                        exclusions_path,
                        *chart_paths,
                        *sensitivity_paths,
                        *(output / "benchmark_report.md",),
                        *policy_report_paths,
                    )
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report_path = output / "benchmark_report.md"
    report_lines = [
        "# GPUlytics scheduler benchmark",
        "",
        f"- Input: `{input_filename}`",
        f"- Source rows: {source_row_count}; selected rows: {selected_row_count}",
        f"- Policies: {', '.join(results)}",
        f"- Cluster: `{cluster.cluster_name}` (configured assumptions retained)",
        f"- Time unit: `{time_unit}`; no conversion performed.",
        f"- Seed: {seed}; schedulers are deterministic.",
        f"- Generated (UTC): {generated_at}",
        "",
        "All policies received identical ordered workload records and each run started with a fresh copy of the configured cluster.",
        "",
        "## Scheduler outcomes",
        "",
        "| Policy | Submitted | Allocated | Completed | Pending | Rejected | Attempts | Failed attempts | Scheduler seconds | Capacity violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| "
            + " | ".join(
                str(comparison.iloc[row_index][column])
                for column in (
                "policy",
                "submitted_workloads",
                "allocated_workloads",
                "completed_workloads",
                "pending_workloads",
                "unique_rejected_workloads",
                "placement_attempts",
                "failed_placement_attempts",
                "scheduler_runtime_seconds",
                "resource_capacity_violations",
                )
            )
            + " |"
            for row_index in range(len(comparison))
        ],
        "",
        "## Metric interpretation",
        "",
        "- Queue-delay values are simulated arrival-to-first-allocation waits, in the configured unverified time unit; they are not historical scheduling-delay measurements.",
        "- Pending workloads remain queued after the finite event heap is empty if no release event can make progress. Rejected inputs are excluded from event submission; retries are attempts, not unique workloads.",
        "- Resource allocation ratios represent requested-resource reservations, not actual utilization.",
        "- Scheduler wall-clock time measures policy evaluation, not simulated workload time.",
        "- Snapshot fragmentation metrics are immediate queue-fit diagnostics, not universal fragmentation measures.",
        "- No statistical significance claim is made. Deterministic runs are not repeated to manufacture confidence intervals.",
        "",
        "## Pending workloads and placement failures",
        "",
        "- A pending workload entered the simulator but had no successful placement by the time the finite event heap drained.",
        "- A rejected workload failed source-row or workload validation and was never submitted to the scheduler.",
        "- Failed placement attempts include retries; the following breakdown counts unique pending workloads by their final recorded reason.",
        "",
        *[
            f"### {policy.replace('_', ' ').title()}"
            + (
                "\n"
                + "\n".join(
                    f"- `{reason}`: {count} unique pending workload(s)"
                    for reason, count in result.summary[
                        "placement_failures_by_reason"
                    ].items()
                )
                if result.summary["placement_failures_by_reason"]
                else "\n- No workloads remained pending due to placement failures."
            )
            for policy, result in results.items()
        ],
        "",
        "These reasons are evaluated under the configured capacity and role-compatibility rules. `insufficient_capacity:<resource>` means the workload's request exceeded the remaining capacity of otherwise compatible nodes; `role_incompatible` means a node did not match the configured role mapping. When both are listed, the cluster had no eligible node satisfying the full resource vector. This does not by itself establish that IFAS or another policy is better.",
        "",
        "## Resource invariant check",
        "",
        *[
            f"- {policy.replace('_', ' ').title()}: {int(result.summary['resource_capacity_violation_count'])} capacity violations."
            for policy, result in results.items()
        ],
        "- Every run uses atomic resource accounting; a capacity overrun raises instead of being counted as a successful placement.",
        "",
        "## Exclusions",
        "",
        f"- Records in the selected workload subset: {selected_row_count}",
        f"- Valid workloads submitted: {len(workloads)}",
        f"- Row-level excluded records recorded: {len(selection_exclusions or [])}",
        *[
            f"- `{reason}`: {count}"
            for reason, count in sorted(exclusion_reason_counts.items())
        ],
        "",
        "## Dataset and configuration limitations",
        "",
        *[f"- {item}" for item in cluster.assumptions],
        "- Invalid or missing arrival timestamps are recorded as exclusions; no missing event times are fabricated.",
        "- Queue drains only through resource release events. Permanently unplaceable workloads remain pending without infinite retries.",
        "",
        "## Output files",
        "",
        *[
            f"- `{name}`"
            for name in (
                "experiment_config.json",
                "scheduler_comparison.csv",
                "workload_outcomes.csv",
                "selected_workloads.csv",
                "resource_metrics.csv",
                "queue_delay_metrics.csv",
                "fragmentation_metrics.csv",
                "placement_failure_breakdown.csv",
                "excluded_workloads.csv",
                "experiment_log.json",
            )
        ],
        "",
        "Interactive HTML charts are saved under `charts/`. Per-policy simulation artifacts are saved in policy-named subdirectories.",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return [
        config_path,
        comparison_path,
        outcomes_path,
        selected_path,
        resource_path,
        queue_path,
        fragmentation_path,
        failures_path,
        exclusions_path,
        log_path,
        report_path,
        *chart_paths,
        *sensitivity_paths,
    ]


def _capacity_variant(
    cluster: ClusterState, resource: str, factor: float, *, gpu_only: bool = False
) -> ClusterState:
    variant = copy.deepcopy(cluster)
    for node in variant.nodes.values():
        if gpu_only and node.capacity.gpu_devices <= 0:
            continue
        capacity_values = node.capacity.as_dict()
        if resource == "gpu_devices":
            capacity_values[resource] = max(
                0, int(round(float(capacity_values[resource]) * factor))
            )
        else:
            capacity_values[resource] = float(capacity_values[resource]) * factor
        node.capacity = ResourceCapacity(
            cpu_units=float(capacity_values["cpu_units"]),
            gpu_devices=int(capacity_values["gpu_devices"]),
            memory_units=float(capacity_values["memory_units"]),
            rdma_units=float(capacity_values["rdma_units"]),
            disk_units=float(capacity_values["disk_units"]),
            extra={
                name: float(value)
                for name, value in capacity_values.items()
                if name not in RESOURCE_NAMES
            },
        )
    variant.assumptions.append(
        f"Sensitivity scenario changes {resource} capacity by factor {factor} only."
    )
    return variant


def _add_gpu_node(cluster: ClusterState) -> ClusterState:
    variant = copy.deepcopy(cluster)
    source = next(
        (node for node in variant.nodes.values() if node.capacity.gpu_devices > 0),
        None,
    )
    if source is None:
        return variant
    ordinal = 1
    while f"sensitivity-gpu-node-{ordinal:02d}" in variant.nodes:
        ordinal += 1
    variant.nodes[f"sensitivity-gpu-node-{ordinal:02d}"] = SimulatedNode(
        node_id=f"sensitivity-gpu-node-{ordinal:02d}",
        role=source.role,
        capacity=copy.deepcopy(source.capacity),
    )
    variant.assumptions.append(
        "Sensitivity scenario adds one GPU node cloned from an existing hypothetical GPU node."
    )
    return variant


def run_sensitivity_analysis(
    cluster: ClusterState,
    workloads: list[SimulatedWorkload],
    *,
    policies: Iterable[str] = SUPPORTED_POLICIES,
    rejected_inputs: list[dict[str, object]] | None = None,
    time_unit: str = "trace_units_unverified",
    ifas_config: IFASConfig | None = None,
) -> list[dict[str, Any]]:
    """Run one-factor scenarios; never combine capacity and policy changes."""
    selected_policies = tuple(policies)
    scenarios: list[tuple[str, ClusterState, IFASConfig | None, str]] = [
        (
            f"{resource}_capacity_{factor:g}x",
            _capacity_variant(
                cluster,
                resource,
                factor,
                gpu_only=resource == "gpu_devices",
            ),
            ifas_config,
            f"Only {resource} capacity is scaled by {factor}.",
        )
        for resource in ("gpu_devices", "cpu_units", "memory_units")
        for factor in (0.75, 1.25)
    ]
    if any(node.capacity.gpu_devices > 0 for node in cluster.nodes.values()):
        scenarios.append(
            (
                "gpu_node_count_plus_1",
                _add_gpu_node(cluster),
                ifas_config,
                "One GPU node is added using an existing configured GPU-node capacity.",
            )
        )
    if (
        "ifas" in selected_policies
        and ifas_config is not None
    ):
        for weight_name in (
            "balance_weight",
            "fragmentation_weight",
            "spread_weight",
            "residual_fit_weight",
        ):
            updated = asdict(ifas_config)
            updated[weight_name] = float(updated[weight_name]) + 0.1
            changed = IFASConfig(**updated)
            scenarios.append(
                (
                    f"ifas_{weight_name}_plus_0.1",
                    copy.deepcopy(cluster),
                    changed,
                    f"Only IFAS {weight_name} increases by 0.1.",
                )
            )
    positive_density_present = any(
        workload.max_instance_per_node is not None
        and workload.max_instance_per_node > 0
        for workload in workloads
    )
    if positive_density_present and not cluster.density_limit_semantics_verified:
        density_variant = copy.deepcopy(cluster)
        density_variant.density_limit_semantics_verified = True
        density_variant.assumptions.append(
            "Sensitivity-only assumption: positive max_instance_per_node values are enforced as per-node application caps; negative sentinel semantics remain unchanged."
        )
        scenarios.append(
            (
                "positive_density_caps_enforced_assumption",
                density_variant,
                ifas_config,
                "Only positive density-cap enforcement is toggled as an explicit sensitivity assumption.",
            )
        )

    rows: list[dict[str, Any]] = []
    for name, variant, scenario_ifas, description in scenarios:
        scenario_results = run_experiment(
            variant,
            workloads,
            policies=selected_policies,
            rejected_inputs=rejected_inputs,
            time_unit=time_unit,
            ifas_config=scenario_ifas,
        )
        configuration = {
            "scenario": name,
            "description": description,
            "cluster_configuration": _cluster_configuration(variant),
            "ifas_configuration": (
                asdict(scenario_ifas) if scenario_ifas is not None else None
            ),
            "workload_ids": [workload.workload_id for workload in workloads],
            "time_unit": time_unit,
        }
        for policy_name, result in scenario_results.items():
            metric = benchmark_metrics(result, workloads)
            rows.append(
                {
                    "scenario": name,
                    "policy": policy_name,
                    "allocated_workloads": metric["allocated_workloads"],
                    "pending_workloads": metric["pending_workloads"],
                    "completed_workloads": metric["completed_workloads"],
                    "scheduler_runtime_seconds": metric["scheduler_runtime_seconds"],
                    "resource_capacity_violations": metric[
                        "resource_capacity_violations"
                    ],
                    "configuration": configuration,
                }
            )
    arrival_groups: dict[float, list[SimulatedWorkload]] = {}
    for workload in workloads:
        if workload.arrival_time is not None:
            arrival_groups.setdefault(workload.arrival_time, []).append(workload)
    if any(len(group) > 1 for group in arrival_groups.values()):
        tie_reversed = [
            workload
            for arrival in sorted(arrival_groups)
            for workload in reversed(arrival_groups[arrival])
        ]
        scenario_results = run_experiment(
            cluster,
            tie_reversed,
            policies=selected_policies,
            rejected_inputs=rejected_inputs,
            time_unit=time_unit,
            ifas_config=ifas_config,
        )
        configuration = {
            "scenario": "arrival_tie_order_reversed",
            "description": (
                "Only source order within equal-arrival-time groups is reversed; "
                "observed arrival times and all other inputs are unchanged."
            ),
            "cluster_configuration": _cluster_configuration(cluster),
            "ifas_configuration": (
                asdict(ifas_config) if ifas_config is not None else None
            ),
            "workload_ids": [workload.workload_id for workload in tie_reversed],
            "time_unit": time_unit,
        }
        for policy_name, result in scenario_results.items():
            metric = benchmark_metrics(result, tie_reversed)
            rows.append(
                {
                    "scenario": "arrival_tie_order_reversed",
                    "policy": policy_name,
                    "allocated_workloads": metric["allocated_workloads"],
                    "pending_workloads": metric["pending_workloads"],
                    "completed_workloads": metric["completed_workloads"],
                    "scheduler_runtime_seconds": metric["scheduler_runtime_seconds"],
                    "resource_capacity_violations": metric[
                        "resource_capacity_violations"
                    ],
                    "configuration": configuration,
                }
            )
    return rows


def selection_exclusions(
    frame: pd.DataFrame, selected_positions: set[int], reason: str
) -> list[dict[str, Any]]:
    """List source rows excluded by deterministic subset selection."""
    identifier_column = "instance_sn" if "instance_sn" in frame else None
    excluded: list[dict[str, Any]] = []
    for position in range(len(frame)):
        if position in selected_positions:
            continue
        identifier = (
            frame.iloc[position][identifier_column]
            if identifier_column
            else None
        )
        excluded.append(
            {
                "source_row": position,
                "workload_id": (
                    str(identifier)
                    if identifier is not None and pd.notna(identifier)
                    else f"row-{position}"
                ),
                "reason": reason,
            }
        )
    return excluded


def select_evaluation_rows(
    frame: pd.DataFrame,
    *,
    max_workloads: int | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Select valid-arrival rows reproducibly and report every omitted source row."""
    if max_workloads is not None and max_workloads < 1:
        raise ValueError("max_workloads must be positive")
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive")
    if max_workloads is not None and sample_size is not None:
        raise ValueError("Choose either max_workloads or sample_size, not both")
    arrival_column = (
        "relative_arrival_time_raw_units"
        if "relative_arrival_time_raw_units" in frame.columns
        else "creation_time"
        if "creation_time" in frame.columns
        else None
    )
    if arrival_column is None:
        return frame.iloc[0:0].copy(), [
            {
                "source_row": index,
                "workload_id": str(
                    frame.iloc[index].get("instance_sn", f"row-{index}")
                ),
                "reason": "missing_arrival_time_column",
            }
            for index in range(len(frame))
        ]
    arrivals = pd.to_numeric(frame[arrival_column], errors="coerce")
    if arrival_column == "creation_time":
        arrivals = arrivals - arrivals.min()
    valid = arrivals.notna() & arrivals.map(math.isfinite) & (arrivals >= 0)
    valid_positions = [
        index for index, is_valid in enumerate(valid.tolist()) if is_valid
    ]
    invalid_positions = set(range(len(frame))) - set(valid_positions)
    invalid_exclusions = [
        {
            "source_row": index,
            "workload_id": str(frame.iloc[index].get("instance_sn", f"row-{index}")),
            "reason": (
                "negative_arrival_time"
                if pd.notna(arrivals.iloc[index]) and arrivals.iloc[index] < 0
                else "missing_or_invalid_arrival_time"
            ),
        }
        for index in sorted(invalid_positions)
    ]
    valid_frame = pd.DataFrame(
        {"position": valid_positions, "arrival": arrivals.iloc[valid_positions].to_numpy()}
    )
    if sample_size is not None:
        chosen = valid_frame.sample(
            n=min(sample_size, len(valid_frame)), random_state=seed
        )
    else:
        chosen = valid_frame.sort_values(
            ["arrival", "position"], kind="stable"
        )
        if max_workloads is not None:
            chosen = chosen.head(max_workloads)
    chosen_positions = set(chosen["position"].astype(int).tolist())
    selected = frame.iloc[sorted(chosen_positions)].copy()
    selected["__benchmark_source_row"] = sorted(chosen_positions)
    selected["__benchmark_arrival"] = arrivals.iloc[sorted(chosen_positions)].to_numpy()
    selected = selected.sort_values(
        ["__benchmark_arrival"], kind="stable"
    ).drop(columns=["__benchmark_arrival"])
    unselected_valid = set(valid_positions) - chosen_positions
    selection_reason = (
        "not_selected_by_fixed_seed_sample"
        if sample_size is not None
        else "beyond_max_workloads_limit"
        if max_workloads is not None
        else "not_selected"
    )
    omitted = invalid_exclusions + selection_exclusions(
        frame,
        chosen_positions | invalid_positions,
        selection_reason,
    )
    if not unselected_valid and max_workloads is None and sample_size is None:
        omitted = invalid_exclusions
    return selected, omitted
