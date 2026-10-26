"""Run and report controlled First Fit, Best Fit, and IFAS comparisons."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.cluster_models import ClusterState, SimulatedWorkload
from src.schedulers import (
    BestFitScheduler,
    FirstFitScheduler,
    IFASConfig,
    IntelligentFragmentationAwareScheduler,
)
from src.simulation_reports import save_simulation_report
from src.simulator import SimulationResult, run_simulation


def compare_schedulers(
    cluster: ClusterState,
    workloads: list[SimulatedWorkload],
    *,
    rejected_inputs: list[dict[str, object]] | None = None,
    time_unit: str = "trace_units_unverified",
    ifas_config: IFASConfig | None = None,
) -> dict[str, SimulationResult]:
    """Run all policies with cloned initial cluster state and identical inputs."""
    results: dict[str, SimulationResult] = {}
    for scheduler in (
        FirstFitScheduler(),
        BestFitScheduler(),
        IntelligentFragmentationAwareScheduler(ifas_config),
    ):
        policy_cluster = copy.deepcopy(cluster)
        results[scheduler.name] = run_simulation(
            policy_cluster,
            workloads,
            rejected_inputs=copy.deepcopy(rejected_inputs or []),
            policy=scheduler,
            time_unit=time_unit,
        )
    return results


def comparison_metrics(result: SimulationResult) -> dict[str, Any]:
    """Select comparable counters and metric definitions from a simulation."""
    summary = result.summary
    queue_delay = summary["queue_delay"]
    return {
        "policy": result.policy_name,
        "submitted_workloads": summary["accepted_workload_count"],
        "rejected_input_workloads": summary["input_rejected_workload_count"],
        "allocated_workloads": summary["allocated_workload_count"],
        "completed_workloads": summary["completed_workload_count"],
        "pending_workloads": summary["pending_workload_count"],
        "ever_queued_workloads": summary["queued_workload_count"],
        "unique_placement_failures": summary["unique_placement_failure_count"],
        "placement_failures_by_reason": summary["placement_failures_by_reason"],
        "placement_attempts": summary["placement_attempt_count"],
        "failed_placement_attempts": summary["placement_attempt_failure_count"],
        "queue_retries": summary["queue_retry_count"],
        "allocation_ratios": summary["time_weighted_allocation_ratios"],
        "average_queue_delay": queue_delay["mean"],
        "p95_queue_delay": queue_delay["p95"],
        "queue_delay_observation_count": queue_delay["count"],
        "scheduler_runtime_seconds": summary["scheduler_runtime_seconds"],
        "resource_capacity_violations": summary["resource_capacity_violation_count"],
        "time_weighted_fragmentation_metrics": summary[
            "time_weighted_fragmentation_metrics"
        ],
        "average_active_node_count": summary["average_active_node_count"],
        "max_active_node_count": summary["max_active_node_count"],
        "scheduler_decision_count": sum(
            bool(event.get("decision_diagnostics"))
            for event in result.events
        ),
    }


def save_comparison_report(
    results: dict[str, SimulationResult],
    cluster: ClusterState,
    output_dir: str | Path,
    *,
    input_filename: str,
    seed: int,
    source_row_count: int,
    time_unit: str,
    ifas_config: IFASConfig | None = None,
) -> list[Path]:
    """Save individual run artifacts and a side-by-side comparison report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for policy_name, result in results.items():
        save_simulation_report(
            result,
            cluster,
            output / policy_name,
            input_filename=input_filename,
            seed=seed,
            source_row_count=source_row_count,
        )
    metrics = {
        policy_name: comparison_metrics(result)
        for policy_name, result in results.items()
    }
    rejected_rows = next(iter(results.values())).summary.get("rejected_inputs", [])
    input_exclusion_counts = Counter(
        reason
        for entry in rejected_rows
        for reason in entry.get("reasons", [])
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "analysis_timestamp_utc": generated_at,
        "input_filename": input_filename,
        "source_row_count": source_row_count,
        "cluster_name": cluster.cluster_name,
        "cluster_configuration": {
            node_id: {"role": node.role, "capacity": node.capacity.as_dict()}
            for node_id, node in cluster.nodes.items()
        },
        "time_unit": time_unit,
        "seed": seed,
        "assumptions": cluster.assumptions,
        "input_exclusions_by_reason": dict(sorted(input_exclusion_counts.items())),
        "comparison_controls": [
            "All policies receive the same ordered workload list and rejected input rows.",
            "Each policy starts from a deep copy of the same unallocated cluster state.",
            "Resource constraints, role compatibility, density semantics, and event ordering are unchanged.",
        ],
        "metric_definitions": {
            "submitted_workloads": "Workloads passing input arrival/duration validation and entering the event queue.",
            "unique_placement_failures": "Workloads still pending at end of simulation.",
            "failed_placement_attempts": "All policy decisions that returned no node, including retries.",
            "allocation_ratios": "Time-weighted requested-resource reservations divided by capacity times simulated duration; not hardware utilization.",
            "queue_delay": "Arrival to first successful allocation, only for workloads allocated during the run.",
            "scheduler_runtime_seconds": "Wall-clock time spent evaluating policy choices; excludes event processing and IO.",
            "resource_capacity_violations": "Invariant violations; atomic accounting raises rather than continuing if one occurs.",
        },
        "policy_configurations": {
            policy_name: result.policy_configuration
            for policy_name, result in results.items()
        },
        "ifas_configuration": (
            ifas_config.__dict__ if ifas_config is not None else None
        ),
        "results": metrics,
    }
    json_path = output / "comparison_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = output / "comparison_metrics.csv"
    flat_rows = []
    for metric_values in metrics.values():
        row = {
            key: json.dumps(value, sort_keys=True, allow_nan=False)
            if isinstance(value, (dict, list))
            else value
            for key, value in metric_values.items()
        }
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    md_path = output / "comparison_report.md"
    lines = [
        "# First Fit, Best Fit, and IFAS Simulation Comparison",
        "",
        f"- Input: `{input_filename}`",
        f"- Source rows: {source_row_count}",
        f"- Cluster: `{cluster.cluster_name}`",
        f"- Time unit: `{time_unit}`; values were not converted.",
        f"- Random seed metadata: {seed} (these policies are deterministic).",
        f"- Generated (UTC): {generated_at}",
        "",
        "All runs used the same workload ordering, cluster configuration, resource semantics, and time unit. Each began with a fresh cluster state.",
        "",
        "| Metric | First Fit | Best Fit | IFAS |",
        "|---|---:|---:|---:|",
    ]
    ordered_metrics = [
        ("submitted_workloads", "Submitted workloads"),
        ("rejected_input_workloads", "Rejected input workloads"),
        ("allocated_workloads", "Allocated workloads"),
        ("completed_workloads", "Completed workloads"),
        ("pending_workloads", "Pending workloads"),
        ("unique_placement_failures", "Unique unresolved placement failures"),
        ("placement_attempts", "Placement attempts"),
        ("failed_placement_attempts", "Failed placement attempts"),
        ("queue_retries", "Queue retries"),
        ("average_queue_delay", "Average queue delay"),
        ("p95_queue_delay", "P95 queue delay"),
        ("queue_delay_observation_count", "Queue delay observations"),
        ("scheduler_runtime_seconds", "Scheduler runtime (seconds)"),
        ("average_active_node_count", "Time-weighted active nodes (mean)"),
        ("max_active_node_count", "Peak active nodes"),
        ("resource_capacity_violations", "Resource capacity violations"),
        ("scheduler_decision_count", "Recorded decisions with diagnostics"),
    ]
    for key, label in ordered_metrics:
        values = [metrics[name].get(key) for name in ("first_fit", "best_fit", "ifas")]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "## Resource allocation ratios",
            "",
            "Ratios represent requested-resource reservations over simulated time, not measured utilization.",
            "",
            "| Resource | First Fit | Best Fit | IFAS |",
            "|---|---:|---:|---:|",
        ]
    )
    first_ratios = metrics.get("first_fit", {}).get("allocation_ratios", {})
    best_ratios = metrics.get("best_fit", {}).get("allocation_ratios", {})
    ifas_ratios = metrics.get("ifas", {}).get("allocation_ratios", {})
    resource_names = sorted(set(first_ratios) | set(best_ratios) | set(ifas_ratios))
    for name in resource_names:
        lines.append(
            f"| {name} | {first_ratios.get(name)} | {best_ratios.get(name)} | "
            f"{ifas_ratios.get(name)} |"
        )
    lines.extend(
        [
            "",
            "## Time-weighted fragmentation and active-node metrics",
            "",
            "GPU stranded-slot fraction uses the earliest queued GPU workload; aggregate fragmentation is the fraction of role-compatible queued workloads with sufficient aggregate residual resources but no single-node fit. Both are time-weighted over snapshots where defined.",
            "",
            "| Metric | First Fit | Best Fit | IFAS |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("stranded_gpu_slot_fraction", "Stranded GPU slot fraction"),
        ("aggregate_fragmentation_fraction", "Aggregate multi-resource fragmentation fraction"),
    ):
        values = [
            metrics[name]["time_weighted_fragmentation_metrics"].get(key)
            for name in ("first_fit", "best_fit", "ifas")
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(["", "## Unique unresolved placement failures", ""])
    for policy_name in ("first_fit", "best_fit", "ifas"):
        failures = metrics.get(policy_name, {}).get("placement_failures_by_reason", {})
        lines.append(f"### {policy_name.replace('_', ' ').title()}")
        lines.append("")
        if failures:
            lines.extend(f"- `{reason}`: {count}" for reason, count in failures.items())
        else:
            lines.append("- None")
        lines.append("")
    lines.extend(
        [
            "## Input exclusions",
            "",
            "These are invalid or unverified source rows excluded before scheduler placement; they are not placement failures.",
            "",
        ]
    )
    if input_exclusion_counts:
        lines.extend(
            f"- `{reason}`: {count} rows"
            for reason, count in sorted(input_exclusion_counts.items())
        )
    else:
        lines.append("- None")
    lines.append("")
    lines.extend(
        [
            "## Metric definitions and limitations",
            "",
            *[f"- **{name.replace('_', ' ')}:** {definition}" for name, definition in payload["metric_definitions"].items()],
            "- First Fit uses ascending node-ID order.",
            "- Best Fit minimizes the sum of post-placement residual/capacity ratios over positive-capacity dimensions; ties use node ID. It is a baseline heuristic, not globally optimal.",
            "- IFAS score weights, limitations, and per-placement diagnostic components are included in policy configuration and scheduler decision outputs.",
            "- Hypothetical capacity and role compatibility assumptions are not claims about Alibaba hardware or placement rules.",
            "- Missing lifetimes do not produce completion events.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [json_path, csv_path, md_path]
