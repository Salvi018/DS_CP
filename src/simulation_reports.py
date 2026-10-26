"""Persist deterministic simulation outputs and their assumptions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cluster_models import ClusterState
from src.simulator import SimulationResult


def save_simulation_report(
    result: SimulationResult,
    cluster: ClusterState,
    output_dir: str | Path,
    *,
    input_filename: str,
    seed: int,
    source_row_count: int,
) -> list[Path]:
    """Write summary, event, resource-history, and Markdown report files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        **result.summary,
        "input_filename": input_filename,
        "source_row_count": source_row_count,
        "analysis_timestamp_utc": generated_at,
        "random_seed": seed,
        "time_unit_assumption": result.time_unit,
        "policy": result.policy_name,
        "policy_configuration": result.policy_configuration,
        "cluster_assumptions": result.assumptions,
        "node_capacities": {
            node.node_id: {
                "role": node.role,
                "capacity": node.capacity.as_dict(),
            }
            for node in cluster.nodes.values()
        },
    }
    summary_path = output / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    events_frame = pd.DataFrame(
        result.events,
        columns=[
            "time",
            "event",
            "workload_id",
            "node_id",
            "reason",
            "queue_length",
            "decision_diagnostics",
        ],
    )
    events_path = output / "simulation_events.csv"
    if "decision_diagnostics" in events_frame:
        events_frame["decision_diagnostics"] = events_frame[
            "decision_diagnostics"
        ].map(lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
    events_frame.to_csv(events_path, index=False)
    history_path = output / "resource_history.csv"
    history_frame = pd.DataFrame(result.resource_history)
    for column in (
        "allocated_requests",
        "configured_capacity",
        "allocation_ratios",
        "fragmentation",
    ):
        if column in history_frame:
            history_frame[column] = history_frame[column].map(
                lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
            )
    history_frame.to_csv(history_path, index=False)

    report_path = output / "simulation_report.md"
    lines = [
        "# GPUlytics Cluster Simulation",
        "",
        f"- Input: `{input_filename}`",
        f"- Source rows considered: {source_row_count}",
        f"- Analysis timestamp (UTC): {generated_at}",
        f"- Cluster: {cluster.cluster_name}",
        f"- Policy interface: `{result.policy_name}`",
        f"- Policy configuration: `{json.dumps(result.policy_configuration, sort_keys=True)}`",
        f"- Simulation time unit: `{result.time_unit}` (no unit conversion performed)",
        f"- Random seed recorded: {seed}; this policy does not use randomness.",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "accepted_workload_count",
        "allocated_workload_count",
        "completed_workload_count",
        "pending_workload_count",
        "rejected_workload_count",
        "queued_workload_count",
        "simulated_duration",
        "final_active_workload_count",
    ):
        lines.append(f"| {key.replace('_', ' ').title()} | {summary.get(key)} |")
    lines.extend(
        [
            "",
            "### Time-weighted allocation ratios",
            "",
            "These ratios describe requested resources reserved by the simulator, not measured hardware utilization.",
            "",
            "| Resource | Ratio |",
            "|---|---:|",
        ]
    )
    ratios = result.summary.get("time_weighted_allocation_ratios", {})
    if isinstance(ratios, dict):
        for name, value in ratios.items():
            lines.append(f"| {name} | {value if value is not None else 'N/A (zero capacity or zero duration)'} |")
    lines.extend(["", "## Node capacities", ""])
    for node in cluster.nodes.values():
        lines.append(f"- `{node.node_id}` ({node.role}): `{json.dumps(node.capacity.as_dict(), sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Assumptions and limitations",
            "",
            *[f"- {assumption}" for assumption in result.assumptions],
            "- Only workloads with valid observed resource requests and arrival times enter the event queue.",
            "- Missing observed lifetimes are treated as censored workloads with no fabricated completion event.",
            "- Negative or unverified negative density limits are rejected unless a sentinel is explicitly configured.",
            "- First Fit uses ascending node-ID order; Best Fit minimizes normalized residual capacity and breaks ties by node ID.",
            "- IFAS score components are normalized heuristic estimates. Queued-workload fit terms use only the currently visible queue; metrics are not global optimality guarantees.",
            "- The legacy unique-feasible-node policy remains available to callers that omit a scheduler; it queues ambiguous placements.",
            "- IFAS is a deterministic heuristic; it is not globally optimal and does not model GPU architecture, topology, or actual utilization.",
            "- No machine-learning prediction is implemented.",
            "- A simulation completion means the workload's observed scheduled lifetime elapsed; it is not a real-world completion forecast.",
            "",
            "## Output files",
            "",
            "- `simulation_summary.json`",
            "- `simulation_events.csv`",
            "- `resource_history.csv`",
            "- `simulation_report.md`",
            "- `scheduler_decisions.csv` (IFAS placement score diagnostics when enabled)",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    decisions_path = output / "scheduler_decisions.csv"
    decision_rows = [
        {
            "event": event.get("event"),
            "time": event["time"],
            "workload_id": event["workload_id"],
            "selected_node_id": event["node_id"],
            "reason": event.get("reason"),
            "decision_diagnostics": json.dumps(
                event.get("decision_diagnostics", {}), sort_keys=True, allow_nan=False
            ),
        }
        for event in result.events
        if event.get("decision_diagnostics")
    ]
    pd.DataFrame(
        decision_rows,
        columns=[
            "event",
            "time",
            "workload_id",
            "selected_node_id",
            "reason",
            "decision_diagnostics",
        ],
    ).to_csv(decisions_path, index=False)
    return [summary_path, events_path, history_path, report_path, decisions_path]
