"""Reproducible residual-capacity fragmentation metrics."""

from __future__ import annotations

from typing import Any

from src.cluster_models import ClusterState, SimulatedWorkload

RESOURCE_NAMES = ("cpu_units", "gpu_devices", "memory_units", "rdma_units", "disk_units")


def _node_residual(cluster: ClusterState, node_id: str) -> dict[str, float]:
    node = cluster.nodes[node_id]
    capacity = node.capacity.as_dict()
    allocated = node.allocated.as_dict()
    return {
        name: float(value) - float(allocated.get(name, 0.0))
        for name, value in capacity.items()
    }


def _fits(workload: SimulatedWorkload, residual: dict[str, float]) -> bool:
    return all(
        float(demand) <= residual.get(name, 0.0) + 1e-9
        for name, demand in workload.demand.as_dict().items()
    )


def fragmentation_metrics(
    cluster: ClusterState,
    pending_workloads: list[SimulatedWorkload],
) -> dict[str, Any]:
    """Measure queued-workload stranded capacity without assuming future arrivals.

    The representative GPU workload is the earliest currently queued GPU job,
    with workload ID as deterministic tie-break. Aggregate multi-resource
    fragmentation counts queued workloads whose per-resource aggregate capacity
    is sufficient on role-compatible nodes but which fit on no single node.
    Aggregate sufficiency is only a diagnostic, not proof that a placement is
    valid; this resource-only metric does not account for density caps.
    """
    gpu_nodes = [
        node_id
        for node_id, node in cluster.nodes.items()
        if float(node.capacity.gpu_devices) > 0
    ]
    residuals = {node_id: _node_residual(cluster, node_id) for node_id in cluster.nodes}
    residual_gpu_slots = sum(
        max(0.0, residuals[node_id].get("gpu_devices", 0.0))
        for node_id in gpu_nodes
    )
    queued_gpu = sorted(
        (
            workload
            for workload in pending_workloads
            if workload.demand.gpu_devices > 0
        ),
        key=lambda workload: (
            workload.arrival_time if workload.arrival_time is not None else float("inf"),
            workload.workload_id,
        ),
    )
    representative = queued_gpu[0] if queued_gpu else None
    stranded_gpu_slots = 0.0
    if representative is not None:
        for node_id in gpu_nodes:
            node = cluster.nodes[node_id]
            role_compatible = node.role in cluster.role_compatibility.get(
                representative.role, []
            )
            if not role_compatible or not _fits(representative, residuals[node_id]):
                stranded_gpu_slots += max(
                    0.0, residuals[node_id].get("gpu_devices", 0.0)
                )

    aggregate_capacity_blocked = 0
    aggregate_capacity_insufficient = 0
    role_compatible_pending_count = 0
    gpu_role_compatible_count = 0
    gpu_no_single_node_fit = 0
    gpu_aggregate_insufficient = 0
    gpu_fragmentation_blocked = 0
    for workload in pending_workloads:
        compatible_nodes = [
            node_id
            for node_id, node in cluster.nodes.items()
            if node.role in cluster.role_compatibility.get(workload.role, [])
        ]
        if not compatible_nodes:
            continue
        role_compatible_pending_count += 1
        is_gpu_workload = workload.demand.gpu_devices > 0
        if is_gpu_workload:
            gpu_role_compatible_count += 1
        aggregate_residual = {
            name: sum(residuals[node_id].get(name, 0.0) for node_id in compatible_nodes)
            for name in set().union(
                *(set(cluster.nodes[node_id].capacity.as_dict()) for node_id in compatible_nodes)
            )
        }
        aggregate_resource_fit = _fits(workload, aggregate_residual)
        individual_resource_fit = any(
            _fits(workload, residuals[node_id]) for node_id in compatible_nodes
        )
        if not individual_resource_fit:
            if is_gpu_workload:
                gpu_no_single_node_fit += 1
            if aggregate_resource_fit:
                aggregate_capacity_blocked += 1
                if is_gpu_workload:
                    gpu_fragmentation_blocked += 1
            else:
                aggregate_capacity_insufficient += 1
                if is_gpu_workload:
                    gpu_aggregate_insufficient += 1

    return {
        "pending_workload_count": len(pending_workloads),
        "queued_gpu_workload_count": len(queued_gpu),
        "gpu_workloads_with_compatible_nodes": gpu_role_compatible_count,
        "gpu_workloads_without_single_node_fit": gpu_no_single_node_fit,
        "gpu_workloads_fragmentation_blocked": gpu_fragmentation_blocked,
        "gpu_workloads_aggregate_capacity_insufficient": gpu_aggregate_insufficient,
        "gpu_residual_fit_fragmentation_fraction": (
            gpu_no_single_node_fit / gpu_role_compatible_count
            if gpu_role_compatible_count
            else None
        ),
        "gpu_fragmentation_blocked_fraction": (
            gpu_fragmentation_blocked / gpu_role_compatible_count
            if gpu_role_compatible_count
            else None
        ),
        "representative_gpu_workload_id": (
            representative.workload_id if representative is not None else None
        ),
        "residual_gpu_slots": residual_gpu_slots,
        "stranded_gpu_slots_for_representative": stranded_gpu_slots,
        "stranded_gpu_slot_fraction": (
            stranded_gpu_slots / residual_gpu_slots if residual_gpu_slots > 0 else None
        ),
        "role_compatible_pending_workload_count": role_compatible_pending_count,
        "aggregate_capacity_sufficient_but_no_single_node_count": aggregate_capacity_blocked,
        "aggregate_capacity_insufficient_count": aggregate_capacity_insufficient,
        "aggregate_fragmentation_fraction": (
            aggregate_capacity_blocked / role_compatible_pending_count
            if role_compatible_pending_count
            else None
        ),
        "active_node_count": sum(bool(node.allocations) for node in cluster.nodes.values()),
    }
