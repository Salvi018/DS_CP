"""Atomic capacity accounting and allocation invariants."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.cluster_models import ActiveAllocation, ClusterState, SimulatedWorkload


class ResourceAccountingError(ValueError):
    """Raised when a resource allocation or release is invalid."""


@dataclass(frozen=True)
class PlacementCheck:
    feasible: bool
    reason: str


def _validate_demand(workload: SimulatedWorkload, cluster: ClusterState) -> str | None:
    demand = workload.demand.as_dict()
    if not workload.workload_id:
        return "missing_workload_id"
    for name, value in demand.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"invalid_demand:{name}"
        if not math.isfinite(float(value)) or value < 0:
            return f"invalid_demand:{name}"
    if not isinstance(demand["gpu_devices"], int):
        return "gpu_demand_not_integer"
    if workload.max_instance_per_node is not None:
        limit = workload.max_instance_per_node
        if isinstance(limit, bool) or not isinstance(limit, int):
            return "invalid_density_limit"
        if limit in cluster.density_unlimited_values:
            pass
        elif limit < 0:
            return "invalid_density_limit"
        elif cluster.density_limit_semantics_verified and limit == 0:
            return "invalid_density_limit"
        elif cluster.density_limit_semantics_verified and not workload.app_name:
            return "missing_application_for_density_limit"
    return None


def check_invariants(cluster: ClusterState) -> None:
    """Raise if allocations are duplicated, negative, or exceed node capacity."""
    workload_ids: set[str] = set()
    for node in cluster.nodes.values():
        allocated = node.allocated.as_dict()
        capacity = node.capacity.as_dict()
        for name, amount in allocated.items():
            if not math.isfinite(float(amount)) or amount < -1e-9:
                raise ResourceAccountingError(f"Invalid allocated amount for {node.node_id}.{name}")
            if float(amount) > float(capacity.get(name, 0.0)) + 1e-9:
                raise ResourceAccountingError(f"Capacity invariant violated for {node.node_id}.{name}")
        for workload_id, allocation in node.allocations.items():
            if workload_id != allocation.workload.workload_id or workload_id in workload_ids:
                raise ResourceAccountingError(f"Duplicate or inconsistent allocation: {workload_id}")
            workload_ids.add(workload_id)


def can_allocate(
    cluster: ClusterState, node_id: str, workload: SimulatedWorkload
) -> PlacementCheck:
    """Check role, density, and all resource dimensions without mutating state."""
    node = cluster.nodes.get(node_id)
    if node is None:
        return PlacementCheck(False, "unknown_node")
    demand_error = _validate_demand(workload, cluster)
    if demand_error:
        return PlacementCheck(False, demand_error)
    permitted_roles = cluster.role_compatibility.get(workload.role)
    if permitted_roles is None or node.role not in permitted_roles:
        return PlacementCheck(False, "role_incompatible")
    if any(workload.workload_id in candidate.allocations for candidate in cluster.nodes.values()):
        return PlacementCheck(False, "workload_already_allocated")

    limit = workload.max_instance_per_node
    if (
        limit is not None
        and cluster.density_limit_semantics_verified
        and limit not in cluster.density_unlimited_values
    ):
        same_application_count = sum(
            allocation.workload.app_name == workload.app_name
            for allocation in node.allocations.values()
        )
        if same_application_count >= limit:
            return PlacementCheck(False, "application_density_limit")

    allocated = node.allocated.as_dict()
    capacity = node.capacity.as_dict()
    for name, amount in workload.demand.as_dict().items():
        if name not in capacity:
            return PlacementCheck(False, f"unconfigured_resource:{name}")
        if float(allocated.get(name, 0.0)) + float(amount) > float(capacity[name]) + 1e-9:
            return PlacementCheck(False, f"insufficient_capacity:{name}")
    return PlacementCheck(True, "feasible")


def allocate(
    cluster: ClusterState, node_id: str, workload: SimulatedWorkload, at_time: float
) -> None:
    """Atomically reserve every demanded resource, or leave cluster unchanged."""
    if not math.isfinite(at_time):
        raise ResourceAccountingError("Allocation time must be finite")
    result = can_allocate(cluster, node_id, workload)
    if not result.feasible:
        raise ResourceAccountingError(
            f"Cannot allocate workload {workload.workload_id} on {node_id}: {result.reason}"
        )
    node = cluster.nodes[node_id]
    node.allocations[workload.workload_id] = ActiveAllocation(
        workload=workload, allocated_at=at_time
    )
    try:
        check_invariants(cluster)
    except ResourceAccountingError:
        del node.allocations[workload.workload_id]
        raise


def release(cluster: ClusterState, workload_id: str) -> str:
    """Release a workload's complete allocation and return its node identifier."""
    for node in cluster.nodes.values():
        if workload_id in node.allocations:
            allocation = node.allocations.pop(workload_id)
            try:
                check_invariants(cluster)
            except ResourceAccountingError:
                node.allocations[workload_id] = allocation
                raise
            return node.node_id
    raise ResourceAccountingError(f"Workload {workload_id} has no active allocation")
