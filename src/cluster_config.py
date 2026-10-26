"""Load and validate explicit JSON cluster configurations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT
from src.cluster_models import ClusterState, ResourceCapacity, SimulatedNode

DEFAULT_CLUSTER_CONFIG = PROJECT_ROOT / "config" / "demo_cluster.json"


class ClusterConfigError(ValueError):
    """Raised when a cluster configuration is missing or invalid."""


def _finite_number(value: Any, label: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClusterConfigError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ClusterConfigError(f"{label} must be finite and non-negative")
    if integer and not number.is_integer():
        raise ClusterConfigError(f"{label} must be a whole number")
    return int(number) if integer else number


def load_cluster_config(path: str | Path = DEFAULT_CLUSTER_CONFIG) -> ClusterState:
    """Load nodes, compatibility, density semantics, and assumptions from JSON."""
    source = Path(path)
    if not source.is_file():
        raise ClusterConfigError(f"Cluster configuration not found: {source}")
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClusterConfigError(f"Could not read cluster configuration {source}: {exc}") from exc
    if not isinstance(config, dict):
        raise ClusterConfigError("Cluster configuration must be a JSON object")

    cluster_name = config.get("cluster_name")
    node_specs = config.get("nodes")
    role_compatibility = config.get("role_compatibility", {})
    if not isinstance(cluster_name, str) or not cluster_name.strip():
        raise ClusterConfigError("cluster_name must be a non-empty string")
    if not isinstance(node_specs, list) or not node_specs:
        raise ClusterConfigError("nodes must be a non-empty list")
    if not isinstance(role_compatibility, dict):
        raise ClusterConfigError("role_compatibility must be an object")

    nodes: dict[str, SimulatedNode] = {}
    for index, spec in enumerate(node_specs):
        label = f"nodes[{index}]"
        if not isinstance(spec, dict):
            raise ClusterConfigError(f"{label} must be an object")
        node_id, role = spec.get("node_id"), spec.get("role")
        resources = spec.get("capacity")
        if not isinstance(node_id, str) or not node_id.strip() or node_id in nodes:
            raise ClusterConfigError(f"{label}.node_id must be unique and non-empty")
        if not isinstance(role, str) or not role.strip():
            raise ClusterConfigError(f"{label}.role must be a non-empty string")
        if not isinstance(resources, dict):
            raise ClusterConfigError(f"{label}.capacity must be an object")
        gpu_devices = _finite_number(resources.get("gpu_devices"), f"{label}.capacity.gpu_devices", integer=True)
        known = {"cpu_units", "gpu_devices", "memory_units", "rdma_units", "disk_units"}
        missing = known - resources.keys()
        if missing:
            raise ClusterConfigError(f"{label}.capacity is missing: {', '.join(sorted(missing))}")
        extra_source = resources.get("extra", {})
        if not isinstance(extra_source, dict):
            raise ClusterConfigError(f"{label}.capacity.extra must be an object")
        if known.intersection(extra_source):
            raise ClusterConfigError(
                f"{label}.capacity.extra cannot redefine standard resources"
            )
        extra = {
            str(name): float(_finite_number(value, f"{label}.capacity.extra.{name}"))
            for name, value in extra_source.items()
        }
        capacity = ResourceCapacity(
            cpu_units=float(_finite_number(resources["cpu_units"], f"{label}.capacity.cpu_units")),
            gpu_devices=int(gpu_devices),
            memory_units=float(_finite_number(resources["memory_units"], f"{label}.capacity.memory_units")),
            rdma_units=float(_finite_number(resources["rdma_units"], f"{label}.capacity.rdma_units")),
            disk_units=float(_finite_number(resources["disk_units"], f"{label}.capacity.disk_units")),
            extra=extra,
        )
        nodes[node_id] = SimulatedNode(node_id=node_id, role=role, capacity=capacity)

    compatibility: dict[str, list[str]] = {}
    for workload_role, node_roles in role_compatibility.items():
        if not isinstance(workload_role, str) or not isinstance(node_roles, list) or not all(
            isinstance(role, str) for role in node_roles
        ):
            raise ClusterConfigError("role_compatibility must map role names to lists of node role names")
        compatibility[workload_role] = node_roles
    assumptions = config.get("assumptions", [])
    if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
        raise ClusterConfigError("assumptions must be a list of strings")
    sentinels = config.get("density_unlimited_values", [])
    if not isinstance(sentinels, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in sentinels
    ):
        raise ClusterConfigError("density_unlimited_values must be a list of integers")
    density_verified = config.get("density_limit_semantics_verified", False)
    if not isinstance(density_verified, bool):
        raise ClusterConfigError("density_limit_semantics_verified must be a boolean")
    extra_units = config.get("extra_resource_units", {})
    if not isinstance(extra_units, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in extra_units.items()
    ):
        raise ClusterConfigError("extra_resource_units must map resource names to unit strings")

    return ClusterState(
        cluster_name=cluster_name,
        nodes=nodes,
        role_compatibility=compatibility,
        assumptions=assumptions,
        density_unlimited_values=sentinels,
        density_limit_semantics_verified=density_verified,
        extra_resource_units=extra_units,
    )
