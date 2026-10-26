"""Typed state models for the GPUlytics cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResourceVector:
    """Capacities or demands using canonical resource names."""

    cpu_units: float
    gpu_devices: int
    memory_units: float
    rdma_units: float
    disk_units: float
    extra: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        standard_names = {
            "cpu_units",
            "gpu_devices",
            "memory_units",
            "rdma_units",
            "disk_units",
        }
        if standard_names.intersection(self.extra):
            raise ValueError("Extra resource names cannot replace standard resources")
        if any(not name for name in self.extra):
            raise ValueError("Extra resource names must be non-empty")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cpu_units": self.cpu_units,
            "gpu_devices": self.gpu_devices,
            "memory_units": self.memory_units,
            "rdma_units": self.rdma_units,
            "disk_units": self.disk_units,
            **self.extra,
        }


@dataclass(frozen=True)
class ResourceCapacity(ResourceVector):
    """Node resource capacity."""


@dataclass(frozen=True)
class ResourceDemand(ResourceVector):
    """Atomic workload resource requirements."""


@dataclass(frozen=True)
class PolicyDecision:
    """A scheduler's selected node or explicit no-placement reason."""

    node_id: str | None
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatedWorkload:
    """A workload with validated demands and simulation time in configured units."""

    workload_id: str
    demand: ResourceDemand
    role: str
    app_name: str | None = None
    max_instance_per_node: int | None = None
    arrival_time: float | None = None
    duration: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveAllocation:
    """Allocation record indexed by workload identifier."""

    workload: SimulatedWorkload
    allocated_at: float


@dataclass
class SimulatedNode:
    """Compute node with configured role, capacity, and active allocations."""

    node_id: str
    role: str
    capacity: ResourceCapacity
    allocations: dict[str, ActiveAllocation] = field(default_factory=dict)

    @property
    def allocated(self) -> ResourceVector:
        totals: dict[str, float] = {
            "cpu_units": 0.0,
            "gpu_devices": 0,
            "memory_units": 0.0,
            "rdma_units": 0.0,
            "disk_units": 0.0,
        }
        extras: dict[str, float] = {}
        for allocation in self.allocations.values():
            for name, value in allocation.workload.demand.as_dict().items():
                if name == "gpu_devices":
                    totals[name] += int(value)
                elif name in totals:
                    totals[name] += float(value)
                else:
                    extras[name] = extras.get(name, 0.0) + float(value)
        return ResourceVector(
            cpu_units=totals["cpu_units"],
            gpu_devices=int(totals["gpu_devices"]),
            memory_units=totals["memory_units"],
            rdma_units=totals["rdma_units"],
            disk_units=totals["disk_units"],
            extra=extras,
        )


@dataclass
class ClusterState:
    """Mutable cluster runtime state and configuration assumptions."""

    cluster_name: str
    nodes: dict[str, SimulatedNode]
    role_compatibility: dict[str, list[str]]
    assumptions: list[str] = field(default_factory=list)
    density_unlimited_values: list[int] = field(default_factory=list)
    density_limit_semantics_verified: bool = False
    extra_resource_units: dict[str, str] = field(default_factory=dict)
