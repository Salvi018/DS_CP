"""Deterministic baseline placement policies using authoritative accounting."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.cluster_models import ClusterState, PolicyDecision, SimulatedWorkload
from src.resource_accounting import PlacementCheck, can_allocate

DEFAULT_IFAS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ifas.json"


class Scheduler(Protocol):
    """Existing simulator policy contract, adapted to authoritative accounting."""

    name: str

    def choose_node(
        self,
        workload: SimulatedWorkload,
        feasible_node_ids: list[str],
        cluster: ClusterState,
    ) -> PolicyDecision: ...


def _eligible_nodes(
    workload: SimulatedWorkload, cluster: ClusterState
) -> tuple[list[str], dict[str, PlacementCheck]]:
    """Evaluate every node with the resource-accounting engine."""
    checks = {
        node_id: can_allocate(cluster, node_id, workload)
        for node_id in sorted(cluster.nodes)
    }
    return [node_id for node_id, result in checks.items() if result.feasible], checks


def _failure_reason(checks: dict[str, PlacementCheck]) -> str:
    reasons = sorted({result.reason for result in checks.values()})
    if not reasons:
        return "no_configured_nodes"
    return f"no_feasible_node:{'|'.join(reasons)}"


class FirstFitScheduler:
    """Select the first feasible node in ascending node-ID order."""

    name = "first_fit"
    configuration = {"node_order": "ascending node_id"}

    def select_node(
        self, workload: SimulatedWorkload, cluster: ClusterState
    ) -> PolicyDecision:
        feasible, checks = _eligible_nodes(workload, cluster)
        if not feasible:
            return PolicyDecision(None, _failure_reason(checks))
        return PolicyDecision(feasible[0], "first_feasible_node_by_node_id")

    def choose_node(
        self,
        workload: SimulatedWorkload,
        feasible_node_ids: list[str],
        cluster: ClusterState,
    ) -> PolicyDecision:
        del feasible_node_ids
        return self.select_node(workload, cluster)


class BestFitScheduler:
    """Minimize equal-weight normalized residual capacity after placement.

    Score = sum((capacity - allocated - demand) / capacity) over configured
    resource dimensions with positive capacity. Smaller scores pack remaining
    resources more tightly. Zero-capacity dimensions are omitted after the
    accounting engine has already enforced that their demand must fit (normally
    zero). Ties resolve by ascending node ID. This local heuristic does not
    account for future arrivals, topology, fragmentation across devices, or GPU
    compute/memory capability, and is not globally optimal.
    """

    name = "best_fit"
    configuration = {
        "score": "sum((capacity - allocated - demand) / capacity) over positive-capacity dimensions",
        "selection": "minimum score",
        "tie_break": "ascending node_id",
        "dimensions": "all configured resource dimensions, including integral GPU device count",
        "limitations": "equal weights; no topology, future arrivals, device fragmentation, or GPU capability model; not globally optimal",
    }

    @staticmethod
    def score(
        workload: SimulatedWorkload, cluster: ClusterState, node_id: str
    ) -> float:
        node = cluster.nodes[node_id]
        allocated = node.allocated.as_dict()
        demand = workload.demand.as_dict()
        capacity = node.capacity.as_dict()
        dimensions = [
            name for name, value in capacity.items()
            if float(value) > 0
        ]
        if not dimensions:
            return 0.0
        return sum(
            (
                float(capacity[name])
                - float(allocated.get(name, 0.0))
                - float(demand.get(name, 0.0))
            )
            / float(capacity[name])
            for name in dimensions
        )

    def select_node(
        self, workload: SimulatedWorkload, cluster: ClusterState
    ) -> PolicyDecision:
        feasible, checks = _eligible_nodes(workload, cluster)
        if not feasible:
            return PolicyDecision(None, _failure_reason(checks))
        selected = min(
            feasible,
            key=lambda node_id: (self.score(workload, cluster, node_id), node_id),
        )
        return PolicyDecision(selected, "minimum_normalized_residual_capacity")

    def choose_node(
        self,
        workload: SimulatedWorkload,
        feasible_node_ids: list[str],
        cluster: ClusterState,
    ) -> PolicyDecision:
        del feasible_node_ids
        return self.select_node(workload, cluster)


@dataclass(frozen=True)
class IFASConfig:
    """Normalized score weights and optional deterministic queue-fit behavior."""

    balance_weight: float = 0.25
    fragmentation_weight: float = 0.35
    spread_weight: float = 0.20
    residual_fit_weight: float = 0.20
    enable_future_fit: bool = False
    future_fit_weight: float = 0.20
    tie_breaking: str = "node_id"
    diagnostics_enabled: bool = True

    def __post_init__(self) -> None:
        weights = (
            self.balance_weight,
            self.fragmentation_weight,
            self.spread_weight,
            self.residual_fit_weight,
            self.future_fit_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in weights
        ):
            raise ValueError("IFAS weights must be finite non-negative numbers")
        if not math.isfinite(sum(float(value) for value in weights)):
            raise ValueError("Total IFAS score weight must be finite")
        base_weight_sum = sum(
            (
                self.balance_weight,
                self.fragmentation_weight,
                self.spread_weight,
                self.residual_fit_weight,
            )
        )
        if self.enable_future_fit and self.future_fit_weight <= 0:
            raise ValueError(
                "future_fit_weight must be positive when enable_future_fit is true"
            )
        if base_weight_sum <= 0 and not self.enable_future_fit:
            raise ValueError("At least one enabled IFAS score component must have positive weight")
        if self.tie_breaking not in {"node_id", "configured_order"}:
            raise ValueError("tie_breaking must be 'node_id' or 'configured_order'")
        if not isinstance(self.enable_future_fit, bool):
            raise ValueError("enable_future_fit must be a boolean")
        if not isinstance(self.diagnostics_enabled, bool):
            raise ValueError("diagnostics_enabled must be a boolean")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "IFASConfig":
        valid_names = set(cls.__dataclass_fields__)
        unknown = set(values) - valid_names
        if unknown:
            raise ValueError(f"Unknown IFAS configuration keys: {', '.join(sorted(unknown))}")

        def number(name: str, default: float) -> float:
            value = values.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            return float(value)

        enable_future_fit = values.get("enable_future_fit", False)
        diagnostics_enabled = values.get("diagnostics_enabled", True)
        tie_breaking = values.get("tie_breaking", "node_id")
        if not isinstance(enable_future_fit, bool):
            raise ValueError("enable_future_fit must be a boolean")
        if not isinstance(diagnostics_enabled, bool):
            raise ValueError("diagnostics_enabled must be a boolean")
        if not isinstance(tie_breaking, str):
            raise ValueError("tie_breaking must be a string")
        return cls(
            balance_weight=number("balance_weight", cls.balance_weight),
            fragmentation_weight=number(
                "fragmentation_weight", cls.fragmentation_weight
            ),
            spread_weight=number("spread_weight", cls.spread_weight),
            residual_fit_weight=number(
                "residual_fit_weight", cls.residual_fit_weight
            ),
            enable_future_fit=enable_future_fit,
            future_fit_weight=number("future_fit_weight", cls.future_fit_weight),
            tie_breaking=tie_breaking,
            diagnostics_enabled=diagnostics_enabled,
        )


def load_ifas_config(path: str | Path = DEFAULT_IFAS_CONFIG_PATH) -> IFASConfig:
    """Load validated IFAS weights, using a local JSON configuration file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if source == DEFAULT_IFAS_CONFIG_PATH:
            return IFASConfig()
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read IFAS configuration {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("IFAS configuration must be a JSON object")
    return IFASConfig.from_mapping(payload)


class IntelligentFragmentationAwareScheduler:
    """Deterministic multi-resource fragmentation-aware placement heuristic."""

    name = "ifas"

    def __init__(self, config: IFASConfig | None = None) -> None:
        self.config = config or IFASConfig()
        self.configuration = asdict(self.config)
        self._pending_workloads: tuple[SimulatedWorkload, ...] = ()

    def set_pending_workloads(self, workloads: list[SimulatedWorkload]) -> None:
        """Set only the queue already observed by the simulator at this decision."""
        self._pending_workloads = tuple(workloads)

    @staticmethod
    def _residual(
        workload: SimulatedWorkload, cluster: ClusterState, node_id: str
    ) -> dict[str, float]:
        node = cluster.nodes[node_id]
        capacity = node.capacity.as_dict()
        allocated = node.allocated.as_dict()
        demand = workload.demand.as_dict()
        return {
            name: float(value) - float(allocated.get(name, 0.0)) - float(demand.get(name, 0.0))
            for name, value in capacity.items()
        }

    @staticmethod
    def _normalized_residual(
        residual: dict[str, float], cluster: ClusterState, node_id: str
    ) -> dict[str, float]:
        capacities = cluster.nodes[node_id].capacity.as_dict()
        return {
            name: max(0.0, value / float(capacities[name]))
            for name, value in residual.items()
            if float(capacities[name]) > 0
        }

    @staticmethod
    def _fits_residual(workload: SimulatedWorkload, residual: dict[str, float]) -> bool:
        demand = workload.demand.as_dict()
        return all(
            float(amount) <= residual.get(name, 0.0) + 1e-9
            for name, amount in demand.items()
        )

    def _fragmentation_penalty(
        self, residual: dict[str, float], cluster: ClusterState, node_id: str
    ) -> tuple[float, str | None]:
        normalized = self._normalized_residual(residual, cluster, node_id)
        if not normalized:
            return 0.0, "no_positive_capacity_dimensions"
        pending = [
            candidate
            for candidate in self._pending_workloads
            if cluster.role_compatibility.get(candidate.role)
            and cluster.nodes[node_id].role
            in cluster.role_compatibility[candidate.role]
        ]
        if not pending:
            return 0.0, "no_known_queued_workload_for_fit_estimate"
        if any(self._fits_residual(candidate, residual) for candidate in pending):
            return 0.0, None
        return sum(normalized.values()) / len(normalized), None

    @staticmethod
    def _spread_penalty(
        cluster: ClusterState,
        node_id: str,
        feasible_node_ids: list[str],
    ) -> float:
        active_feasible = [
            candidate_id
            for candidate_id in feasible_node_ids
            if cluster.nodes[candidate_id].allocations
        ]
        if active_feasible and not cluster.nodes[node_id].allocations:
            return 1.0
        return 0.0

    def _future_fit_penalty(
        self,
        workload: SimulatedWorkload,
        cluster: ClusterState,
        node_id: str,
    ) -> tuple[float, dict[str, int]]:
        if not self.config.enable_future_fit:
            return 0.0, {"known_pending_count": len(self._pending_workloads), "at_risk_count": 0}
        at_risk = 0
        considered = 0
        target_node = cluster.nodes[node_id]
        target_residual = self._residual(workload, cluster, node_id)
        for waiting in self._pending_workloads:
            currently_fits = any(
                can_allocate(cluster, candidate_id, waiting).feasible
                for candidate_id in cluster.nodes
            )
            if not currently_fits:
                continue
            considered += 1
            remains_feasible = False
            for candidate_id, candidate_node in cluster.nodes.items():
                if candidate_id != node_id:
                    if can_allocate(cluster, candidate_id, waiting).feasible:
                        remains_feasible = True
                        break
                    continue
                allowed_roles = cluster.role_compatibility.get(waiting.role, [])
                if candidate_node.role not in allowed_roles:
                    continue
                if (
                    cluster.density_limit_semantics_verified
                    and waiting.max_instance_per_node is not None
                    and waiting.max_instance_per_node not in cluster.density_unlimited_values
                ):
                    same_app_count = sum(
                        allocation.workload.app_name == waiting.app_name
                        for allocation in candidate_node.allocations.values()
                    )
                    if candidate_node is target_node and workload.app_name == waiting.app_name:
                        same_app_count += 1
                    if same_app_count >= waiting.max_instance_per_node:
                        continue
                if self._fits_residual(waiting, target_residual):
                    remains_feasible = True
                    break
            if not remains_feasible:
                at_risk += 1
        return (
            at_risk / considered if considered else 0.0,
            {"known_pending_count": len(self._pending_workloads), "at_risk_count": at_risk, "considered_count": considered},
        )

    def choose_node(
        self,
        workload: SimulatedWorkload,
        feasible_node_ids: list[str],
        cluster: ClusterState,
    ) -> PolicyDecision:
        del feasible_node_ids
        checks = {
            node_id: can_allocate(cluster, node_id, workload)
            for node_id in cluster.nodes
        }
        authoritative_feasible = [
            node_id for node_id, check in checks.items() if check.feasible
        ]
        candidate_order = (
            sorted(authoritative_feasible)
            if self.config.tie_breaking == "node_id"
            else authoritative_feasible
        )
        rejected = {
            node_id: check.reason
            for node_id, check in checks.items()
            if not check.feasible
        }
        if not candidate_order:
            return PolicyDecision(
                None,
                _failure_reason(checks),
                {
                    "workload_id": workload.workload_id,
                    "selected_node_id": None,
                    "feasible_candidate_count": 0,
                    "rejected_candidate_reasons": rejected,
                    "configured_weights": self.configuration,
                    "score_components": {},
                    "final_score": None,
                }
                if self.config.diagnostics_enabled
                else {
                    "workload_id": workload.workload_id,
                    "selected_node_id": None,
                    "feasible_candidate_count": 0,
                    "final_score": None,
                },
            )

        raw_weights = {
            "balance_penalty": self.config.balance_weight,
            "fragmentation_penalty": self.config.fragmentation_weight,
            "spread_penalty": self.config.spread_weight,
            "residual_fit_penalty": self.config.residual_fit_weight,
        }
        if self.config.enable_future_fit:
            raw_weights["future_fit_penalty"] = self.config.future_fit_weight
        weight_total = sum(raw_weights.values())
        evaluated: list[tuple[float, str, dict[str, float], dict[str, str], dict[str, int]]] = []
        for node_id in candidate_order:
            residual = self._residual(workload, cluster, node_id)
            normalized = self._normalized_residual(residual, cluster, node_id)
            balance = statistics.pstdev(normalized.values()) if len(normalized) > 1 else 0.0
            fragmentation, fragmentation_note = self._fragmentation_penalty(
                residual, cluster, node_id
            )
            spread = self._spread_penalty(cluster, node_id, candidate_order)
            residual_fit = (
                sum(normalized.values()) / len(normalized) if normalized else 0.0
            )
            future_fit, future_details = self._future_fit_penalty(
                workload, cluster, node_id
            )
            components = {
                "balance_penalty": balance,
                "fragmentation_penalty": fragmentation,
                "spread_penalty": spread,
                "residual_fit_penalty": residual_fit,
            }
            if self.config.enable_future_fit:
                components["future_fit_penalty"] = future_fit
            score = sum(
                raw_weights[name] * components[name] for name in raw_weights
            ) / weight_total
            notes: dict[str, str] = {}
            if fragmentation_note:
                notes["fragmentation_penalty"] = fragmentation_note
            evaluated.append((score, node_id, components, notes, future_details))

        final_score, selected_node_id, components, notes, future_details = min(
            evaluated,
            key=lambda item: (
                item[0],
                item[1] if self.config.tie_breaking == "node_id" else candidate_order.index(item[1]),
            ),
        )
        diagnostics: dict[str, object] = {
            "workload_id": workload.workload_id,
            "selected_node_id": selected_node_id,
            "feasible_candidate_count": len(candidate_order),
            "rejected_candidate_reasons": rejected,
            "score_components": components,
            "final_score": final_score,
            "configured_weights": self.configuration,
        }
        if self.config.diagnostics_enabled:
            diagnostics.update(
                {
                "selected_candidate_notes": notes,
                "future_fit_details": future_details,
                "candidate_scores": {
                    node_id: {
                        "final_score": score,
                        "score_components": candidate_components,
                    }
                    for score, node_id, candidate_components, _, _ in evaluated
                },
                "known_pending_workload_count": len(self._pending_workloads),
                "future_fit_queue_order_assumption": (
                    "only workloads already queued at the decision time; insertion order"
                ),
                }
            )
        return PolicyDecision(
            selected_node_id,
            "minimum_ifas_multi_resource_score",
            diagnostics,
        )


def create_scheduler(name: str, config: IFASConfig | None = None) -> Scheduler:
    """Return a supported scheduler or raise a helpful ValueError."""
    if name == FirstFitScheduler.name:
        return FirstFitScheduler()
    if name == BestFitScheduler.name:
        return BestFitScheduler()
    if name == IntelligentFragmentationAwareScheduler.name:
        return IntelligentFragmentationAwareScheduler(config)
    raise ValueError(
        f"Unsupported scheduling policy {name!r}; choose first_fit, best_fit, or ifas"
    )
