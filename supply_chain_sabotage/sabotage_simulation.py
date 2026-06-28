import copy
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ROOT = Path(__file__).resolve().parent
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SupplyChainConfig:
    steps: int = 180
    attack_probability: float = 0.78
    attack_burst_probability: float = 0.28
    failure_probability: float = 0.34
    node_recovery_rate: float = 0.024
    edge_recovery_rate: float = 0.030
    max_ship_fraction: float = 0.56
    demand_noise: float = 0.28
    demand_surge_probability: float = 0.16
    demand_surge_min: float = 1.35
    demand_surge_max: float = 2.15
    action_budget: float = 1.0
    max_buffer_units: float = 46.0
    max_protected_stock_add: float = 42.0
    max_reinforce_amount: float = 0.60
    max_edge_health_repair: float = 0.45
    max_expedite_priority: float = 1.0
    max_expedite_multiplier: float = 2.0
    buffer_budget_weight: float = 0.45
    reinforce_budget_weight: float = 0.35
    expedite_budget_weight: float = 0.20
    observation_update_interval: int = 3
    inventory_report_noise: float = 0.08
    inventory_report_granularity: float = 5.0
    health_report_noise: float = 0.035
    health_report_granularity: float = 0.05
    protected_stock_report_granularity: float = 5.0
    action_budget_score_weight: float = 0.35
    control_update_interval: int = 1
    monte_carlo_samples: int = 24
    hybrid_codex_candidates: int = 3
    admin_max_sample_multiplier: float = 2.0
    judge_shortlist_size: int = 8
    judge_override_tolerance: float = 0.12


@dataclass
class Node:
    name: str
    kind: str
    inventory: float
    capacity: float
    max_inventory: float
    demand: float = 0.0
    health: float = 1.0
    protected_stock: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class Edge:
    source: str
    target: str
    capacity: float
    health: float = 1.0
    reinforced: float = 0.0


@dataclass
class Intervention:
    buffer_node: Optional[str] = None
    buffer_amount: float = 0.0
    reinforce_edge: Optional[Tuple[str, str]] = None
    reinforce_amount: float = 0.0
    expedite_edge: Optional[Tuple[str, str]] = None
    expedite_priority: float = 0.0


@dataclass
class RunResult:
    time: np.ndarray
    unmet_demand: np.ndarray
    service_level: np.ndarray
    economic_loss: np.ndarray
    attack_loss: np.ndarray
    total_inventory: np.ndarray
    active_capacity: np.ndarray
    buffer_actions: np.ndarray
    reinforce_actions: np.ndarray
    expedite_actions: np.ndarray
    action_budget_used: np.ndarray
    final_nodes: Dict[str, Node]
    final_edges: List[Edge]
    policy_name: str
    policy_stats: Dict[str, float]
    decision_log: List[Dict[str, Any]]


class SupplyChainNetwork:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}
        self._build_default()

    def copy(self) -> "SupplyChainNetwork":
        return copy.deepcopy(self)

    def edge(self, source: str, target: str) -> Edge:
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        raise KeyError((source, target))

    def outgoing(self, node: str) -> List[Edge]:
        return self._outgoing.get(node, [])

    def incoming(self, node: str) -> List[Edge]:
        return self._incoming.get(node, [])

    def retailers(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.kind == "retailer"]

    def warehouses(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.kind == "warehouse"]

    def factories(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.kind == "factory"]

    def suppliers(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.kind == "supplier"]

    def _add_node(
        self,
        name: str,
        kind: str,
        inventory: float,
        capacity: float,
        max_inventory: float,
        demand: float,
        position: Tuple[float, float],
    ) -> None:
        self.nodes[name] = Node(
            name=name,
            kind=kind,
            inventory=inventory,
            capacity=capacity,
            max_inventory=max_inventory,
            demand=demand,
            position=position,
        )

    def _add_edge(self, source: str, target: str, capacity: float) -> None:
        edge = Edge(source=source, target=target, capacity=capacity)
        self.edges.append(edge)
        self._outgoing.setdefault(source, []).append(edge)
        self._incoming.setdefault(target, []).append(edge)

    def _build_default(self) -> None:
        suppliers = [
            ("S1 rare earths", 82.0, 30.0, 190.0, (0.0, 3.2)),
            ("S2 battery chem", 74.0, 27.0, 175.0, (0.0, 2.2)),
            ("S3 semiconductors", 68.0, 24.0, 160.0, (0.0, 1.2)),
            ("S4 metalworks", 78.0, 29.0, 185.0, (0.0, 0.2)),
            ("S5 packaging", 62.0, 22.0, 150.0, (0.0, -0.8)),
            ("S6 fuel depot", 58.0, 20.0, 145.0, (0.0, -1.8)),
        ]
        factories = [
            ("F1 north assembly", 46.0, 31.0, 155.0, (1.5, 2.8)),
            ("F2 electronics", 42.0, 28.0, 145.0, (1.7, 1.6)),
            ("F3 vehicle kits", 48.0, 33.0, 165.0, (1.6, 0.2)),
            ("F4 emergency packs", 40.0, 27.0, 140.0, (1.5, -1.2)),
        ]
        warehouses = [
            ("W1 north hub", 64.0, 0.0, 210.0, (3.0, 3.0)),
            ("W2 metro hub", 70.0, 0.0, 230.0, (3.2, 1.9)),
            ("W3 central hub", 74.0, 0.0, 240.0, (3.3, 0.8)),
            ("W4 south hub", 62.0, 0.0, 205.0, (3.1, -0.4)),
            ("W5 coastal hub", 54.0, 0.0, 180.0, (3.0, -1.5)),
            ("W6 reserve depot", 78.0, 0.0, 250.0, (2.8, 0.1)),
        ]
        retailers = [
            ("R1 trauma hospital", 28.0, 0.0, 105.0, 35.0, (4.9, 3.2)),
            ("R2 airfield", 30.0, 0.0, 115.0, 38.0, (5.2, 2.5)),
            ("R3 command post", 32.0, 0.0, 120.0, 42.0, (5.3, 1.7)),
            ("R4 city district", 34.0, 0.0, 130.0, 45.0, (5.2, 0.9)),
            ("R5 logistics yard", 30.0, 0.0, 115.0, 40.0, (5.0, 0.1)),
            ("R6 port terminal", 28.0, 0.0, 110.0, 36.0, (5.1, -0.8)),
            ("R7 shelter network", 26.0, 0.0, 105.0, 34.0, (4.8, -1.6)),
            ("R8 island outpost", 22.0, 0.0, 90.0, 29.0, (5.4, -2.2)),
        ]

        for name, inventory, capacity, max_inventory, position in suppliers:
            self._add_node(name, "supplier", inventory, capacity, max_inventory, 0.0, position)
        for name, inventory, capacity, max_inventory, position in factories:
            self._add_node(name, "factory", inventory, capacity, max_inventory, 0.0, position)
        for name, inventory, capacity, max_inventory, position in warehouses:
            self._add_node(name, "warehouse", inventory, capacity, max_inventory, 0.0, position)
        for name, inventory, capacity, max_inventory, demand, position in retailers:
            self._add_node(name, "retailer", inventory, capacity, max_inventory, demand, position)

        supplier_factory_edges = [
            ("S1 rare earths", "F1 north assembly", 24.0),
            ("S1 rare earths", "F2 electronics", 18.0),
            ("S2 battery chem", "F2 electronics", 26.0),
            ("S2 battery chem", "F3 vehicle kits", 18.0),
            ("S3 semiconductors", "F1 north assembly", 16.0),
            ("S3 semiconductors", "F2 electronics", 28.0),
            ("S3 semiconductors", "F4 emergency packs", 12.0),
            ("S4 metalworks", "F1 north assembly", 22.0),
            ("S4 metalworks", "F3 vehicle kits", 28.0),
            ("S5 packaging", "F3 vehicle kits", 16.0),
            ("S5 packaging", "F4 emergency packs", 24.0),
            ("S6 fuel depot", "F3 vehicle kits", 20.0),
            ("S6 fuel depot", "F4 emergency packs", 18.0),
        ]
        factory_warehouse_edges = [
            ("F1 north assembly", "W1 north hub", 28.0),
            ("F1 north assembly", "W2 metro hub", 24.0),
            ("F2 electronics", "W2 metro hub", 26.0),
            ("F2 electronics", "W3 central hub", 24.0),
            ("F2 electronics", "W6 reserve depot", 14.0),
            ("F3 vehicle kits", "W3 central hub", 28.0),
            ("F3 vehicle kits", "W4 south hub", 24.0),
            ("F3 vehicle kits", "W6 reserve depot", 16.0),
            ("F4 emergency packs", "W4 south hub", 26.0),
            ("F4 emergency packs", "W5 coastal hub", 24.0),
            ("F4 emergency packs", "W6 reserve depot", 12.0),
        ]
        warehouse_retail_edges = [
            ("W1 north hub", "R1 trauma hospital", 24.0),
            ("W1 north hub", "R2 airfield", 18.0),
            ("W2 metro hub", "R1 trauma hospital", 16.0),
            ("W2 metro hub", "R2 airfield", 26.0),
            ("W2 metro hub", "R3 command post", 24.0),
            ("W3 central hub", "R3 command post", 18.0),
            ("W3 central hub", "R4 city district", 28.0),
            ("W3 central hub", "R5 logistics yard", 20.0),
            ("W4 south hub", "R4 city district", 18.0),
            ("W4 south hub", "R5 logistics yard", 26.0),
            ("W4 south hub", "R6 port terminal", 20.0),
            ("W5 coastal hub", "R6 port terminal", 24.0),
            ("W5 coastal hub", "R7 shelter network", 20.0),
            ("W5 coastal hub", "R8 island outpost", 18.0),
            ("W6 reserve depot", "R2 airfield", 12.0),
            ("W6 reserve depot", "R4 city district", 14.0),
            ("W6 reserve depot", "R7 shelter network", 16.0),
        ]
        transfer_edges = [
            ("W1 north hub", "W2 metro hub", 14.0),
            ("W2 metro hub", "W1 north hub", 10.0),
            ("W2 metro hub", "W3 central hub", 18.0),
            ("W3 central hub", "W2 metro hub", 14.0),
            ("W3 central hub", "W4 south hub", 18.0),
            ("W4 south hub", "W3 central hub", 14.0),
            ("W4 south hub", "W5 coastal hub", 16.0),
            ("W5 coastal hub", "W4 south hub", 12.0),
            ("W6 reserve depot", "W2 metro hub", 12.0),
            ("W6 reserve depot", "W3 central hub", 14.0),
            ("W6 reserve depot", "W4 south hub", 12.0),
        ]
        for source, target, capacity in (
            supplier_factory_edges
            + factory_warehouse_edges
            + warehouse_retail_edges
            + transfer_edges
        ):
            self._add_edge(source, target, capacity)


class Policy:
    name = "policy"

    def reset_for_run(self) -> None:
        pass

    def decision_stats(self) -> Dict[str, float]:
        return {}

    def decision_log(self) -> List[Dict[str, Any]]:
        return []

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        raise NotImplementedError


class BaselinePolicy(Policy):
    name = "baseline"

    def choose_action(self, *args, **kwargs) -> Intervention:
        return Intervention()


class RuleBasedPolicy(Policy):
    name = "rule_based"

    def reset_for_run(self) -> None:
        self._last_target_step: Dict[Tuple[str, str], int] = {}

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        if not hasattr(self, "_last_target_step"):
            self._last_target_step = {}
        recent = history[-1] if history else {}
        previous = history[-min(len(history), 6)] if len(history) >= 2 else recent
        service_level = float(recent.get("service_level", 1.0))
        unmet_demand = float(recent.get("unmet_demand", 0.0))
        service_drop = float(previous.get("service_level", service_level)) - service_level

        storage_nodes = [
            node
            for node in network.nodes.values()
            if node.kind in {"warehouse", "retailer"} and node.max_inventory > 0.0
        ]
        demand_nodes = [node for node in network.retailers() if node.demand > 0.0]
        weak_edge = min(network.edges, key=lambda edge: edge.health, default=None)

        buffer_target = min(
            storage_nodes,
            key=lambda node: node.inventory / max(node.max_inventory, 1.0),
            default=None,
        )
        buffer_fill = (
            buffer_target.inventory / max(buffer_target.max_inventory, 1.0)
            if buffer_target is not None
            else 1.0
        )

        shortage_target = min(
            demand_nodes,
            key=lambda node: node.inventory / max(node.demand, 1.0),
            default=None,
        )
        shortage_days = (
            shortage_target.inventory / max(shortage_target.demand, 1.0)
            if shortage_target is not None
            else 99.0
        )
        expedite_edge = (
            _generic_expedite_edge(network, shortage_target)
            if shortage_target is not None
            else None
        )
        weak_health = weak_edge.health if weak_edge is not None else 1.0

        options: List[Tuple[float, str, Intervention]] = []
        if buffer_target is not None and not self._recently_targeted(("buffer", buffer_target.name), step, 12):
            buffer_score = max(0.0, (0.40 - buffer_fill) / 0.40)
            if service_level < 0.80:
                buffer_score = max(buffer_score, 0.70)
            if buffer_score > 0.0:
                target_inventory = 0.40 * buffer_target.max_inventory
                gap = max(0.0, target_inventory - buffer_target.inventory)
                amount = min(0.55 * config.max_buffer_units, gap)
                if amount > 0.0:
                    options.append(
                        (
                            buffer_score,
                            "buffer",
                            Intervention(buffer_node=buffer_target.name, buffer_amount=amount),
                        )
                    )

        if weak_edge is not None and not self._recently_targeted(
            ("reinforce", f"{weak_edge.source}->{weak_edge.target}"),
            step,
            18,
        ):
            reinforce_score = max(0.0, (0.65 - weak_health) / 0.65)
            if unmet_demand > 65.0 and weak_health < 0.75:
                reinforce_score = max(reinforce_score, 0.35)
            if reinforce_score > 0.0:
                amount_fraction = 0.45 if weak_health < 0.45 else 0.30
                options.append(
                    (
                        reinforce_score,
                        "reinforce",
                        Intervention(
                            reinforce_edge=(weak_edge.source, weak_edge.target),
                            reinforce_amount=amount_fraction * config.max_reinforce_amount,
                        ),
                    )
                )

        if expedite_edge is not None and not self._recently_targeted(
            ("expedite", f"{expedite_edge.source}->{expedite_edge.target}"),
            step,
            8,
        ):
            expedite_score = 0.0
            if service_drop > 0.06:
                expedite_score = max(expedite_score, 0.55)
            if shortage_days < 0.50 and service_level < 0.68:
                expedite_score = max(expedite_score, 0.45)
            if service_level < 0.60 and unmet_demand > 80.0:
                expedite_score = max(expedite_score, 0.50)
            if expedite_score > 0.0:
                priority = 0.45 if shortage_days < 0.50 or service_level < 0.65 else 0.30
                options.append(
                    (
                        expedite_score,
                        "expedite",
                        Intervention(
                            expedite_edge=(expedite_edge.source, expedite_edge.target),
                            expedite_priority=priority * config.max_expedite_priority,
                        ),
                    )
                )

        if not options:
            return Intervention()

        _, action_type, action = max(options, key=lambda item: (item[0], -_rule_action_cost_rank(item[1])))
        self._remember_targets(action_type, action, step)
        return action

    def _recently_targeted(self, key: Tuple[str, str], step: int, cooldown: int) -> bool:
        return step - self._last_target_step.get(key, -10_000) < cooldown

    def _remember_targets(self, action_type: str, action: Intervention, step: int) -> None:
        if action_type == "buffer" and action.buffer_node:
            self._last_target_step[("buffer", action.buffer_node)] = step
        elif action_type == "reinforce" and action.reinforce_edge:
            edge_key = f"{action.reinforce_edge[0]}->{action.reinforce_edge[1]}"
            self._last_target_step[("reinforce", edge_key)] = step
        elif action_type == "expedite" and action.expedite_edge:
            edge_key = f"{action.expedite_edge[0]}->{action.expedite_edge[1]}"
            self._last_target_step[("expedite", edge_key)] = step


class MonteCarloPolicy(Policy):
    name = "monte_carlo"

    def __init__(self, samples: int = 24) -> None:
        self.samples = samples

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        candidates = [("native", action) for action in _candidate_interventions(network, rng, config)]
        if not candidates:
            return Intervention()
        selected, _ = _select_monte_carlo_action(
            network=network,
            tagged_candidates=candidates,
            config=config,
            rng=rng,
            max_evaluations=self.samples,
        )
        return selected


class CodexSupplyChainPolicy(RuleBasedPolicy):
    def __init__(
        self,
        name: str,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
    ) -> None:
        self.name = name
        self.codex_command = codex_command
        self.decision_interval = max(1, decision_interval)
        self.timeout_seconds = max(1, timeout_seconds)
        self.current_intervention = Intervention()
        self._session_id: Optional[str] = None
        self._stats: Dict[str, float] = {}
        self._decision_log: List[Dict[str, Any]] = []

    def reset_for_run(self) -> None:
        self.current_intervention = Intervention()
        self._session_id = None
        self._stats = {
            "codex_calls": 0.0,
            "codex_valid_responses": 0.0,
            "codex_invalid_responses": 0.0,
            "codex_failures": 0.0,
            "codex_noop_responses": 0.0,
        }
        self._decision_log = []

    def decision_stats(self) -> Dict[str, float]:
        return dict(self._stats)

    def decision_log(self) -> List[Dict[str, Any]]:
        return list(self._decision_log)

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        if self._should_consult(step, history):
            self.current_intervention = self._consult_codex(network, history, config)
        return self.current_intervention

    def _should_consult(self, step: int, history: List[Dict[str, float]]) -> bool:
        if self.name == "codex_steady":
            return step % self.decision_interval == 0
        if self.name == "codex_guardian":
            if step % max(1, self.decision_interval // 2) != 0 or len(history) < 5:
                return False
            recent = history[-1]
            previous = history[-5]
            return (
                recent["unmet_demand"] > previous["unmet_demand"] + 5.0
                or recent["service_level"] < previous["service_level"] - 0.05
                or recent["economic_loss"] > previous["economic_loss"] + 80.0
            )
        return False

    def _consult_codex(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        config: SupplyChainConfig,
    ) -> Intervention:
        self._stats["codex_calls"] += 1.0
        payload = {
            "system": "directed supply-chain sabotage response",
            "objective": "minimize unmet demand and economic loss under random failures and adversarial attacks",
            "score_note": (
                "Unnecessary action budget use increases the lower-is-better score. "
                "Do not spend the full budget unless the expected resilience benefit justifies it."
            ),
            "threat_model": {
                "attack_probability": config.attack_probability,
                "attack_burst_probability": config.attack_burst_probability,
                "failure_probability": config.failure_probability,
                "demand_surge_probability": config.demand_surge_probability,
            },
            "telemetry_note": (
                "Node inventories and health values are delayed, noisy, and quantized reports. "
                "They are the same controller-visible telemetry given to all policies, not hidden ground truth."
            ),
            "action_budget": config.action_budget,
            "control_update_interval": config.control_update_interval,
            "budget_weights": {
                "buffer_full": config.buffer_budget_weight,
                "reinforce_full": config.reinforce_budget_weight,
                "expedite_full": config.expedite_budget_weight,
            },
            "action_limits": {
                "buffer_amount": [0.0, config.max_buffer_units],
                "reinforce_amount": [0.0, config.max_reinforce_amount],
                "expedite_priority": [0.0, config.max_expedite_priority],
            },
            "state": history[-1] if history else {},
            "recent_history": history[-8:],
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind,
                    "inventory": round(node.inventory, 2),
                    "capacity": round(node.capacity, 2),
                    "max_inventory": round(node.max_inventory, 2),
                    "demand": round(node.demand, 2),
                    "health": round(node.health, 3),
                    "protected_stock": round(node.protected_stock, 2),
                }
                for node in network.nodes.values()
            ],
            "candidate_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "capacity": round(edge.capacity, 2),
                    "health": round(edge.health, 3),
                    "reinforced": round(edge.reinforced, 3),
                }
                for edge in sorted(network.edges, key=lambda item: item.capacity * item.health)[:6]
            ],
            "shortage_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "capacity": round(edge.capacity, 2),
                    "health": round(edge.health, 3),
                    "target_inventory": round(network.nodes[edge.target].inventory, 2),
                }
                for edge in _shortage_edges(network)[:6]
            ],
        }
        prompt = (
            "You are a supply-chain sabotage response advisor in a directed-graph "
            "agent simulation. Choose concrete budgeted interventions for the next "
            "control interval. Use only node and edge names from the payload. Stay "
            "within the normalized action_budget. A full buffer uses buffer_full "
            "budget, a full reinforce uses reinforce_full budget, and a full "
            "expedite uses expedite_full budget. Use at most one buffer action, "
            "one reinforce action, and one expedite action. Do not return "
            "duplicate action types. Prefer combined actions when the threat "
            "model is severe, reported retailer coverage is thin, or route "
            "health is degraded. Unnecessary action budget use increases the "
            "lower-is-better score, but spending 0.7-1.0 budget is acceptable "
            "when the expected unmet-demand reduction justifies it. The "
            "selected intervention is applied each simulation step until the "
            "next control update interval.\n\n"
            "Return only JSON with this shape:\n"
            "{\"actions\":["
            "{\"type\":\"buffer\",\"node\":\"NODE_NAME\",\"amount\":0.0},"
            "{\"type\":\"reinforce\",\"edge\":[\"SOURCE\",\"TARGET\"],\"amount\":0.0},"
            "{\"type\":\"expedite\",\"edge\":[\"SOURCE\",\"TARGET\"],\"priority\":0.0}"
            "],\"reason\":\"short reason\"}\n\n"
            f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
        )
        step = int(payload["state"].get("step", -1)) if payload["state"] else -1
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error": _format_codex_exception(exc),
                    "intervention": _intervention_to_dict(Intervention()),
                }
            )
            print(
                f"  {self.name}: Codex unavailable after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); applying no intervention.",
                flush=True,
            )
            return Intervention()
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            self._stats["codex_invalid_responses"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "invalid_json",
                    "elapsed_seconds": time.monotonic() - started_at,
                    "intervention": _intervention_to_dict(Intervention()),
                }
            )
            print(f"  {self.name}: Codex returned no JSON; applying no intervention.", flush=True)
            return Intervention()
        try:
            payload_json = json.loads(match.group(0))
        except json.JSONDecodeError:
            self._stats["codex_invalid_responses"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "invalid_json",
                    "elapsed_seconds": time.monotonic() - started_at,
                    "intervention": _intervention_to_dict(Intervention()),
                }
            )
            print(f"  {self.name}: Codex JSON parse failed; applying no intervention.", flush=True)
            return Intervention()

        intervention, valid_actions, errors = _intervention_from_codex_payload(
            payload_json,
            network,
            config,
        )
        elapsed = time.monotonic() - started_at
        normalized, budget_used = _normalize_intervention(network, intervention, config)
        status = "valid" if valid_actions > 0 else "noop"
        if errors:
            self._stats["codex_invalid_responses"] += 1.0
            status = "partially_valid" if valid_actions > 0 else "invalid_action"
        if valid_actions > 0:
            self._stats["codex_valid_responses"] += 1.0
        else:
            self._stats["codex_noop_responses"] += 1.0
        self._decision_log.append(
            {
                "step": step,
                "status": status,
                "elapsed_seconds": elapsed,
                "budget_used": budget_used,
                "errors": errors,
                "reason": str(payload_json.get("reason", ""))[:240],
                "intervention": _intervention_to_dict(normalized),
            }
        )
        print(
            f"  {self.name}: Codex returned {status} intervention "
            f"(budget={budget_used:.2f}) in {elapsed:.1f}s.",
            flush=True,
        )
        return normalized

    def _ask_codex(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as output_file:
            output_path = Path(output_file.name)
        advisor_root = Path(tempfile.gettempdir()) / "supply_chain_codex_policy_workspace"
        advisor_root.mkdir(parents=True, exist_ok=True)
        command = self._codex_command(output_path, advisor_root)
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=True,
            )
            self._remember_session_id(completed.stdout)
            return output_path.read_text().strip()
        finally:
            try:
                output_path.unlink()
            except OSError:
                pass

    def _codex_command(self, output_path: Path, advisor_root: Path) -> List[str]:
        return [
            self.codex_command,
            "exec",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--cd",
            str(advisor_root),
            "--json",
            "--output-last-message",
            str(output_path),
            "--color",
            "never",
            "-",
        ]

    def _remember_session_id(self, stdout: str) -> None:
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = _extract_session_id(event)
            if session_id is not None:
                self._session_id = session_id


class CodexMonteCarloPolicy(CodexSupplyChainPolicy):
    def __init__(
        self,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
        samples: int = 24,
        max_codex_candidates: int = 3,
    ) -> None:
        super().__init__(
            name="codex_monte_carlo",
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.samples = samples
        self.max_codex_candidates = max(1, max_codex_candidates)

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self._stats.update(
            {
                "hybrid_native_candidates_evaluated": 0.0,
                "hybrid_codex_candidates_evaluated": 0.0,
                "hybrid_selected_native": 0.0,
                "hybrid_selected_codex": 0.0,
                "hybrid_selected_noop": 0.0,
            }
        )

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        if step % self.decision_interval != 0:
            return self.current_intervention

        codex_candidates, consultation = self._consult_codex_candidates(network, history, config)
        native_candidates = _candidate_interventions(network, rng, config)
        native_evaluations = min(
            self.samples,
            _unique_intervention_count(network, native_candidates, config),
        )
        extra_evaluations = _unique_intervention_count(
            network,
            codex_candidates[: self.max_codex_candidates],
            config,
        )
        max_evaluations = native_evaluations + extra_evaluations
        tagged_candidates = (
            [("native", action) for action in native_candidates]
            + [("codex", action) for action in codex_candidates[: self.max_codex_candidates]]
        )
        selected, scoring = _select_monte_carlo_action(
            network=network,
            tagged_candidates=tagged_candidates,
            config=config,
            rng=rng,
            max_evaluations=max_evaluations,
        )
        self.current_intervention = selected

        selected_source = str(scoring.get("selected_source", "native"))
        if _intervention_budget(selected, config) <= 0.0:
            self._stats["hybrid_selected_noop"] += 1.0
        elif selected_source == "codex":
            self._stats["hybrid_selected_codex"] += 1.0
        else:
            self._stats["hybrid_selected_native"] += 1.0
        self._stats["hybrid_native_candidates_evaluated"] += float(
            scoring.get("native_candidates_evaluated", 0)
        )
        self._stats["hybrid_codex_candidates_evaluated"] += float(
            scoring.get("codex_candidates_evaluated", 0)
        )

        consultation.update(
            {
                "selected_source": selected_source,
                "selected_score": scoring.get("selected_score"),
                "native_evaluations": native_evaluations,
                "extra_evaluations": extra_evaluations,
                "max_evaluations": max_evaluations,
                "verifier": "default_monte_carlo_risk",
                "evaluated_candidates": scoring.get("evaluated_candidates", 0),
                "native_candidates_evaluated": scoring.get("native_candidates_evaluated", 0),
                "codex_candidates_evaluated": scoring.get("codex_candidates_evaluated", 0),
                "selected_intervention": _intervention_to_dict(selected),
                "selected_budget": _intervention_budget(selected, config),
            }
        )
        self._decision_log.append(consultation)
        print(
            f"  {self.name}: selected {selected_source} candidate "
            f"(score={scoring.get('selected_score', 0.0):.2f}, "
            f"budget={_intervention_budget(selected, config):.2f}).",
            flush=True,
        )
        return self.current_intervention

    def _consult_codex_candidates(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        config: SupplyChainConfig,
    ) -> Tuple[List[Intervention], Dict[str, Any]]:
        self._stats["codex_calls"] += 1.0
        step = int(history[-1].get("step", -1)) if history else -1
        prompt = _codex_candidate_prompt(network, history, config, self.max_codex_candidates)
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex for candidates at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex candidate generation failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using native Monte Carlo candidates.",
                flush=True,
            )
            return [], {
                "step": step,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": _format_codex_exception(exc),
                "codex_candidates": [],
            }

        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            self._stats["codex_invalid_responses"] += 1.0
            return [], {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": time.monotonic() - started_at,
                "codex_candidates": [],
            }
        try:
            payload_json = json.loads(match.group(0))
        except json.JSONDecodeError:
            self._stats["codex_invalid_responses"] += 1.0
            return [], {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": time.monotonic() - started_at,
                "codex_candidates": [],
            }

        candidates, errors, reasons = _codex_candidate_interventions_from_payload(
            payload_json,
            network,
            config,
            self.max_codex_candidates,
        )
        elapsed = time.monotonic() - started_at
        if errors:
            self._stats["codex_invalid_responses"] += 1.0
        if candidates:
            self._stats["codex_valid_responses"] += 1.0
            status = "valid" if not errors else "partially_valid"
        else:
            self._stats["codex_noop_responses"] += 1.0
            status = "invalid_action" if errors else "noop"
        print(
            f"  {self.name}: Codex returned {len(candidates)} candidate(s) "
            f"with status {status} in {elapsed:.1f}s.",
            flush=True,
        )
        return candidates, {
            "step": step,
            "status": status,
            "elapsed_seconds": elapsed,
            "errors": errors,
            "reasons": reasons,
            "codex_candidates": [_intervention_to_dict(action) for action in candidates],
        }


class CodexMonteCarloAdminPolicy(CodexSupplyChainPolicy):
    def __init__(
        self,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
        base_samples: int = 24,
        max_sample_multiplier: float = 2.0,
    ) -> None:
        super().__init__(
            name="codex_monte_carlo_admin",
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.base_samples = max(1, base_samples)
        self.max_sample_multiplier = max(1.0, max_sample_multiplier)

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self._stats.update(
            {
                "admin_native_candidates_evaluated": 0.0,
                "admin_guided_candidates_evaluated": 0.0,
                "admin_selected_native": 0.0,
                "admin_selected_guided": 0.0,
                "admin_selected_noop": 0.0,
                "admin_requested_sample_multiplier_total": 0.0,
                "admin_decisions": 0.0,
            }
        )

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        if step % self.decision_interval != 0:
            return self.current_intervention

        plan, consultation = self._consult_admin_plan(network, history, config)
        native_candidates = _candidate_interventions(network, rng, config)
        admin_candidates = _admin_guided_candidates(network, plan, rng, config)
        sample_multiplier = _clip(
            _safe_float(plan.get("sample_multiplier"), 1.0),
            1.0,
            self.max_sample_multiplier,
        )
        native_evaluations = min(
            self.base_samples,
            _unique_intervention_count(network, native_candidates, config),
        )
        extra_evaluations = max(0, int(round(self.base_samples * (sample_multiplier - 1.0))))
        max_evaluations = max(1, native_evaluations + extra_evaluations)
        tagged_candidates = (
            [("native", action) for action in native_candidates]
            + [("admin", action) for action in admin_candidates]
        )
        selected, scoring = _select_monte_carlo_action(
            network=network,
            tagged_candidates=tagged_candidates,
            config=config,
            rng=rng,
            max_evaluations=max_evaluations,
        )
        self.current_intervention = selected

        selected_source = str(scoring.get("selected_source", "native"))
        if _intervention_budget(selected, config) <= 0.0:
            self._stats["admin_selected_noop"] += 1.0
        elif selected_source == "admin":
            self._stats["admin_selected_guided"] += 1.0
        else:
            self._stats["admin_selected_native"] += 1.0
        self._stats["admin_native_candidates_evaluated"] += float(
            scoring.get("native_candidates_evaluated", 0)
        )
        self._stats["admin_guided_candidates_evaluated"] += float(
            scoring.get("admin_candidates_evaluated", 0)
        )
        self._stats["admin_requested_sample_multiplier_total"] += sample_multiplier
        self._stats["admin_decisions"] += 1.0

        consultation.update(
            {
                "selected_source": selected_source,
                "selected_score": scoring.get("selected_score"),
                "sample_multiplier": sample_multiplier,
                "native_evaluations": native_evaluations,
                "extra_evaluations": extra_evaluations,
                "max_evaluations": max_evaluations,
                "verifier": "default_monte_carlo_risk",
                "evaluated_candidates": scoring.get("evaluated_candidates", 0),
                "native_candidates_evaluated": scoring.get("native_candidates_evaluated", 0),
                "admin_candidates_evaluated": scoring.get("admin_candidates_evaluated", 0),
                "selected_intervention": _intervention_to_dict(selected),
                "selected_budget": _intervention_budget(selected, config),
            }
        )
        self._decision_log.append(consultation)
        print(
            f"  {self.name}: selected {selected_source} candidate "
            f"(score={scoring.get('selected_score', 0.0):.2f}, "
            f"evals={scoring.get('evaluated_candidates', 0)}, "
            f"budget={_intervention_budget(selected, config):.2f}).",
            flush=True,
        )
        return self.current_intervention

    def _consult_admin_plan(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        config: SupplyChainConfig,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._stats["codex_calls"] += 1.0
        step = int(history[-1].get("step", -1)) if history else -1
        prompt = _codex_admin_prompt(network, history, config, self.max_sample_multiplier)
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex for Monte Carlo admin plan at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex admin plan failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using native Monte Carlo.",
                flush=True,
            )
            return _default_admin_plan(active=False), {
                "step": step,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": _format_codex_exception(exc),
                "admin_plan": _default_admin_plan(active=False),
            }

        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            self._stats["codex_invalid_responses"] += 1.0
            return _default_admin_plan(active=False), {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": time.monotonic() - started_at,
                "admin_plan": _default_admin_plan(active=False),
            }
        try:
            payload_json = json.loads(match.group(0))
        except json.JSONDecodeError:
            self._stats["codex_invalid_responses"] += 1.0
            return _default_admin_plan(active=False), {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": time.monotonic() - started_at,
                "admin_plan": _default_admin_plan(active=False),
            }

        plan, errors = _admin_plan_from_payload(payload_json, network, self.max_sample_multiplier)
        elapsed = time.monotonic() - started_at
        if errors:
            self._stats["codex_invalid_responses"] += 1.0
            status = "partially_valid"
        else:
            status = "valid"
        self._stats["codex_valid_responses"] += 1.0
        print(
            f"  {self.name}: Codex returned {status} admin plan "
            f"(risk={plan['risk_mode']}, sample_multiplier={plan['sample_multiplier']:.2f}) "
            f"in {elapsed:.1f}s.",
            flush=True,
        )
        return plan, {
            "step": step,
            "status": status,
            "elapsed_seconds": elapsed,
            "errors": errors,
            "admin_plan": _admin_plan_to_dict(plan),
        }


class CodexMonteCarloJudgePolicy(CodexSupplyChainPolicy):
    def __init__(
        self,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
        base_samples: int = 24,
        shortlist_size: int = 8,
        override_tolerance: float = 0.12,
    ) -> None:
        super().__init__(
            name="codex_monte_carlo_judge",
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.base_samples = max(1, base_samples)
        self.shortlist_size = max(2, shortlist_size)
        self.override_tolerance = max(0.0, override_tolerance)

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self._stats.update(
            {
                "judge_native_candidates_evaluated": 0.0,
                "judge_extra_candidates_evaluated": 0.0,
                "judge_selected_default": 0.0,
                "judge_selected_override": 0.0,
                "judge_selected_native": 0.0,
                "judge_selected_extra": 0.0,
                "judge_selected_noop": 0.0,
                "judge_guardrail_rejections": 0.0,
                "judge_decisions": 0.0,
                "judge_default_score_total": 0.0,
                "judge_selected_score_total": 0.0,
            }
        )

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
        config: SupplyChainConfig,
    ) -> Intervention:
        if step % self.decision_interval != 0:
            return self.current_intervention

        native_candidates = _candidate_interventions(network, rng, config)
        judge_candidates = _judge_extra_candidates(network, rng, config)
        native_evaluations = min(
            self.base_samples,
            _unique_intervention_count(network, native_candidates, config),
        )
        extra_evaluations = min(
            self.base_samples,
            _unique_intervention_count(network, judge_candidates, config),
        )
        tagged_candidates = (
            [("native", action) for action in native_candidates]
            + [("judge", action) for action in judge_candidates]
        )
        scored, scoring = _score_monte_carlo_candidates(
            network=network,
            tagged_candidates=tagged_candidates,
            config=config,
            rng=rng,
            max_evaluations=native_evaluations + extra_evaluations,
        )
        self._stats["judge_decisions"] += 1.0
        if not scored:
            self.current_intervention = Intervention()
            self._decision_log.append(
                {
                    "step": step,
                    "status": "no_candidates",
                    "selected_intervention": _intervention_to_dict(self.current_intervention),
                }
            )
            return self.current_intervention

        default_entry = next(
            (item for item in scored if item["source"] == "native"),
            scored[0],
        )
        shortlist = _shortlist_scored_candidates(scored, self.shortlist_size)
        if default_entry["candidate_id"] not in {item["candidate_id"] for item in shortlist}:
            shortlist = [default_entry] + shortlist[: self.shortlist_size - 1]
        choice_id, consultation = self._consult_judge(
            network=network,
            history=history,
            config=config,
            shortlist=shortlist,
            default_entry=default_entry,
        )
        selected_entry = default_entry
        guardrail_rejected = False
        if choice_id is not None:
            candidate_by_id = {item["candidate_id"]: item for item in shortlist}
            proposed_entry = candidate_by_id.get(choice_id)
            if proposed_entry is not None:
                max_allowed_score = default_entry["default_score"] + (
                    abs(default_entry["default_score"]) * self.override_tolerance
                )
                if proposed_entry["default_score"] <= max_allowed_score:
                    selected_entry = proposed_entry
                else:
                    guardrail_rejected = True

        selected = selected_entry["intervention"]
        self.current_intervention = selected
        selected_source = str(selected_entry["source"])
        selected_is_default = selected_entry["candidate_id"] == default_entry["candidate_id"]
        if guardrail_rejected:
            self._stats["judge_guardrail_rejections"] += 1.0
        if selected_is_default:
            self._stats["judge_selected_default"] += 1.0
        else:
            self._stats["judge_selected_override"] += 1.0
        if _intervention_budget(selected, config) <= 0.0:
            self._stats["judge_selected_noop"] += 1.0
        elif selected_source == "native":
            self._stats["judge_selected_native"] += 1.0
        else:
            self._stats["judge_selected_extra"] += 1.0
        self._stats["judge_native_candidates_evaluated"] += float(
            scoring.get("native_candidates_evaluated", 0)
        )
        self._stats["judge_extra_candidates_evaluated"] += float(
            scoring.get("judge_candidates_evaluated", 0)
        )
        self._stats["judge_default_score_total"] += float(default_entry["default_score"])
        self._stats["judge_selected_score_total"] += float(selected_entry["default_score"])

        consultation.update(
            {
                "selected_source": selected_source,
                "selected_candidate_id": selected_entry["candidate_id"],
                "default_candidate_id": default_entry["candidate_id"],
                "judge_choice_id": choice_id,
                "judge_overrode_default": not selected_is_default,
                "judge_guardrail_rejected": guardrail_rejected,
                "default_score": default_entry["default_score"],
                "selected_score": selected_entry["default_score"],
                "native_evaluations": native_evaluations,
                "extra_evaluations": extra_evaluations,
                "max_evaluations": native_evaluations + extra_evaluations,
                "evaluated_candidates": scoring.get("evaluated_candidates", 0),
                "native_candidates_evaluated": scoring.get("native_candidates_evaluated", 0),
                "judge_candidates_evaluated": scoring.get("judge_candidates_evaluated", 0),
                "shortlist": [_scored_candidate_to_prompt_dict(item) for item in shortlist],
                "selected_intervention": _intervention_to_dict(selected),
                "selected_budget": _intervention_budget(selected, config),
            }
        )
        self._decision_log.append(consultation)
        print(
            f"  {self.name}: selected {selected_source} candidate "
            f"(score={selected_entry['default_score']:.2f}, "
            f"default={default_entry['default_score']:.2f}, "
            f"evals={scoring.get('evaluated_candidates', 0)}, "
            f"budget={_intervention_budget(selected, config):.2f}).",
            flush=True,
        )
        return self.current_intervention

    def _consult_judge(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        config: SupplyChainConfig,
        shortlist: Sequence[Dict[str, Any]],
        default_entry: Dict[str, Any],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        self._stats["codex_calls"] += 1.0
        step = int(history[-1].get("step", -1)) if history else -1
        prompt = _codex_judge_prompt(
            network=network,
            history=history,
            config=config,
            shortlist=shortlist,
            default_entry=default_entry,
            override_tolerance=self.override_tolerance,
        )
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex to judge MC shortlist at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex judge failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using default Monte Carlo winner.",
                flush=True,
            )
            return None, {
                "step": step,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": _format_codex_exception(exc),
            }

        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            return None, {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": time.monotonic() - started_at,
            }
        try:
            payload_json = json.loads(match.group(0))
        except json.JSONDecodeError:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            return None, {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": time.monotonic() - started_at,
            }

        allowed_ids = {str(item["candidate_id"]) for item in shortlist}
        choice = str(payload_json.get("choice", payload_json.get("candidate_id", ""))).strip()
        elapsed = time.monotonic() - started_at
        if choice not in allowed_ids:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            print(
                f"  {self.name}: Codex judge returned invalid choice {choice!r}; "
                "using default Monte Carlo winner.",
                flush=True,
            )
            return None, {
                "step": step,
                "status": "invalid_choice",
                "elapsed_seconds": elapsed,
                "choice": choice,
                "reason": str(payload_json.get("reason", ""))[:240],
            }

        self._stats["codex_valid_responses"] += 1.0
        print(
            f"  {self.name}: Codex judge chose {choice} in {elapsed:.1f}s.",
            flush=True,
        )
        return choice, {
            "step": step,
            "status": "valid",
            "elapsed_seconds": elapsed,
            "choice": choice,
            "reason": str(payload_json.get("reason", ""))[:240],
        }


def run_simulation(
    config: SupplyChainConfig,
    policy: Policy,
    seed: int,
    progress_interval: int = 0,
) -> RunResult:
    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + 317_503)
    observation_rng = np.random.default_rng(seed + 884_911)
    network = SupplyChainNetwork()
    observed_network = _observed_network(network, config, observation_rng)
    time = np.arange(config.steps + 1)
    unmet = np.zeros(config.steps + 1)
    service = np.ones(config.steps + 1)
    loss = np.zeros(config.steps + 1)
    attack_loss = np.zeros(config.steps + 1)
    inventory = np.zeros(config.steps + 1)
    active_capacity = np.zeros(config.steps + 1)
    buffer_actions = np.zeros(config.steps + 1)
    reinforce_actions = np.zeros(config.steps + 1)
    expedite_actions = np.zeros(config.steps + 1)
    action_budget_used = np.zeros(config.steps + 1)
    true_history: List[Dict[str, float]] = []
    controller_history: List[Dict[str, float]] = []
    current_action = Intervention()
    current_budget_used = 0.0
    control_update_interval = max(1, config.control_update_interval)
    controller_decisions = 0

    for step in range(config.steps + 1):
        _record(
            network,
            true_history,
            step,
            unmet,
            service,
            loss,
            attack_loss,
            inventory,
            active_capacity,
        )
        if progress_interval > 0 and step % progress_interval == 0:
            print(
                f"  {policy.name}: step {step}/{config.steps} "
                f"unmet={unmet[step]:.1f} service={service[step]:.2f} "
                f"loss={loss[step]:.1f}.",
                flush=True,
            )
        if step == config.steps:
            break

        if step % max(1, config.observation_update_interval) == 0:
            observed_network = _observed_network(network, config, observation_rng)
        _record_controller_history(
            observed_network,
            controller_history,
            step,
            unmet,
            service,
            loss,
            attack_loss,
        )

        if step % control_update_interval == 0:
            raw_action = policy.choose_action(
                observed_network,
                controller_history,
                step,
                policy_rng,
                config,
            )
            current_action, current_budget_used = _normalize_intervention(network, raw_action, config)
            controller_decisions += 1
        action = current_action
        budget_used = current_budget_used
        action_cost = _apply_intervention(network, action, config, rng)
        buffer_actions[step + 1] = action.buffer_amount / max(config.max_buffer_units, 1.0)
        reinforce_actions[step + 1] = action.reinforce_amount / max(config.max_reinforce_amount, 1.0e-9)
        expedite_actions[step + 1] = action.expedite_priority
        action_budget_used[step + 1] = budget_used
        attack_loss[step + 1] = _apply_disruptions(network, config, rng)
        _recover(network, config)
        _produce(network)
        _ship(network, config, rng, action)
        demand, unmet_step = _consume_demand(network, config, rng)
        unmet[step + 1] = unmet_step
        service[step + 1] = 1.0 - unmet_step / max(demand, 1.0)
        loss[step + 1] = 8.0 * unmet_step + attack_loss[step + 1] + action_cost

    return RunResult(
        time=time,
        unmet_demand=unmet,
        service_level=service,
        economic_loss=loss,
        attack_loss=attack_loss,
        total_inventory=inventory,
        active_capacity=active_capacity,
        buffer_actions=buffer_actions,
        reinforce_actions=reinforce_actions,
        expedite_actions=expedite_actions,
        action_budget_used=action_budget_used,
        final_nodes=network.nodes,
        final_edges=network.edges,
        policy_name=policy.name,
        policy_stats={
            **policy.decision_stats(),
            "controller_decisions": float(controller_decisions),
        },
        decision_log=policy.decision_log(),
    )


def run_repeated(
    config: SupplyChainConfig,
    policies: Sequence[Policy],
    runs: Union[int, Dict[str, int]],
    seed: int,
    progress_interval: int = 0,
) -> Dict[str, List[RunResult]]:
    results: Dict[str, List[RunResult]] = {}
    for policy in policies:
        policy_runs = runs.get(policy.name, 1) if isinstance(runs, dict) else runs
        policy_runs = max(1, int(policy_runs))
        results[policy.name] = []
        for run_index in range(policy_runs):
            policy.reset_for_run()
            print(f"Running {policy.name} run {run_index + 1}/{policy_runs}...", flush=True)
            results[policy.name].append(
                run_simulation(
                    config=config,
                    policy=policy,
                    seed=seed + run_index,
                    progress_interval=progress_interval,
                )
            )
    return results


def summarize(
    results: Dict[str, List[RunResult]],
    config: Optional[SupplyChainConfig] = None,
) -> Dict[str, object]:
    score_config = config if config is not None else SupplyChainConfig()
    metrics: Dict[str, Dict[str, Any]] = {}
    per_run_metrics: Dict[str, List[Dict[str, float]]] = {}
    for policy_name, runs in results.items():
        run_metrics = [_run_metrics(run, score_config) for run in runs]
        per_run_metrics[policy_name] = run_metrics
        total_unmet = np.array([item["total_unmet_demand"] for item in run_metrics])
        total_loss = np.array([item["economic_loss"] for item in run_metrics])
        service = np.array([item["service_level"] for item in run_metrics])
        service_cvar = np.array([item["service_level_cvar10"] for item in run_metrics])
        terminal_inventory = np.array([item["terminal_inventory"] for item in run_metrics])
        action_budget = np.array([item["mean_action_budget_used"] for item in run_metrics])
        scores = np.array([item["sabotage_impact_score_lower_is_better"] for item in run_metrics])
        policy_stats = _aggregate_policy_stats(runs)
        metrics[policy_name] = {
            "runs": len(runs),
            "total_unmet_demand": _mean_interval(total_unmet),
            "economic_loss": _mean_interval(total_loss),
            "service_level": _mean_interval(service),
            "service_level_cvar10": _mean_interval(service_cvar),
            "terminal_inventory": _mean_interval(terminal_inventory),
            "mean_action_budget_used": _mean_interval(action_budget),
            "sabotage_impact_score_lower_is_better": float(np.mean(scores)),
            "sabotage_impact_score_ci95": _mean_interval(scores),
            "decision_stats": policy_stats,
        }
        # Backward-compatible scalar fields used by old dashboards/readmes.
        metrics[policy_name]["total_unmet_demand_mean"] = float(np.mean(total_unmet))
        metrics[policy_name]["economic_loss_mean"] = float(np.mean(total_loss))
        metrics[policy_name]["service_level_mean"] = float(np.mean(service))
        metrics[policy_name]["terminal_inventory_mean"] = float(np.mean(terminal_inventory))
    ranking = [
        {
            "policy": name,
            "sabotage_impact_score_lower_is_better": values["sabotage_impact_score_lower_is_better"],
            "score_ci95_low": values["sabotage_impact_score_ci95"]["ci95_low"],
            "score_ci95_high": values["sabotage_impact_score_ci95"]["ci95_high"],
        }
        for name, values in metrics.items()
    ]
    ranking.sort(key=lambda item: item["sabotage_impact_score_lower_is_better"])
    comparisons = _paired_comparisons(per_run_metrics)
    decision_logs = {
        policy_name: [
            {"run_index": index, "decisions": run.decision_log}
            for index, run in enumerate(runs)
            if run.decision_log
        ]
        for policy_name, runs in results.items()
    }
    return {
        "metrics": metrics,
        "per_run_metrics": per_run_metrics,
        "comparisons": comparisons,
        "decision_logs": decision_logs,
        "ranking_lower_is_better": ranking,
    }


def save_outputs(
    results: Dict[str, List[RunResult]],
    output_dir: Path,
    config: Optional[SupplyChainConfig] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for policy_name, runs in results.items():
        print(f"Saving outputs for {policy_name}...", flush=True)
        written.extend(_save_policy_plot(policy_name, runs, output_dir))
    written.extend(_save_dashboard(results, output_dir, config=config))
    written.extend(_save_network_map(results, output_dir))
    summary = summarize(results, config=config)
    if config is not None:
        summary["config"] = asdict(config)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    written.append(summary_path)
    return written


def _observed_network(
    network: SupplyChainNetwork,
    config: SupplyChainConfig,
    rng: np.random.Generator,
) -> SupplyChainNetwork:
    """Return the shared telemetry snapshot visible to every controller."""

    observed = network.copy()
    for name, reported in observed.nodes.items():
        actual = network.nodes[name]
        inventory_multiplier = 1.0 + float(rng.normal(0.0, config.inventory_report_noise))
        reported.inventory = _quantize(
            _clip(actual.inventory * inventory_multiplier, 0.0, actual.max_inventory),
            config.inventory_report_granularity,
        )
        reported.health = _quantize(
            _clip(actual.health + float(rng.normal(0.0, config.health_report_noise)), 0.0, 1.0),
            config.health_report_granularity,
        )
        reported.protected_stock = _quantize(
            _clip(actual.protected_stock, 0.0, actual.max_inventory),
            config.protected_stock_report_granularity,
        )

    for reported in observed.edges:
        actual = network.edge(reported.source, reported.target)
        reported.health = _quantize(
            _clip(actual.health + float(rng.normal(0.0, config.health_report_noise)), 0.0, 1.0),
            config.health_report_granularity,
        )
        reported.reinforced = _quantize(
            _clip(actual.reinforced + float(rng.normal(0.0, config.health_report_noise * 0.5)), 0.0, 1.0),
            config.health_report_granularity,
        )
    return observed


def _run_metrics(run: RunResult, config: SupplyChainConfig) -> Dict[str, float]:
    total_unmet = float(np.sum(run.unmet_demand))
    total_loss = float(np.sum(run.economic_loss))
    service = float(np.mean(run.service_level))
    terminal_inventory = float(run.total_inventory[-1])
    mean_action_budget = float(np.mean(run.action_budget_used))
    score = _impact_score(total_unmet, total_loss, service, terminal_inventory, mean_action_budget, config)
    return {
        "total_unmet_demand": total_unmet,
        "economic_loss": total_loss,
        "service_level": service,
        "service_level_cvar10": _low_tail_mean(run.service_level, 0.10),
        "terminal_inventory": terminal_inventory,
        "mean_action_budget_used": mean_action_budget,
        "buffer_action_rate": float(np.mean(run.buffer_actions > 0.0)),
        "reinforce_action_rate": float(np.mean(run.reinforce_actions > 0.0)),
        "expedite_action_rate": float(np.mean(run.expedite_actions > 0.0)),
        "sabotage_impact_score_lower_is_better": score,
    }


def _impact_score(
    total_unmet: float,
    total_loss: float,
    service: float,
    terminal_inventory: float,
    mean_action_budget: float,
    config: SupplyChainConfig,
) -> float:
    return float(
        total_unmet / 2500.0
        + total_loss / 25000.0
        + 1.4 * (1.0 - service)
        + 0.15 * (1.0 / max(terminal_inventory / 500.0, 0.1))
        + config.action_budget_score_weight * mean_action_budget
    )


def _mean_interval(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values)) if values.size else 0.0
    if values.size <= 1:
        return {
            "mean": mean,
            "std": 0.0,
            "sem": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    std = float(np.std(values, ddof=1))
    sem = std / float(np.sqrt(values.size))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
    }


def _low_tail_mean(values: np.ndarray, fraction: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.mean(values[:count]))


def _aggregate_policy_stats(runs: List[RunResult]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for run in runs:
        for key, value in run.policy_stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _paired_comparisons(
    per_run_metrics: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Any]:
    if "baseline" not in per_run_metrics:
        return {"vs_baseline": {}, "vs_best_non_codex": {}, "best_non_codex_policy": None}
    non_codex = [
        name
        for name in per_run_metrics
        if not name.startswith("codex_") and name != "baseline"
    ]
    if non_codex:
        best_non_codex = min(
            non_codex,
            key=lambda name: np.mean(
                [item["sabotage_impact_score_lower_is_better"] for item in per_run_metrics[name]]
            ),
        )
    else:
        best_non_codex = "baseline"

    return {
        "best_non_codex_policy": best_non_codex,
        "vs_baseline": {
            name: _paired_score_delta(items, per_run_metrics["baseline"])
            for name, items in per_run_metrics.items()
            if name != "baseline"
        },
        "vs_best_non_codex": {
            name: _paired_score_delta(items, per_run_metrics[best_non_codex])
            for name, items in per_run_metrics.items()
            if name != best_non_codex
        },
    }


def _paired_score_delta(
    policy_metrics: List[Dict[str, float]],
    reference_metrics: List[Dict[str, float]],
) -> Dict[str, Optional[float]]:
    count = min(len(policy_metrics), len(reference_metrics))
    if count <= 0:
        return {
            "paired_runs": 0,
            "mean_delta_policy_minus_reference": None,
            "delta_ci95_low": None,
            "delta_ci95_high": None,
            "win_rate_lower_score": None,
            "relative_improvement_mean": None,
        }
    policy_scores = np.array(
        [item["sabotage_impact_score_lower_is_better"] for item in policy_metrics[:count]],
        dtype=float,
    )
    reference_scores = np.array(
        [item["sabotage_impact_score_lower_is_better"] for item in reference_metrics[:count]],
        dtype=float,
    )
    delta = policy_scores - reference_scores
    interval = _mean_interval(delta)
    relative = (reference_scores - policy_scores) / np.maximum(np.abs(reference_scores), 1.0e-9)
    return {
        "paired_runs": count,
        "mean_delta_policy_minus_reference": interval["mean"],
        "delta_ci95_low": interval["ci95_low"],
        "delta_ci95_high": interval["ci95_high"],
        "win_rate_lower_score": float(np.mean(policy_scores < reference_scores)),
        "relative_improvement_mean": float(np.mean(relative)),
    }


def _select_monte_carlo_action(
    network: SupplyChainNetwork,
    tagged_candidates: Sequence[Tuple[str, Intervention]],
    config: SupplyChainConfig,
    rng: np.random.Generator,
    max_evaluations: int,
    score_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[Intervention, Dict[str, Any]]:
    scored, scoring = _score_monte_carlo_candidates(
        network=network,
        tagged_candidates=tagged_candidates,
        config=config,
        rng=rng,
        max_evaluations=max_evaluations,
        score_plan=score_plan,
    )
    if not scored:
        scoring.update(
            {
                "selected_source": "none",
                "selected_score": None,
            }
        )
        return Intervention(), scoring

    best = scored[0]
    scoring.update(
        {
            "selected_source": best["source"],
            "selected_score": best["primary_score"],
        }
    )
    return best["intervention"], scoring


def _score_monte_carlo_candidates(
    network: SupplyChainNetwork,
    tagged_candidates: Sequence[Tuple[str, Intervention]],
    config: SupplyChainConfig,
    rng: np.random.Generator,
    max_evaluations: int,
    score_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    seen = set()
    source_counts: Dict[str, int] = {}
    for source, action in tagged_candidates:
        if len(scored) >= max(1, max_evaluations):
            break
        normalized, _ = _normalize_intervention(network, action, config)
        key = _intervention_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        trial = network.copy()
        _apply_intervention(trial, normalized, config, rng, dry_run=True)
        scored.append(
            {
                "source": source,
                "intervention": normalized,
                "budget_used": _intervention_budget(normalized, config),
                "primary_score": _candidate_score(trial, normalized, config, score_plan),
                "default_score": _candidate_score(trial, normalized, config),
                "service_score": _candidate_score(
                    trial,
                    normalized,
                    config,
                    _candidate_score_plan("service"),
                ),
                "tail_score": _candidate_score(
                    trial,
                    normalized,
                    config,
                    _candidate_score_plan("tail"),
                ),
                "inventory_score": _candidate_score(
                    trial,
                    normalized,
                    config,
                    _candidate_score_plan("inventory"),
                ),
                "budget_score": _candidate_score(
                    trial,
                    normalized,
                    config,
                    _candidate_score_plan("budget"),
                ),
            }
        )
        source_counts[source] = source_counts.get(source, 0) + 1

    scored.sort(key=lambda item: item["primary_score"])
    for rank, item in enumerate(scored):
        item["candidate_id"] = f"c{rank}"
        item["default_rank"] = rank + 1
    return scored, {
        "evaluated_candidates": len(scored),
        "native_candidates_evaluated": source_counts.get("native", 0),
        "codex_candidates_evaluated": source_counts.get("codex", 0),
        "admin_candidates_evaluated": source_counts.get("admin", 0),
        "judge_candidates_evaluated": source_counts.get("judge", 0),
    }


def _unique_intervention_count(
    network: SupplyChainNetwork,
    candidates: Sequence[Intervention],
    config: SupplyChainConfig,
) -> int:
    seen = set()
    for action in candidates:
        normalized, _ = _normalize_intervention(network, action, config)
        seen.add(_intervention_key(normalized))
    return len(seen)


def _candidate_score(
    network: SupplyChainNetwork,
    action: Intervention,
    config: SupplyChainConfig,
    score_plan: Optional[Dict[str, Any]] = None,
) -> float:
    if not score_plan:
        return _short_horizon_risk(network) + 65.0 * _intervention_budget(action, config)

    weights = score_plan.get("objective_weights", {})
    inventory_weight = _safe_float(weights.get("inventory_gap"), 1.0)
    node_health_weight = _safe_float(weights.get("node_health"), 80.0)
    edge_health_weight = _safe_float(weights.get("edge_health"), 55.0)
    retailer_gap_weight = _safe_float(weights.get("retailer_gap"), 2.0)
    action_budget_weight = _safe_float(weights.get("action_budget"), 65.0)
    priority_nodes = set(score_plan.get("priority_nodes", []))
    priority_edges = {
        tuple(edge)
        for edge in score_plan.get("priority_edges", [])
        if isinstance(edge, (list, tuple)) and len(edge) == 2
    }

    desired = _desired_inventory(network)
    inventory_gap = 0.0
    unhealthy_nodes = 0.0
    retailer_gap = 0.0
    for name, node in network.nodes.items():
        multiplier = 1.45 if name in priority_nodes else 1.0
        inventory_gap += multiplier * max(0.0, desired[name] - node.inventory)
        unhealthy_nodes += multiplier * (1.0 - node.health)
        if node.kind == "retailer":
            retailer_gap += multiplier * max(0.0, node.demand * 2.0 - node.inventory)

    unhealthy_edges = 0.0
    for edge in network.edges:
        multiplier = 1.55 if (edge.source, edge.target) in priority_edges else 1.0
        unhealthy_edges += multiplier * (1.0 - edge.health)

    risk_mode = str(score_plan.get("risk_mode", "balanced")).lower()
    if risk_mode == "service":
        retailer_gap_weight *= 1.35
    elif risk_mode == "tail":
        max_retailer_gap = max(
            [max(0.0, node.demand * 2.5 - node.inventory) for node in network.retailers()],
            default=0.0,
        )
        retailer_gap += 2.0 * max_retailer_gap
        edge_health_weight *= 1.15
    elif risk_mode == "inventory":
        inventory_weight *= 1.30
    elif risk_mode == "budget":
        action_budget_weight *= 1.35

    return float(
        inventory_weight * inventory_gap
        + node_health_weight * unhealthy_nodes
        + edge_health_weight * unhealthy_edges
        + retailer_gap_weight * retailer_gap
        + action_budget_weight * _intervention_budget(action, config)
    )


def _candidate_score_plan(risk_mode: str) -> Dict[str, Any]:
    plan = _default_admin_plan(active=True)
    return {
        "risk_mode": risk_mode,
        "priority_nodes": [],
        "priority_edges": [],
        "objective_weights": dict(plan["objective_weights"]),
    }


def _codex_candidate_prompt(
    network: SupplyChainNetwork,
    history: List[Dict[str, float]],
    config: SupplyChainConfig,
    max_candidates: int,
) -> str:
    payload = {
        "system": "directed supply-chain sabotage response",
        "objective": (
            "propose diverse candidate interventions; a Monte Carlo evaluator will "
            "choose the best candidate under the same simulator budget"
        ),
        "candidate_count": max_candidates,
        "score_note": (
            "Unnecessary action budget use increases the lower-is-better score. "
            "Propose candidates that trade resilience benefit against budget use."
        ),
        "threat_model": {
            "attack_probability": config.attack_probability,
            "attack_burst_probability": config.attack_burst_probability,
            "failure_probability": config.failure_probability,
            "demand_surge_probability": config.demand_surge_probability,
        },
        "telemetry_note": (
            "Node inventories and health values are delayed, noisy, and quantized reports. "
            "They are the same controller-visible telemetry given to all policies, not hidden ground truth."
        ),
        "action_budget": config.action_budget,
        "control_update_interval": config.control_update_interval,
        "budget_weights": {
            "buffer_full": config.buffer_budget_weight,
            "reinforce_full": config.reinforce_budget_weight,
            "expedite_full": config.expedite_budget_weight,
        },
        "action_limits": {
            "buffer_amount": [0.0, config.max_buffer_units],
            "reinforce_amount": [0.0, config.max_reinforce_amount],
            "expedite_priority": [0.0, config.max_expedite_priority],
        },
        "state": history[-1] if history else {},
        "recent_history": history[-8:],
        "nodes": [
            {
                "name": node.name,
                "kind": node.kind,
                "inventory": round(node.inventory, 2),
                "capacity": round(node.capacity, 2),
                "max_inventory": round(node.max_inventory, 2),
                "demand": round(node.demand, 2),
                "health": round(node.health, 3),
                "protected_stock": round(node.protected_stock, 2),
            }
            for node in network.nodes.values()
        ],
        "candidate_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "capacity": round(edge.capacity, 2),
                "health": round(edge.health, 3),
                "reinforced": round(edge.reinforced, 3),
            }
            for edge in sorted(network.edges, key=lambda item: item.capacity * item.health)[:8]
        ],
        "shortage_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "capacity": round(edge.capacity, 2),
                "health": round(edge.health, 3),
                "target_inventory": round(network.nodes[edge.target].inventory, 2),
            }
            for edge in _shortage_edges(network)[:8]
        ],
    }
    return (
        "You are a candidate generator for a Monte Carlo supply-chain controller. "
        "Propose diverse, plausible interventions. The simulator will evaluate "
        "your candidates against native Monte Carlo candidates and execute only "
        "the lowest-scoring one, so include high-value actions that a generic "
        "candidate generator might miss. Use only node and edge names from the "
        "payload. Each candidate can include at most one buffer action, one "
        "reinforce action, and one expedite action. Do not duplicate action "
        "types inside a candidate.\n\n"
        "Return only JSON with this shape:\n"
        "{\"candidates\":["
        "{\"actions\":["
        "{\"type\":\"buffer\",\"node\":\"NODE_NAME\",\"amount\":0.0},"
        "{\"type\":\"reinforce\",\"edge\":[\"SOURCE\",\"TARGET\"],\"amount\":0.0},"
        "{\"type\":\"expedite\",\"edge\":[\"SOURCE\",\"TARGET\"],\"priority\":0.0}"
        "],\"reason\":\"short reason\"}"
        "]}\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _codex_candidate_interventions_from_payload(
    payload: Dict[str, Any],
    network: SupplyChainNetwork,
    config: SupplyChainConfig,
    max_candidates: int,
) -> Tuple[List[Intervention], List[str], List[str]]:
    raw_candidates = payload.get("candidates")
    if raw_candidates is None and "actions" in payload:
        raw_candidates = [payload]
    errors: List[str] = []
    reasons: List[str] = []
    if not isinstance(raw_candidates, list):
        return [], ["candidates must be a list"], reasons

    candidates: List[Intervention] = []
    seen = set()
    for index, item in enumerate(raw_candidates[: max(1, max_candidates)]):
        if not isinstance(item, dict):
            errors.append(f"candidates[{index}] is not an object")
            continue
        intervention, valid_actions, candidate_errors = _intervention_from_codex_payload(
            item,
            network,
            config,
        )
        errors.extend(f"candidates[{index}]: {error}" for error in candidate_errors)
        normalized, budget = _normalize_intervention(network, intervention, config)
        key = _intervention_key(normalized)
        if valid_actions <= 0 or budget <= 0.0:
            continue
        if key in seen:
            errors.append(f"candidates[{index}]: duplicate candidate ignored")
            continue
        seen.add(key)
        candidates.append(normalized)
        reasons.append(str(item.get("reason", ""))[:240])
    return candidates, errors, reasons


def _shortlist_scored_candidates(
    scored: Sequence[Dict[str, Any]],
    shortlist_size: int,
) -> List[Dict[str, Any]]:
    limit = max(1, shortlist_size)
    selected: List[Dict[str, Any]] = []
    seen = set()

    def add(items: Sequence[Dict[str, Any]]) -> None:
        for item in items:
            if len(selected) >= limit:
                return
            candidate_id = item["candidate_id"]
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            selected.append(item)

    add(list(scored)[: max(3, limit // 2)])
    for score_key in ["service_score", "tail_score", "inventory_score", "budget_score"]:
        add(sorted(scored, key=lambda item: item[score_key])[:2])
    add(list(scored))
    return selected


def _scored_candidate_to_prompt_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["candidate_id"],
        "source": item["source"],
        "default_rank": item["default_rank"],
        "default_score": round(float(item["default_score"]), 3),
        "service_score": round(float(item["service_score"]), 3),
        "tail_score": round(float(item["tail_score"]), 3),
        "inventory_score": round(float(item["inventory_score"]), 3),
        "budget_score": round(float(item["budget_score"]), 3),
        "budget_used": round(float(item["budget_used"]), 3),
        "intervention": _intervention_to_dict(item["intervention"]),
    }


def _codex_judge_prompt(
    network: SupplyChainNetwork,
    history: List[Dict[str, float]],
    config: SupplyChainConfig,
    shortlist: Sequence[Dict[str, Any]],
    default_entry: Dict[str, Any],
    override_tolerance: float,
) -> str:
    payload = {
        "system": "directed supply-chain sabotage response",
        "objective": "choose one already evaluated Monte Carlo intervention candidate",
        "telemetry_note": (
            "Node inventories and health values are delayed, noisy, and quantized reports. "
            "They are the same controller-visible telemetry given to all policies, not hidden ground truth."
        ),
        "score_note": (
            "All scores are lower-is-better. default_score is the normal Monte Carlo risk objective. "
            "service_score, tail_score, inventory_score, and budget_score are alternate stress views. "
            "A guardrail rejects overrides whose default_score is more than override_tolerance worse "
            "than the default winner."
        ),
        "default_winner_id": default_entry["candidate_id"],
        "override_tolerance": override_tolerance,
        "threat_model": {
            "attack_probability": config.attack_probability,
            "attack_burst_probability": config.attack_burst_probability,
            "failure_probability": config.failure_probability,
            "demand_surge_probability": config.demand_surge_probability,
        },
        "action_budget": config.action_budget,
        "control_update_interval": config.control_update_interval,
        "state": history[-1] if history else {},
        "recent_history": history[-8:],
        "candidate_count": len(shortlist),
        "candidates": [_scored_candidate_to_prompt_dict(item) for item in shortlist],
    }
    return (
        "You are a high-level judge over a Monte Carlo supply-chain controller. "
        "Do not invent actions. Choose exactly one candidate id from the "
        "shortlist. Prefer the default winner unless another candidate has a "
        "credible robustness advantage in service, tail, inventory, or budget "
        "scores under the reported telemetry. Avoid overrides that spend more "
        "budget without a clear resilience benefit.\n\n"
        "Return only JSON with this shape:\n"
        "{\"choice\":\"c0\",\"reason\":\"short reason\"}\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _codex_admin_prompt(
    network: SupplyChainNetwork,
    history: List[Dict[str, float]],
    config: SupplyChainConfig,
    max_sample_multiplier: float,
) -> str:
    payload = {
        "system": "directed supply-chain sabotage response",
        "objective": "configure a Monte Carlo search controller under noisy reported telemetry",
        "allowed_risk_modes": ["balanced", "service", "tail", "inventory", "budget"],
        "sample_multiplier_limit": [1.0, max_sample_multiplier],
        "telemetry_note": (
            "Node inventories and health values are delayed, noisy, and quantized reports. "
            "They are the same controller-visible telemetry given to all policies, not hidden ground truth."
        ),
        "threat_model": {
            "attack_probability": config.attack_probability,
            "attack_burst_probability": config.attack_burst_probability,
            "failure_probability": config.failure_probability,
            "demand_surge_probability": config.demand_surge_probability,
        },
        "action_budget": config.action_budget,
        "control_update_interval": config.control_update_interval,
        "state": history[-1] if history else {},
        "recent_history": history[-8:],
        "nodes": [
            {
                "name": node.name,
                "kind": node.kind,
                "inventory": round(node.inventory, 2),
                "capacity": round(node.capacity, 2),
                "max_inventory": round(node.max_inventory, 2),
                "demand": round(node.demand, 2),
                "health": round(node.health, 3),
            }
            for node in network.nodes.values()
        ],
        "weak_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "capacity": round(edge.capacity, 2),
                "health": round(edge.health, 3),
                "reinforced": round(edge.reinforced, 3),
            }
            for edge in sorted(network.edges, key=lambda item: item.capacity * item.health)[:8]
        ],
        "shortage_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "capacity": round(edge.capacity, 2),
                "health": round(edge.health, 3),
                "target_inventory": round(network.nodes[edge.target].inventory, 2),
            }
            for edge in _shortage_edges(network)[:8]
        ],
    }
    return (
        "You are the high-level administrator of a Monte Carlo supply-chain "
        "controller. Do not choose the final action. Instead configure the "
        "search: candidate mix, risk mode, priority nodes/edges, objective "
        "weights, and bounded compute multiplier. The simulator will generate "
        "and evaluate actions from your plan. Use only node and edge names from "
        "the payload.\n\n"
        "Return only JSON with this shape:\n"
        "{\"candidate_mix\":{\"buffer\":0.25,\"reinforce\":0.25,"
        "\"expedite\":0.20,\"combined\":0.30},"
        "\"risk_mode\":\"balanced|service|tail|inventory|budget\","
        "\"sample_multiplier\":1.0,"
        "\"priority_nodes\":[\"NODE_NAME\"],"
        "\"priority_edges\":[[\"SOURCE\",\"TARGET\"]],"
        "\"objective_weights\":{\"inventory_gap\":1.0,\"node_health\":80.0,"
        "\"edge_health\":55.0,\"retailer_gap\":2.0,\"action_budget\":65.0},"
        "\"reason\":\"short reason\"}\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _admin_plan_from_payload(
    payload: Dict[str, Any],
    network: SupplyChainNetwork,
    max_sample_multiplier: float,
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    plan = _default_admin_plan(active=True)

    mix = payload.get("candidate_mix", {})
    if isinstance(mix, dict):
        plan["candidate_mix"] = _normalized_candidate_mix(mix)
    else:
        errors.append("candidate_mix must be an object")

    risk_mode = str(payload.get("risk_mode", plan["risk_mode"])).lower()
    if risk_mode in {"balanced", "service", "tail", "inventory", "budget"}:
        plan["risk_mode"] = risk_mode
    else:
        errors.append(f"unsupported risk_mode: {risk_mode}")

    plan["sample_multiplier"] = _clip(
        _safe_float(payload.get("sample_multiplier"), 1.0),
        1.0,
        max(1.0, max_sample_multiplier),
    )

    priority_nodes = []
    for name in payload.get("priority_nodes", []):
        node_name = str(name).strip()
        if node_name in network.nodes:
            priority_nodes.append(node_name)
        else:
            errors.append(f"unknown priority node: {node_name}")
    plan["priority_nodes"] = priority_nodes[:6]

    priority_edges = []
    for item in payload.get("priority_edges", []):
        edge = _parse_edge({"edge": item})
        if _edge_exists(network, edge):
            priority_edges.append(edge)
        else:
            errors.append(f"unknown priority edge: {item}")
    plan["priority_edges"] = priority_edges[:8]

    raw_weights = payload.get("objective_weights", {})
    if isinstance(raw_weights, dict):
        defaults = plan["objective_weights"]
        plan["objective_weights"] = {
            "inventory_gap": _clip(_safe_float(raw_weights.get("inventory_gap"), defaults["inventory_gap"]), 0.2, 4.0),
            "node_health": _clip(_safe_float(raw_weights.get("node_health"), defaults["node_health"]), 20.0, 180.0),
            "edge_health": _clip(_safe_float(raw_weights.get("edge_health"), defaults["edge_health"]), 20.0, 160.0),
            "retailer_gap": _clip(_safe_float(raw_weights.get("retailer_gap"), defaults["retailer_gap"]), 0.5, 8.0),
            "action_budget": _clip(_safe_float(raw_weights.get("action_budget"), defaults["action_budget"]), 20.0, 160.0),
        }
    else:
        errors.append("objective_weights must be an object")

    plan["reason"] = str(payload.get("reason", ""))[:240]
    return plan, errors


def _default_admin_plan(active: bool) -> Dict[str, Any]:
    return {
        "active": active,
        "candidate_mix": {"buffer": 0.35, "reinforce": 0.25, "expedite": 0.15, "combined": 0.25},
        "risk_mode": "balanced",
        "sample_multiplier": 1.0,
        "priority_nodes": [],
        "priority_edges": [],
        "objective_weights": {
            "inventory_gap": 1.0,
            "node_health": 80.0,
            "edge_health": 55.0,
            "retailer_gap": 2.0,
            "action_budget": 65.0,
        },
        "reason": "",
    }


def _admin_plan_to_dict(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "active": bool(plan.get("active", False)),
        "candidate_mix": dict(plan.get("candidate_mix", {})),
        "risk_mode": str(plan.get("risk_mode", "balanced")),
        "sample_multiplier": _safe_float(plan.get("sample_multiplier"), 1.0),
        "priority_nodes": list(plan.get("priority_nodes", [])),
        "priority_edges": [
            list(edge)
            for edge in plan.get("priority_edges", [])
            if isinstance(edge, (list, tuple)) and len(edge) == 2
        ],
        "objective_weights": dict(plan.get("objective_weights", {})),
        "reason": str(plan.get("reason", ""))[:240],
    }


def _normalized_candidate_mix(raw_mix: Dict[str, Any]) -> Dict[str, float]:
    keys = ["buffer", "reinforce", "expedite", "combined"]
    values = {key: max(0.0, _safe_float(raw_mix.get(key), 0.0)) for key in keys}
    total = sum(values.values())
    if total <= 0.0:
        return _default_admin_plan(active=True)["candidate_mix"]
    return {key: values[key] / total for key in keys}


def _admin_guided_candidates(
    network: SupplyChainNetwork,
    plan: Dict[str, Any],
    rng: np.random.Generator,
    config: SupplyChainConfig,
) -> List[Intervention]:
    if not plan.get("active", True):
        return []

    mix = plan.get("candidate_mix", _default_admin_plan(active=True)["candidate_mix"])
    priority_nodes = [
        network.nodes[name]
        for name in plan.get("priority_nodes", [])
        if name in network.nodes and network.nodes[name].kind in {"warehouse", "retailer"}
    ]
    storage_nodes = [
        node
        for node in network.nodes.values()
        if node.kind in {"warehouse", "retailer"} and node.max_inventory > 0.0
    ]
    storage_nodes = _unique_nodes(
        priority_nodes
        + sorted(storage_nodes, key=lambda node: node.inventory / max(node.max_inventory, 1.0))
    )

    priority_edges = [
        network.edge(*edge)
        for edge in plan.get("priority_edges", [])
        if _edge_exists(network, edge)
    ]
    weak_edges = sorted(network.edges, key=lambda edge: edge.health * edge.capacity)
    shortage_edges = _shortage_edges(network)
    edges = _unique_edges(priority_edges + weak_edges + shortage_edges)

    intensity = 0.70 if plan.get("risk_mode") == "budget" else 1.0
    candidates: List[Intervention] = []
    counts = {
        key: max(1, int(round(14.0 * float(mix.get(key, 0.0)))))
        for key in ["buffer", "reinforce", "expedite", "combined"]
    }

    for node in storage_nodes[: counts["buffer"]]:
        candidates.append(
            Intervention(
                buffer_node=node.name,
                buffer_amount=config.max_buffer_units * intensity,
            )
        )

    for edge in edges[: counts["reinforce"]]:
        candidates.append(
            Intervention(
                reinforce_edge=(edge.source, edge.target),
                reinforce_amount=config.max_reinforce_amount * intensity,
            )
        )

    expedite_edges = _unique_edges(priority_edges + shortage_edges + weak_edges)
    for edge in expedite_edges[: counts["expedite"]]:
        candidates.append(
            Intervention(
                expedite_edge=(edge.source, edge.target),
                expedite_priority=config.max_expedite_priority * intensity,
            )
        )

    combined_nodes = storage_nodes[: max(1, counts["combined"])]
    combined_edges = _unique_edges(priority_edges + shortage_edges + weak_edges)[: max(1, counts["combined"])]
    for index, edge in enumerate(combined_edges):
        node = combined_nodes[index % len(combined_nodes)] if combined_nodes else None
        candidates.append(
            Intervention(
                buffer_node=node.name if node else None,
                buffer_amount=config.max_buffer_units * intensity if node else 0.0,
                reinforce_edge=(edge.source, edge.target),
                reinforce_amount=config.max_reinforce_amount * intensity,
                expedite_edge=(edge.source, edge.target),
                expedite_priority=config.max_expedite_priority * intensity,
            )
        )

    rng.shuffle(candidates)
    return candidates


def _judge_extra_candidates(
    network: SupplyChainNetwork,
    rng: np.random.Generator,
    config: SupplyChainConfig,
) -> List[Intervention]:
    storage_nodes = [
        node
        for node in network.nodes.values()
        if node.kind in {"warehouse", "retailer"} and node.max_inventory > 0.0
    ]
    low_storage = sorted(
        storage_nodes,
        key=lambda node: node.inventory / max(node.max_inventory, 1.0),
    )
    shortage_edges = _shortage_edges(network)
    weak_edges = sorted(network.edges, key=lambda edge: edge.health * edge.capacity)

    service_plan = _default_admin_plan(active=True)
    service_plan.update(
        {
            "candidate_mix": {"buffer": 0.20, "reinforce": 0.20, "expedite": 0.25, "combined": 0.35},
            "risk_mode": "service",
            "priority_nodes": [node.name for node in low_storage[:5]],
            "priority_edges": [(edge.source, edge.target) for edge in shortage_edges[:6]],
        }
    )
    tail_plan = _default_admin_plan(active=True)
    tail_plan.update(
        {
            "candidate_mix": {"buffer": 0.25, "reinforce": 0.30, "expedite": 0.15, "combined": 0.30},
            "risk_mode": "tail",
            "priority_nodes": [node.name for node in low_storage[:5]],
            "priority_edges": [
                (edge.source, edge.target)
                for edge in _unique_edges(weak_edges[:5] + shortage_edges[:5])
            ],
        }
    )

    candidates = _admin_guided_candidates(network, service_plan, rng, config)
    candidates.extend(_admin_guided_candidates(network, tail_plan, rng, config))

    for node in low_storage[:4]:
        candidates.append(
            Intervention(
                buffer_node=node.name,
                buffer_amount=config.max_buffer_units * 0.65,
            )
        )

    for edge in _unique_edges(shortage_edges[:4] + weak_edges[:4]):
        candidates.append(
            Intervention(
                reinforce_edge=(edge.source, edge.target),
                reinforce_amount=config.max_reinforce_amount * 0.65,
                expedite_edge=(edge.source, edge.target),
                expedite_priority=config.max_expedite_priority * 0.75,
            )
        )

    rng.shuffle(candidates)
    return candidates


def _unique_nodes(nodes: Sequence[Node]) -> List[Node]:
    seen = set()
    unique: List[Node] = []
    for node in nodes:
        if node.name in seen:
            continue
        seen.add(node.name)
        unique.append(node)
    return unique


def _unique_edges(edges: Sequence[Edge]) -> List[Edge]:
    seen = set()
    unique: List[Edge] = []
    for edge in edges:
        key = (edge.source, edge.target)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def _intervention_key(action: Intervention) -> Tuple[Any, ...]:
    return (
        action.buffer_node,
        round(action.buffer_amount, 6),
        action.reinforce_edge,
        round(action.reinforce_amount, 6),
        action.expedite_edge,
        round(action.expedite_priority, 6),
    )


def _normalize_intervention(
    network: SupplyChainNetwork,
    action: Intervention,
    config: SupplyChainConfig,
) -> Tuple[Intervention, float]:
    buffer_node = action.buffer_node if action.buffer_node in network.nodes else None
    buffer_amount = _clip(action.buffer_amount, 0.0, config.max_buffer_units) if buffer_node else 0.0

    reinforce_edge = action.reinforce_edge if _edge_exists(network, action.reinforce_edge) else None
    reinforce_amount = (
        _clip(action.reinforce_amount, 0.0, config.max_reinforce_amount)
        if reinforce_edge
        else 0.0
    )

    expedite_edge = action.expedite_edge if _edge_exists(network, action.expedite_edge) else None
    expedite_priority = (
        _clip(action.expedite_priority, 0.0, config.max_expedite_priority)
        if expedite_edge
        else 0.0
    )

    normalized = Intervention(
        buffer_node=buffer_node if buffer_amount > 0.0 else None,
        buffer_amount=buffer_amount,
        reinforce_edge=reinforce_edge if reinforce_amount > 0.0 else None,
        reinforce_amount=reinforce_amount,
        expedite_edge=expedite_edge if expedite_priority > 0.0 else None,
        expedite_priority=expedite_priority,
    )
    budget = _intervention_budget(normalized, config)
    if budget <= config.action_budget or budget <= 0.0:
        return normalized, min(budget, config.action_budget)

    scale = config.action_budget / budget
    scaled = Intervention(
        buffer_node=normalized.buffer_node,
        buffer_amount=normalized.buffer_amount * scale,
        reinforce_edge=normalized.reinforce_edge,
        reinforce_amount=normalized.reinforce_amount * scale,
        expedite_edge=normalized.expedite_edge,
        expedite_priority=normalized.expedite_priority * scale,
    )
    return scaled, _intervention_budget(scaled, config)


def _intervention_budget(action: Intervention, config: SupplyChainConfig) -> float:
    budget = 0.0
    if action.buffer_node and action.buffer_amount > 0.0:
        budget += config.buffer_budget_weight * action.buffer_amount / max(config.max_buffer_units, 1.0)
    if action.reinforce_edge and action.reinforce_amount > 0.0:
        budget += (
            config.reinforce_budget_weight
            * action.reinforce_amount
            / max(config.max_reinforce_amount, 1.0e-9)
        )
    if action.expedite_edge and action.expedite_priority > 0.0:
        budget += (
            config.expedite_budget_weight
            * action.expedite_priority
            / max(config.max_expedite_priority, 1.0e-9)
        )
    return float(budget)


def _intervention_from_codex_payload(
    payload: Dict[str, Any],
    network: SupplyChainNetwork,
    config: SupplyChainConfig,
) -> Tuple[Intervention, int, List[str]]:
    actions = payload.get("actions", [])
    errors: List[str] = []
    if not isinstance(actions, list):
        return Intervention(), 0, ["actions must be a list"]

    intervention = Intervention()
    valid_actions = 0
    for index, item in enumerate(actions):
        if not isinstance(item, dict):
            errors.append(f"actions[{index}] is not an object")
            continue
        action_type = str(item.get("type", "")).strip().lower()
        if action_type == "buffer":
            if intervention.buffer_node is not None:
                errors.append("duplicate buffer action ignored")
                continue
            node_name = str(item.get("node", "")).strip()
            amount = _safe_float(item.get("amount"), 0.0)
            if node_name not in network.nodes or amount <= 0.0:
                errors.append(f"invalid buffer action at index {index}")
                continue
            intervention.buffer_node = node_name
            intervention.buffer_amount = min(amount, config.max_buffer_units)
            valid_actions += 1
        elif action_type == "reinforce":
            if intervention.reinforce_edge is not None:
                errors.append("duplicate reinforce action ignored")
                continue
            edge = _parse_edge(item)
            amount = _safe_float(item.get("amount"), 0.0)
            if not _edge_exists(network, edge) or amount <= 0.0:
                errors.append(f"invalid reinforce action at index {index}")
                continue
            intervention.reinforce_edge = edge
            intervention.reinforce_amount = min(amount, config.max_reinforce_amount)
            valid_actions += 1
        elif action_type == "expedite":
            if intervention.expedite_edge is not None:
                errors.append("duplicate expedite action ignored")
                continue
            edge = _parse_edge(item)
            priority = _safe_float(item.get("priority", item.get("amount")), 0.0)
            if not _edge_exists(network, edge) or priority <= 0.0:
                errors.append(f"invalid expedite action at index {index}")
                continue
            intervention.expedite_edge = edge
            intervention.expedite_priority = min(priority, config.max_expedite_priority)
            valid_actions += 1
        else:
            errors.append(f"unknown action type at index {index}: {action_type}")
    return intervention, valid_actions, errors


def _parse_edge(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    edge = item.get("edge")
    if isinstance(edge, list) and len(edge) == 2:
        return str(edge[0]).strip(), str(edge[1]).strip()
    source = item.get("source")
    target = item.get("target")
    if source is not None and target is not None:
        return str(source).strip(), str(target).strip()
    return None


def _intervention_to_dict(action: Intervention) -> Dict[str, Any]:
    return {
        "buffer_node": action.buffer_node,
        "buffer_amount": action.buffer_amount,
        "reinforce_edge": list(action.reinforce_edge) if action.reinforce_edge else None,
        "reinforce_amount": action.reinforce_amount,
        "expedite_edge": list(action.expedite_edge) if action.expedite_edge else None,
        "expedite_priority": action.expedite_priority,
    }


def _edge_exists(network: SupplyChainNetwork, edge: Optional[Tuple[str, str]]) -> bool:
    if edge is None:
        return False
    try:
        network.edge(*edge)
    except KeyError:
        return False
    return True


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _quantize(value: float, granularity: float) -> float:
    if granularity <= 0.0:
        return float(value)
    return float(round(float(value) / granularity) * granularity)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_intervention(
    network: SupplyChainNetwork,
    action: Intervention,
    config: SupplyChainConfig,
    rng: np.random.Generator,
    dry_run: bool = False,
) -> float:
    cost = 0.0
    if action.buffer_node and action.buffer_node in network.nodes:
        node = network.nodes[action.buffer_node]
        add = min(action.buffer_amount, node.max_inventory - node.inventory)
        node.inventory += max(0.0, add)
        protected_add = config.max_protected_stock_add * (
            action.buffer_amount / max(config.max_buffer_units, 1.0)
        )
        node.protected_stock = min(node.max_inventory * 0.55, node.protected_stock + protected_add)
        cost += 18.0 * action.buffer_amount / max(config.max_buffer_units, 1.0)
    if action.reinforce_edge:
        try:
            edge = network.edge(*action.reinforce_edge)
            edge.reinforced = min(1.0, edge.reinforced + action.reinforce_amount)
            health_repair = config.max_edge_health_repair * (
                action.reinforce_amount / max(config.max_reinforce_amount, 1.0e-9)
            )
            edge.health = min(1.0, edge.health + health_repair)
            cost += 14.0 * action.reinforce_amount / max(config.max_reinforce_amount, 1.0e-9)
        except KeyError:
            pass
    if action.expedite_edge:
        cost += 10.0 * action.expedite_priority / max(config.max_expedite_priority, 1.0e-9)
    return 0.0 if dry_run else cost


def _apply_disruptions(
    network: SupplyChainNetwork,
    config: SupplyChainConfig,
    rng: np.random.Generator,
) -> float:
    loss = 0.0
    if rng.random() < config.attack_probability:
        attack_count = 1
        if rng.random() < config.attack_burst_probability:
            attack_count += 1
        if rng.random() < config.attack_burst_probability * 0.35:
            attack_count += 1
        for _ in range(attack_count):
            if rng.random() < 0.68:
                target = _weighted_node_target(network, rng)
                destroyed = min(target.inventory, rng.uniform(24.0, 66.0))
                protected = min(target.protected_stock, destroyed * 0.82)
                actual_loss = max(0.0, destroyed - protected)
                target.inventory -= actual_loss
                target.protected_stock = max(0.0, target.protected_stock - protected)
                target.health = max(0.10, target.health - rng.uniform(0.24, 0.58))
                loss += 4.0 * actual_loss + 85.0 * (1.0 - target.health)
            else:
                edge = _weighted_edge_target(network, rng)
                mitigation = 1.0 - 0.78 * edge.reinforced
                damage = rng.uniform(0.28, 0.72) * max(0.12, mitigation)
                edge.health = max(0.08, edge.health - damage)
                loss += 120.0 * damage
    if rng.random() < config.failure_probability:
        node = str(rng.choice(list(network.nodes)))
        network.nodes[node].health = max(0.22, network.nodes[node].health - rng.uniform(0.10, 0.26))
        loss += 35.0
    return loss


def _recover(network: SupplyChainNetwork, config: SupplyChainConfig) -> None:
    for node in network.nodes.values():
        node.health = min(1.0, node.health + config.node_recovery_rate)
        node.protected_stock *= 0.995
    for edge in network.edges:
        edge.health = min(1.0, edge.health + config.edge_recovery_rate + 0.02 * edge.reinforced)
        edge.reinforced *= 0.998


def _produce(network: SupplyChainNetwork) -> None:
    for supplier in network.suppliers():
        supplier.inventory = min(
            supplier.max_inventory,
            supplier.inventory + supplier.capacity * supplier.health,
        )
    for factory in network.factories():
        inbound_stock = factory.inventory
        produced = min(factory.capacity * factory.health, inbound_stock * 0.55)
        factory.inventory = min(factory.max_inventory, factory.inventory - produced * 0.45 + produced)


def _ship(
    network: SupplyChainNetwork,
    config: SupplyChainConfig,
    rng: np.random.Generator,
    action: Intervention,
) -> None:
    desired = _desired_inventory(network)
    edge_order = sorted(
        network.edges,
        key=lambda edge: desired.get(edge.target, 0.0) - network.nodes[edge.target].inventory,
        reverse=True,
    )
    for edge in edge_order:
        source = network.nodes[edge.source]
        target = network.nodes[edge.target]
        deficit = max(0.0, desired.get(edge.target, 0.0) - target.inventory)
        if deficit <= 0.0 or source.inventory <= 0.0:
            continue
        expedite = action.expedite_edge == (edge.source, edge.target)
        expedite_multiplier = 1.0
        if expedite:
            expedite_multiplier += (
                config.max_expedite_multiplier - 1.0
            ) * action.expedite_priority
        capacity = edge.capacity * edge.health * expedite_multiplier
        amount = min(source.inventory * config.max_ship_fraction, capacity, deficit)
        source.inventory -= amount
        target.inventory = min(target.max_inventory, target.inventory + amount)


def _consume_demand(
    network: SupplyChainNetwork,
    config: SupplyChainConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    total_demand = 0.0
    total_unmet = 0.0
    surge_target = None
    if rng.random() < config.demand_surge_probability:
        surge_target = rng.choice(network.retailers()).name
    for retailer in network.retailers():
        demand = max(0.0, retailer.demand * rng.normal(1.0, config.demand_noise))
        if retailer.name == surge_target:
            demand *= rng.uniform(config.demand_surge_min, config.demand_surge_max)
        served = min(retailer.inventory, demand)
        retailer.inventory -= served
        total_demand += demand
        total_unmet += demand - served
    return total_demand, total_unmet


def _record(
    network: SupplyChainNetwork,
    history: List[Dict[str, float]],
    step: int,
    unmet: np.ndarray,
    service: np.ndarray,
    loss: np.ndarray,
    attack_loss: np.ndarray,
    inventory: np.ndarray,
    active_capacity: np.ndarray,
) -> None:
    inventory[step] = sum(node.inventory for node in network.nodes.values())
    active_capacity[step] = sum(node.capacity * node.health for node in network.nodes.values())
    history.append(
        {
            "step": float(step),
            "unmet_demand": float(unmet[step]),
            "service_level": float(service[step]),
            "economic_loss": float(loss[step]),
            "attack_loss": float(attack_loss[step]),
            "inventory": float(inventory[step]),
            "active_capacity": float(active_capacity[step]),
        }
    )


def _record_controller_history(
    observed_network: SupplyChainNetwork,
    history: List[Dict[str, float]],
    step: int,
    unmet: np.ndarray,
    service: np.ndarray,
    loss: np.ndarray,
    attack_loss: np.ndarray,
) -> None:
    reported_inventory = sum(node.inventory for node in observed_network.nodes.values())
    reported_capacity = sum(
        node.capacity * node.health for node in observed_network.nodes.values()
    )
    history.append(
        {
            "step": float(step),
            "unmet_demand": float(unmet[step]),
            "service_level": float(service[step]),
            "economic_loss": float(loss[step]),
            "attack_loss": float(attack_loss[step]),
            "reported_inventory": float(reported_inventory),
            "reported_active_capacity": float(reported_capacity),
        }
    )


def _generic_expedite_edge(
    network: SupplyChainNetwork,
    target: Optional[Node],
) -> Optional[Edge]:
    if target is None:
        return None
    inbound = network.incoming(target.name)
    if not inbound:
        return None
    stocked = [
        edge
        for edge in inbound
        if network.nodes[edge.source].inventory > max(5.0, target.demand * 0.25)
    ]
    candidates = stocked if stocked else inbound
    return max(
        candidates,
        key=lambda edge: (
            edge.health
            * edge.capacity
            * network.nodes[edge.source].inventory
            / max(network.nodes[edge.source].max_inventory, 1.0)
        ),
    )


def _rule_action_cost_rank(action_type: str) -> int:
    ranks = {
        "expedite": 1,
        "reinforce": 2,
        "buffer": 3,
    }
    return ranks.get(action_type, 99)


def _desired_inventory(network: SupplyChainNetwork) -> Dict[str, float]:
    desired: Dict[str, float] = {}
    downstream_demand = sum(node.demand for node in network.retailers())
    for node in network.nodes.values():
        if node.kind == "retailer":
            desired[node.name] = min(node.max_inventory, node.demand * 3.0)
        elif node.kind == "warehouse":
            desired[node.name] = min(node.max_inventory, downstream_demand * 1.15)
        elif node.kind == "factory":
            desired[node.name] = min(node.max_inventory, node.capacity * 2.2)
        else:
            desired[node.name] = min(node.max_inventory, node.capacity * 2.5)
    return desired


def _critical_buffer_node(network: SupplyChainNetwork) -> Optional[Node]:
    candidates = network.warehouses() + network.retailers()
    if not candidates:
        return None
    return min(candidates, key=lambda node: node.inventory / max(node.max_inventory, 1.0))


def _candidate_interventions(
    network: SupplyChainNetwork,
    rng: np.random.Generator,
    config: SupplyChainConfig,
) -> List[Intervention]:
    candidates = [Intervention()]
    critical = _critical_buffer_node(network)
    weak_edges = sorted(network.edges, key=lambda edge: edge.health * edge.capacity)
    demand_edges = sorted(
        network.edges,
        key=lambda edge: network.nodes[edge.target].demand if edge.target in network.nodes else 0.0,
        reverse=True,
    )
    if critical is not None:
        candidates.append(
            Intervention(buffer_node=critical.name, buffer_amount=config.max_buffer_units)
        )
    for edge in weak_edges[:5]:
        candidates.append(
            Intervention(
                reinforce_edge=(edge.source, edge.target),
                reinforce_amount=config.max_reinforce_amount,
            )
        )
    for edge in demand_edges[:5]:
        candidates.append(
            Intervention(
                expedite_edge=(edge.source, edge.target),
                expedite_priority=config.max_expedite_priority,
            )
        )
    if critical is not None:
        for edge in weak_edges[:4]:
            candidates.append(
                Intervention(
                    buffer_node=critical.name,
                    buffer_amount=config.max_buffer_units,
                    reinforce_edge=(edge.source, edge.target),
                    reinforce_amount=config.max_reinforce_amount,
                    expedite_edge=(edge.source, edge.target),
                    expedite_priority=config.max_expedite_priority,
                )
            )
    rng.shuffle(candidates)
    return candidates


def _shortage_edge(network: SupplyChainNetwork) -> Optional[Edge]:
    retailer = min(
        network.retailers(),
        key=lambda node: node.inventory / max(node.demand, 1.0),
        default=None,
    )
    if retailer is None:
        return None
    inbound = network.incoming(retailer.name)
    if not inbound:
        return None
    return max(inbound, key=lambda edge: edge.capacity * edge.health)


def _shortage_edges(network: SupplyChainNetwork) -> List[Edge]:
    retailers = sorted(
        network.retailers(),
        key=lambda node: node.inventory / max(node.demand, 1.0),
    )
    edges: List[Edge] = []
    seen = set()
    for retailer in retailers:
        inbound = sorted(
            network.incoming(retailer.name),
            key=lambda edge: edge.capacity * edge.health,
            reverse=True,
        )
        for edge in inbound:
            key = (edge.source, edge.target)
            if key not in seen:
                edges.append(edge)
                seen.add(key)
    return edges


def _bottleneck_edge(network: SupplyChainNetwork) -> Optional[Edge]:
    demand_pressure = _desired_inventory(network)
    candidates = sorted(
        network.edges,
        key=lambda edge: (
            demand_pressure.get(edge.target, 0.0) - network.nodes[edge.target].inventory,
            edge.capacity * edge.health,
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _short_horizon_risk(network: SupplyChainNetwork) -> float:
    desired = _desired_inventory(network)
    inventory_gap = sum(max(0.0, desired[name] - node.inventory) for name, node in network.nodes.items())
    unhealthy_nodes = sum(1.0 - node.health for node in network.nodes.values())
    unhealthy_edges = sum(1.0 - edge.health for edge in network.edges)
    retailer_gap = sum(
        max(0.0, node.demand * 2.0 - node.inventory)
        for node in network.retailers()
    )
    return inventory_gap + 80.0 * unhealthy_nodes + 55.0 * unhealthy_edges + 2.0 * retailer_gap


def _weighted_node_target(network: SupplyChainNetwork, rng: np.random.Generator) -> Node:
    nodes = list(network.nodes.values())
    weights = np.array([
        1.0 + node.inventory / 80.0 + (2.0 if node.kind in ("factory", "warehouse") else 0.0)
        for node in nodes
    ])
    weights = weights / np.sum(weights)
    return nodes[int(rng.choice(len(nodes), p=weights))]


def _weighted_edge_target(network: SupplyChainNetwork, rng: np.random.Generator) -> Edge:
    weights = np.array([edge.capacity * (1.0 - 0.45 * edge.reinforced) for edge in network.edges])
    weights = weights / np.sum(weights)
    return network.edges[int(rng.choice(len(network.edges), p=weights))]


def _mean_sem(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _save_policy_plot(policy_name: str, runs: List[RunResult], output_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"Supply-chain sabotage: {policy_name}", fontsize=15)
    series = [
        ("unmet_demand", "Unmet demand", "#B23A48"),
        ("service_level", "Service level", "#2B8C67"),
        ("economic_loss", "Economic loss", "#9C5A2E"),
        ("total_inventory", "Total inventory", "#4F5D75"),
    ]
    for axis, (name, label, color) in zip(axes, series):
        values = np.vstack([getattr(run, name) for run in runs])
        mean, sem = _mean_sem(values)
        axis.plot(runs[0].time, mean, color=color, linewidth=2.0)
        axis.fill_between(runs[0].time, mean - sem, mean + sem, color=color, alpha=0.16, linewidth=0)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
    axes[-1].set_xlabel("Simulation step")
    return _save(fig, output_dir / policy_name)


def _save_dashboard(
    results: Dict[str, List[RunResult]],
    output_dir: Path,
    config: Optional[SupplyChainConfig] = None,
) -> List[Path]:
    fig = plt.figure(figsize=(13.5, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    fig.suptitle("Supply-chain sabotage response dashboard", fontsize=16)
    colors = ["#4F5D75", "#2B8C67", "#9C5A2E", "#6C5B7B", "#B23A48"]
    panels = [
        (grid[0, 0], "unmet_demand", "Unmet demand"),
        (grid[1, 0], "service_level", "Service level"),
    ]
    for panel, attribute, ylabel in panels:
        axis = fig.add_subplot(panel)
        for index, (policy_name, runs) in enumerate(results.items()):
            values = np.vstack([getattr(run, attribute) for run in runs])
            mean, sem = _mean_sem(values)
            axis.plot(runs[0].time, mean, color=colors[index % len(colors)], linewidth=2.0, label=_title(policy_name))
            axis.fill_between(runs[0].time, mean - sem, mean + sem, color=colors[index % len(colors)], alpha=0.12, linewidth=0)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
        axis.legend(frameon=False, fontsize=8)
    summary = summarize(results, config=config)
    policies = [item["policy"] for item in summary["ranking_lower_is_better"]]
    scores = [summary["metrics"][policy]["sabotage_impact_score_lower_is_better"] for policy in policies]
    losses = [summary["metrics"][policy]["economic_loss_mean"] for policy in policies]

    loss_axis = fig.add_subplot(grid[0, 1])
    loss_axis.bar([_title(name) for name in policies], losses, color=colors[: len(policies)], alpha=0.86)
    loss_axis.set_title("Economic loss\n(lower is better)")
    loss_axis.set_ylabel("Mean total loss")
    loss_axis.tick_params(axis="x", rotation=25)
    loss_axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)

    score_axis = fig.add_subplot(grid[1, 1])
    score_axis.bar([_title(name) for name in policies], scores, color=colors[: len(policies)], alpha=0.86)
    score_axis.set_title("Sabotage impact score\n(lower is better)")
    score_axis.set_ylabel("Impact score")
    score_axis.tick_params(axis="x", rotation=25)
    score_axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)
    return _save(fig, output_dir / "dashboard")


def _save_network_map(results: Dict[str, List[RunResult]], output_dir: Path) -> List[Path]:
    network = SupplyChainNetwork()
    fig, axis = plt.subplots(figsize=(13.0, 8.0), constrained_layout=True)
    kind_colors = {
        "supplier": "#4F5D75",
        "factory": "#9C5A2E",
        "warehouse": "#2B8C67",
        "retailer": "#B23A48",
    }
    for edge in network.edges:
        start = network.nodes[edge.source].position
        end = network.nodes[edge.target].position
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#AAB3BD", "lw": 1.3, "shrinkA": 10, "shrinkB": 10},
        )
    for node in network.nodes.values():
        axis.scatter(
            [node.position[0]],
            [node.position[1]],
            s=250,
            color=kind_colors[node.kind],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        axis.text(node.position[0], node.position[1] - 0.16, node.name, ha="center", va="top", fontsize=7)
    axis.set_title(
        f"Directed supply-chain network ({len(network.nodes)} nodes, {len(network.edges)} shipment routes)"
    )
    axis.set_xticks([])
    axis.set_yticks([])
    x_values = [node.position[0] for node in network.nodes.values()]
    y_values = [node.position[1] for node in network.nodes.values()]
    axis.set_xlim(min(x_values) - 0.45, max(x_values) + 0.45)
    axis.set_ylim(min(y_values) - 0.55, max(y_values) + 0.45)
    return _save(fig, output_dir / "network_map")


def _save(fig: plt.Figure, base_path: Path) -> List[Path]:
    png = base_path.with_suffix(".png")
    pdf = base_path.with_suffix(".pdf")
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def _extract_session_id(value) -> Optional[str]:
    uuid_pattern = (
        r"[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}"
    )
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and ("session" in key_lower or "thread" in key_lower):
                if re.fullmatch(uuid_pattern, item.strip()):
                    return item.strip()
            nested = _extract_session_id(item)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _extract_session_id(item)
            if nested is not None:
                return nested
    return None


def _format_codex_exception(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or "").strip()
        if stderr:
            return f"{type(exc).__name__}: {stderr.splitlines()[-1]}"
    return type(exc).__name__
