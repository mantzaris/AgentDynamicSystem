import copy
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    reinforce_edge: Optional[Tuple[str, str]] = None
    expedite_edge: Optional[Tuple[str, str]] = None


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
    final_nodes: Dict[str, Node]
    final_edges: List[Edge]
    policy_name: str


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

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
    ) -> Intervention:
        raise NotImplementedError


class BaselinePolicy(Policy):
    name = "baseline"

    def choose_action(self, *args, **kwargs) -> Intervention:
        return Intervention()


class RuleBasedPolicy(Policy):
    name = "rule_based"

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
    ) -> Intervention:
        retailer = min(network.retailers(), key=lambda node: node.inventory / max(node.demand, 1.0))
        inbound = network.incoming(retailer.name)
        edge = max(inbound, key=lambda item: item.capacity * item.health, default=None)
        buffer_node = _critical_buffer_node(network)
        return Intervention(
            buffer_node=buffer_node.name if buffer_node is not None else None,
            reinforce_edge=(edge.source, edge.target) if edge is not None else None,
            expedite_edge=(edge.source, edge.target) if edge is not None else None,
        )


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
    ) -> Intervention:
        candidates = _candidate_interventions(network, rng)
        if not candidates:
            return Intervention()
        scored: List[Tuple[float, Intervention]] = []
        for action in candidates[: self.samples]:
            trial = network.copy()
            _apply_intervention(trial, action, SupplyChainConfig(), rng, dry_run=True)
            score = _short_horizon_risk(trial)
            scored.append((score, action))
        scored.sort(key=lambda item: item[0])
        return scored[0][1]


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
        self.current_tactic = "combined_response"
        self._session_id: Optional[str] = None

    def reset_for_run(self) -> None:
        self.current_tactic = "combined_response"
        self._session_id = None

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, float]],
        step: int,
        rng: np.random.Generator,
    ) -> Intervention:
        if self._should_consult(step, history):
            self.current_tactic = self._consult_codex(network, history)
        return self._action_for_tactic(network, self.current_tactic)

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
    ) -> str:
        payload = {
            "system": "directed supply-chain sabotage response",
            "objective": "minimize unmet demand and economic loss under random failures and adversarial attacks",
            "allowed_tactics": [
                "buffer_critical",
                "reinforce_bottleneck",
                "expedite_shortage",
                "combined_response",
            ],
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
            "weakest_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "capacity": round(edge.capacity, 2),
                    "health": round(edge.health, 3),
                    "reinforced": round(edge.reinforced, 3),
                }
                for edge in sorted(network.edges, key=lambda item: item.capacity * item.health)[:6]
            ],
        }
        prompt = (
            "You are a supply-chain sabotage response advisor in a directed-graph "
            "agent simulation. Choose one high-level intervention tactic for the "
            "next interval. Return only JSON: "
            "{\"tactic\":\"buffer_critical|reinforce_bottleneck|expedite_shortage|combined_response\","
            "\"reason\":\"short reason\"}.\n\n"
            f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
        )
        step = int(payload["state"].get("step", -1)) if payload["state"] else -1
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            print(
                f"  {self.name}: Codex unavailable after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using combined_response.",
                flush=True,
            )
            return "combined_response"
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            print(f"  {self.name}: Codex returned no JSON; using combined_response.", flush=True)
            return "combined_response"
        try:
            tactic = json.loads(match.group(0)).get("tactic", "combined_response")
        except json.JSONDecodeError:
            print(f"  {self.name}: Codex JSON parse failed; using combined_response.", flush=True)
            return "combined_response"
        allowed = {"buffer_critical", "reinforce_bottleneck", "expedite_shortage", "combined_response"}
        if tactic not in allowed:
            print(f"  {self.name}: Codex tactic was invalid; using combined_response.", flush=True)
            return "combined_response"
        elapsed = time.monotonic() - started_at
        print(f"  {self.name}: Codex chose {tactic} in {elapsed:.1f}s.", flush=True)
        return tactic

    def _ask_codex(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as output_file:
            output_path = Path(output_file.name)
        command = self._codex_command(output_path)
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

    def _codex_command(self, output_path: Path) -> List[str]:
        if self._session_id is None:
            return [
                self.codex_command,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--json",
                "--output-last-message",
                str(output_path),
                "--color",
                "never",
                "-",
            ]
        return [
            self.codex_command,
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(output_path),
            self._session_id,
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

    def _action_for_tactic(self, network: SupplyChainNetwork, tactic: str) -> Intervention:
        critical = _critical_buffer_node(network)
        shortage_edge = _shortage_edge(network)
        bottleneck = _bottleneck_edge(network)
        if tactic == "buffer_critical":
            return Intervention(buffer_node=critical.name if critical is not None else None)
        if tactic == "reinforce_bottleneck":
            return Intervention(
                reinforce_edge=(bottleneck.source, bottleneck.target) if bottleneck is not None else None
            )
        if tactic == "expedite_shortage":
            return Intervention(
                expedite_edge=(shortage_edge.source, shortage_edge.target) if shortage_edge is not None else None
            )
        return Intervention(
            buffer_node=critical.name if critical is not None else None,
            reinforce_edge=(bottleneck.source, bottleneck.target) if bottleneck is not None else None,
            expedite_edge=(shortage_edge.source, shortage_edge.target) if shortage_edge is not None else None,
        )


def run_simulation(
    config: SupplyChainConfig,
    policy: Policy,
    seed: int,
    progress_interval: int = 0,
) -> RunResult:
    rng = np.random.default_rng(seed)
    network = SupplyChainNetwork()
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
    history: List[Dict[str, float]] = []

    for step in range(config.steps + 1):
        _record(network, history, step, unmet, service, loss, attack_loss, inventory, active_capacity)
        if progress_interval > 0 and step % progress_interval == 0:
            print(
                f"  {policy.name}: step {step}/{config.steps} "
                f"unmet={unmet[step]:.1f} service={service[step]:.2f} "
                f"loss={loss[step]:.1f}.",
                flush=True,
            )
        if step == config.steps:
            break

        action = policy.choose_action(network, history, step, rng)
        action_cost = _apply_intervention(network, action, config, rng)
        buffer_actions[step + 1] = 1.0 if action.buffer_node else 0.0
        reinforce_actions[step + 1] = 1.0 if action.reinforce_edge else 0.0
        expedite_actions[step + 1] = 1.0 if action.expedite_edge else 0.0
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
        final_nodes=network.nodes,
        final_edges=network.edges,
        policy_name=policy.name,
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


def summarize(results: Dict[str, List[RunResult]]) -> Dict[str, object]:
    metrics: Dict[str, Dict[str, float]] = {}
    for policy_name, runs in results.items():
        total_unmet = np.array([np.sum(run.unmet_demand) for run in runs])
        total_loss = np.array([np.sum(run.economic_loss) for run in runs])
        service = np.array([np.mean(run.service_level) for run in runs])
        terminal_inventory = np.array([run.total_inventory[-1] for run in runs])
        score = (
            np.mean(total_unmet) / 2500.0
            + np.mean(total_loss) / 25000.0
            + 1.4 * np.mean(1.0 - service)
            + 0.15 * np.mean(1.0 / np.maximum(terminal_inventory / 500.0, 0.1))
        )
        metrics[policy_name] = {
            "total_unmet_demand_mean": float(np.mean(total_unmet)),
            "economic_loss_mean": float(np.mean(total_loss)),
            "service_level_mean": float(np.mean(service)),
            "terminal_inventory_mean": float(np.mean(terminal_inventory)),
            "sabotage_impact_score_lower_is_better": float(score),
        }
    ranking = [
        {
            "policy": name,
            "sabotage_impact_score_lower_is_better": values[
                "sabotage_impact_score_lower_is_better"
            ],
        }
        for name, values in metrics.items()
    ]
    ranking.sort(key=lambda item: item["sabotage_impact_score_lower_is_better"])
    return {"metrics": metrics, "ranking_lower_is_better": ranking}


def save_outputs(results: Dict[str, List[RunResult]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for policy_name, runs in results.items():
        print(f"Saving outputs for {policy_name}...", flush=True)
        written.extend(_save_policy_plot(policy_name, runs, output_dir))
    written.extend(_save_dashboard(results, output_dir))
    written.extend(_save_network_map(results, output_dir))
    summary = summarize(results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    written.append(summary_path)
    return written


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
        add = min(46.0, node.max_inventory - node.inventory)
        node.inventory += max(0.0, add)
        node.protected_stock = min(node.max_inventory * 0.55, node.protected_stock + 42.0)
        cost += 18.0
    if action.reinforce_edge:
        try:
            edge = network.edge(*action.reinforce_edge)
            edge.reinforced = min(1.0, edge.reinforced + 0.60)
            edge.health = min(1.0, edge.health + 0.45)
            cost += 14.0
        except KeyError:
            pass
    if action.expedite_edge:
        cost += 10.0
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
        capacity = edge.capacity * edge.health * (2.00 if expedite else 1.0)
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


def _candidate_interventions(network: SupplyChainNetwork, rng: np.random.Generator) -> List[Intervention]:
    candidates = [Intervention()]
    critical = _critical_buffer_node(network)
    weak_edges = sorted(network.edges, key=lambda edge: edge.health * edge.capacity)
    demand_edges = sorted(
        network.edges,
        key=lambda edge: network.nodes[edge.target].demand if edge.target in network.nodes else 0.0,
        reverse=True,
    )
    if critical is not None:
        candidates.append(Intervention(buffer_node=critical.name))
    for edge in weak_edges[:5]:
        candidates.append(Intervention(reinforce_edge=(edge.source, edge.target)))
    for edge in demand_edges[:5]:
        candidates.append(Intervention(expedite_edge=(edge.source, edge.target)))
    if critical is not None:
        for edge in weak_edges[:4]:
            candidates.append(
                Intervention(
                    buffer_node=critical.name,
                    reinforce_edge=(edge.source, edge.target),
                    expedite_edge=(edge.source, edge.target),
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


def _save_dashboard(results: Dict[str, List[RunResult]], output_dir: Path) -> List[Path]:
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
    summary = summarize(results)
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
