import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parent
DEFAULT_MAP = ROOT / "data" / "orlando_convention_roads.json"
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class UrbanResponseConfig:
    steps: int = 180
    initial_zombies: int = 150
    initial_civilians: int = 300
    initial_defenders: int = 30
    civilian_speed: float = 17.0
    zombie_speed: float = 34.0
    defender_speed: float = 29.0
    defender_visibility_m: float = 200.0
    zombie_detection_m: float = 230.0
    bite_radius_m: float = 52.0
    engagement_radius_m: float = 28.0
    neutralization_probability: float = 0.46
    defender_casualty_probability: float = 0.72
    noise_probability: float = 0.05
    max_steps_without_zombies: int = 3
    control_update_interval: int = 25
    monte_carlo_samples: int = 24


@dataclass
class Agent:
    kind: str
    node: str
    target: str
    progress: float
    previous_node: Optional[str] = None
    alive: bool = True


@dataclass
class RunResult:
    time: np.ndarray
    zombies: np.ndarray
    civilians: np.ndarray
    defenders: np.ndarray
    neutralized: np.ndarray
    converted: np.ndarray
    defender_losses: np.ndarray
    clearance_time: int
    zombie_victory: bool
    zombie_victory_time: int
    final_agents: List[Agent]
    policy_name: str
    map_name: str
    policy_stats: Dict[str, float]
    decision_log: List[Dict[str, Any]]


class RoadNetwork:
    def __init__(self, path: Path = DEFAULT_MAP) -> None:
        payload = json.loads(path.read_text())
        self.name = payload["name"]
        self.source_note = payload.get("source_note", "")
        self.latlon = {key: tuple(value) for key, value in payload["nodes"].items()}
        self.nodes = list(self.latlon)
        self.xy = self._project(self.latlon)
        self.edges = [(a, b, name) for a, b, name in payload["edges"]]
        self.neighbors: Dict[str, List[str]] = {node: [] for node in self.nodes}
        self.edge_names: Dict[Tuple[str, str], str] = {}
        self.edge_lengths: Dict[Tuple[str, str], float] = {}
        for a, b, name in self.edges:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
            self.edge_names[(a, b)] = name
            self.edge_names[(b, a)] = name
            length = float(np.linalg.norm(np.array(self.xy[a]) - np.array(self.xy[b])))
            self.edge_lengths[(a, b)] = max(length, 1.0)
            self.edge_lengths[(b, a)] = max(length, 1.0)
        self._shortest_paths: Dict[str, Dict[str, float]] = {}

    def position(self, agent: Agent) -> np.ndarray:
        start = np.array(self.xy[agent.node], dtype=float)
        end = np.array(self.xy[agent.target], dtype=float)
        return start + agent.progress * (end - start)

    def distance(self, a: Agent, b: Agent) -> float:
        return float(np.linalg.norm(self.position(a) - self.position(b)))

    def random_agent(self, kind: str, rng: np.random.Generator) -> Agent:
        node = str(rng.choice(self.nodes))
        target = self.random_neighbor(node, rng)
        return Agent(kind=kind, node=node, target=target, progress=float(rng.random()))

    def random_neighbor(self, node: str, rng: np.random.Generator) -> str:
        return str(rng.choice(self.neighbors[node]))

    def move(
        self,
        agent: Agent,
        speed: float,
        rng: np.random.Generator,
        next_target: Optional[Callable[[Agent, str, np.random.Generator], str]] = None,
    ) -> None:
        remaining = speed
        while remaining > 0.0:
            length = self.edge_lengths[(agent.node, agent.target)]
            distance_to_target = (1.0 - agent.progress) * length
            if remaining < distance_to_target:
                agent.progress += remaining / length
                return
            remaining -= distance_to_target
            previous = agent.node
            agent.previous_node = previous
            agent.node = agent.target
            agent.progress = 0.0
            if next_target is not None:
                agent.target = next_target(agent, previous, rng)
            else:
                choices = [node for node in self.neighbors[agent.node] if node != previous]
                if not choices:
                    choices = self.neighbors[agent.node]
                agent.target = str(rng.choice(choices))

    def route_next_node(self, start: str, goal: str) -> str:
        if start == goal:
            return goal
        distances, previous = self._dijkstra(start)
        if goal not in previous and goal != start:
            return start
        current = goal
        while previous.get(current) != start:
            current = previous.get(current, start)
            if current == start:
                return goal
        return current

    def nearest_node(self, point: np.ndarray) -> str:
        return min(self.nodes, key=lambda node: float(np.linalg.norm(np.array(self.xy[node]) - point)))

    def _dijkstra(self, start: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        distances = {node: float("inf") for node in self.nodes}
        distances[start] = 0.0
        previous: Dict[str, str] = {}
        pending = set(self.nodes)
        while pending:
            current = min(pending, key=lambda node: distances[node])
            pending.remove(current)
            for neighbor in self.neighbors[current]:
                distance = distances[current] + self.edge_lengths[(current, neighbor)]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
        return distances, previous

    @staticmethod
    def _project(latlon: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        lat0 = float(np.mean([lat for lat, _ in latlon.values()]))
        lon0 = float(np.mean([lon for _, lon in latlon.values()]))
        meters_per_lat = 111_320.0
        meters_per_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
        return {
            node: ((lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat)
            for node, (lat, lon) in latlon.items()
        }


class Policy:
    name = "policy"
    defender_mobility_factor = 1.0

    def reset_for_run(self) -> None:
        pass

    def decision_stats(self) -> Dict[str, float]:
        return {}

    def decision_log(self) -> List[Dict[str, Any]]:
        return []

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        raise NotImplementedError


class BaselinePatrolPolicy(Policy):
    name = "baseline"
    defender_mobility_factor = 0.35

    def choose_defender_targets(self, *args, **kwargs) -> Dict[int, Optional[Agent]]:
        defenders = args[1]
        return {index: None for index, _ in enumerate(defenders)}


class RuleBasedResponsePolicy(Policy):
    name = "rule_based"

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        targets = {}
        fallback_threat: Optional[Agent] = None
        for index, defender in enumerate(defenders):
            visible = _visible_zombies(network, defender, zombies, config.defender_visibility_m)
            if visible:
                targets[index] = min(visible, key=lambda zombie: network.distance(defender, zombie))
            else:
                if fallback_threat is None:
                    fallback_threat = _nearest_threat_to_civilians(network, zombies, civilians)
                targets[index] = fallback_threat
        return targets


class MonteCarloResponsePolicy(Policy):
    name = "monte_carlo"

    def __init__(self, samples: int = 24) -> None:
        self.samples = max(1, samples)

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        civilian_risk_by_zombie = _zombie_civilian_risk_map(network, zombies, civilians)
        candidates_by_defender = [
            _monte_carlo_target_candidates(
                network,
                defender,
                zombies,
                civilians,
                config,
                civilian_risk_by_zombie,
            )
            for defender in defenders
        ]
        if not any(candidates_by_defender):
            return {index: None for index, _ in enumerate(defenders)}

        best_targets: Dict[int, Optional[Agent]] = {}
        best_score = float("inf")
        deterministic = {
            index: (candidates[0] if candidates else None)
            for index, candidates in enumerate(candidates_by_defender)
        }
        for sample_index in range(self.samples):
            if sample_index == 0:
                sampled = deterministic
            else:
                sampled = {
                    index: _sample_target_candidate(candidates, rng)
                    for index, candidates in enumerate(candidates_by_defender)
                }
            score = _urban_tactic_score(
                network,
                defenders,
                zombies,
                civilians,
                sampled,
                config,
                civilian_risk_by_zombie=civilian_risk_by_zombie,
            )
            if score < best_score:
                best_score = score
                best_targets = sampled
        return best_targets


class CodexResponsePolicy(RuleBasedResponsePolicy):
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
        self.current_tactic = "nearest_threat"
        self._session_id: Optional[str] = None
        self.defender_mobility_factor = 1.0
        self._stats: Dict[str, float] = {}
        self._decision_log: List[Dict[str, Any]] = []

    def reset_for_run(self) -> None:
        self.current_tactic = "nearest_threat"
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

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        if self._should_consult(step, history):
            self.current_tactic = self._consult_codex(network, defenders, zombies, civilians, config, history)
        if self.current_tactic == "protect_civilians":
            return self._protect_civilians(network, defenders, zombies, civilians, config)
        if self.current_tactic == "contain_hotspot":
            return self._contain_hotspot(network, defenders, zombies, civilians, config)
        return super().choose_defender_targets(network, defenders, zombies, civilians, config, rng, step, history)

    def _should_consult(self, step: int, history: List[Dict[str, float]]) -> bool:
        if self.name == "codex_steady":
            return step % self.decision_interval == 0
        if self.name == "codex_guardian":
            if step == 0:
                return True
            if step % self.decision_interval != 0 or len(history) < 2:
                return False
            lookback = min(len(history), self.decision_interval + 1)
            recent = history[-1]
            previous = history[-lookback]
            return (
                recent["zombies"] > previous["zombies"]
                or recent["civilians"] < previous["civilians"] - 3
                or recent["defenders"] < previous["defenders"] - 1
                or recent["zombies"] > 1.25 * max(recent["defenders"], 1.0)
            )
        return False

    def _consult_codex(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        history: List[Dict[str, float]],
    ) -> str:
        self._stats["codex_calls"] += 1.0
        payload = {
            "system": "urban hostile-contagion response on a real road graph",
            "map": network.name,
            "objective": "minimize zombie clearance time while preserving civilians and defenders",
            "allowed_tactics": ["nearest_threat", "protect_civilians", "contain_hotspot"],
            "state": history[-1] if history else {},
            "recent_history": history[-8:],
            "agent_counts": {
                "zombies": len(zombies),
                "civilians": len(civilians),
                "defenders": len(defenders),
            },
        }
        prompt = (
            "You are an urban response tactical advisor in a road-network agent "
            "simulation. Choose one defender tactic for the next interval. "
            "Return only JSON: {\"tactic\":\"nearest_threat|protect_civilians|contain_hotspot\","
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
            self._stats["codex_failures"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error": _format_codex_exception(exc),
                    "tactic": "nearest_threat",
                }
            )
            print(
                f"  {self.name}: Codex unavailable after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using nearest_threat.",
                flush=True,
            )
            return "nearest_threat"
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "invalid_json",
                    "elapsed_seconds": time.monotonic() - started_at,
                    "tactic": "nearest_threat",
                }
            )
            print(f"  {self.name}: Codex returned no JSON; using nearest_threat.", flush=True)
            return "nearest_threat"
        try:
            payload_json = json.loads(match.group(0))
            tactic = payload_json.get("tactic", "nearest_threat")
        except json.JSONDecodeError:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "invalid_json",
                    "elapsed_seconds": time.monotonic() - started_at,
                    "tactic": "nearest_threat",
                }
            )
            print(f"  {self.name}: Codex JSON parse failed; using nearest_threat.", flush=True)
            return "nearest_threat"
        if tactic not in {"nearest_threat", "protect_civilians", "contain_hotspot"}:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            self._decision_log.append(
                {
                    "step": step,
                    "status": "invalid_tactic",
                    "elapsed_seconds": time.monotonic() - started_at,
                    "tactic": "nearest_threat",
                    "raw_tactic": str(tactic),
                }
            )
            print(f"  {self.name}: Codex tactic was invalid; using nearest_threat.", flush=True)
            return "nearest_threat"
        elapsed = time.monotonic() - started_at
        self._stats["codex_valid_responses"] += 1.0
        self._decision_log.append(
            {
                "step": step,
                "status": "valid",
                "elapsed_seconds": elapsed,
                "tactic": tactic,
                "reason": str(payload_json.get("reason", ""))[:240],
            }
        )
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

    def _protect_civilians(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
    ) -> Dict[int, Optional[Agent]]:
        return _protect_civilians_targets(network, defenders, zombies, civilians, config)

    def _contain_hotspot(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
    ) -> Dict[int, Optional[Agent]]:
        if not zombies:
            return {index: None for index, _ in enumerate(defenders)}
        centroid = np.mean([network.position(zombie) for zombie in zombies], axis=0)
        hotspot = min(zombies, key=lambda zombie: float(np.linalg.norm(network.position(zombie) - centroid)))
        return {index: hotspot for index, _ in enumerate(defenders)}


class CodexMonteCarloResponsePolicy(CodexResponsePolicy):
    def __init__(
        self,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
    ) -> None:
        super().__init__(
            name="codex_monte_carlo",
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.current_tactic = "monte_carlo"

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self.current_tactic = "monte_carlo"
        self._stats.update(
            {
                "hybrid_selected_native": 0.0,
                "hybrid_selected_codex": 0.0,
                "hybrid_evaluated_candidates": 0.0,
            }
        )

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        if step % self.decision_interval == 0:
            codex_candidates, consultation = self._consult_codex_tactic_candidates(
                network,
                defenders,
                zombies,
                civilians,
                config,
                history,
            )
            tagged = [("native", "monte_carlo")]
            tagged.extend(("codex", tactic) for tactic in codex_candidates)
            tactic, scoring = _select_urban_tactic(
                network,
                defenders,
                zombies,
                civilians,
                config,
                rng,
                tagged,
            )
            self.current_tactic = tactic
            if scoring["selected_source"] == "codex":
                self._stats["hybrid_selected_codex"] += 1.0
            else:
                self._stats["hybrid_selected_native"] += 1.0
            self._stats["hybrid_evaluated_candidates"] += float(scoring["evaluated_candidates"])
            consultation.update(scoring)
            consultation["selected_tactic"] = tactic
            self._decision_log.append(consultation)
            print(
                f"  {self.name}: selected {scoring['selected_source']} tactic "
                f"{tactic} (score={scoring['selected_score']:.2f}).",
                flush=True,
            )
        return _targets_for_tactic(
            self.current_tactic,
            network,
            defenders,
            zombies,
            civilians,
            config,
            rng,
        )

    def _consult_codex_tactic_candidates(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        history: List[Dict[str, float]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        self._stats["codex_calls"] += 1.0
        step = int(history[-1].get("step", -1)) if history else -1
        prompt = _codex_urban_candidate_prompt(network, defenders, zombies, civilians, history)
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex for tactic candidates at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex candidate call failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using native tactics.",
                flush=True,
            )
            return [], {
                "step": step,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": _format_codex_exception(exc),
            }
        payload = _parse_json_object(response)
        elapsed = time.monotonic() - started_at
        if payload is None:
            self._stats["codex_invalid_responses"] += 1.0
            return [], {"step": step, "status": "invalid_json", "elapsed_seconds": elapsed}
        candidates = _tactics_from_payload(payload)
        if candidates:
            self._stats["codex_valid_responses"] += 1.0
            status = "valid"
        else:
            self._stats["codex_noop_responses"] += 1.0
            status = "noop"
        return candidates, {
            "step": step,
            "status": status,
            "elapsed_seconds": elapsed,
            "codex_candidates": candidates,
            "reason": str(payload.get("reason", ""))[:240],
        }


class CodexMonteCarloAdminResponsePolicy(CodexResponsePolicy):
    def __init__(
        self,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
    ) -> None:
        super().__init__(
            name="codex_monte_carlo_admin",
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.current_tactic = "monte_carlo"
        self.current_score_plan: Dict[str, Any] = {"risk_mode": "balanced"}

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self.current_tactic = "monte_carlo"
        self.current_score_plan = {"risk_mode": "balanced"}
        self._stats.update(
            {
                "admin_selected_native": 0.0,
                "admin_selected_guided": 0.0,
                "admin_evaluated_candidates": 0.0,
            }
        )

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        if step % self.decision_interval == 0:
            plan, consultation = self._consult_admin_plan(network, defenders, zombies, civilians, history)
            native = [("native", "monte_carlo")]
            guided_tactics = _admin_guided_tactics(plan) if consultation.get("status") == "valid" else []
            guided = [("admin", tactic) for tactic in guided_tactics]
            tactic, scoring = _select_urban_tactic(
                network,
                defenders,
                zombies,
                civilians,
                config,
                rng,
                native + guided,
                score_plan=plan,
            )
            self.current_tactic = tactic
            self.current_score_plan = plan
            if scoring["selected_source"] == "admin":
                self._stats["admin_selected_guided"] += 1.0
            else:
                self._stats["admin_selected_native"] += 1.0
            self._stats["admin_evaluated_candidates"] += float(scoring["evaluated_candidates"])
            consultation.update(scoring)
            consultation["selected_tactic"] = tactic
            self._decision_log.append(consultation)
            print(
                f"  {self.name}: selected {scoring['selected_source']} tactic "
                f"{tactic} (risk={plan.get('risk_mode')}, score={scoring['selected_score']:.2f}).",
                flush=True,
            )
        return _targets_for_tactic(
            self.current_tactic,
            network,
            defenders,
            zombies,
            civilians,
            config,
            rng,
        )

    def _consult_admin_plan(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        history: List[Dict[str, float]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._stats["codex_calls"] += 1.0
        step = int(history[-1].get("step", -1)) if history else -1
        prompt = _codex_urban_admin_prompt(network, defenders, zombies, civilians, history)
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex for urban MC admin plan at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex admin call failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using balanced MC.",
                flush=True,
            )
            return _default_urban_admin_plan(), {
                "step": step,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": _format_codex_exception(exc),
            }
        payload = _parse_json_object(response)
        elapsed = time.monotonic() - started_at
        if payload is None:
            self._stats["codex_invalid_responses"] += 1.0
            return _default_urban_admin_plan(), {
                "step": step,
                "status": "invalid_json",
                "elapsed_seconds": elapsed,
            }
        plan = _urban_admin_plan_from_payload(payload)
        self._stats["codex_valid_responses"] += 1.0
        return plan, {
            "step": step,
            "status": "valid",
            "elapsed_seconds": elapsed,
            "admin_plan": plan,
            "reason": str(payload.get("reason", ""))[:240],
        }


class CodexMonteCarloJudgeResponsePolicy(CodexResponsePolicy):
    def __init__(
        self,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
        override_tolerance: float = 0.10,
    ) -> None:
        super().__init__(
            name="codex_monte_carlo_judge",
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.override_tolerance = max(0.0, override_tolerance)
        self.current_tactic = "monte_carlo"

    def reset_for_run(self) -> None:
        super().reset_for_run()
        self.current_tactic = "monte_carlo"
        self._stats.update(
            {
                "judge_selected_default": 0.0,
                "judge_selected_override": 0.0,
                "judge_guardrail_rejections": 0.0,
            }
        )

    def choose_defender_targets(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
        rng: np.random.Generator,
        step: int,
        history: List[Dict[str, float]],
    ) -> Dict[int, Optional[Agent]]:
        if step % self.decision_interval == 0:
            scored = _score_urban_tactics(
                network,
                defenders,
                zombies,
                civilians,
                config,
                rng,
                [("native", tactic) for tactic in _urban_candidate_tactics()],
            )
            default = scored[0] if scored else {"candidate_id": "c0", "tactic": "monte_carlo", "score": 0.0}
            choice, consultation = self._consult_judge(network, defenders, zombies, civilians, history, scored)
            selected = default
            guardrail_rejected = False
            if choice is not None:
                by_id = {item["candidate_id"]: item for item in scored}
                proposed = by_id.get(choice)
                if proposed is not None and proposed["score"] <= default["score"] * (1.0 + self.override_tolerance):
                    selected = proposed
                elif proposed is not None:
                    guardrail_rejected = True
            self.current_tactic = str(selected["tactic"])
            if selected["candidate_id"] == default["candidate_id"]:
                self._stats["judge_selected_default"] += 1.0
            else:
                self._stats["judge_selected_override"] += 1.0
            if guardrail_rejected:
                self._stats["judge_guardrail_rejections"] += 1.0
            consultation.update(
                {
                    "selected_tactic": self.current_tactic,
                    "default_tactic": default["tactic"],
                    "selected_score": selected["score"],
                    "default_score": default["score"],
                    "guardrail_rejected": guardrail_rejected,
                }
            )
            self._decision_log.append(consultation)
            print(
                f"  {self.name}: selected tactic {self.current_tactic} "
                f"(score={selected['score']:.2f}, default={default['score']:.2f}).",
                flush=True,
            )
        return _targets_for_tactic(
            self.current_tactic,
            network,
            defenders,
            zombies,
            civilians,
            config,
            rng,
        )

    def _consult_judge(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        history: List[Dict[str, float]],
        scored: Sequence[Dict[str, Any]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        self._stats["codex_calls"] += 1.0
        step = int(history[-1].get("step", -1)) if history else -1
        prompt = _codex_urban_judge_prompt(network, defenders, zombies, civilians, history, scored)
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex to judge urban MC shortlist at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex judge call failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using default MC tactic.",
                flush=True,
            )
            return None, {
                "step": step,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": _format_codex_exception(exc),
            }
        payload = _parse_json_object(response)
        elapsed = time.monotonic() - started_at
        if payload is None:
            self._stats["codex_invalid_responses"] += 1.0
            return None, {"step": step, "status": "invalid_json", "elapsed_seconds": elapsed}
        choice = str(payload.get("choice", payload.get("candidate_id", ""))).strip()
        allowed = {item["candidate_id"] for item in scored}
        if choice not in allowed:
            self._stats["codex_invalid_responses"] += 1.0
            return None, {
                "step": step,
                "status": "invalid_choice",
                "elapsed_seconds": elapsed,
                "choice": choice,
                "reason": str(payload.get("reason", ""))[:240],
            }
        self._stats["codex_valid_responses"] += 1.0
        return choice, {
            "step": step,
            "status": "valid",
            "elapsed_seconds": elapsed,
            "choice": choice,
            "reason": str(payload.get("reason", ""))[:240],
        }


def _urban_candidate_tactics() -> List[str]:
    return ["nearest_threat", "protect_civilians", "contain_hotspot", "monte_carlo"]


def _targets_for_tactic(
    tactic: str,
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
) -> Dict[int, Optional[Agent]]:
    if tactic == "protect_civilians":
        return _protect_civilians_targets(network, defenders, zombies, civilians, config)
    if tactic == "contain_hotspot":
        return _contain_hotspot_targets(network, defenders, zombies)
    if tactic == "monte_carlo":
        return MonteCarloResponsePolicy(samples=config.monte_carlo_samples).choose_defender_targets(
            network,
            defenders,
            zombies,
            civilians,
            config,
            rng,
            0,
            [],
        )
    return RuleBasedResponsePolicy().choose_defender_targets(
        network,
        defenders,
        zombies,
        civilians,
        config,
        rng,
        0,
        [],
    )


def _monte_carlo_target_candidates(
    network: RoadNetwork,
    defender: Agent,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    config: UrbanResponseConfig,
    civilian_risk_by_zombie: Dict[int, float],
) -> List[Agent]:
    if not zombies:
        return []
    visible = _visible_zombies(network, defender, zombies, config.defender_visibility_m * 1.35)
    candidates = visible if visible else list(zombies)
    scored = []
    for zombie in candidates:
        civilian_risk = civilian_risk_by_zombie.get(id(zombie), 900.0)
        score = network.distance(defender, zombie) + 0.50 * civilian_risk
        scored.append((score, zombie))
    scored.sort(key=lambda item: item[0])
    shortlist = [zombie for _, zombie in scored[:8]]
    civilian_threat = _nearest_threat_to_civilians(network, zombies, civilians)
    if civilian_threat is not None:
        shortlist.append(civilian_threat)
    return _dedupe_agents(shortlist)


def _sample_target_candidate(candidates: Sequence[Agent], rng: np.random.Generator) -> Optional[Agent]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    weights = np.linspace(float(len(candidates)), 1.0, len(candidates))
    weights = weights / np.sum(weights)
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _dedupe_agents(agents: Sequence[Agent]) -> List[Agent]:
    seen = set()
    deduped = []
    for agent in agents:
        marker = id(agent)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(agent)
    return deduped


def _protect_civilians_targets(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    config: UrbanResponseConfig,
) -> Dict[int, Optional[Agent]]:
    targets = {}
    risk_by_zombie = _zombie_civilian_risk_map(network, zombies, civilians)
    high_risk = sorted(zombies, key=lambda zombie: risk_by_zombie.get(id(zombie), 9999.0))
    for index, defender in enumerate(defenders):
        visible = _visible_zombies(network, defender, high_risk, config.defender_visibility_m * 1.6)
        targets[index] = visible[0] if visible else (high_risk[0] if high_risk else None)
    return targets


def _contain_hotspot_targets(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
) -> Dict[int, Optional[Agent]]:
    if not zombies:
        return {index: None for index, _ in enumerate(defenders)}
    centroid = np.mean([network.position(zombie) for zombie in zombies], axis=0)
    hotspot = min(zombies, key=lambda zombie: float(np.linalg.norm(network.position(zombie) - centroid)))
    return {index: hotspot for index, _ in enumerate(defenders)}


def _select_urban_tactic(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
    tagged_tactics: Sequence[Tuple[str, str]],
    score_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    scored = _score_urban_tactics(
        network,
        defenders,
        zombies,
        civilians,
        config,
        rng,
        tagged_tactics,
        score_plan=score_plan,
    )
    if not scored:
        return "nearest_threat", {
            "evaluated_candidates": 0,
            "selected_source": "none",
            "selected_score": 0.0,
        }
    selected = scored[0]
    return selected["tactic"], {
        "evaluated_candidates": len(scored),
        "selected_source": selected["source"],
        "selected_score": selected["score"],
        "candidate_scores": [
            {
                "id": item["candidate_id"],
                "source": item["source"],
                "tactic": item["tactic"],
                "score": item["score"],
            }
            for item in scored
        ],
    }


def _score_urban_tactics(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
    tagged_tactics: Sequence[Tuple[str, str]],
    score_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    seen = set()
    for source, tactic in tagged_tactics:
        if tactic not in _urban_candidate_tactics() or tactic in seen:
            continue
        seen.add(tactic)
        targets = _targets_for_tactic(tactic, network, defenders, zombies, civilians, config, rng)
        score = _urban_tactic_score(network, defenders, zombies, civilians, targets, config, score_plan)
        scored.append(
            {
                "source": source,
                "tactic": tactic,
                "score": score,
            }
        )
    scored.sort(key=lambda item: item["score"])
    for index, item in enumerate(scored):
        item["candidate_id"] = f"c{index}"
    return scored


def _urban_tactic_score(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    targets: Dict[int, Optional[Agent]],
    config: UrbanResponseConfig,
    score_plan: Optional[Dict[str, Any]] = None,
    civilian_risk_by_zombie: Optional[Dict[int, float]] = None,
) -> float:
    if not zombies:
        return 0.0
    risk_mode = str((score_plan or {}).get("risk_mode", "balanced"))
    civilian_weight = 0.55
    defender_weight = 60.0
    clearance_weight = 1.0
    coverage_weight = 35.0
    if risk_mode == "civilian_protection":
        civilian_weight = 0.95
        coverage_weight = 25.0
    elif risk_mode == "force_preservation":
        defender_weight = 105.0
    elif risk_mode == "clearance":
        clearance_weight = 1.35
        coverage_weight = 55.0
    elif risk_mode == "containment":
        civilian_weight = 0.70
        defender_weight = 80.0

    target_distances = []
    target_civilian_risks = []
    unique_targets = set()
    civilian_positions = [network.position(civilian) for civilian in civilians if civilian.alive]
    for index, defender in enumerate(defenders):
        target = targets.get(index)
        if target is None or not target.alive:
            continue
        unique_targets.add(id(target))
        target_distances.append(network.distance(defender, target))
        if civilian_risk_by_zombie is not None:
            target_civilian_risks.append(civilian_risk_by_zombie.get(id(target), 900.0))
        else:
            zombie_position = network.position(target)
            target_civilian_risks.append(
                min(
                    (float(np.linalg.norm(zombie_position - civ)) for civ in civilian_positions),
                    default=900.0,
                )
            )
    mean_distance = float(np.mean(target_distances)) if target_distances else 900.0
    mean_civilian_risk = float(np.mean(target_civilian_risks)) if target_civilian_risks else 900.0
    defender_exposure = 0.0
    for defender in defenders:
        nearby = sum(
            1
            for zombie in zombies
            if zombie.alive and network.distance(defender, zombie) <= 1.35 * config.bite_radius_m
        )
        defender_exposure += nearby
    coverage = len(unique_targets) / max(len(defenders), 1)
    return float(
        clearance_weight * mean_distance
        + civilian_weight * mean_civilian_risk
        + defender_weight * defender_exposure / max(len(defenders), 1)
        - coverage_weight * coverage
    )


def _codex_urban_candidate_prompt(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    history: List[Dict[str, float]],
) -> str:
    payload = _urban_prompt_payload(network, defenders, zombies, civilians, history)
    payload["allowed_tactics"] = _urban_candidate_tactics()
    return (
        "You are an urban hostile-contagion response candidate generator. "
        "Propose up to three tactic candidates. A local Monte Carlo verifier "
        "will choose the final tactic, so do not invent new tactic names. "
        "Return only JSON: {\"candidates\":[\"nearest_threat\"],"
        "\"reason\":\"short reason\"}.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _codex_urban_admin_prompt(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    history: List[Dict[str, float]],
) -> str:
    payload = _urban_prompt_payload(network, defenders, zombies, civilians, history)
    payload["allowed_risk_modes"] = ["balanced", "civilian_protection", "force_preservation", "clearance", "containment"]
    payload["allowed_tactics"] = _urban_candidate_tactics()
    return (
        "You are the administrator of a Monte Carlo urban response controller. "
        "Configure the risk mode and rank tactic priorities; do not invent new "
        "tactic names. Return only JSON: {\"risk_mode\":\"balanced\","
        "\"tactic_priorities\":[\"protect_civilians\"],\"reason\":\"short reason\"}.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _codex_urban_judge_prompt(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    history: List[Dict[str, float]],
    scored: Sequence[Dict[str, Any]],
) -> str:
    payload = _urban_prompt_payload(network, defenders, zombies, civilians, history)
    payload["candidates"] = [
        {
            "id": item["candidate_id"],
            "tactic": item["tactic"],
            "score": round(float(item["score"]), 3),
            "source": item["source"],
        }
        for item in scored
    ]
    payload["score_note"] = "scores are lower-is-better current-state risk proxies"
    return (
        "You are a high-level judge over scored urban response tactics. Choose "
        "one candidate id from the payload; do not invent tactics. Prefer the "
        "lowest score unless qualitative tactical context justifies another "
        "candidate. Return only JSON: {\"choice\":\"c0\",\"reason\":\"short reason\"}.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _urban_prompt_payload(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    history: List[Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "system": "urban hostile-contagion response on a road graph",
        "map": network.name,
        "objective": "minimize clearance time, civilian loss, defender loss, and hostile victory",
        "state": history[-1] if history else {},
        "recent_history": history[-8:],
        "agent_counts": {
            "zombies": len(zombies),
            "civilians": len(civilians),
            "defenders": len(defenders),
        },
        "nearest_zombie_to_civilian_m": _nearest_zombie_civilian_distance(network, zombies, civilians),
        "mean_zombie_defender_distance_m": _mean_nearest_distance(network, zombies, defenders),
        "mean_zombie_civilian_distance_m": _mean_nearest_distance(network, zombies, civilians),
    }


def _default_urban_admin_plan() -> Dict[str, Any]:
    return {"risk_mode": "balanced", "tactic_priorities": []}


def _urban_admin_plan_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    risk_mode = str(payload.get("risk_mode", "balanced")).strip().lower()
    if risk_mode not in {"balanced", "civilian_protection", "force_preservation", "clearance", "containment"}:
        risk_mode = "balanced"
    priorities = [
        tactic
        for tactic in _tactics_from_payload({"candidates": payload.get("tactic_priorities", [])})
        if tactic in _urban_candidate_tactics()
    ]
    return {"risk_mode": risk_mode, "tactic_priorities": priorities}


def _admin_guided_tactics(plan: Dict[str, Any]) -> List[str]:
    priorities = [
        tactic
        for tactic in plan.get("tactic_priorities", [])
        if tactic in _urban_candidate_tactics()
    ]
    return priorities + [tactic for tactic in _urban_candidate_tactics() if tactic not in priorities]


def _tactics_from_payload(payload: Dict[str, Any]) -> List[str]:
    raw = payload.get("candidates", payload.get("tactics", []))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    tactics = []
    for item in raw:
        tactic = str(item).strip()
        if tactic in _urban_candidate_tactics() and tactic not in tactics:
            tactics.append(tactic)
    return tactics[:3]


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _nearest_zombie_civilian_distance(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
) -> Optional[float]:
    values = [
        network.distance(zombie, civilian)
        for zombie in zombies
        for civilian in civilians
        if zombie.alive and civilian.alive
    ]
    return float(min(values)) if values else None


def _mean_nearest_distance(
    network: RoadNetwork,
    sources: Sequence[Agent],
    targets: Sequence[Agent],
) -> Optional[float]:
    values = [
        min((network.distance(source, target) for target in targets if target.alive), default=0.0)
        for source in sources
        if source.alive
    ]
    return float(np.mean(values)) if values else None


def run_simulation(
    network: RoadNetwork,
    config: UrbanResponseConfig,
    policy: Policy,
    seed: int,
    progress_interval: int = 0,
) -> RunResult:
    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + 518_911)
    zombies = [network.random_agent("zombie", rng) for _ in range(config.initial_zombies)]
    civilians = [network.random_agent("civilian", rng) for _ in range(config.initial_civilians)]
    defenders = [network.random_agent("defender", rng) for _ in range(config.initial_defenders)]

    time = np.arange(config.steps + 1)
    zombie_series = np.zeros(config.steps + 1)
    civilian_series = np.zeros(config.steps + 1)
    defender_series = np.zeros(config.steps + 1)
    neutralized_series = np.zeros(config.steps + 1)
    converted_series = np.zeros(config.steps + 1)
    defender_loss_series = np.zeros(config.steps + 1)
    history: List[Dict[str, float]] = []
    clearance_time = config.steps
    zombie_victory_time = config.steps
    clear_streak = 0
    neutralized_total = 0
    converted_total = 0
    defender_losses_total = 0
    zombie_victory = False
    cached_target_by_defender_id: Dict[int, Optional[Agent]] = {}
    control_update_interval = max(1, int(config.control_update_interval))

    for step in range(config.steps + 1):
        living_zombies = [agent for agent in zombies if agent.alive]
        living_civilians = [agent for agent in civilians if agent.alive]
        living_defenders = [agent for agent in defenders if agent.alive]
        _record(
            history,
            step,
            living_zombies,
            living_civilians,
            living_defenders,
            neutralized_total,
            converted_total,
            defender_losses_total,
            zombie_series,
            civilian_series,
            defender_series,
            neutralized_series,
            converted_series,
            defender_loss_series,
        )
        if not living_zombies:
            clear_streak += 1
            if clear_streak >= config.max_steps_without_zombies:
                clearance_time = step
                zombie_series[step:] = 0
                civilian_series[step:] = len(living_civilians)
                defender_series[step:] = len(living_defenders)
                neutralized_series[step:] = neutralized_total
                converted_series[step:] = converted_total
                defender_loss_series[step:] = defender_losses_total
                print(
                    f"  {policy.name}: cleared at step {step} "
                    f"(civilians={len(living_civilians)}, defenders={len(living_defenders)}).",
                    flush=True,
                )
                break
        else:
            clear_streak = 0
        if not living_civilians or not living_defenders:
            zombie_victory = True
            zombie_victory_time = step
            zombie_series[step:] = len(living_zombies)
            civilian_series[step:] = len(living_civilians)
            defender_series[step:] = len(living_defenders)
            neutralized_series[step:] = neutralized_total
            converted_series[step:] = converted_total
            defender_loss_series[step:] = defender_losses_total
            print(
                f"  {policy.name}: zombie victory at step {step} "
                f"(civilians={len(living_civilians)}, defenders={len(living_defenders)}).",
                flush=True,
            )
            break
        if step == config.steps:
            break

        if progress_interval > 0 and step % progress_interval == 0:
            print(
                f"  {policy.name}: step {step}/{config.steps} "
                f"zombies={len(living_zombies)} civilians={len(living_civilians)} "
                f"defenders={len(living_defenders)}.",
                flush=True,
            )

        if step % control_update_interval == 0 or not cached_target_by_defender_id:
            planned_targets = policy.choose_defender_targets(
                network,
                living_defenders,
                living_zombies,
                living_civilians,
                config,
                policy_rng,
                step,
                history,
            )
            cached_target_by_defender_id = {
                id(defender): planned_targets.get(index)
                for index, defender in enumerate(living_defenders)
            }
        targets = {
            index: cached_target_by_defender_id.get(id(defender))
            for index, defender in enumerate(living_defenders)
        }
        _move_zombies(network, living_zombies, living_civilians, living_defenders, config, rng)
        _move_civilians(network, living_civilians, living_zombies, config, rng)
        _move_defenders(network, living_defenders, targets, config, rng, policy)
        converted = _zombie_contacts(network, living_zombies, living_civilians, config, rng)
        contact_defender_losses = _zombie_defender_contacts(
            network,
            living_zombies,
            living_defenders,
            config,
            rng,
            policy,
        )
        neutralized, defender_losses = _defender_engagements(
            network,
            living_zombies,
            living_defenders,
            config,
            rng,
            policy,
        )
        zombies.extend(converted)
        neutralized_total += neutralized
        converted_total += len(converted)
        defender_losses_total += contact_defender_losses + defender_losses

    final_agents = [agent for agent in zombies + civilians + defenders if agent.alive]
    return RunResult(
        time=time,
        zombies=zombie_series,
        civilians=civilian_series,
        defenders=defender_series,
        neutralized=neutralized_series,
        converted=converted_series,
        defender_losses=defender_loss_series,
        clearance_time=clearance_time,
        zombie_victory=zombie_victory,
        zombie_victory_time=zombie_victory_time,
        final_agents=final_agents,
        policy_name=policy.name,
        map_name=network.name,
        policy_stats=policy.decision_stats(),
        decision_log=policy.decision_log(),
    )


def run_repeated(
    config: UrbanResponseConfig,
    policies: Sequence[Policy],
    runs: Union[int, Dict[str, int]],
    seed: int,
    map_path: Path = DEFAULT_MAP,
    progress_interval: int = 0,
) -> Dict[str, List[RunResult]]:
    results = {}
    for policy in policies:
        network = RoadNetwork(map_path)
        policy_runs = runs.get(policy.name, 1) if isinstance(runs, dict) else runs
        policy_runs = max(1, int(policy_runs))
        results[policy.name] = []
        for run_index in range(policy_runs):
            policy.reset_for_run()
            print(
                f"Running {policy.name} run {run_index + 1}/{policy_runs}...",
                flush=True,
            )
            results[policy.name].append(
                run_simulation(
                    network,
                    config,
                    policy,
                    seed + run_index,
                    progress_interval=progress_interval,
                )
            )
    return results


def save_outputs(
    results: Dict[str, List[RunResult]],
    output_dir: Path,
    map_path: Path = DEFAULT_MAP,
    config: Optional[UrbanResponseConfig] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    network = RoadNetwork(map_path)
    written = []
    for policy_name, runs in results.items():
        print(f"Saving outputs for {policy_name}...", flush=True)
        written.extend(_save_policy_plot(policy_name, runs, output_dir))
        written.extend(_save_final_map(network, policy_name, runs[0], output_dir))
    print("Saving dashboard and summary...", flush=True)
    written.extend(_save_dashboard(results, output_dir))
    summary = summarize(results)
    if config is not None:
        summary["config"] = asdict(config)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    written.append(summary_path)
    return written


def summarize(results: Dict[str, List[RunResult]]) -> Dict[str, object]:
    metrics = {}
    per_run_metrics: Dict[str, List[Dict[str, float]]] = {}
    for policy_name, runs in results.items():
        run_metrics = [_run_metrics(run) for run in runs]
        per_run_metrics[policy_name] = run_metrics
        clearance = np.array([item["clearance_time"] for item in run_metrics], dtype=float)
        zombie_victory = np.array([item["zombie_victory"] for item in run_metrics], dtype=float)
        zombie_victory_time = np.array([item["zombie_victory_time"] for item in run_metrics], dtype=float)
        survival = np.array([item["civilian_survival_fraction"] for item in run_metrics], dtype=float)
        defender_survival = np.array([item["defender_survival_fraction"] for item in run_metrics], dtype=float)
        final_zombies = np.array([item["final_zombies"] for item in run_metrics], dtype=float)
        scores = np.array([item["elimination_score_lower_is_better"] for item in run_metrics], dtype=float)
        victory_times = zombie_victory_time[zombie_victory > 0.5]
        metrics[policy_name] = {
            "runs": len(runs),
            "mean_clearance_time": float(np.mean(clearance)),
            "zombie_victory_rate": float(np.mean(zombie_victory)),
            "mean_zombie_victory_time": float(np.mean(victory_times)) if len(victory_times) else None,
            "mean_civilian_survival_fraction": float(np.mean(survival)),
            "mean_defender_survival_fraction": float(np.mean(defender_survival)),
            "mean_final_zombies": float(np.mean(final_zombies)),
            "elimination_score_lower_is_better": float(np.mean(scores)),
            "elimination_score_ci95": _mean_interval(scores),
            "decision_stats": _aggregate_policy_stats(runs),
        }
    ranking = [
        {
            "policy": name,
            "elimination_score_lower_is_better": values["elimination_score_lower_is_better"],
        }
        for name, values in metrics.items()
    ]
    ranking.sort(key=lambda item: item["elimination_score_lower_is_better"])
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
        "comparisons": _paired_comparisons(per_run_metrics),
        "decision_logs": decision_logs,
        "ranking_lower_is_better": ranking,
    }


def _run_metrics(run: RunResult) -> Dict[str, float]:
    horizon = max(run.time[-1], 1.0)
    zombie_victory = 1.0 if run.zombie_victory else 0.0
    zombie_victory_time = float(run.zombie_victory_time)
    clearance_component = (
        2.0 + (1.0 - zombie_victory_time / horizon)
        if run.zombie_victory
        else float(run.clearance_time) / horizon
    )
    survival = float(run.civilians[-1] / max(run.civilians[0], 1.0))
    defender_survival = float(run.defenders[-1] / max(run.defenders[0], 1.0))
    final_zombies = float(run.zombies[-1])
    initial_zombies = max(float(run.zombies[0]), 1.0)
    score = (
        clearance_component
        + 0.25 * (1.0 - survival)
        + 0.15 * (1.0 - defender_survival)
        + 0.20 * final_zombies / initial_zombies
        + 0.50 * zombie_victory
    )
    return {
        "clearance_time": float(run.clearance_time),
        "zombie_victory": zombie_victory,
        "zombie_victory_time": zombie_victory_time,
        "civilian_survival_fraction": survival,
        "defender_survival_fraction": defender_survival,
        "final_zombies": final_zombies,
        "elimination_score_lower_is_better": float(score),
    }


def _aggregate_policy_stats(runs: Sequence[RunResult]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for run in runs:
        for key, value in run.policy_stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _mean_interval(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values)) if values.size else 0.0
    if values.size <= 1:
        return {"mean": mean, "std": 0.0, "sem": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(values, ddof=1))
    sem = std / float(np.sqrt(values.size))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
    }


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
    best_non_codex = min(
        non_codex,
        key=lambda name: np.mean(
            [item["elimination_score_lower_is_better"] for item in per_run_metrics[name]]
        ),
    ) if non_codex else "baseline"
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
            "win_rate_lower_score": None,
        }
    policy_scores = np.array(
        [item["elimination_score_lower_is_better"] for item in policy_metrics[:count]],
        dtype=float,
    )
    reference_scores = np.array(
        [item["elimination_score_lower_is_better"] for item in reference_metrics[:count]],
        dtype=float,
    )
    delta = policy_scores - reference_scores
    interval = _mean_interval(delta)
    return {
        "paired_runs": count,
        "mean_delta_policy_minus_reference": interval["mean"],
        "delta_ci95_low": interval["ci95_low"],
        "delta_ci95_high": interval["ci95_high"],
        "win_rate_lower_score": float(np.mean(policy_scores < reference_scores)),
    }


def _agent_positions(network: RoadNetwork, agents: Sequence[Agent]) -> np.ndarray:
    if not agents:
        return np.empty((0, 2), dtype=float)
    return np.array([network.position(agent) for agent in agents], dtype=float)


def _pairwise_distances(source_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    if len(source_positions) == 0 or len(target_positions) == 0:
        return np.empty((len(source_positions), len(target_positions)), dtype=float)
    deltas = source_positions[:, None, :] - target_positions[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _move_zombies(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    defenders: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
) -> None:
    _zombie_next_target.network = network  # type: ignore[attr-defined]
    prey_agents = list(civilians) + list(defenders)
    if prey_agents and zombies:
        distances = _pairwise_distances(_agent_positions(network, zombies), _agent_positions(network, prey_agents))
    else:
        distances = np.empty((len(zombies), 0), dtype=float)
    for index, zombie in enumerate(zombies):
        if prey_agents and rng.random() > config.noise_probability:
            nearest_index = int(np.argmin(distances[index]))
            if distances[index, nearest_index] < config.zombie_detection_m:
                _route_toward(network, zombie, prey_agents[nearest_index])
        network.move(zombie, config.zombie_speed, rng, next_target=_zombie_next_target)


def _move_civilians(
    network: RoadNetwork,
    civilians: Sequence[Agent],
    zombies: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
) -> None:
    if civilians and zombies:
        distances = _pairwise_distances(_agent_positions(network, civilians), _agent_positions(network, zombies))
    else:
        distances = np.empty((len(civilians), 0), dtype=float)
    for index, civilian in enumerate(civilians):
        if zombies:
            nearest_index = int(np.argmin(distances[index]))
            if distances[index, nearest_index] <= config.zombie_detection_m:
                _route_away(network, civilian, zombies[nearest_index])
        network.move(civilian, config.civilian_speed * 0.64, rng)


def _move_defenders(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    targets: Dict[int, Optional[Agent]],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
    policy: Policy,
) -> None:
    for index, defender in enumerate(defenders):
        target = targets.get(index)
        if target is not None and target.alive:
            _route_toward(network, defender, target)
        network.move(defender, config.defender_speed * 0.76 * getattr(policy, "defender_mobility_factor", 1.0), rng)


def _zombie_next_target(agent: Agent, previous: str, rng: np.random.Generator) -> str:
    network = _zombie_next_target.network  # type: ignore[attr-defined]
    choices = [node for node in network.neighbors[agent.node] if node != previous]
    if not choices:
        choices = network.neighbors[agent.node]
    if len(choices) == 1:
        return choices[0]
    current_position = np.array(network.xy[agent.node], dtype=float)
    previous_position = np.array(network.xy[previous], dtype=float)
    forward = current_position - previous_position
    if np.allclose(forward, 0.0):
        return str(rng.choice(choices))
    scored = []
    for node in choices:
        delta = np.array(network.xy[node], dtype=float) - current_position
        score = float(np.dot(delta, forward))
        scored.append((score, node))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_choices = [node for _, node in scored[: min(2, len(scored))]]
    return str(rng.choice(top_choices))


_zombie_next_target.network = None  # type: ignore[attr-defined]


def _route_toward(network: RoadNetwork, mover: Agent, target: Agent) -> None:
    goal = network.nearest_node(network.position(target))
    if mover.node != goal:
        mover.target = network.route_next_node(mover.node, goal)
        mover.progress = min(mover.progress, 0.35)


def _route_away(network: RoadNetwork, mover: Agent, threat: Agent) -> None:
    threat_position = network.position(threat)
    best = max(
        network.neighbors[mover.node],
        key=lambda node: float(np.linalg.norm(np.array(network.xy[node]) - threat_position)),
    )
    mover.target = best
    mover.progress = min(mover.progress, 0.35)


def _zombie_contacts(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
) -> List[Agent]:
    converted = []
    if not zombies or not civilians:
        return converted
    distances = _pairwise_distances(_agent_positions(network, civilians), _agent_positions(network, zombies))
    infected = np.any(distances <= config.bite_radius_m, axis=1)
    for index, civilian in enumerate(civilians):
        if not civilian.alive:
            continue
        if infected[index]:
            civilian.alive = False
            if rng.random() < 0.5:
                converted.append(Agent("zombie", civilian.node, civilian.target, civilian.progress))
    return converted


def _zombie_defender_contacts(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    defenders: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
    policy: Policy,
) -> int:
    losses = 0
    baseline_mode = getattr(policy, "name", "") == "baseline"
    living_zombies = [zombie for zombie in zombies if zombie.alive]
    living_defender_count = max(1, sum(1 for defender in defenders if defender.alive))
    force_ratio = len(living_zombies) / living_defender_count
    if living_zombies and defenders:
        distances = _pairwise_distances(_agent_positions(network, defenders), _agent_positions(network, living_zombies))
    else:
        distances = np.empty((len(defenders), 0), dtype=float)
    for index, defender in enumerate(defenders):
        if not defender.alive:
            continue
        pressure_radius = config.zombie_detection_m * (4.0 if baseline_mode else 1.0)
        contact_count = int(np.sum(distances[index] <= config.bite_radius_m)) if living_zombies else 0
        pressure_count = int(np.sum(distances[index] <= pressure_radius)) if living_zombies else 0
        if contact_count == 0 and pressure_count == 0:
            continue
        if contact_count > 0:
            density_scale = 1.0 + 1.85 * max(0, contact_count - 1)
            base_risk = 0.55 if baseline_mode else 0.18
            ratio_scale = 1.0 + (0.35 * force_ratio if baseline_mode else 0.0)
            risk = min(0.995, base_risk * density_scale * ratio_scale)
        else:
            if baseline_mode:
                risk = min(0.35, 0.012 * pressure_count + 0.018 * force_ratio)
            else:
                risk = min(0.08, 0.002 * pressure_count)
        if rng.random() < risk:
            defender.alive = False
            losses += 1
    return losses


def _defender_engagements(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    defenders: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
    policy: Policy,
) -> Tuple[int, int]:
    neutralized = 0
    defender_losses = 0
    baseline_mode = getattr(policy, "name", "") == "baseline"
    if zombies and defenders:
        distances = _pairwise_distances(_agent_positions(network, defenders), _agent_positions(network, zombies))
    else:
        distances = np.empty((len(defenders), 0), dtype=float)
    for defender_index, defender in enumerate(defenders):
        if not defender.alive:
            continue
        nearby_indices = [
            zombie_index
            for zombie_index, zombie in enumerate(zombies)
            if zombie.alive and distances[defender_index, zombie_index] <= config.engagement_radius_m
        ]
        if not nearby_indices:
            continue
        density = len(nearby_indices)
        density_scale = 1.0 + 0.35 * max(0, density - 1)
        target_index = min(nearby_indices, key=lambda zombie_index: distances[defender_index, zombie_index])
        target = zombies[target_index]
        neutralize_prob = min(0.95, config.neutralization_probability * (0.20 if baseline_mode else 1.0) / density_scale)
        casualty_prob = min(0.995, config.defender_casualty_probability * density_scale * (1.25 if baseline_mode else 1.0))
        if rng.random() < neutralize_prob:
            target.alive = False
            neutralized += 1
        if rng.random() < casualty_prob:
            defender.alive = False
            defender_losses += 1
        elif baseline_mode and rng.random() < min(0.95, 0.35 * density_scale):
            defender.alive = False
            defender_losses += 1
    return neutralized, defender_losses


def _visible_zombies(
    network: RoadNetwork,
    observer: Agent,
    zombies: Sequence[Agent],
    radius: float,
) -> List[Agent]:
    visible = []
    for zombie in zombies:
        if not zombie.alive:
            continue
        distance = network.distance(observer, zombie)
        if distance <= radius:
            visible.append((distance, zombie))
    visible.sort(key=lambda item: item[0])
    return [zombie for _, zombie in visible]


def _nearest_threat_to_civilians(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
) -> Optional[Agent]:
    if not zombies:
        return None
    risk_by_zombie = _zombie_civilian_risk_map(network, zombies, civilians)
    return min(zombies, key=lambda zombie: risk_by_zombie.get(id(zombie), 9999.0))


def _zombie_civilian_risk_map(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
) -> Dict[int, float]:
    civilian_positions = [network.position(civilian) for civilian in civilians if civilian.alive]
    risks = {}
    for zombie in zombies:
        if not zombie.alive:
            continue
        zombie_position = network.position(zombie)
        risks[id(zombie)] = min(
            (float(np.linalg.norm(zombie_position - position)) for position in civilian_positions),
            default=9999.0,
        )
    return risks


def _record(
    history: List[Dict[str, float]],
    step: int,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    defenders: Sequence[Agent],
    neutralized_total: int,
    converted_total: int,
    defender_losses_total: int,
    zombie_series: np.ndarray,
    civilian_series: np.ndarray,
    defender_series: np.ndarray,
    neutralized_series: np.ndarray,
    converted_series: np.ndarray,
    defender_loss_series: np.ndarray,
) -> None:
    zombie_series[step] = len(zombies)
    civilian_series[step] = len(civilians)
    defender_series[step] = len(defenders)
    neutralized_series[step] = neutralized_total
    converted_series[step] = converted_total
    defender_loss_series[step] = defender_losses_total
    history.append(
        {
            "step": float(step),
            "zombies": float(len(zombies)),
            "civilians": float(len(civilians)),
            "defenders": float(len(defenders)),
            "neutralized": float(neutralized_total),
            "converted": float(converted_total),
            "defender_losses": float(defender_losses_total),
        }
    )


def _save_policy_plot(policy_name: str, runs: List[RunResult], output_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"Urban response: {policy_name}", fontsize=15)
    series = [
        ("zombies", "Zombies remaining", "#A84632"),
        ("civilians", "Civilians alive", "#3F7CAC"),
        ("defenders", "Defenders active", "#2B8C67"),
        ("neutralized", "Zombies neutralized", "#6C5B7B"),
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
    fig.suptitle("Urban zombie elimination dashboard", fontsize=16)
    colors = ["#4F5D75", "#2B8C67", "#9C5A2E", "#6C5B7B", "#B23A48"]
    for panel, attribute, ylabel in [
        (grid[0, 0], "zombies", "Zombies remaining"),
        (grid[1, 0], "civilians", "Civilians alive"),
    ]:
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
    victory_axis = fig.add_subplot(grid[0, 1])
    score_axis = fig.add_subplot(grid[1, 1])
    summary = summarize(results)
    policies = [item["policy"] for item in summary["ranking_lower_is_better"]]
    scores = [summary["metrics"][policy]["elimination_score_lower_is_better"] for policy in policies]
    victory_rates = [summary["metrics"][policy]["zombie_victory_rate"] for policy in policies]
    victory_axis.bar([_title(name) for name in policies], victory_rates, color=colors[: len(policies)], alpha=0.86)
    victory_axis.set_title("Zombie victory rate\n(lower is better)")
    victory_axis.set_ylabel("Fraction of runs")
    victory_axis.set_ylim(0.0, 1.0)
    victory_axis.tick_params(axis="x", rotation=25)
    victory_axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)
    score_axis.bar([_title(name) for name in policies], scores, color=colors[: len(policies)], alpha=0.86)
    score_axis.set_title("Failure-adjusted elimination score\n(lower is better)")
    score_axis.set_ylabel("Failure-adjusted score")
    score_axis.tick_params(axis="x", rotation=25)
    score_axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)
    return _save(fig, output_dir / "dashboard")


def _save_final_map(
    network: RoadNetwork,
    policy_name: str,
    run: RunResult,
    output_dir: Path,
) -> List[Path]:
    fig, axis = plt.subplots(figsize=(9.5, 9.0), constrained_layout=True)
    for a, b, road_name in network.edges:
        x = [network.xy[a][0], network.xy[b][0]]
        y = [network.xy[a][1], network.xy[b][1]]
        axis.plot(x, y, color="#B9C2CC", linewidth=1.4, zorder=1)
    for kind, color, marker, label in [
        ("zombie", "#A84632", "x", "Zombies"),
        ("civilian", "#3F7CAC", ".", "Civilians"),
        ("defender", "#2B8C67", "^", "Defenders"),
    ]:
        points = np.array([network.position(agent) for agent in run.final_agents if agent.kind == kind and agent.alive])
        if len(points):
            axis.scatter(points[:, 0], points[:, 1], s=30 if kind != "civilian" else 14, c=color, marker=marker, label=label, zorder=3)
    axis.set_title(f"{_title(policy_name)} final populated road map")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=False, loc="upper right")
    axis.text(0.01, 0.01, network.name, transform=axis.transAxes, fontsize=8, color="#4E5964")
    return _save(fig, output_dir / f"{policy_name}_final_map")


def _mean_sem(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _save(fig: plt.Figure, path_without_suffix: Path) -> List[Path]:
    png = path_without_suffix.with_suffix(".png")
    pdf = path_without_suffix.with_suffix(".pdf")
    fig.savefig(png, dpi=170)
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
