import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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
        for index, defender in enumerate(defenders):
            visible = _visible_zombies(network, defender, zombies, config.defender_visibility_m)
            if visible:
                targets[index] = min(visible, key=lambda zombie: network.distance(defender, zombie))
            else:
                targets[index] = _nearest_threat_to_civilians(network, zombies, civilians)
        return targets


class MonteCarloResponsePolicy(Policy):
    name = "monte_carlo"

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
        civilian_positions = [network.position(civilian) for civilian in civilians if civilian.alive]
        for index, defender in enumerate(defenders):
            candidates = _visible_zombies(network, defender, zombies, config.defender_visibility_m * 1.25)
            if not candidates:
                candidates = list(zombies)
            if not candidates:
                targets[index] = None
                continue
            scored = []
            for zombie in candidates:
                defender_distance = network.distance(defender, zombie)
                zombie_position = network.position(zombie)
                civilian_risk = min(
                    (float(np.linalg.norm(zombie_position - civ)) for civ in civilian_positions),
                    default=800.0,
                )
                score = defender_distance + 0.45 * civilian_risk
                scored.append((score, zombie))
            scored.sort(key=lambda item: item[0])
            targets[index] = scored[0][1]
        return targets


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

    def reset_for_run(self) -> None:
        self.current_tactic = "nearest_threat"
        self._session_id = None

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
            if step % max(1, self.decision_interval // 2) != 0 or len(history) < 5:
                return False
            recent = history[-1]
            previous = history[-5]
            return (
                recent["zombies"] > previous["zombies"]
                or recent["civilians"] < previous["civilians"] - 3
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
            print(
                f"  {self.name}: Codex unavailable after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using nearest_threat.",
                flush=True,
            )
            return "nearest_threat"
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            print(f"  {self.name}: Codex returned no JSON; using nearest_threat.", flush=True)
            return "nearest_threat"
        try:
            tactic = json.loads(match.group(0)).get("tactic", "nearest_threat")
        except json.JSONDecodeError:
            print(f"  {self.name}: Codex JSON parse failed; using nearest_threat.", flush=True)
            return "nearest_threat"
        if tactic not in {"nearest_threat", "protect_civilians", "contain_hotspot"}:
            print(f"  {self.name}: Codex tactic was invalid; using nearest_threat.", flush=True)
            return "nearest_threat"
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

    def _protect_civilians(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        config: UrbanResponseConfig,
    ) -> Dict[int, Optional[Agent]]:
        targets = {}
        high_risk = sorted(
            zombies,
            key=lambda zombie: min((network.distance(zombie, civ) for civ in civilians), default=9999.0),
        )
        for index, defender in enumerate(defenders):
            visible = _visible_zombies(network, defender, high_risk, config.defender_visibility_m * 1.6)
            targets[index] = visible[0] if visible else (high_risk[0] if high_risk else None)
        return targets

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


def run_simulation(
    network: RoadNetwork,
    config: UrbanResponseConfig,
    policy: Policy,
    seed: int,
    progress_interval: int = 0,
) -> RunResult:
    rng = np.random.default_rng(seed)
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

        targets = policy.choose_defender_targets(
            network,
            living_defenders,
            living_zombies,
            living_civilians,
            config,
            rng,
            step,
            history,
        )
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
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    written.append(summary_path)
    return written


def summarize(results: Dict[str, List[RunResult]]) -> Dict[str, object]:
    metrics = {}
    for policy_name, runs in results.items():
        horizon = max(runs[0].time[-1], 1.0)
        clearance = np.array([run.clearance_time for run in runs], dtype=float)
        zombie_victory = np.array([run.zombie_victory for run in runs], dtype=bool)
        zombie_victory_time = np.array([run.zombie_victory_time for run in runs], dtype=float)
        survival = np.array([
            run.civilians[-1] / max(run.civilians[0], 1.0) for run in runs
        ])
        defender_survival = np.array([
            run.defenders[-1] / max(run.defenders[0], 1.0) for run in runs
        ])
        final_zombies = np.array([run.zombies[-1] for run in runs], dtype=float)
        clearance_component = np.where(
            zombie_victory,
            2.0 + (1.0 - zombie_victory_time / horizon),
            clearance / horizon,
        )
        elimination_score = (
            np.mean(clearance_component)
            + 0.25 * np.mean(1.0 - survival)
            + 0.15 * np.mean(1.0 - defender_survival)
            + 0.20 * np.mean(final_zombies / max(runs[0].zombies[0], 1.0))
            + 0.50 * np.mean(zombie_victory)
        )
        victory_times = zombie_victory_time[zombie_victory]
        metrics[policy_name] = {
            "mean_clearance_time": float(np.mean(clearance)),
            "zombie_victory_rate": float(np.mean(zombie_victory)),
            "mean_zombie_victory_time": float(np.mean(victory_times)) if len(victory_times) else None,
            "mean_civilian_survival_fraction": float(np.mean(survival)),
            "mean_defender_survival_fraction": float(np.mean(defender_survival)),
            "mean_final_zombies": float(np.mean(final_zombies)),
            "elimination_score_lower_is_better": float(elimination_score),
        }
    ranking = [
        {
            "policy": name,
            "elimination_score_lower_is_better": values["elimination_score_lower_is_better"],
        }
        for name, values in metrics.items()
    ]
    ranking.sort(key=lambda item: item["elimination_score_lower_is_better"])
    return {"metrics": metrics, "ranking_lower_is_better": ranking}


def _move_zombies(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    defenders: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
) -> None:
    _zombie_next_target.network = network  # type: ignore[attr-defined]
    for zombie in zombies:
        prey = [
            agent for agent in list(civilians) + list(defenders)
            if network.distance(zombie, agent) < config.zombie_detection_m
        ]
        if prey and rng.random() > config.noise_probability:
            target_agent = min(prey, key=lambda agent: network.distance(zombie, agent))
            _route_toward(network, zombie, target_agent)
        network.move(zombie, config.zombie_speed, rng, next_target=_zombie_next_target)


def _move_civilians(
    network: RoadNetwork,
    civilians: Sequence[Agent],
    zombies: Sequence[Agent],
    config: UrbanResponseConfig,
    rng: np.random.Generator,
) -> None:
    for civilian in civilians:
        nearby = _visible_zombies(network, civilian, zombies, config.zombie_detection_m)
        if nearby:
            _route_away(network, civilian, nearby[0])
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
    for civilian in civilians:
        if not civilian.alive:
            continue
        if any(network.distance(zombie, civilian) <= config.bite_radius_m for zombie in zombies if zombie.alive):
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
    for defender in defenders:
        if not defender.alive:
            continue
        contact_zombies = [
            zombie for zombie in zombies
            if zombie.alive and network.distance(defender, zombie) <= config.bite_radius_m
        ]
        pressure_radius = config.zombie_detection_m * (4.0 if baseline_mode else 1.0)
        pressure_zombies = [
            zombie for zombie in zombies
            if zombie.alive and network.distance(defender, zombie) <= pressure_radius
        ]
        if not contact_zombies and not pressure_zombies:
            continue
        if contact_zombies:
            density_scale = 1.0 + 1.85 * max(0, len(contact_zombies) - 1)
            base_risk = 0.55 if baseline_mode else 0.18
            ratio_scale = 1.0 + (0.35 * force_ratio if baseline_mode else 0.0)
            risk = min(0.995, base_risk * density_scale * ratio_scale)
        else:
            pressure_count = len(pressure_zombies)
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
    for defender in defenders:
        if not defender.alive:
            continue
        nearby = [
            zombie for zombie in zombies
            if zombie.alive and network.distance(defender, zombie) <= config.engagement_radius_m
        ]
        if not nearby:
            continue
        density = len(nearby)
        density_scale = 1.0 + 0.35 * max(0, density - 1)
        target = min(nearby, key=lambda zombie: network.distance(defender, zombie))
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
    visible = [zombie for zombie in zombies if zombie.alive and network.distance(observer, zombie) <= radius]
    visible.sort(key=lambda zombie: network.distance(observer, zombie))
    return visible


def _nearest_threat_to_civilians(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
) -> Optional[Agent]:
    if not zombies:
        return None
    return min(
        zombies,
        key=lambda zombie: min((network.distance(zombie, civilian) for civilian in civilians), default=9999.0),
    )


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
