import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from urban_response import (
    Agent,
    CodexResponsePolicy,
    DEFAULT_MAP,
    RoadNetwork,
    UrbanResponseConfig,
    _agent_route_hazard,
    _agent_positions,
    _format_codex_exception,
    _mean_interval,
    _network_context,
    _move_zombies,
    _pairwise_distances,
    _parse_json_object,
    _route_away,
    _route_toward,
    _select_urban_tactic,
    _targets_for_tactic,
    _urban_candidate_tactics,
)


@dataclass(frozen=True)
class UrbanQualitativeConfig:
    base: UrbanResponseConfig
    social_budget_weight: float = 0.30
    physical_score_weight: float = 8.00
    human_score_weight: float = 5.00
    report_count: int = 5
    report_noise_probability: float = 0.15
    report_mode: str = "full"
    trust_recovery_rate: float = 0.012
    compliance_recovery_rate: float = 0.016
    panic_decay_rate: float = 0.026
    rumor_decay_rate: float = 0.030
    route_clarity_recovery_rate: float = 0.018
    message_fatigue_decay_rate: float = 0.035
    responder_fatigue_recovery_rate: float = 0.020


@dataclass
class UrbanQualitativeAction:
    tactic: str = "monte_carlo"
    social_action: Optional[str] = None
    target_node: Optional[str] = None
    social_intensity: float = 0.0
    risk_mode: str = "balanced"
    tactic_priorities: Optional[List[str]] = None


@dataclass
class UrbanHumanState:
    trust: Dict[str, float]
    compliance: Dict[str, float]
    panic: Dict[str, float]
    rumor_pressure: Dict[str, float]
    route_clarity: Dict[str, float]
    message_fatigue: Dict[str, float]
    responder_fatigue: float
    institutional_friction: float

    @classmethod
    def initialize(cls, network: RoadNetwork, rng: np.random.Generator) -> "UrbanHumanState":
        state = cls(
            trust={node: float(rng.uniform(0.56, 0.82)) for node in network.nodes},
            compliance={node: float(rng.uniform(0.50, 0.76)) for node in network.nodes},
            panic={node: float(rng.uniform(0.08, 0.24)) for node in network.nodes},
            rumor_pressure={node: float(rng.uniform(0.06, 0.20)) for node in network.nodes},
            route_clarity={node: float(rng.uniform(0.56, 0.82)) for node in network.nodes},
            message_fatigue={node: float(rng.uniform(0.02, 0.10)) for node in network.nodes},
            responder_fatigue=float(rng.uniform(0.08, 0.20)),
            institutional_friction=float(rng.uniform(0.08, 0.18)),
        )
        if network.scenario == "social_complex_large":
            _initialize_social_complex_large_human_state(network, state, rng)
        return state

    def copy(self) -> "UrbanHumanState":
        return UrbanHumanState(
            trust=dict(self.trust),
            compliance=dict(self.compliance),
            panic=dict(self.panic),
            rumor_pressure=dict(self.rumor_pressure),
            route_clarity=dict(self.route_clarity),
            message_fatigue=dict(self.message_fatigue),
            responder_fatigue=float(self.responder_fatigue),
            institutional_friction=float(self.institutional_friction),
        )


@dataclass
class UrbanQualitativeRunResult:
    time: np.ndarray
    zombies: np.ndarray
    civilians: np.ndarray
    defenders: np.ndarray
    neutralized: np.ndarray
    converted: np.ndarray
    defender_losses: np.ndarray
    action_budget_used: np.ndarray
    social_budget_used: np.ndarray
    mean_trust: np.ndarray
    mean_compliance: np.ndarray
    mean_panic: np.ndarray
    mean_rumor_pressure: np.ndarray
    mean_route_clarity: np.ndarray
    mean_message_fatigue: np.ndarray
    responder_fatigue: np.ndarray
    institutional_friction: np.ndarray
    clearance_time: int
    zombie_victory: bool
    zombie_victory_time: int
    policy_name: str
    policy_stats: Dict[str, float]
    decision_log: List[Dict[str, Any]]


class UrbanQualitativePolicy:
    name = "q_urban_policy"
    defender_mobility_factor = 1.0

    def reset_for_run(self) -> None:
        pass

    def decision_stats(self) -> Dict[str, float]:
        return {}

    def decision_log(self) -> List[Dict[str, Any]]:
        return []

    def choose_action(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        history: List[Dict[str, Any]],
        reports: Sequence[str],
        step: int,
        rng: np.random.Generator,
        config: UrbanQualitativeConfig,
        structured_human_state: Optional[UrbanHumanState] = None,
    ) -> UrbanQualitativeAction:
        raise NotImplementedError


class UrbanQualitativeBaselinePolicy(UrbanQualitativePolicy):
    name = "q_baseline"
    defender_mobility_factor = 0.35

    def choose_action(self, *args, **kwargs) -> UrbanQualitativeAction:
        return UrbanQualitativeAction(tactic="patrol")


class UrbanTacticalMonteCarloPolicy(UrbanQualitativePolicy):
    name = "q_monte_carlo_tactical"

    def choose_action(self, *args, **kwargs) -> UrbanQualitativeAction:
        return UrbanQualitativeAction(tactic="monte_carlo")


class UrbanKeywordMonteCarloPolicy(UrbanQualitativePolicy):
    name = "q_keyword_monte_carlo"

    def choose_action(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        history: List[Dict[str, Any]],
        reports: Sequence[str],
        step: int,
        rng: np.random.Generator,
        config: UrbanQualitativeConfig,
        structured_human_state: Optional[UrbanHumanState] = None,
    ) -> UrbanQualitativeAction:
        action = _keyword_social_action(network, reports)
        action.tactic = "monte_carlo"
        return action


class UrbanStructuredHumanStateMonteCarloPolicy(UrbanQualitativePolicy):
    name = "q_structured_human_state_monte_carlo"

    def choose_action(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        history: List[Dict[str, Any]],
        reports: Sequence[str],
        step: int,
        rng: np.random.Generator,
        config: UrbanQualitativeConfig,
        structured_human_state: Optional[UrbanHumanState] = None,
    ) -> UrbanQualitativeAction:
        if structured_human_state is None:
            return UrbanQualitativeAction(tactic="monte_carlo")
        return _structured_human_state_action(network, zombies, civilians, structured_human_state)


class CodexUrbanQualitativePolicy(CodexResponsePolicy):
    def __init__(
        self,
        name: str,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
        admin_mode: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.admin_mode = admin_mode

    def choose_action(
        self,
        network: RoadNetwork,
        defenders: Sequence[Agent],
        zombies: Sequence[Agent],
        civilians: Sequence[Agent],
        history: List[Dict[str, Any]],
        reports: Sequence[str],
        step: int,
        rng: np.random.Generator,
        config: UrbanQualitativeConfig,
        structured_human_state: Optional[UrbanHumanState] = None,
    ) -> UrbanQualitativeAction:
        if step % self.decision_interval != 0:
            return getattr(self, "current_qualitative_action", UrbanQualitativeAction())

        self._stats["codex_calls"] += 1.0
        prompt = _codex_qualitative_prompt(
            network,
            defenders,
            zombies,
            civilians,
            history,
            reports,
            admin_mode=self.admin_mode,
        )
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex qualitative controller at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            action = UrbanQualitativeAction(tactic="monte_carlo")
            self.current_qualitative_action = action
            self._decision_log.append(
                {
                    "step": step,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error": _format_codex_exception(exc),
                    "action": _qualitative_action_to_dict(action),
                }
            )
            print(
                f"  {self.name}: Codex qualitative call failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using Monte Carlo.",
                flush=True,
            )
            return action

        payload = _parse_json_object(response)
        elapsed = time.monotonic() - started_at
        if payload is None:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            action = UrbanQualitativeAction(tactic="monte_carlo")
            self.current_qualitative_action = action
            self._decision_log.append(
                {
                    "step": step,
                    "status": "invalid_json",
                    "elapsed_seconds": elapsed,
                    "action": _qualitative_action_to_dict(action),
                }
            )
            return action

        action, errors = _qualitative_action_from_payload(payload, network)
        if self.admin_mode:
            action.tactic = _admin_qualitative_tactic(
                network,
                defenders,
                zombies,
                civilians,
                action,
                rng,
                config,
            )

        if errors:
            self._stats["codex_invalid_responses"] += 1.0
            status = "partially_valid"
        else:
            status = "valid"
        self._stats["codex_valid_responses"] += 1.0
        self.current_qualitative_action = action
        self._decision_log.append(
            {
                "step": step,
                "status": status,
                "elapsed_seconds": elapsed,
                "errors": errors,
                "reason": str(payload.get("reason", ""))[:240],
                "latent_estimates": payload.get("latent_estimates", {}),
                "action": _qualitative_action_to_dict(action),
            }
        )
        print(
            f"  {self.name}: Codex returned {status} qualitative action "
            f"(tactic={action.tactic}, social={action.social_action or 'none'}) in {elapsed:.1f}s.",
            flush=True,
        )
        return action


def run_qualitative_repeated(
    config: UrbanQualitativeConfig,
    policies: Sequence[UrbanQualitativePolicy],
    runs: Union[int, Dict[str, int]],
    seed: int,
    map_path: Path = DEFAULT_MAP,
    progress_interval: int = 0,
) -> Dict[str, List[UrbanQualitativeRunResult]]:
    results: Dict[str, List[UrbanQualitativeRunResult]] = {}
    for policy in policies:
        policy_runs = runs.get(policy.name, 1) if isinstance(runs, dict) else runs
        policy_runs = max(1, int(policy_runs))
        results[policy.name] = []
        for run_index in range(policy_runs):
            policy.reset_for_run()
            print(
                f"Running qualitative {policy.name} run {run_index + 1}/{policy_runs}...",
                flush=True,
            )
            results[policy.name].append(
                run_qualitative_simulation(
                    config=config,
                    policy=policy,
                    seed=seed + run_index,
                    map_path=map_path,
                    progress_interval=progress_interval,
                )
            )
    return results


def run_qualitative_simulation(
    config: UrbanQualitativeConfig,
    policy: UrbanQualitativePolicy,
    seed: int,
    map_path: Path = DEFAULT_MAP,
    progress_interval: int = 0,
) -> UrbanQualitativeRunResult:
    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + 518_911)
    human_rng = np.random.default_rng(seed + 743_117)
    network = RoadNetwork(map_path, scenario=config.base.scenario)
    human = UrbanHumanState.initialize(network, human_rng)

    zombies = [network.random_agent("zombie", rng) for _ in range(config.base.initial_zombies)]
    civilians = [network.random_agent("civilian", rng) for _ in range(config.base.initial_civilians)]
    defenders = [network.random_agent("defender", rng) for _ in range(config.base.initial_defenders)]

    time_values = np.arange(config.base.steps + 1)
    zombie_series = np.zeros(config.base.steps + 1)
    civilian_series = np.zeros(config.base.steps + 1)
    defender_series = np.zeros(config.base.steps + 1)
    neutralized_series = np.zeros(config.base.steps + 1)
    converted_series = np.zeros(config.base.steps + 1)
    defender_loss_series = np.zeros(config.base.steps + 1)
    action_budget = np.zeros(config.base.steps + 1)
    social_budget = np.zeros(config.base.steps + 1)
    mean_trust = np.zeros(config.base.steps + 1)
    mean_compliance = np.zeros(config.base.steps + 1)
    mean_panic = np.zeros(config.base.steps + 1)
    mean_rumor = np.zeros(config.base.steps + 1)
    mean_route_clarity = np.zeros(config.base.steps + 1)
    mean_message_fatigue = np.zeros(config.base.steps + 1)
    fatigue = np.zeros(config.base.steps + 1)
    friction = np.zeros(config.base.steps + 1)

    history: List[Dict[str, Any]] = []
    reports: List[str] = []
    current_action = UrbanQualitativeAction(tactic="patrol" if policy.name == "q_baseline" else "monte_carlo")
    current_social_budget = 0.0
    controller_decisions = 0
    clearance_time = config.base.steps
    zombie_victory_time = config.base.steps
    clear_streak = 0
    neutralized_total = 0
    converted_total = 0
    defender_losses_total = 0
    zombie_victory = False

    for step in range(config.base.steps + 1):
        living_zombies = [agent for agent in zombies if agent.alive]
        living_civilians = [agent for agent in civilians if agent.alive]
        living_defenders = [agent for agent in defenders if agent.alive]
        _record_state(
            network,
            human,
            history,
            reports,
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
            mean_trust,
            mean_compliance,
            mean_panic,
            mean_rumor,
            mean_route_clarity,
            mean_message_fatigue,
            fatigue,
            friction,
        )
        action_budget[step] = current_social_budget
        social_budget[step] = current_social_budget

        if not living_zombies:
            clear_streak += 1
            if clear_streak >= config.base.max_steps_without_zombies:
                clearance_time = step
                zombie_series[step:] = 0
                civilian_series[step:] = len(living_civilians)
                defender_series[step:] = len(living_defenders)
                neutralized_series[step:] = neutralized_total
                converted_series[step:] = converted_total
                defender_loss_series[step:] = defender_losses_total
                _fill_terminal_human_series(
                    step,
                    human,
                    mean_trust,
                    mean_compliance,
                    mean_panic,
                    mean_rumor,
                    mean_route_clarity,
                    mean_message_fatigue,
                    fatigue,
                    friction,
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
            _fill_terminal_human_series(
                step,
                human,
                mean_trust,
                mean_compliance,
                mean_panic,
                mean_rumor,
                mean_route_clarity,
                mean_message_fatigue,
                fatigue,
                friction,
            )
            break
        if step == config.base.steps:
            break

        if progress_interval > 0 and step % progress_interval == 0:
            print(
                f"  qualitative {policy.name}: step {step}/{config.base.steps} "
                f"zombies={len(living_zombies)} civilians={len(living_civilians)} "
                f"trust={mean_trust[step]:.2f} panic={mean_panic[step]:.2f}.",
                flush=True,
            )

        if step % max(1, config.base.control_update_interval) == 0:
            reports = _generate_reports(network, human, living_zombies, living_civilians, human_rng, config)
            raw_action = policy.choose_action(
                network,
                living_defenders,
                living_zombies,
                living_civilians,
                history,
                reports,
                step,
                policy_rng,
                config,
                structured_human_state=human.copy()
                if policy.name == "q_structured_human_state_monte_carlo"
                else None,
            )
            if network.scenario == "social_complex_large" and policy.name != "q_baseline":
                raw_action.tactic = "monte_carlo"
            current_action, current_social_budget = _normalize_action(raw_action, network, config)
            controller_decisions += 1

        _apply_social_action(network, human, current_action, config)
        targets = _targets_for_qualitative_action(
            network,
            living_defenders,
            living_zombies,
            living_civilians,
            current_action,
            config,
            policy_rng,
        )
        _move_zombies(network, living_zombies, living_civilians, living_defenders, config.base, rng)
        _move_civilians_qualitative(network, living_civilians, living_zombies, human, config, rng)
        _move_defenders_qualitative(
            network,
            living_defenders,
            targets,
            human,
            config,
            rng,
            policy.defender_mobility_factor,
        )
        converted = _zombie_contacts_qualitative(network, living_zombies, living_civilians, human, config, rng)
        contact_defender_losses = _zombie_defender_contacts_qualitative(
            network,
            living_zombies,
            living_defenders,
            human,
            config,
            rng,
            policy.defender_mobility_factor < 1.0,
        )
        neutralized, engagement_losses = _defender_engagements_qualitative(
            network,
            living_zombies,
            living_defenders,
            human,
            config,
            rng,
            policy.defender_mobility_factor < 1.0,
        )
        zombies.extend(converted)
        neutralized_total += neutralized
        converted_total += len(converted)
        defender_losses_total += contact_defender_losses + engagement_losses
        _update_human_state(
            network,
            human,
            living_zombies,
            living_civilians,
            converted,
            contact_defender_losses + engagement_losses,
            current_action,
            config,
            human_rng,
        )
        _recover_human_state(human, config)

    return UrbanQualitativeRunResult(
        time=time_values,
        zombies=zombie_series,
        civilians=civilian_series,
        defenders=defender_series,
        neutralized=neutralized_series,
        converted=converted_series,
        defender_losses=defender_loss_series,
        action_budget_used=action_budget,
        social_budget_used=social_budget,
        mean_trust=mean_trust,
        mean_compliance=mean_compliance,
        mean_panic=mean_panic,
        mean_rumor_pressure=mean_rumor,
        mean_route_clarity=mean_route_clarity,
        mean_message_fatigue=mean_message_fatigue,
        responder_fatigue=fatigue,
        institutional_friction=friction,
        clearance_time=clearance_time,
        zombie_victory=zombie_victory,
        zombie_victory_time=zombie_victory_time,
        policy_name=policy.name,
        policy_stats={
            **policy.decision_stats(),
            "controller_decisions": float(controller_decisions),
        },
        decision_log=policy.decision_log(),
    )


def summarize_qualitative(
    results: Dict[str, List[UrbanQualitativeRunResult]],
    config: UrbanQualitativeConfig,
) -> Dict[str, Any]:
    metrics: Dict[str, Dict[str, Any]] = {}
    per_run_metrics: Dict[str, List[Dict[str, float]]] = {}
    for policy_name, runs in results.items():
        run_metrics = [_qualitative_run_metrics(run, config) for run in runs]
        per_run_metrics[policy_name] = run_metrics
        metrics[policy_name] = {
            "runs": len(runs),
            "physical_score_lower_is_better": _mean_interval(
                [item["physical_score_lower_is_better"] for item in run_metrics]
            ),
            "human_score_lower_is_better": _mean_interval(
                [item["human_score_lower_is_better"] for item in run_metrics]
            ),
            "qualitative_response_score_lower_is_better": float(
                np.mean([item["qualitative_response_score_lower_is_better"] for item in run_metrics])
            ),
            "qualitative_response_score_ci95": _mean_interval(
                [item["qualitative_response_score_lower_is_better"] for item in run_metrics]
            ),
            "zombie_victory_rate": _mean_interval([item["zombie_victory"] for item in run_metrics]),
            "civilian_survival_fraction": _mean_interval(
                [item["civilian_survival_fraction"] for item in run_metrics]
            ),
            "defender_survival_fraction": _mean_interval(
                [item["defender_survival_fraction"] for item in run_metrics]
            ),
            "mean_trust": _mean_interval([item["mean_trust"] for item in run_metrics]),
            "mean_compliance": _mean_interval([item["mean_compliance"] for item in run_metrics]),
            "mean_panic": _mean_interval([item["mean_panic"] for item in run_metrics]),
            "mean_rumor_pressure": _mean_interval([item["mean_rumor_pressure"] for item in run_metrics]),
            "mean_route_clarity": _mean_interval([item["mean_route_clarity"] for item in run_metrics]),
            "mean_message_fatigue": _mean_interval([item["mean_message_fatigue"] for item in run_metrics]),
            "responder_fatigue": _mean_interval([item["responder_fatigue"] for item in run_metrics]),
            "human_collapse_penalty": _mean_interval(
                [item["human_collapse_penalty"] for item in run_metrics]
            ),
            "human_survival_penalty": _mean_interval(
                [item["human_survival_penalty"] for item in run_metrics]
            ),
            "mean_action_budget_used": _mean_interval(
                [item["mean_action_budget_used"] for item in run_metrics]
            ),
            "mean_social_budget_used": _mean_interval(
                [item["mean_social_budget_used"] for item in run_metrics]
            ),
            "mean_institutional_friction": _mean_interval(
                [item["mean_institutional_friction"] for item in run_metrics]
            ),
            "decision_stats": _aggregate_policy_stats(runs),
        }
    ranking = [
        {
            "policy": policy_name,
            "qualitative_response_score_lower_is_better": row[
                "qualitative_response_score_lower_is_better"
            ],
            "score_ci95_low": row["qualitative_response_score_ci95"]["ci95_low"],
            "score_ci95_high": row["qualitative_response_score_ci95"]["ci95_high"],
        }
        for policy_name, row in metrics.items()
    ]
    ranking.sort(key=lambda item: item["qualitative_response_score_lower_is_better"])
    return {
        "study": "urban_qualitative_response",
        "config": asdict(config),
        "metrics": metrics,
        "per_run_metrics": per_run_metrics,
        "comparisons": _paired_qualitative_comparisons(per_run_metrics),
        "decision_logs": {
            policy_name: [
                {"run_index": index, "decisions": run.decision_log}
                for index, run in enumerate(runs)
                if run.decision_log
            ]
            for policy_name, runs in results.items()
        },
        "ranking_lower_is_better": ranking,
    }


def save_qualitative_outputs(
    results: Dict[str, List[UrbanQualitativeRunResult]],
    output_dir: Path,
    config: UrbanQualitativeConfig,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_qualitative(results, config)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return [summary_path]


def print_qualitative_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    width = max(28, *(len(policy) for policy in metrics))
    print("Qualitative sociotechnical urban ranking lower-is-better:")
    print(
        f"{'policy':{width}s} {'score':>9s} {'physical':>9s} {'human':>9s} "
        f"{'victory':>8s} {'civ':>8s} {'trust':>8s} {'route':>8s} "
        f"{'panic':>8s} {'budget':>8s}"
    )
    for item in summary["ranking_lower_is_better"]:
        policy = item["policy"]
        row = metrics[policy]
        print(
            f"{policy:{width}s} "
            f"{row['qualitative_response_score_lower_is_better']:9.3f} "
            f"{row['physical_score_lower_is_better']['mean']:9.3f} "
            f"{row['human_score_lower_is_better']['mean']:9.3f} "
            f"{row['zombie_victory_rate']['mean']:8.3f} "
            f"{row['civilian_survival_fraction']['mean']:8.3f} "
            f"{row['mean_trust']['mean']:8.3f} "
            f"{row['mean_route_clarity']['mean']:8.3f} "
            f"{row['mean_panic']['mean']:8.3f} "
            f"{row['mean_action_budget_used']['mean']:8.3f}"
        )

    comparisons = summary.get("comparisons", {})
    best_deployable = comparisons.get("best_deployable_non_codex_policy")
    structured_state = comparisons.get("structured_human_state_policy")
    if best_deployable:
        print(f"Best deployable non-Codex qualitative policy: {best_deployable}")
    if structured_state:
        print(f"Structured human-state heuristic baseline: {structured_state}")


def _record_state(
    network: RoadNetwork,
    human: UrbanHumanState,
    history: List[Dict[str, Any]],
    reports: Sequence[str],
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
    mean_trust: np.ndarray,
    mean_compliance: np.ndarray,
    mean_panic: np.ndarray,
    mean_rumor: np.ndarray,
    mean_route_clarity: np.ndarray,
    mean_message_fatigue: np.ndarray,
    fatigue: np.ndarray,
    friction: np.ndarray,
) -> None:
    zombie_series[step] = len(zombies)
    civilian_series[step] = len(civilians)
    defender_series[step] = len(defenders)
    neutralized_series[step] = neutralized_total
    converted_series[step] = converted_total
    defender_loss_series[step] = defender_losses_total
    mean_trust[step] = _mean_dict(human.trust)
    mean_compliance[step] = _mean_dict(human.compliance)
    mean_panic[step] = _mean_dict(human.panic)
    mean_rumor[step] = _mean_dict(human.rumor_pressure)
    mean_route_clarity[step] = _mean_dict(human.route_clarity)
    mean_message_fatigue[step] = _mean_dict(human.message_fatigue)
    fatigue[step] = human.responder_fatigue
    friction[step] = human.institutional_friction
    history.append(
        {
            "step": float(step),
            "zombies": float(len(zombies)),
            "civilians": float(len(civilians)),
            "defenders": float(len(defenders)),
            "neutralized": float(neutralized_total),
            "converted": float(converted_total),
            "defender_losses": float(defender_losses_total),
            "mean_trust_reported": round(float(mean_trust[step]), 2),
            "mean_compliance_reported": round(float(mean_compliance[step]), 2),
            "mean_panic_reported": round(float(mean_panic[step]), 2),
            "mean_rumor_reported": round(float(mean_rumor[step]), 2),
            "mean_route_clarity_reported": round(float(mean_route_clarity[step]), 2),
            "mean_message_fatigue_reported": round(float(mean_message_fatigue[step]), 2),
            "responder_fatigue_reported": round(float(fatigue[step]), 2),
            "recent_reports": list(reports),
        }
    )


def _initialize_social_complex_large_human_state(
    network: RoadNetwork,
    human: UrbanHumanState,
    rng: np.random.Generator,
) -> None:
    for node in network.nodes:
        zone = network.node_zones.get(node, "")
        roles = set(network.roles(node))
        route_risk = network.node_risk.get(node, 0.0)
        if zone in {"downtown core", "airport gate"}:
            human.rumor_pressure[node] = _clip(human.rumor_pressure[node] + 0.18 + 0.10 * route_risk, 0.0, 1.0)
            human.route_clarity[node] = _clip(human.route_clarity[node] - 0.12 - 0.12 * route_risk, 0.0, 1.0)
            human.message_fatigue[node] = _clip(human.message_fatigue[node] + 0.05, 0.0, 1.0)
        if zone in {"hospital med", "university campus", "suburb shelter"}:
            human.panic[node] = _clip(human.panic[node] + 0.15 + 0.08 * route_risk, 0.0, 1.0)
            human.route_clarity[node] = _clip(human.route_clarity[node] - 0.10, 0.0, 1.0)
        if zone in {"industrial port", "river island"}:
            human.trust[node] = _clip(human.trust[node] - 0.16 - 0.06 * route_risk, 0.0, 1.0)
            human.rumor_pressure[node] = _clip(human.rumor_pressure[node] + 0.16 + 0.12 * route_risk, 0.0, 1.0)
            human.route_clarity[node] = _clip(human.route_clarity[node] - 0.16 - 0.10 * route_risk, 0.0, 1.0)
            human.panic[node] = _clip(human.panic[node] + 0.10, 0.0, 1.0)
        if {"shelter", "hospital", "evacuation_hub"} & roles:
            human.panic[node] = _clip(human.panic[node] + 0.08, 0.0, 1.0)
            human.compliance[node] = _clip(human.compliance[node] - 0.08, 0.0, 1.0)
        if {"command", "staging_base", "reserve_depot"} & roles:
            human.message_fatigue[node] = _clip(human.message_fatigue[node] + 0.06, 0.0, 1.0)
        human.compliance[node] = _clip(
            0.13
            + 0.72 * human.trust[node]
            + 0.20 * human.route_clarity[node]
            - 0.33 * human.panic[node]
            - 0.30 * human.rumor_pressure[node]
            + float(rng.normal(0.0, 0.015)),
            0.0,
            1.0,
        )
    human.responder_fatigue = float(rng.uniform(0.28, 0.42))
    human.institutional_friction = float(rng.uniform(0.22, 0.34))


def _fill_terminal_human_series(
    start: int,
    human: UrbanHumanState,
    mean_trust: np.ndarray,
    mean_compliance: np.ndarray,
    mean_panic: np.ndarray,
    mean_rumor: np.ndarray,
    mean_route_clarity: np.ndarray,
    mean_message_fatigue: np.ndarray,
    fatigue: np.ndarray,
    friction: np.ndarray,
) -> None:
    mean_trust[start:] = _mean_dict(human.trust)
    mean_compliance[start:] = _mean_dict(human.compliance)
    mean_panic[start:] = _mean_dict(human.panic)
    mean_rumor[start:] = _mean_dict(human.rumor_pressure)
    mean_route_clarity[start:] = _mean_dict(human.route_clarity)
    mean_message_fatigue[start:] = _mean_dict(human.message_fatigue)
    fatigue[start:] = human.responder_fatigue
    friction[start:] = human.institutional_friction


def _generate_reports(
    network: RoadNetwork,
    human: UrbanHumanState,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    rng: np.random.Generator,
    config: UrbanQualitativeConfig,
) -> List[str]:
    if config.report_mode == "removed":
        return []
    if config.report_mode == "shuffled":
        human = _shuffled_urban_human_state(human, rng)
    zombie_counts = _agent_counts_by_node(zombies)
    civilian_counts = _agent_counts_by_node(civilians)
    panic_node = max(human.panic, key=human.panic.get)
    rumor_node = max(human.rumor_pressure, key=human.rumor_pressure.get)
    low_trust_node = min(human.trust, key=human.trust.get)
    low_compliance_node = min(human.compliance, key=human.compliance.get)
    low_route_node = min(human.route_clarity, key=human.route_clarity.get)
    fatigue_message_node = max(human.message_fatigue, key=human.message_fatigue.get)
    exposure_node = max(
        network.nodes,
        key=lambda node: zombie_counts.get(node, 0) + 0.35 * civilian_counts.get(node, 0),
    )
    if config.report_mode == "explicit":
        reports: List[Tuple[float, str]] = [
            (
                human.panic[panic_node],
                f"EXPLICIT_LABEL panic high at {panic_node}; use medical_triage or shelter_opening if civilians are freezing.",
            ),
            (
                human.rumor_pressure[rumor_node],
                f"EXPLICIT_LABEL rumor high at {rumor_node}; use public_message if trust is adequate.",
            ),
            (
                1.0 - human.trust[low_trust_node],
                f"EXPLICIT_LABEL low_trust distrust at {low_trust_node}; use community_liaison before broadcasts.",
            ),
            (
                1.0 - human.route_clarity[low_route_node],
                f"EXPLICIT_LABEL route_confusion evacuation at {low_route_node}; use evacuation_guidance.",
            ),
            (
                human.message_fatigue[fatigue_message_node],
                f"EXPLICIT_LABEL message_fatigue at {fatigue_message_node}; avoid repeated broadcasts and use liaison.",
            ),
            (
                human.responder_fatigue,
                "EXPLICIT_LABEL responder fatigue high; use responder_rotation.",
            ),
        ]
        reports.sort(key=lambda item: item[0], reverse=True)
        return [report for _, report in reports[: config.report_count]]
    reports: List[Tuple[float, str]] = [
        (
            human.panic[panic_node],
            (
                f"Field volunteers near {panic_node} say families are standing "
                "in place, asking for help, and not entering the marked corridors."
            ),
        ),
        (
            human.rumor_pressure[rumor_node],
            (
                f"Residents around {rumor_node} are repeating two different stories "
                "about which streets are open, and the bus staging point has become unclear."
            ),
        ),
        (
            1.0 - human.trust[low_trust_node],
            (
                f"Block captains in {low_trust_node} say residents are waiting for "
                "confirmation from familiar local voices before following city instructions."
            ),
        ),
        (
            1.0 - human.compliance[low_compliance_node],
            (
                f"Observers near {low_compliance_node} say many households remain on "
                "porches or in cars despite repeated movement instructions."
            ),
        ),
        (
            1.0 - human.route_clarity[low_route_node],
            (
                f"At {low_route_node}, families say the signs, buses, and responder "
                "hand signals point to different routes through the district."
            ),
        ),
        (
            human.message_fatigue[fatigue_message_node],
            (
                f"Coordinators near {fatigue_message_node} say residents have heard "
                "too many official updates and are tuning out repeated announcements."
            ),
        ),
        (
            human.responder_fatigue,
            (
                "Mobile-team supervisors say radio traffic is clipped, crews are missing "
                "handoffs, and another push without relief may reduce engagement quality."
            ),
        ),
        (
            zombie_counts.get(exposure_node, 0) / 20.0,
            (
                f"Dispatch notes high hostile density near {exposure_node} with "
                f"{civilian_counts.get(exposure_node, 0)} civilians still exposed."
            ),
        ),
    ]
    if rng.random() < config.report_noise_probability:
        noisy_node = str(rng.choice(network.nodes))
        reports.append(
            (
                0.35,
                (
                    f"A neighborhood chat says buses already cleared {noisy_node}, "
                    "but responder logs do not confirm the pickup."
                ),
            )
        )
    reports.sort(key=lambda item: item[0], reverse=True)
    return [report for _, report in reports[: config.report_count]]


def _shuffled_urban_human_state(
    human: UrbanHumanState,
    rng: np.random.Generator,
) -> UrbanHumanState:
    def shuffled(values: Dict[str, float]) -> Dict[str, float]:
        keys = list(values)
        vals = [values[key] for key in keys]
        rng.shuffle(vals)
        return dict(zip(keys, vals))

    return UrbanHumanState(
        trust=shuffled(human.trust),
        compliance=shuffled(human.compliance),
        panic=shuffled(human.panic),
        rumor_pressure=shuffled(human.rumor_pressure),
        route_clarity=shuffled(human.route_clarity),
        message_fatigue=shuffled(human.message_fatigue),
        responder_fatigue=float(human.responder_fatigue),
        institutional_friction=float(human.institutional_friction),
    )


def _normalize_action(
    action: UrbanQualitativeAction,
    network: RoadNetwork,
    config: UrbanQualitativeConfig,
) -> Tuple[UrbanQualitativeAction, float]:
    tactic = action.tactic if action.tactic in set(_urban_candidate_tactics() + ["patrol"]) else "monte_carlo"
    social_action = _normalize_social_action(action.social_action)
    target_node = action.target_node if action.target_node in network.nodes else None
    if social_action in {"public_message", "community_liaison", "evacuation_guidance", "shelter_opening", "medical_triage"}:
        if target_node is None:
            social_action = None
    intensity = _clip(action.social_intensity, 0.0, 1.0) if social_action else 0.0
    if social_action and intensity <= 0.0:
        intensity = _default_social_intensity(social_action)
    budget = config.social_budget_weight * intensity
    return (
        UrbanQualitativeAction(
            tactic=tactic,
            social_action=social_action,
            target_node=target_node,
            social_intensity=intensity,
            risk_mode=action.risk_mode,
            tactic_priorities=action.tactic_priorities or [],
        ),
        budget,
    )


def _apply_social_action(
    network: RoadNetwork,
    human: UrbanHumanState,
    action: UrbanQualitativeAction,
    config: UrbanQualitativeConfig,
) -> None:
    if not action.social_action or action.social_intensity <= 0.0:
        return
    intensity = action.social_intensity
    effect_scale = 1.32 if network.scenario == "social_complex_large" else 1.0
    target = action.target_node
    social_targets = _social_target_weights(network, target)
    if action.social_action == "public_message" and social_targets:
        for node, weight in social_targets.items():
            profile = _district_response_profile(node)
            scaled = intensity * weight * effect_scale
            trust = human.trust[node]
            fatigue = human.message_fatigue[node]
            effectiveness = profile["broadcast"] * scaled * _clip(
                0.48 + 0.70 * trust - 0.82 * fatigue,
                0.0,
                1.10,
            )
            backlash = scaled * (
                0.85 * max(0.0, 0.42 - trust)
                + 0.75 * max(0.0, fatigue - 0.34)
            )
            human.rumor_pressure[node] = _clip(
                human.rumor_pressure[node] - 0.34 * effectiveness + 0.11 * backlash,
                0.0,
                1.0,
            )
            human.panic[node] = _clip(
                human.panic[node] - 0.05 * effectiveness + 0.07 * backlash,
                0.0,
                1.0,
            )
            human.trust[node] = _clip(
                human.trust[node] + 0.07 * effectiveness - 0.09 * backlash,
                0.0,
                1.0,
            )
            human.compliance[node] = _clip(human.compliance[node] + 0.05 * effectiveness, 0.0, 1.0)
            human.route_clarity[node] = _clip(human.route_clarity[node] + 0.10 * effectiveness, 0.0, 1.0)
            human.message_fatigue[node] = _clip(human.message_fatigue[node] + 0.17 * scaled, 0.0, 1.0)
    elif action.social_action == "community_liaison" and social_targets:
        for node, weight in social_targets.items():
            profile = _district_response_profile(node)
            scaled = intensity * weight * effect_scale
            effectiveness = profile["liaison"] * scaled * _clip(
                0.78 + 0.52 * (1.0 - human.trust[node]) + 0.18 * human.message_fatigue[node],
                0.55,
                1.30,
            )
            human.trust[node] = _clip(human.trust[node] + 0.22 * effectiveness, 0.0, 1.0)
            human.compliance[node] = _clip(human.compliance[node] + 0.12 * effectiveness, 0.0, 1.0)
            human.panic[node] = _clip(human.panic[node] - 0.10 * effectiveness, 0.0, 1.0)
            human.rumor_pressure[node] = _clip(human.rumor_pressure[node] - 0.11 * effectiveness, 0.0, 1.0)
            human.route_clarity[node] = _clip(human.route_clarity[node] + 0.05 * effectiveness, 0.0, 1.0)
            human.message_fatigue[node] = _clip(human.message_fatigue[node] - 0.13 * effectiveness, 0.0, 1.0)
    elif action.social_action == "evacuation_guidance" and social_targets:
        for node, weight in social_targets.items():
            profile = _district_response_profile(node)
            scaled = intensity * weight * effect_scale
            route_need = 1.0 - human.route_clarity[node]
            trust_gate = 0.34 + 0.80 * human.trust[node]
            panic_gate = 1.05 - 0.42 * human.panic[node]
            effectiveness = profile["guidance"] * scaled * route_need * _clip(
                trust_gate * panic_gate,
                0.0,
                1.25,
            )
            confusion_backlash = scaled * (
                0.55 * max(0.0, 0.38 - human.trust[node])
                + 0.35 * max(0.0, human.message_fatigue[node] - 0.44)
            )
            human.route_clarity[node] = _clip(human.route_clarity[node] + 0.34 * effectiveness, 0.0, 1.0)
            human.compliance[node] = _clip(
                human.compliance[node] + 0.22 * effectiveness - 0.05 * confusion_backlash,
                0.0,
                1.0,
            )
            human.panic[node] = _clip(
                human.panic[node] - 0.10 * effectiveness + 0.09 * confusion_backlash,
                0.0,
                1.0,
            )
            human.rumor_pressure[node] = _clip(
                human.rumor_pressure[node] - 0.04 * effectiveness + 0.06 * confusion_backlash,
                0.0,
                1.0,
            )
            human.message_fatigue[node] = _clip(human.message_fatigue[node] + 0.07 * scaled, 0.0, 1.0)
    elif action.social_action == "shelter_opening" and social_targets:
        for node, weight in social_targets.items():
            profile = _district_response_profile(node)
            scaled = intensity * weight * effect_scale
            effectiveness = profile["shelter"] * scaled * _clip(
                0.65 + 0.45 * human.panic[node] + 0.25 * (1.0 - human.route_clarity[node]),
                0.55,
                1.25,
            )
            human.trust[node] = _clip(human.trust[node] + 0.11 * effectiveness, 0.0, 1.0)
            human.compliance[node] = _clip(human.compliance[node] + 0.13 * effectiveness, 0.0, 1.0)
            human.panic[node] = _clip(human.panic[node] - 0.21 * effectiveness, 0.0, 1.0)
            human.rumor_pressure[node] = _clip(human.rumor_pressure[node] - 0.06 * effectiveness, 0.0, 1.0)
            human.route_clarity[node] = _clip(human.route_clarity[node] + 0.12 * effectiveness, 0.0, 1.0)
            human.message_fatigue[node] = _clip(human.message_fatigue[node] - 0.04 * effectiveness, 0.0, 1.0)
    elif action.social_action == "medical_triage" and social_targets:
        for node, weight in social_targets.items():
            profile = _district_response_profile(node)
            scaled = intensity * weight * effect_scale
            effectiveness = profile["medical"] * scaled * _clip(
                0.70 + 0.70 * human.panic[node],
                0.60,
                1.35,
            )
            human.panic[node] = _clip(human.panic[node] - 0.29 * effectiveness, 0.0, 1.0)
            human.trust[node] = _clip(human.trust[node] + 0.09 * effectiveness, 0.0, 1.0)
            human.compliance[node] = _clip(human.compliance[node] + 0.05 * effectiveness, 0.0, 1.0)
            human.rumor_pressure[node] = _clip(human.rumor_pressure[node] - 0.04 * effectiveness, 0.0, 1.0)
            human.message_fatigue[node] = _clip(human.message_fatigue[node] - 0.03 * effectiveness, 0.0, 1.0)
        human.responder_fatigue = _clip(human.responder_fatigue + 0.014 * intensity, 0.0, 1.0)
    elif action.social_action == "responder_rotation":
        human.responder_fatigue = _clip(human.responder_fatigue - 0.50 * intensity, 0.0, 1.0)
        friction_cost = 0.014 if network.scenario == "social_complex_large" else 0.024
        human.institutional_friction = _clip(human.institutional_friction + friction_cost * intensity, 0.0, 1.0)
        for node in human.trust:
            human.trust[node] = _clip(human.trust[node] + 0.010 * intensity, 0.0, 1.0)
            human.message_fatigue[node] = _clip(human.message_fatigue[node] - 0.006 * intensity, 0.0, 1.0)


def _targets_for_qualitative_action(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    action: UrbanQualitativeAction,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
) -> Dict[int, Optional[Agent]]:
    if action.tactic == "patrol":
        return {index: None for index, _ in enumerate(defenders)}
    return _targets_for_tactic(action.tactic, network, defenders, zombies, civilians, config.base, rng)


def _move_civilians_qualitative(
    network: RoadNetwork,
    civilians: Sequence[Agent],
    zombies: Sequence[Agent],
    human: UrbanHumanState,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
) -> None:
    if civilians and zombies:
        distances = _pairwise_distances(_agent_positions(network, civilians), _agent_positions(network, zombies))
    else:
        distances = np.empty((len(civilians), 0), dtype=float)
    for index, civilian in enumerate(civilians):
        node = civilian.node
        trust = human.trust.get(node, _mean_dict(human.trust))
        compliance = human.compliance.get(node, _mean_dict(human.compliance))
        panic = human.panic.get(node, _mean_dict(human.panic))
        rumor = human.rumor_pressure.get(node, _mean_dict(human.rumor_pressure))
        route_clarity = human.route_clarity.get(node, _mean_dict(human.route_clarity))
        if zombies:
            nearest_index = int(np.argmin(distances[index]))
            if distances[index, nearest_index] <= config.base.zombie_detection_m:
                freeze_probability = min(
                    0.58,
                    0.18 * panic
                    + 0.12 * rumor
                    + 0.08 * (1.0 - trust)
                    + 0.14 * (1.0 - route_clarity),
                )
                if rng.random() > freeze_probability:
                    _route_away(network, civilian, zombies[nearest_index])
            elif network.scenario == "social_complex_large" and compliance > 0.50 and route_clarity > 0.52:
                refuge = network.nearest_role_node(
                    network.position(civilian),
                    ["shelter", "hospital", "evacuation_hub"],
                )
                if refuge is not None and civilian.node != refuge:
                    civilian.target = network.route_next_node(civilian.node, refuge)
                    civilian.progress = min(civilian.progress, 0.35)
        elif network.scenario == "social_complex_large" and compliance > 0.56 and route_clarity > 0.56:
            refuge = network.nearest_role_node(
                network.position(civilian),
                ["shelter", "hospital", "evacuation_hub"],
            )
            if refuge is not None and civilian.node != refuge:
                civilian.target = network.route_next_node(civilian.node, refuge)
                civilian.progress = min(civilian.progress, 0.35)
        speed_factor = _clip(
            0.62
            + 0.32 * compliance
            + 0.10 * trust
            + 0.24 * route_clarity
            - 0.34 * panic
            - 0.18 * rumor,
            0.26,
            1.16,
        )
        network.move(civilian, config.base.civilian_speed * 0.64 * speed_factor, rng)


def _move_defenders_qualitative(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    targets: Dict[int, Optional[Agent]],
    human: UrbanHumanState,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
    mobility_factor: float,
) -> None:
    fatigue_factor = _clip(1.0 - 0.34 * human.responder_fatigue, 0.58, 1.0)
    for index, defender in enumerate(defenders):
        target = targets.get(index)
        if target is not None and target.alive:
            _route_toward(network, defender, target)
        network.move(defender, config.base.defender_speed * 0.76 * mobility_factor * fatigue_factor, rng)


def _zombie_contacts_qualitative(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    human: UrbanHumanState,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
) -> List[Agent]:
    converted = []
    if not zombies or not civilians:
        return converted
    distances = _pairwise_distances(_agent_positions(network, civilians), _agent_positions(network, zombies))
    infected = np.any(distances <= config.base.bite_radius_m, axis=1)
    for index, civilian in enumerate(civilians):
        if not civilian.alive or not infected[index]:
            continue
        node = civilian.node
        panic = human.panic.get(node, _mean_dict(human.panic))
        compliance = human.compliance.get(node, _mean_dict(human.compliance))
        rumor = human.rumor_pressure.get(node, _mean_dict(human.rumor_pressure))
        route_clarity = human.route_clarity.get(node, _mean_dict(human.route_clarity))
        civilian.alive = False
        conversion_probability = _clip(
            0.44
            + 0.18 * panic
            + 0.08 * rumor
            + 0.09 * (1.0 - route_clarity)
            - 0.10 * compliance,
            0.18,
            0.82,
        )
        conversion_probability = _clip(
            conversion_probability
            + 0.14 * _agent_route_hazard(network, civilian)
            + 0.06 * network.node_risk.get(node, 0.0),
            0.18,
            0.90,
        )
        if rng.random() < conversion_probability:
            converted.append(Agent("zombie", civilian.node, civilian.target, civilian.progress))
    return converted


def _zombie_defender_contacts_qualitative(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    defenders: Sequence[Agent],
    human: UrbanHumanState,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
    baseline_mode: bool,
) -> int:
    losses = 0
    if zombies and defenders:
        distances = _pairwise_distances(_agent_positions(network, defenders), _agent_positions(network, zombies))
    else:
        distances = np.empty((len(defenders), 0), dtype=float)
    force_ratio = len(zombies) / max(1, len(defenders))
    fatigue_scale = 1.0 + 0.52 * human.responder_fatigue
    for index, defender in enumerate(defenders):
        contact_count = int(np.sum(distances[index] <= config.base.bite_radius_m)) if zombies else 0
        pressure_radius = config.base.zombie_detection_m * (4.0 if baseline_mode else 1.0)
        pressure_count = int(np.sum(distances[index] <= pressure_radius)) if zombies else 0
        if contact_count == 0 and pressure_count == 0:
            continue
        if contact_count > 0:
            density_scale = 1.0 + 1.85 * max(0, contact_count - 1)
            base_risk = 0.55 if baseline_mode else 0.18
            ratio_scale = 1.0 + (0.35 * force_ratio if baseline_mode else 0.0)
            hazard_scale = 1.0 + 0.34 * _agent_route_hazard(network, defender)
            risk = min(0.995, base_risk * density_scale * ratio_scale * fatigue_scale * hazard_scale)
        else:
            risk = min(
                0.35 if baseline_mode else 0.10,
                0.012 * pressure_count * fatigue_scale
                + (0.026 if baseline_mode else 0.008) * _agent_route_hazard(network, defender),
            )
        if rng.random() < risk:
            defender.alive = False
            losses += 1
    return losses


def _defender_engagements_qualitative(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    defenders: Sequence[Agent],
    human: UrbanHumanState,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
    baseline_mode: bool,
) -> Tuple[int, int]:
    neutralized = 0
    defender_losses = 0
    if zombies and defenders:
        distances = _pairwise_distances(_agent_positions(network, defenders), _agent_positions(network, zombies))
    else:
        distances = np.empty((len(defenders), 0), dtype=float)
    fatigue = human.responder_fatigue
    for defender_index, defender in enumerate(defenders):
        if not defender.alive:
            continue
        nearby_indices = [
            zombie_index
            for zombie_index, zombie in enumerate(zombies)
            if zombie.alive and distances[defender_index, zombie_index] <= config.base.engagement_radius_m
        ]
        if not nearby_indices:
            continue
        density = len(nearby_indices)
        density_scale = 1.0 + 0.35 * max(0, density - 1)
        target_index = min(nearby_indices, key=lambda zombie_index: distances[defender_index, zombie_index])
        target = zombies[target_index]
        hazard = _agent_route_hazard(network, defender)
        neutralize_prob = min(
            0.95,
            config.base.neutralization_probability
            * (0.20 if baseline_mode else 1.0)
            * max(0.42, 1.0 - 0.46 * fatigue)
            * max(0.60, 1.0 - 0.22 * hazard)
            / density_scale,
        )
        casualty_prob = min(
            0.995,
            config.base.defender_casualty_probability
            * density_scale
            * (1.25 if baseline_mode else 1.0)
            * (1.0 + 0.38 * fatigue)
            * (1.0 + 0.30 * hazard),
        )
        if rng.random() < neutralize_prob:
            target.alive = False
            neutralized += 1
        if rng.random() < casualty_prob:
            defender.alive = False
            defender_losses += 1
    return neutralized, defender_losses


def _update_human_state(
    network: RoadNetwork,
    human: UrbanHumanState,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    converted: Sequence[Agent],
    defender_losses: int,
    action: UrbanQualitativeAction,
    config: UrbanQualitativeConfig,
    rng: np.random.Generator,
) -> None:
    zombie_counts = _agent_counts_by_node(zombies)
    civilian_counts = _agent_counts_by_node(civilians)
    converted_counts = _agent_counts_by_node(converted)
    unmanaged_social_crisis = network.scenario == "social_complex_large" and not action.social_action
    for node in network.nodes:
        exposure = min(1.0, zombie_counts.get(node, 0) / 12.0)
        crowding = min(1.0, civilian_counts.get(node, 0) / 24.0)
        loss_shock = min(1.0, converted_counts.get(node, 0) / 4.0)
        route_hazard = network.node_risk.get(node, 0.0)
        no_action_pressure = 1.0 if unmanaged_social_crisis and (exposure > 0.0 or crowding > 0.12) else 0.0
        human.panic[node] = _clip(
            human.panic[node]
            + 0.030 * exposure
            + 0.020 * crowding
            + 0.090 * loss_shock
            + no_action_pressure * (0.014 + 0.014 * route_hazard)
            + float(rng.normal(0.0, 0.003)),
            0.0,
            1.0,
        )
        human.rumor_pressure[node] = _clip(
            human.rumor_pressure[node]
            + 0.024 * exposure
            + 0.020 * loss_shock
            + 0.010 * (1.0 - human.trust[node])
            + 0.008 * (1.0 - human.route_clarity[node])
            + no_action_pressure * (0.015 + 0.018 * (1.0 - human.trust[node]))
            + float(rng.normal(0.0, 0.003)),
            0.0,
            1.0,
        )
        human.route_clarity[node] = _clip(
            human.route_clarity[node]
            - 0.026 * exposure
            - 0.024 * loss_shock
            - 0.020 * human.rumor_pressure[node]
            - 0.010 * human.message_fatigue[node]
            - no_action_pressure * (0.018 + 0.010 * route_hazard)
            + float(rng.normal(0.0, 0.002)),
            0.0,
            1.0,
        )
        human.message_fatigue[node] = _clip(
            human.message_fatigue[node]
            + 0.010 * exposure
            + 0.010 * human.rumor_pressure[node]
            + 0.006 * (1.0 - human.trust[node])
            + no_action_pressure * 0.006
            + float(rng.normal(0.0, 0.002)),
            0.0,
            1.0,
        )
        human.trust[node] = _clip(
            human.trust[node]
            - 0.026 * loss_shock
            - 0.008 * exposure
            - 0.010 * human.rumor_pressure[node]
            - 0.006 * human.message_fatigue[node]
            - no_action_pressure * 0.012
            + float(rng.normal(0.0, 0.002)),
            0.0,
            1.0,
        )
        human.compliance[node] = _clip(
            0.15
            + 0.72 * human.trust[node]
            + 0.18 * human.route_clarity[node]
            - 0.35 * human.panic[node]
            - 0.28 * human.rumor_pressure[node],
            0.0,
            1.0,
        )
    _diffuse_urban_social_state(network, human)
    human.responder_fatigue = _clip(
        human.responder_fatigue
        + 0.010 * min(1.0, len(zombies) / max(config.base.initial_zombies, 1))
        + 0.018 * defender_losses
        - 0.004,
        0.0,
        1.0,
    )
    if action.tactic == "protect_civilians":
        human.responder_fatigue = _clip(human.responder_fatigue + 0.006, 0.0, 1.0)
    human.institutional_friction = _clip(
        human.institutional_friction
        + 0.008 * _mean_dict(human.rumor_pressure)
        + 0.006 * (1.0 - _mean_dict(human.trust))
        + (0.010 if unmanaged_social_crisis else 0.0)
        - 0.003,
        0.0,
        1.0,
    )


def _recover_human_state(human: UrbanHumanState, config: UrbanQualitativeConfig) -> None:
    for node in human.trust:
        human.trust[node] = _clip(
            human.trust[node] + config.trust_recovery_rate * (0.72 - human.trust[node]),
            0.0,
            1.0,
        )
        human.compliance[node] = _clip(
            human.compliance[node]
            + config.compliance_recovery_rate * (0.65 - human.compliance[node]),
            0.0,
            1.0,
        )
        human.panic[node] = _clip(human.panic[node] * (1.0 - config.panic_decay_rate), 0.0, 1.0)
        human.rumor_pressure[node] = _clip(
            human.rumor_pressure[node] * (1.0 - config.rumor_decay_rate),
            0.0,
            1.0,
        )
        human.route_clarity[node] = _clip(
            human.route_clarity[node]
            + config.route_clarity_recovery_rate * (0.74 - human.route_clarity[node]),
            0.0,
            1.0,
        )
        human.message_fatigue[node] = _clip(
            human.message_fatigue[node] * (1.0 - config.message_fatigue_decay_rate),
            0.0,
            1.0,
        )
    human.responder_fatigue = _clip(
        human.responder_fatigue * (1.0 - config.responder_fatigue_recovery_rate),
        0.0,
        1.0,
    )


def _keyword_social_action(
    network: RoadNetwork,
    reports: Sequence[str],
) -> UrbanQualitativeAction:
    text = " ".join(reports).lower()
    target = _first_report_node(network, reports)
    if any(word in text for word in ["unverified", "claim", "rumor", "false"]):
        return UrbanQualitativeAction(social_action="public_message", target_node=target, social_intensity=0.65)
    if any(word in text for word in ["leader", "distrust", "trust", "legitimacy"]):
        return UrbanQualitativeAction(social_action="community_liaison", target_node=target, social_intensity=0.65)
    if any(word in text for word in ["evacuation", "shelter", "compliance"]):
        return UrbanQualitativeAction(social_action="evacuation_guidance", target_node=target, social_intensity=0.65)
    if any(word in text for word in ["injury", "medical", "triage"]):
        return UrbanQualitativeAction(social_action="medical_triage", target_node=target, social_intensity=0.60)
    if any(word in text for word in ["fatigue", "rotation"]):
        return UrbanQualitativeAction(social_action="responder_rotation", social_intensity=0.60)
    return UrbanQualitativeAction()


def _structured_human_state_action(
    network: RoadNetwork,
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    human: UrbanHumanState,
) -> UrbanQualitativeAction:
    zombie_counts = _agent_counts_by_node(zombies)
    civilian_counts = _agent_counts_by_node(civilians)
    target = max(
        network.nodes,
        key=lambda node: (
            0.8 * human.panic[node]
            + 0.7 * human.rumor_pressure[node]
            + 0.6 * (1.0 - human.trust[node])
            + 0.5 * (1.0 - human.compliance[node])
            + 0.65 * (1.0 - human.route_clarity[node])
            + 0.45 * human.message_fatigue[node]
            + 0.05 * zombie_counts.get(node, 0)
            + 0.01 * civilian_counts.get(node, 0)
        ),
    )
    if human.responder_fatigue > 0.58:
        return UrbanQualitativeAction(
            tactic="contain_hotspot",
            social_action="responder_rotation",
            social_intensity=0.90,
        )
    if human.message_fatigue[target] > 0.45 and human.trust[target] < 0.55:
        social = "community_liaison"
    elif human.route_clarity[target] < 0.48 and human.trust[target] > 0.42 and human.panic[target] < 0.58:
        social = "evacuation_guidance"
    elif human.route_clarity[target] < 0.48 and human.panic[target] >= 0.58:
        social = "shelter_opening"
    elif human.rumor_pressure[target] > human.panic[target] and human.rumor_pressure[target] > 0.34:
        social = "public_message"
    elif human.trust[target] < 0.44:
        social = "community_liaison"
    elif human.compliance[target] < 0.42:
        social = "evacuation_guidance"
    elif human.panic[target] > 0.34:
        social = "medical_triage"
    else:
        social = "shelter_opening"
    tactic = "protect_civilians" if civilian_counts.get(target, 0) > 8 else "monte_carlo"
    return UrbanQualitativeAction(
        tactic=tactic,
        social_action=social,
        target_node=target,
        social_intensity=0.86,
    )


def _admin_qualitative_tactic(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    action: UrbanQualitativeAction,
    rng: np.random.Generator,
    config: UrbanQualitativeConfig,
) -> str:
    priorities = [
        tactic
        for tactic in (action.tactic_priorities or [])
        if tactic in _urban_candidate_tactics()
    ]
    if action.tactic in _urban_candidate_tactics() and action.tactic not in priorities:
        priorities.insert(0, action.tactic)
    tagged = [("admin", tactic) for tactic in priorities]
    tagged.append(("native", "monte_carlo"))
    tactic, _ = _select_urban_tactic(
        network,
        defenders,
        zombies,
        civilians,
        config.base,
        rng,
        tagged,
        score_plan={"risk_mode": action.risk_mode},
    )
    return tactic


def _qualitative_action_from_payload(
    payload: Dict[str, Any],
    network: RoadNetwork,
) -> Tuple[UrbanQualitativeAction, List[str]]:
    errors: List[str] = []
    tactic = str(payload.get("tactic", "monte_carlo")).strip()
    if tactic not in _urban_candidate_tactics():
        errors.append(f"invalid tactic: {tactic}")
        tactic = "monte_carlo"
    social_payload = payload.get("social_action")
    if isinstance(social_payload, dict):
        social_action = _normalize_social_action(social_payload.get("type"))
        target_value = payload.get("target_node") or social_payload.get("target_node") or social_payload.get("target")
        intensity_value = payload.get("social_intensity", social_payload.get("intensity", 0.0))
    else:
        social_action = _normalize_social_action(social_payload)
        target_value = payload.get("target_node", "")
        intensity_value = payload.get("social_intensity", 0.0)
    if social_payload and not social_action:
        errors.append(f"invalid social_action: {social_payload}")
    target_node = str(target_value).strip()
    if target_node not in network.nodes:
        target_node = _first_known_node(network, json.dumps(payload))
    social_intensity = _safe_float(intensity_value, 0.0)
    if social_action and social_intensity <= 0.0:
        social_intensity = _default_social_intensity(social_action)
    risk_mode = str(payload.get("risk_mode", "balanced")).strip()
    if risk_mode not in {"balanced", "civilian_protection", "force_preservation", "clearance", "containment"}:
        risk_mode = "balanced"
    priorities = [
        str(item)
        for item in payload.get("tactic_priorities", [])
        if str(item) in _urban_candidate_tactics()
    ]
    return (
        UrbanQualitativeAction(
            tactic=tactic,
            social_action=social_action,
            target_node=target_node,
            social_intensity=social_intensity,
            risk_mode=risk_mode,
            tactic_priorities=priorities,
        ),
        errors,
    )


def _codex_qualitative_prompt(
    network: RoadNetwork,
    defenders: Sequence[Agent],
    zombies: Sequence[Agent],
    civilians: Sequence[Agent],
    history: List[Dict[str, Any]],
    reports: Sequence[str],
    admin_mode: bool,
) -> str:
    payload = {
        "system": "sociotechnical urban hostile-contagion response",
        "map": network.name,
        "objective": (
            "minimize hostile clearance time and casualties while preserving "
            "civilian trust, compliance, low panic, low rumor pressure, and "
            "responder capacity"
        ),
        "scoring_note": (
            "The final score penalizes human-state degradation. A response that "
            "clears hostiles but leaves high panic, high rumor pressure, low "
            "compliance, low route clarity, high message fatigue, or high "
            "responder fatigue can lose."
        ),
        "network_context": _network_context(network),
        "social_action_guidance": {
            "public_message": (
                "best for conflicting stories when trust is adequate and residents "
                "are not saturated with repeated announcements; can backfire under "
                "low trust or high message fatigue"
            ),
            "community_liaison": (
                "best when residents wait for familiar voices, trust is low, or "
                "official updates are being tuned out; slower but more durable"
            ),
            "evacuation_guidance": (
                "best when signs, buses, hand signals, or corridors are ambiguous "
                "and trust is still adequate; avoid repeating it when reports say "
                "residents are tuning out official updates"
            ),
            "shelter_opening": (
                "best when panic and route ambiguity mean people need a concrete "
                "safe destination before they will move"
            ),
            "medical_triage": "best for acute panic, injuries, and crowd freezing; slightly increases responder load",
            "responder_rotation": "best when clipped radio traffic, missed handoffs, or fatigue threaten tactical quality",
        },
        "district_context": {
            node: _district_context(node)
            for node in network.nodes
        },
        "state": history[-1] if history else {},
        "recent_history": history[-6:],
        "reports": list(reports),
        "allowed_tactics": _urban_candidate_tactics(),
        "allowed_social_actions": _allowed_social_actions(),
        "available_target_nodes": network.nodes,
        "agent_counts": {
            "zombies": len(zombies),
            "civilians": len(civilians),
            "defenders": len(defenders),
        },
    }
    if admin_mode:
        payload["allowed_risk_modes"] = [
            "balanced",
            "civilian_protection",
            "force_preservation",
            "clearance",
            "containment",
        ]
        return (
            "You are a qualitative administrator for a Monte Carlo urban response "
            "controller. Interpret the reports, estimate latent human conditions, "
            "choose one social intervention, and set tactical priorities. Match "
            "the social action to the causal condition, not just a keyword. Avoid "
            "broadcasts or repeated guidance when reports imply low trust or "
            "message saturation; use liaison for legitimacy, guidance for route "
            "ambiguity, shelter or triage for immobilizing panic, and rotation "
            "for responder overload. If you choose a social_action, set "
            "social_intensity between 0.55 and 0.95 rather than 0. "
            "Return only JSON "
            "with keys tactic, tactic_priorities, risk_mode, social_action, "
            "target_node, social_intensity, latent_estimates, and reason.\n\n"
            f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    return (
        "You are a sociotechnical urban response controller. Interpret the reports "
        "and choose a bounded tactical and social response. Match the social "
        "action to the causal condition, not just a keyword. Avoid broadcasts "
        "or repeated guidance when reports imply low trust or message saturation; "
        "use liaison for legitimacy, guidance for route ambiguity, shelter or triage for "
        "immobilizing panic, and rotation for responder overload. If you choose "
        "a social_action, set social_intensity between 0.55 and 0.95 rather than 0. "
        "Return only JSON with keys tactic, "
        "social_action, target_node, social_intensity, latent_estimates, and "
        "reason.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _qualitative_run_metrics(
    run: UrbanQualitativeRunResult,
    config: UrbanQualitativeConfig,
) -> Dict[str, float]:
    horizon = max(run.time[-1], 1.0)
    zombie_victory = 1.0 if run.zombie_victory else 0.0
    victory_time_fraction = (
        float(run.zombie_victory_time) / horizon if run.zombie_victory else 1.0
    )
    clearance_component = (
        2.0 + (1.0 - float(run.zombie_victory_time) / horizon)
        if run.zombie_victory
        else float(run.clearance_time) / horizon
    )
    observed_civilian_survival = float(run.civilians[-1] / max(run.civilians[0], 1.0))
    defender_survival = float(run.defenders[-1] / max(run.defenders[0], 1.0))
    final_zombies = float(run.zombies[-1] / max(run.zombies[0], 1.0))
    if config.base.scenario == "social_complex_large" and run.zombie_victory:
        collapse_exposure = _clip(
            1.0 - 0.72 * (1.0 - victory_time_fraction) - 0.20 * final_zombies,
            0.0,
            1.0,
        )
        civilian_survival = observed_civilian_survival * collapse_exposure
    else:
        civilian_survival = observed_civilian_survival
    physical_score = (
        clearance_component
        + 0.35 * (1.0 - civilian_survival)
        + 0.20 * (1.0 - defender_survival)
        + 0.25 * final_zombies
        + 0.65 * zombie_victory
    )
    mean_trust = float(np.mean(run.mean_trust))
    mean_compliance = float(np.mean(run.mean_compliance))
    mean_panic = float(np.mean(run.mean_panic))
    mean_rumor = float(np.mean(run.mean_rumor_pressure))
    mean_route_clarity = float(np.mean(run.mean_route_clarity))
    mean_message_fatigue = float(np.mean(run.mean_message_fatigue))
    fatigue = float(np.mean(run.responder_fatigue))
    if config.base.scenario == "social_complex_large":
        human_collapse_penalty = zombie_victory * (
            1.15 + 0.95 * (1.0 - victory_time_fraction) + 0.35 * final_zombies
        )
    else:
        human_collapse_penalty = zombie_victory * (
            0.90 + 0.60 * (1.0 - victory_time_fraction)
        )
    human_survival_penalty = (
        0.45 * (1.0 - civilian_survival)
        + 0.35 * (1.0 - defender_survival)
    )
    human_score = (
        1.25 * (1.0 - mean_trust)
        + 1.30 * (1.0 - mean_compliance)
        + 1.70 * mean_panic
        + 1.60 * mean_rumor
        + 1.20 * (1.0 - mean_route_clarity)
        + 0.95 * mean_message_fatigue
        + 1.15 * fatigue
        + 0.55 * float(np.mean(run.institutional_friction))
        + human_collapse_penalty
        + human_survival_penalty
    )
    score = (
        config.physical_score_weight * physical_score
        + config.human_score_weight * human_score
        + 0.70 * float(np.mean(run.action_budget_used))
    )
    return {
        "physical_score_lower_is_better": float(physical_score),
        "human_score_lower_is_better": float(human_score),
        "qualitative_response_score_lower_is_better": float(score),
        "zombie_victory": zombie_victory,
        "civilian_survival_fraction": float(civilian_survival),
        "observed_civilian_survival_fraction": float(observed_civilian_survival),
        "defender_survival_fraction": defender_survival,
        "mean_trust": mean_trust,
        "mean_compliance": mean_compliance,
        "mean_panic": mean_panic,
        "mean_rumor_pressure": mean_rumor,
        "mean_route_clarity": mean_route_clarity,
        "mean_message_fatigue": mean_message_fatigue,
        "responder_fatigue": fatigue,
        "human_collapse_penalty": float(human_collapse_penalty),
        "human_survival_penalty": float(human_survival_penalty),
        "mean_action_budget_used": float(np.mean(run.action_budget_used)),
        "mean_social_budget_used": float(np.mean(run.social_budget_used)),
        "mean_institutional_friction": float(np.mean(run.institutional_friction)),
    }


def _paired_qualitative_comparisons(
    per_run_metrics: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Any]:
    deployable = [
        name
        for name in per_run_metrics
        if name not in {"q_baseline", "q_structured_human_state_monte_carlo"}
        and not name.startswith("q_codex_")
    ]
    best_deployable = min(
        deployable,
        key=lambda name: np.mean(
            [item["qualitative_response_score_lower_is_better"] for item in per_run_metrics[name]]
        ),
    ) if deployable else None
    structured = "q_structured_human_state_monte_carlo" if "q_structured_human_state_monte_carlo" in per_run_metrics else None
    return {
        "best_deployable_non_codex_policy": best_deployable,
        "structured_human_state_policy": structured,
        "vs_baseline": {
            name: _paired_score_delta(items, per_run_metrics["q_baseline"])
            for name, items in per_run_metrics.items()
            if name != "q_baseline" and "q_baseline" in per_run_metrics
        },
        "vs_best_deployable_non_codex": {
            name: _paired_score_delta(items, per_run_metrics[best_deployable])
            for name, items in per_run_metrics.items()
            if best_deployable and name != best_deployable
        },
        "vs_structured_human_state": {
            name: _paired_score_delta(items, per_run_metrics[structured])
            for name, items in per_run_metrics.items()
            if structured and name != structured
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
        [item["qualitative_response_score_lower_is_better"] for item in policy_metrics[:count]],
        dtype=float,
    )
    reference_scores = np.array(
        [item["qualitative_response_score_lower_is_better"] for item in reference_metrics[:count]],
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


def _aggregate_policy_stats(runs: Sequence[UrbanQualitativeRunResult]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for run in runs:
        for key, value in run.policy_stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _agent_counts_by_node(agents: Sequence[Agent]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for agent in agents:
        if not agent.alive:
            continue
        counts[agent.node] = counts.get(agent.node, 0) + 1
    return counts


def _district_response_profile(node: str) -> Dict[str, float]:
    profile = {
        "broadcast": 1.00,
        "liaison": 1.00,
        "guidance": 1.00,
        "shelter": 1.00,
        "medical": 1.00,
    }
    if any(token in node for token in ["airport_gate", "airport", "international", "downtown", "lake_eola"]):
        profile.update({"broadcast": 1.18, "guidance": 1.18, "liaison": 0.86, "shelter": 0.92})
    elif any(token in node for token in ["hospital_med"]):
        profile.update({"medical": 1.28, "guidance": 1.10, "liaison": 1.04, "shelter": 0.94})
    elif any(token in node for token in ["industrial_port"]):
        profile.update({"guidance": 1.16, "broadcast": 0.90, "liaison": 0.96, "medical": 0.92})
    elif any(token in node for token in ["river_island"]):
        profile.update({"guidance": 1.24, "shelter": 0.86, "broadcast": 0.92, "liaison": 1.06})
    elif any(token in node for token in ["north_reserve"]):
        profile.update({"medical": 1.08, "liaison": 1.12, "broadcast": 0.86, "guidance": 0.96})
    elif any(token in node for token in ["campus", "college", "university_campus"]):
        profile.update({"liaison": 1.18, "medical": 1.10, "broadcast": 0.92, "guidance": 1.04})
    elif any(token in node for token in ["parramore", "holden", "millenia", "suburb_shelter"]):
        profile.update({"liaison": 1.24, "shelter": 1.08, "broadcast": 0.82, "guidance": 0.94})
    elif any(token in node for token in ["winter", "audubon", "baldwin", "ivanhoe", "corrine"]):
        profile.update({"shelter": 1.14, "liaison": 1.08, "medical": 1.06, "broadcast": 0.94})
    return profile


def _district_context(node: str) -> str:
    if "hospital_med" in node:
        return "hospital district: triage and reliable corridors protect critical medical capacity"
    if "industrial_port" in node:
        return "industrial port district: freight chokepoints and confusing access roads make route guidance important"
    if "river_island" in node:
        return "bridgehead district: bridges and tunnels are chokepoints where route clarity and panic control matter"
    if "north_reserve" in node:
        return "responder staging district: rotation and interagency coordination affect tactical reliability"
    if any(token in node for token in ["airport_gate", "airport", "international", "downtown", "lake_eola"]):
        return "visitor corridor: broadcasts and clear route guidance work well if message fatigue is low"
    if any(token in node for token in ["campus", "college", "university_campus"]):
        return "campus district: trusted local intermediaries and triage cues are important"
    if any(token in node for token in ["parramore", "holden", "millenia", "suburb_shelter"]):
        return "lower-trust residential district: liaison usually works before repeated official messaging"
    if any(token in node for token in ["winter", "audubon", "baldwin", "ivanhoe", "corrine"]):
        return "residential district: shelter access and local liaison reduce panic more than broadcast repetition"
    return "mixed district: choose action from report context and observed social state"


def _diffuse_urban_social_state(network: RoadNetwork, human: UrbanHumanState) -> None:
    next_rumor = dict(human.rumor_pressure)
    next_panic = dict(human.panic)
    next_route = dict(human.route_clarity)
    for node in network.nodes:
        neighbors = network.neighbors.get(node, [])
        if not neighbors:
            continue
        neighbor_rumor = float(np.mean([human.rumor_pressure[neighbor] for neighbor in neighbors]))
        neighbor_panic = float(np.mean([human.panic[neighbor] for neighbor in neighbors]))
        neighbor_route = float(np.mean([human.route_clarity[neighbor] for neighbor in neighbors]))
        next_rumor[node] = _clip(
            human.rumor_pressure[node] + 0.035 * max(0.0, neighbor_rumor - human.rumor_pressure[node]),
            0.0,
            1.0,
        )
        next_panic[node] = _clip(
            human.panic[node] + 0.022 * max(0.0, neighbor_panic - human.panic[node]),
            0.0,
            1.0,
        )
        next_route[node] = _clip(
            human.route_clarity[node] - 0.024 * max(0.0, human.route_clarity[node] - neighbor_route),
            0.0,
            1.0,
        )
    human.rumor_pressure.update(next_rumor)
    human.panic.update(next_panic)
    human.route_clarity.update(next_route)


def _social_target_weights(network: RoadNetwork, target: Optional[str]) -> Dict[str, float]:
    if target is None or target not in network.nodes:
        return {}
    weights = {target: 1.0}
    for neighbor in network.neighbors.get(target, []):
        weights[neighbor] = max(weights.get(neighbor, 0.0), 0.42)
    return weights


def _first_report_node(network: RoadNetwork, reports: Sequence[str]) -> Optional[str]:
    return _first_known_node(network, " ".join(reports))


def _first_known_node(network: RoadNetwork, text: str) -> Optional[str]:
    for node in network.nodes:
        if node in text:
            return node
    return None


def _normalize_social_action(value: Any) -> Optional[str]:
    if value is None:
        return None
    social = str(value).strip().lower()
    return social if social in _allowed_social_actions() else None


def _default_social_intensity(social_action: str) -> float:
    return {
        "public_message": 0.68,
        "community_liaison": 0.72,
        "evacuation_guidance": 0.70,
        "shelter_opening": 0.72,
        "medical_triage": 0.70,
        "responder_rotation": 0.66,
    }.get(social_action, 0.68)


def _allowed_social_actions() -> List[str]:
    return [
        "public_message",
        "community_liaison",
        "evacuation_guidance",
        "shelter_opening",
        "medical_triage",
        "responder_rotation",
    ]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean_dict(values: Dict[Any, float]) -> float:
    return float(np.mean(list(values.values()))) if values else 0.0


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _qualitative_action_to_dict(action: UrbanQualitativeAction) -> Dict[str, Any]:
    return {
        "tactic": action.tactic,
        "social_action": action.social_action,
        "target_node": action.target_node,
        "social_intensity": action.social_intensity,
        "risk_mode": action.risk_mode,
        "tactic_priorities": action.tactic_priorities or [],
    }
