import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from sabotage_simulation import (
    CodexSupplyChainPolicy,
    Edge,
    Intervention,
    Node,
    SupplyChainConfig,
    SupplyChainNetwork,
    _admin_guided_candidates,
    _apply_disruptions,
    _apply_intervention,
    _candidate_interventions,
    _candidate_score,
    _clip,
    _desired_inventory,
    _edge_exists,
    _format_codex_exception,
    _intervention_budget,
    _intervention_from_codex_payload,
    _intervention_to_dict,
    _network_context,
    _normalize_intervention,
    _observed_network,
    _parse_edge,
    _safe_float,
    _select_monte_carlo_action,
    _shortage_edges,
)


@dataclass(frozen=True)
class QualitativeConfig:
    base: SupplyChainConfig
    social_budget_weight: float = 0.30
    social_action_cost: float = 9.0
    human_score_weight: float = 2.25
    report_count: int = 5
    report_noise_probability: float = 0.15
    trust_recovery_rate: float = 0.012
    rumor_decay_rate: float = 0.030
    fatigue_recovery_rate: float = 0.020
    cooperation_recovery_rate: float = 0.018
    equity_decay_rate: float = 0.014


@dataclass
class QualitativeAction:
    logistics: Intervention
    social_action: Optional[str] = None
    social_target: Optional[str] = None
    social_edge: Optional[Tuple[str, str]] = None
    social_intensity: float = 0.0


@dataclass
class HumanLayerState:
    trust: Dict[str, float]
    compliance: Dict[str, float]
    rumor_pressure: Dict[str, float]
    equity_pressure: Dict[str, float]
    workforce_fatigue: Dict[str, float]
    carrier_cooperation: Dict[Tuple[str, str], float]
    institutional_friction: float

    @classmethod
    def initialize(
        cls,
        network: SupplyChainNetwork,
        rng: np.random.Generator,
    ) -> "HumanLayerState":
        retailers = [node.name for node in network.retailers()]
        return cls(
            trust={name: float(rng.uniform(0.58, 0.82)) for name in retailers},
            compliance={name: float(rng.uniform(0.52, 0.76)) for name in retailers},
            rumor_pressure={name: float(rng.uniform(0.08, 0.24)) for name in retailers},
            equity_pressure={name: float(rng.uniform(0.08, 0.20)) for name in retailers},
            workforce_fatigue={
                name: float(rng.uniform(0.10, 0.24))
                for name, node in network.nodes.items()
                if node.kind in {"factory", "warehouse"}
            },
            carrier_cooperation={
                (edge.source, edge.target): float(rng.uniform(0.70, 0.92))
                for edge in network.edges
            },
            institutional_friction=float(rng.uniform(0.08, 0.18)),
        )

    def copy(self) -> "HumanLayerState":
        return HumanLayerState(
            trust=dict(self.trust),
            compliance=dict(self.compliance),
            rumor_pressure=dict(self.rumor_pressure),
            equity_pressure=dict(self.equity_pressure),
            workforce_fatigue=dict(self.workforce_fatigue),
            carrier_cooperation=dict(self.carrier_cooperation),
            institutional_friction=float(self.institutional_friction),
        )


@dataclass
class QualitativeRunResult:
    time: np.ndarray
    unmet_demand: np.ndarray
    service_level: np.ndarray
    economic_loss: np.ndarray
    total_inventory: np.ndarray
    active_capacity: np.ndarray
    action_budget_used: np.ndarray
    social_budget_used: np.ndarray
    mean_trust: np.ndarray
    mean_compliance: np.ndarray
    mean_rumor_pressure: np.ndarray
    mean_workforce_fatigue: np.ndarray
    mean_carrier_cooperation: np.ndarray
    mean_equity_pressure: np.ndarray
    policy_name: str
    policy_stats: Dict[str, float]
    decision_log: List[Dict[str, Any]]


class QualitativePolicy:
    name = "qualitative_policy"

    def reset_for_run(self) -> None:
        pass

    def decision_stats(self) -> Dict[str, float]:
        return {}

    def decision_log(self) -> List[Dict[str, Any]]:
        return []

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, Any]],
        reports: List[str],
        step: int,
        rng: np.random.Generator,
        config: QualitativeConfig,
        structured_human_state: Optional[HumanLayerState] = None,
    ) -> QualitativeAction:
        raise NotImplementedError


class QualitativeBaselinePolicy(QualitativePolicy):
    name = "q_baseline"

    def choose_action(self, *args, **kwargs) -> QualitativeAction:
        return QualitativeAction(logistics=Intervention())


class LogisticsMonteCarloPolicy(QualitativePolicy):
    name = "q_monte_carlo_logistics"

    def __init__(self, samples: int = 24) -> None:
        self.samples = max(1, samples)

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, Any]],
        reports: List[str],
        step: int,
        rng: np.random.Generator,
        config: QualitativeConfig,
        structured_human_state: Optional[HumanLayerState] = None,
    ) -> QualitativeAction:
        logistics = _monte_carlo_logistics_action(network, rng, config.base, self.samples)
        return QualitativeAction(logistics=logistics)


class KeywordMonteCarloPolicy(LogisticsMonteCarloPolicy):
    name = "q_keyword_monte_carlo"

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, Any]],
        reports: List[str],
        step: int,
        rng: np.random.Generator,
        config: QualitativeConfig,
        structured_human_state: Optional[HumanLayerState] = None,
    ) -> QualitativeAction:
        logistics = _monte_carlo_logistics_action(network, rng, config.base, self.samples)
        social = _keyword_social_action(network, reports)
        social.logistics = logistics
        return social


class StructuredHumanStateMonteCarloPolicy(LogisticsMonteCarloPolicy):
    name = "q_structured_human_state_monte_carlo"

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, Any]],
        reports: List[str],
        step: int,
        rng: np.random.Generator,
        config: QualitativeConfig,
        structured_human_state: Optional[HumanLayerState] = None,
    ) -> QualitativeAction:
        if structured_human_state is None:
            return super().choose_action(network, history, reports, step, rng, config)
        logistics = _structured_human_state_logistics_action(
            network,
            structured_human_state,
            rng,
            config,
            self.samples,
        )
        social = _structured_human_state_social_action(network, structured_human_state)
        social.logistics = logistics
        return social


class CodexQualitativePolicy(CodexSupplyChainPolicy):
    def __init__(
        self,
        name: str,
        codex_command: str,
        decision_interval: int,
        timeout_seconds: int,
        samples: int = 24,
        admin_mode: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            codex_command=codex_command,
            decision_interval=decision_interval,
            timeout_seconds=timeout_seconds,
        )
        self.samples = max(1, samples)
        self.admin_mode = admin_mode

    def choose_action(
        self,
        network: SupplyChainNetwork,
        history: List[Dict[str, Any]],
        reports: List[str],
        step: int,
        rng: np.random.Generator,
        config: QualitativeConfig,
        structured_human_state: Optional[HumanLayerState] = None,
    ) -> QualitativeAction:
        if step % self.decision_interval != 0:
            return getattr(self, "current_qualitative_action", QualitativeAction(Intervention()))

        self._stats["codex_calls"] += 1.0
        prompt = _codex_qualitative_prompt(
            network,
            history,
            reports,
            config,
            admin_mode=self.admin_mode,
        )
        started_at = time.monotonic()
        print(f"  {self.name}: consulting Codex qualitative controller at step {step}...", flush=True)
        try:
            response = self._ask_codex(prompt)
        except Exception as exc:
            elapsed = time.monotonic() - started_at
            self._stats["codex_failures"] += 1.0
            print(
                f"  {self.name}: Codex qualitative call failed after {elapsed:.1f}s "
                f"({_format_codex_exception(exc)}); using no social action.",
                flush=True,
            )
            logistics = _monte_carlo_logistics_action(network, rng, config.base, self.samples)
            action = QualitativeAction(logistics=logistics)
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
            return action

        parsed = _parse_json_object(response)
        elapsed = time.monotonic() - started_at
        if parsed is None:
            self._stats["codex_invalid_responses"] += 1.0
            self._stats["codex_noop_responses"] += 1.0
            logistics = _monte_carlo_logistics_action(network, rng, config.base, self.samples)
            action = QualitativeAction(logistics=logistics)
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

        action, errors = _qualitative_action_from_payload(parsed, network, config)
        if self.admin_mode:
            action.logistics = _admin_qualitative_logistics_action(
                network,
                parsed,
                rng,
                config,
                self.samples,
            )
        elif _intervention_budget(action.logistics, config.base) <= 0.0:
            action.logistics = _monte_carlo_logistics_action(network, rng, config.base, self.samples)

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
                "reason": str(parsed.get("reason", ""))[:240],
                "latent_estimates": parsed.get("latent_estimates", {}),
                "action": _qualitative_action_to_dict(action),
            }
        )
        print(
            f"  {self.name}: Codex returned {status} qualitative action "
            f"(social={action.social_action or 'none'}) in {elapsed:.1f}s.",
            flush=True,
        )
        return action


def run_qualitative_repeated(
    config: QualitativeConfig,
    policies: Sequence[QualitativePolicy],
    runs: Union[int, Dict[str, int]],
    seed: int,
    progress_interval: int = 0,
) -> Dict[str, List[QualitativeRunResult]]:
    results: Dict[str, List[QualitativeRunResult]] = {}
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
                    progress_interval=progress_interval,
                )
            )
    return results


def run_qualitative_simulation(
    config: QualitativeConfig,
    policy: QualitativePolicy,
    seed: int,
    progress_interval: int = 0,
) -> QualitativeRunResult:
    rng = np.random.default_rng(seed)
    policy_rng = np.random.default_rng(seed + 917_503)
    human_rng = np.random.default_rng(seed + 471_221)
    observation_rng = np.random.default_rng(seed + 884_911)
    network = SupplyChainNetwork(config.base.network_topology)
    human = HumanLayerState.initialize(network, human_rng)
    observed_network = _observed_network(network, config.base, observation_rng)

    time_values = np.arange(config.base.steps + 1)
    unmet = np.zeros(config.base.steps + 1)
    service = np.ones(config.base.steps + 1)
    loss = np.zeros(config.base.steps + 1)
    inventory = np.zeros(config.base.steps + 1)
    active_capacity = np.zeros(config.base.steps + 1)
    action_budget = np.zeros(config.base.steps + 1)
    social_budget = np.zeros(config.base.steps + 1)
    mean_trust = np.zeros(config.base.steps + 1)
    mean_compliance = np.zeros(config.base.steps + 1)
    mean_rumor = np.zeros(config.base.steps + 1)
    mean_fatigue = np.zeros(config.base.steps + 1)
    mean_cooperation = np.zeros(config.base.steps + 1)
    mean_equity = np.zeros(config.base.steps + 1)

    history: List[Dict[str, Any]] = []
    reports: List[str] = []
    current_action = QualitativeAction(logistics=Intervention())
    current_logistics_budget = 0.0
    current_social_budget = 0.0
    controller_decisions = 0

    for step in range(config.base.steps + 1):
        _record_qualitative_state(
            network,
            human,
            history,
            step,
            unmet,
            service,
            loss,
            inventory,
            active_capacity,
            mean_trust,
            mean_compliance,
            mean_rumor,
            mean_fatigue,
            mean_cooperation,
            mean_equity,
            reports,
        )
        if progress_interval > 0 and step % progress_interval == 0:
            print(
                f"  qualitative {policy.name}: step {step}/{config.base.steps} "
                f"unmet={unmet[step]:.1f} service={service[step]:.2f} "
                f"trust={mean_trust[step]:.2f} rumor={mean_rumor[step]:.2f}.",
                flush=True,
            )
        if step == config.base.steps:
            break

        if step % max(1, config.base.observation_update_interval) == 0:
            observed_network = _observed_network(network, config.base, observation_rng)

        if step % max(1, config.base.control_update_interval) == 0:
            reports = _generate_reports(network, human, human_rng, config)
            raw_action = policy.choose_action(
                observed_network,
                history,
                reports,
                step,
                policy_rng,
                config,
                structured_human_state=human.copy()
                if policy.name == "q_structured_human_state_monte_carlo"
                else None,
            )
            current_action, current_logistics_budget, current_social_budget = (
                _normalize_qualitative_action(raw_action, network, config)
            )
            controller_decisions += 1

        social_cost = _apply_social_action(network, human, current_action, config, human_rng)
        logistics_cost = _apply_intervention(
            network,
            current_action.logistics,
            config.base,
            rng,
        )
        _apply_action_side_effects(network, human, current_action)
        attack_loss = _apply_disruptions(network, config.base, rng)
        _human_attack_effects(network, human, attack_loss)
        _recover_qualitative(network, human, config)
        _produce_qualitative(network, human)
        _ship_qualitative(network, human, config, rng, current_action)
        demand, unmet_step, per_district = _consume_demand_qualitative(
            network,
            human,
            config,
            rng,
            current_action,
        )
        _update_human_state(human, network, per_district, current_action, config, human_rng)

        unmet[step + 1] = unmet_step
        service[step + 1] = 1.0 - unmet_step / max(demand, 1.0)
        loss[step + 1] = (
            8.0 * unmet_step
            + attack_loss
            + logistics_cost
            + social_cost
            + 35.0 * human.institutional_friction
        )
        action_budget[step + 1] = current_logistics_budget + current_social_budget
        social_budget[step + 1] = current_social_budget

    return QualitativeRunResult(
        time=time_values,
        unmet_demand=unmet,
        service_level=service,
        economic_loss=loss,
        total_inventory=inventory,
        active_capacity=active_capacity,
        action_budget_used=action_budget,
        social_budget_used=social_budget,
        mean_trust=mean_trust,
        mean_compliance=mean_compliance,
        mean_rumor_pressure=mean_rumor,
        mean_workforce_fatigue=mean_fatigue,
        mean_carrier_cooperation=mean_cooperation,
        mean_equity_pressure=mean_equity,
        policy_name=policy.name,
        policy_stats={
            **policy.decision_stats(),
            "controller_decisions": float(controller_decisions),
        },
        decision_log=policy.decision_log(),
    )


def summarize_qualitative(
    results: Dict[str, List[QualitativeRunResult]],
    config: QualitativeConfig,
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
            "qualitative_resilience_score_lower_is_better": float(
                np.mean([item["qualitative_resilience_score_lower_is_better"] for item in run_metrics])
            ),
            "qualitative_resilience_score_ci95": _mean_interval(
                [item["qualitative_resilience_score_lower_is_better"] for item in run_metrics]
            ),
            "total_unmet_demand": _mean_interval([item["total_unmet_demand"] for item in run_metrics]),
            "economic_loss": _mean_interval([item["economic_loss"] for item in run_metrics]),
            "service_level": _mean_interval([item["service_level"] for item in run_metrics]),
            "mean_action_budget_used": _mean_interval(
                [item["mean_action_budget_used"] for item in run_metrics]
            ),
            "mean_social_budget_used": _mean_interval(
                [item["mean_social_budget_used"] for item in run_metrics]
            ),
            "mean_trust": _mean_interval([item["mean_trust"] for item in run_metrics]),
            "mean_compliance": _mean_interval([item["mean_compliance"] for item in run_metrics]),
            "mean_rumor_pressure": _mean_interval(
                [item["mean_rumor_pressure"] for item in run_metrics]
            ),
            "mean_workforce_fatigue": _mean_interval(
                [item["mean_workforce_fatigue"] for item in run_metrics]
            ),
            "mean_carrier_cooperation": _mean_interval(
                [item["mean_carrier_cooperation"] for item in run_metrics]
            ),
            "mean_equity_pressure": _mean_interval(
                [item["mean_equity_pressure"] for item in run_metrics]
            ),
            "decision_stats": _aggregate_policy_stats(runs),
        }
    ranking = [
        {
            "policy": policy_name,
            "qualitative_resilience_score_lower_is_better": row[
                "qualitative_resilience_score_lower_is_better"
            ],
            "score_ci95_low": row["qualitative_resilience_score_ci95"]["ci95_low"],
            "score_ci95_high": row["qualitative_resilience_score_ci95"]["ci95_high"],
        }
        for policy_name, row in metrics.items()
    ]
    ranking.sort(key=lambda item: item["qualitative_resilience_score_lower_is_better"])
    return {
        "study": "supply_chain_qualitative_resilience",
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
    results: Dict[str, List[QualitativeRunResult]],
    output_dir: Path,
    config: QualitativeConfig,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_qualitative(results, config)
    summary["config"] = asdict(config)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return [summary_path]


def print_qualitative_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    width = max(24, *(len(policy) for policy in metrics))
    print("Qualitative sociotechnical ranking lower-is-better:")
    print(
        f"{'policy':{width}s} {'score':>9s} {'physical':>9s} {'human':>9s} "
        f"{'service':>9s} {'trust':>8s} {'rumor':>8s} {'budget':>8s}"
    )
    for item in summary["ranking_lower_is_better"]:
        policy = item["policy"]
        row = metrics[policy]
        print(
            f"{policy:{width}s} "
            f"{row['qualitative_resilience_score_lower_is_better']:9.3f} "
            f"{row['physical_score_lower_is_better']['mean']:9.3f} "
            f"{row['human_score_lower_is_better']['mean']:9.3f} "
            f"{row['service_level']['mean']:9.3f} "
            f"{row['mean_trust']['mean']:8.3f} "
            f"{row['mean_rumor_pressure']['mean']:8.3f} "
            f"{row['mean_action_budget_used']['mean']:8.3f}"
        )

    comparisons = summary.get("comparisons", {})
    best_deployable = comparisons.get("best_deployable_non_codex_policy")
    structured_state = comparisons.get(
        "structured_human_state_policy",
        comparisons.get("oracle_upper_bound_policy"),
    )
    if best_deployable:
        print(f"Best deployable non-Codex qualitative policy: {best_deployable}")
    if structured_state:
        print(f"Structured human-state heuristic baseline: {structured_state}")


def _record_qualitative_state(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    history: List[Dict[str, Any]],
    step: int,
    unmet: np.ndarray,
    service: np.ndarray,
    loss: np.ndarray,
    inventory: np.ndarray,
    active_capacity: np.ndarray,
    mean_trust: np.ndarray,
    mean_compliance: np.ndarray,
    mean_rumor: np.ndarray,
    mean_fatigue: np.ndarray,
    mean_cooperation: np.ndarray,
    mean_equity: np.ndarray,
    reports: List[str],
) -> None:
    inventory[step] = sum(node.inventory for node in network.nodes.values())
    active_capacity[step] = sum(node.capacity * node.health for node in network.nodes.values())
    mean_trust[step] = _mean_dict(human.trust)
    mean_compliance[step] = _mean_dict(human.compliance)
    mean_rumor[step] = _mean_dict(human.rumor_pressure)
    mean_fatigue[step] = _mean_dict(human.workforce_fatigue)
    mean_cooperation[step] = _mean_dict(human.carrier_cooperation)
    mean_equity[step] = _mean_dict(human.equity_pressure)
    history.append(
        {
            "step": float(step),
            "unmet_demand": float(unmet[step]),
            "service_level": float(service[step]),
            "economic_loss": float(loss[step]),
            "inventory": float(inventory[step]),
            "active_capacity": float(active_capacity[step]),
            "mean_trust_reported": round(float(mean_trust[step]), 2),
            "mean_compliance_reported": round(float(mean_compliance[step]), 2),
            "mean_rumor_pressure_reported": round(float(mean_rumor[step]), 2),
            "recent_reports": list(reports),
        }
    )


def _generate_reports(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    rng: np.random.Generator,
    config: QualitativeConfig,
) -> List[str]:
    reports: List[Tuple[float, str]] = []
    rumor_target = max(human.rumor_pressure, key=human.rumor_pressure.get)
    trust_target = min(human.trust, key=human.trust.get)
    equity_target = max(human.equity_pressure, key=human.equity_pressure.get)
    fatigue_target = max(human.workforce_fatigue, key=human.workforce_fatigue.get)
    cooperation_edge = min(human.carrier_cooperation, key=human.carrier_cooperation.get)

    reports.append(
        (
            human.rumor_pressure[rumor_target],
            (
                f"Local radio near {rumor_target} is amplifying claims that "
                "supplies are being diverted before the next delivery window."
            ),
        )
    )
    reports.append(
        (
            1.0 - human.trust[trust_target],
            (
                f"Community coordinators in {trust_target} say residents doubt "
                "the allocation plan and may not follow rationing guidance."
            ),
        )
    )
    reports.append(
        (
            human.equity_pressure[equity_target],
            (
                f"District leaders around {equity_target} argue recent deliveries "
                "have been unfair compared with other neighborhoods."
            ),
        )
    )
    reports.append(
        (
            human.workforce_fatigue[fatigue_target],
            (
                f"Supervisors at {fatigue_target} report crew fatigue and warn "
                "that another surge shift may increase loading errors."
            ),
        )
    )
    reports.append(
        (
            1.0 - human.carrier_cooperation[cooperation_edge],
            (
                f"Dispatchers say drivers are reluctant to use "
                f"{cooperation_edge[0]} -> {cooperation_edge[1]} without clearer "
                "coordination and safety assurances."
            ),
        )
    )

    if rng.random() < config.report_noise_probability:
        random_retailer = str(rng.choice([node.name for node in network.retailers()]))
        reports.append(
            (
                0.35,
                (
                    f"An unverified social post claims {random_retailer} has already "
                    "received extra supplies, but coordinators cannot confirm it."
                ),
            )
        )

    reports.sort(key=lambda item: item[0], reverse=True)
    return [report for _, report in reports[: config.report_count]]


def _normalize_qualitative_action(
    action: QualitativeAction,
    network: SupplyChainNetwork,
    config: QualitativeConfig,
) -> Tuple[QualitativeAction, float, float]:
    logistics, logistics_budget = _normalize_intervention(network, action.logistics, config.base)
    social_action = _normalize_social_action(action.social_action)
    social_target = action.social_target if social_action else None
    social_edge = action.social_edge if social_action and _edge_exists(network, action.social_edge) else None
    if social_action in {"carrier_negotiation", "police_escort"} and social_edge is None:
        social_action = None
    if social_action in {
        "public_message",
        "community_liaison",
        "rationing_policy",
        "equity_rebalance",
    } and social_target not in {node.name for node in network.retailers()}:
        social_action = None
    if social_action in {"staff_rotation", "mutual_aid_request"} and social_target not in network.nodes:
        social_action = None

    intensity = _clip(action.social_intensity, 0.0, 1.0) if social_action else 0.0
    social_budget = config.social_budget_weight * intensity
    total_budget = logistics_budget + social_budget
    if total_budget > config.base.action_budget and total_budget > 0.0:
        scale = config.base.action_budget / total_budget
        logistics = Intervention(
            buffer_node=logistics.buffer_node,
            buffer_amount=logistics.buffer_amount * scale,
            reinforce_edge=logistics.reinforce_edge,
            reinforce_amount=logistics.reinforce_amount * scale,
            expedite_edge=logistics.expedite_edge,
            expedite_priority=logistics.expedite_priority * scale,
        )
        logistics_budget = _intervention_budget(logistics, config.base)
        intensity *= scale
        social_budget = config.social_budget_weight * intensity

    return (
        QualitativeAction(
            logistics=logistics,
            social_action=social_action,
            social_target=social_target,
            social_edge=social_edge,
            social_intensity=intensity,
        ),
        logistics_budget,
        social_budget,
    )


def _apply_social_action(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    action: QualitativeAction,
    config: QualitativeConfig,
    rng: np.random.Generator,
) -> float:
    if not action.social_action or action.social_intensity <= 0.0:
        return 0.0
    intensity = action.social_intensity
    social = action.social_action
    target = action.social_target
    edge = action.social_edge
    if social == "public_message" and target in human.trust:
        human.rumor_pressure[target] = _clip(human.rumor_pressure[target] - 0.24 * intensity, 0.0, 1.0)
        human.trust[target] = _clip(human.trust[target] + 0.07 * intensity, 0.0, 1.0)
        human.compliance[target] = _clip(human.compliance[target] + 0.05 * intensity, 0.0, 1.0)
    elif social == "community_liaison" and target in human.trust:
        human.trust[target] = _clip(human.trust[target] + 0.14 * intensity, 0.0, 1.0)
        human.compliance[target] = _clip(human.compliance[target] + 0.11 * intensity, 0.0, 1.0)
        human.equity_pressure[target] = _clip(
            human.equity_pressure[target] - 0.18 * intensity,
            0.0,
            1.0,
        )
    elif social == "carrier_negotiation" and edge in human.carrier_cooperation:
        human.carrier_cooperation[edge] = _clip(
            human.carrier_cooperation[edge] + 0.24 * intensity,
            0.0,
            1.0,
        )
        human.institutional_friction = _clip(human.institutional_friction - 0.03 * intensity, 0.0, 1.0)
    elif social == "police_escort" and edge in human.carrier_cooperation:
        human.carrier_cooperation[edge] = _clip(
            human.carrier_cooperation[edge] + 0.18 * intensity,
            0.0,
            1.0,
        )
        target_node = edge[1]
        if target_node in human.trust:
            human.trust[target_node] = _clip(human.trust[target_node] - 0.035 * intensity, 0.0, 1.0)
            human.rumor_pressure[target_node] = _clip(
                human.rumor_pressure[target_node] - 0.06 * intensity,
                0.0,
                1.0,
            )
    elif social == "staff_rotation" and target in human.workforce_fatigue:
        human.workforce_fatigue[target] = _clip(
            human.workforce_fatigue[target] - 0.32 * intensity,
            0.0,
            1.0,
        )
    elif social == "mutual_aid_request" and target in network.nodes:
        node = network.nodes[target]
        node.inventory = min(node.max_inventory, node.inventory + 16.0 * intensity)
        human.institutional_friction = _clip(human.institutional_friction + 0.025 * intensity, 0.0, 1.0)
    elif social == "equity_rebalance" and target in human.trust:
        human.equity_pressure[target] = _clip(
            human.equity_pressure[target] - 0.24 * intensity,
            0.0,
            1.0,
        )
        human.trust[target] = _clip(human.trust[target] + 0.09 * intensity, 0.0, 1.0)
    elif social == "rationing_policy" and target in human.trust:
        human.compliance[target] = _clip(
            human.compliance[target] + 0.08 * intensity,
            0.0,
            1.0,
        )
        trust_cost = 0.05 * intensity * (1.0 - human.trust[target])
        human.trust[target] = _clip(human.trust[target] - trust_cost, 0.0, 1.0)
    return config.social_action_cost * intensity


def _apply_action_side_effects(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    action: QualitativeAction,
) -> None:
    if action.logistics.expedite_edge:
        edge = action.logistics.expedite_edge
        human.carrier_cooperation[edge] = _clip(
            human.carrier_cooperation.get(edge, 0.75) - 0.025 * action.logistics.expedite_priority,
            0.0,
            1.0,
        )
        for node_name in edge:
            if node_name in human.workforce_fatigue:
                human.workforce_fatigue[node_name] = _clip(
                    human.workforce_fatigue[node_name] + 0.030 * action.logistics.expedite_priority,
                    0.0,
                    1.0,
                )
    if action.logistics.buffer_node in human.workforce_fatigue:
        node_name = str(action.logistics.buffer_node)
        human.workforce_fatigue[node_name] = _clip(
            human.workforce_fatigue[node_name] + 0.018,
            0.0,
            1.0,
        )


def _human_attack_effects(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    attack_loss: float,
) -> None:
    if attack_loss <= 0.0:
        return
    stress = min(0.08, attack_loss / 5000.0)
    for name in human.rumor_pressure:
        human.rumor_pressure[name] = _clip(human.rumor_pressure[name] + stress * 0.45, 0.0, 1.0)
        human.trust[name] = _clip(human.trust[name] - stress * 0.20, 0.0, 1.0)
    for edge in list(human.carrier_cooperation):
        try:
            route_health = network.edge(*edge).health
        except KeyError:
            route_health = 1.0
        if route_health < 0.55:
            human.carrier_cooperation[edge] = _clip(
                human.carrier_cooperation[edge] - stress * 0.35,
                0.0,
                1.0,
            )


def _recover_qualitative(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    config: QualitativeConfig,
) -> None:
    for node in network.nodes.values():
        node.health = min(1.0, node.health + config.base.node_recovery_rate)
        node.protected_stock *= 0.995
    for edge in network.edges:
        edge.health = min(1.0, edge.health + config.base.edge_recovery_rate + 0.02 * edge.reinforced)
        edge.reinforced *= 0.998

    for name in human.trust:
        human.trust[name] = _clip(
            human.trust[name] + config.trust_recovery_rate * (0.72 - human.trust[name]),
            0.0,
            1.0,
        )
        human.rumor_pressure[name] = _clip(
            human.rumor_pressure[name] * (1.0 - config.rumor_decay_rate),
            0.0,
            1.0,
        )
        human.equity_pressure[name] = _clip(
            human.equity_pressure[name] * (1.0 - config.equity_decay_rate),
            0.0,
            1.0,
        )
    for name in human.workforce_fatigue:
        human.workforce_fatigue[name] = _clip(
            human.workforce_fatigue[name] * (1.0 - config.fatigue_recovery_rate),
            0.0,
            1.0,
        )
    for edge in human.carrier_cooperation:
        human.carrier_cooperation[edge] = _clip(
            human.carrier_cooperation[edge]
            + config.cooperation_recovery_rate * (0.82 - human.carrier_cooperation[edge]),
            0.0,
            1.0,
        )


def _produce_qualitative(network: SupplyChainNetwork, human: HumanLayerState) -> None:
    for supplier in network.suppliers():
        supplier.inventory = min(
            supplier.max_inventory,
            supplier.inventory + supplier.capacity * supplier.health,
        )
    for factory in network.factories():
        fatigue = human.workforce_fatigue.get(factory.name, 0.0)
        effective_health = factory.health * max(0.35, 1.0 - 0.42 * fatigue)
        inbound_stock = factory.inventory
        produced = min(factory.capacity * effective_health, inbound_stock * 0.55)
        factory.inventory = min(factory.max_inventory, factory.inventory - produced * 0.45 + produced)
        human.workforce_fatigue[factory.name] = _clip(fatigue + 0.006 * produced / 30.0, 0.0, 1.0)


def _ship_qualitative(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    config: QualitativeConfig,
    rng: np.random.Generator,
    action: QualitativeAction,
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
        expedite = action.logistics.expedite_edge == (edge.source, edge.target)
        expedite_multiplier = 1.0
        if expedite:
            expedite_multiplier += (
                config.base.max_expedite_multiplier - 1.0
            ) * action.logistics.expedite_priority
        cooperation = human.carrier_cooperation.get((edge.source, edge.target), 0.75)
        source_fatigue = human.workforce_fatigue.get(edge.source, 0.0)
        capacity = (
            edge.capacity
            * edge.health
            * cooperation
            * max(0.45, 1.0 - 0.35 * source_fatigue)
            * expedite_multiplier
        )
        amount = min(source.inventory * config.base.max_ship_fraction, capacity, deficit)
        if expedite and rng.random() > max(0.10, cooperation - 0.20 * source_fatigue):
            amount *= 0.45
        source.inventory -= amount
        target.inventory = min(target.max_inventory, target.inventory + amount)
        if edge.source in human.workforce_fatigue:
            human.workforce_fatigue[edge.source] = _clip(
                human.workforce_fatigue[edge.source] + 0.004 * amount / 20.0,
                0.0,
                1.0,
            )


def _consume_demand_qualitative(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    config: QualitativeConfig,
    rng: np.random.Generator,
    action: QualitativeAction,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    total_demand = 0.0
    total_unmet = 0.0
    per_district: Dict[str, Tuple[float, float]] = {}
    surge_target = None
    if rng.random() < config.base.demand_surge_probability:
        surge_target = rng.choice(network.retailers()).name
    for retailer in network.retailers():
        trust = human.trust[retailer.name]
        compliance = human.compliance[retailer.name]
        rumor = human.rumor_pressure[retailer.name]
        equity = human.equity_pressure[retailer.name]
        demand = max(0.0, retailer.demand * rng.normal(1.0, config.base.demand_noise))
        demand *= 1.0 + 0.46 * rumor + 0.22 * (1.0 - trust) + 0.18 * equity
        demand *= max(0.72, 1.0 - 0.22 * compliance)
        if retailer.name == surge_target:
            demand *= rng.uniform(config.base.demand_surge_min, config.base.demand_surge_max)
        if action.social_action == "rationing_policy" and action.social_target == retailer.name:
            demand *= max(0.65, 1.0 - 0.30 * compliance * action.social_intensity)
        served = min(retailer.inventory, demand)
        retailer.inventory -= served
        unmet = demand - served
        total_demand += demand
        total_unmet += unmet
        per_district[retailer.name] = (demand, unmet)
    return total_demand, total_unmet, per_district


def _update_human_state(
    human: HumanLayerState,
    network: SupplyChainNetwork,
    per_district: Dict[str, Tuple[float, float]],
    action: QualitativeAction,
    config: QualitativeConfig,
    rng: np.random.Generator,
) -> None:
    unmet_rates = {
        name: unmet / max(demand, 1.0)
        for name, (demand, unmet) in per_district.items()
    }
    mean_unmet_rate = float(np.mean(list(unmet_rates.values()))) if unmet_rates else 0.0
    for name, unmet_rate in unmet_rates.items():
        relative_neglect = max(0.0, unmet_rate - mean_unmet_rate)
        human.trust[name] = _clip(
            human.trust[name]
            - 0.055 * unmet_rate
            - 0.010 * human.rumor_pressure[name]
            - 0.018 * human.equity_pressure[name]
            + float(rng.normal(0.0, 0.004)),
            0.0,
            1.0,
        )
        human.rumor_pressure[name] = _clip(
            human.rumor_pressure[name]
            + 0.070 * unmet_rate
            + 0.018 * (1.0 - human.trust[name])
            + float(rng.normal(0.0, 0.004)),
            0.0,
            1.0,
        )
        human.equity_pressure[name] = _clip(
            human.equity_pressure[name] + 0.085 * relative_neglect - 0.010,
            0.0,
            1.0,
        )
        human.compliance[name] = _clip(
            0.18
            + 0.70 * human.trust[name]
            - 0.38 * human.rumor_pressure[name]
            - 0.24 * human.equity_pressure[name]
            + float(rng.normal(0.0, 0.015)),
            0.0,
            1.0,
        )

    for edge in list(human.carrier_cooperation):
        try:
            route_health = network.edge(*edge).health
        except KeyError:
            continue
        target_rumor = human.rumor_pressure.get(edge[1], _mean_dict(human.rumor_pressure))
        human.carrier_cooperation[edge] = _clip(
            human.carrier_cooperation[edge]
            - 0.020 * max(0.0, 0.70 - route_health)
            - 0.010 * target_rumor,
            0.0,
            1.0,
        )


def _monte_carlo_logistics_action(
    network: SupplyChainNetwork,
    rng: np.random.Generator,
    config: SupplyChainConfig,
    samples: int,
) -> Intervention:
    candidates = [("native", action) for action in _candidate_interventions(network, rng, config)]
    selected, _ = _select_monte_carlo_action(network, candidates, config, rng, max_evaluations=samples)
    return selected


def _structured_human_state_logistics_action(
    network: SupplyChainNetwork,
    human: HumanLayerState,
    rng: np.random.Generator,
    config: QualitativeConfig,
    samples: int,
) -> Intervention:
    scored: List[Tuple[float, Intervention]] = []
    seen = set()
    for action in _candidate_interventions(network, rng, config.base):
        if len(scored) >= samples:
            break
        normalized, _ = _normalize_intervention(network, action, config.base)
        key = tuple(json.dumps(_intervention_to_dict(normalized), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        trial = network.copy()
        _apply_intervention(trial, normalized, config.base, rng, dry_run=True)
        score = _qualitative_candidate_score(trial, normalized, human, config)
        scored.append((score, normalized))
    if not scored:
        return Intervention()
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _admin_qualitative_logistics_action(
    network: SupplyChainNetwork,
    payload: Dict[str, Any],
    rng: np.random.Generator,
    config: QualitativeConfig,
    samples: int,
) -> Intervention:
    priority_nodes = [
        str(name)
        for name in payload.get("priority_nodes", [])
        if str(name) in network.nodes
    ][:6]
    priority_edges = [
        edge
        for edge in (_parse_edge({"edge": item}) for item in payload.get("priority_edges", []))
        if _edge_exists(network, edge)
    ][:8]
    plan = {
        "active": True,
        "candidate_mix": {"buffer": 0.30, "reinforce": 0.25, "expedite": 0.20, "combined": 0.25},
        "risk_mode": "service",
        "sample_multiplier": 1.0,
        "priority_nodes": priority_nodes,
        "priority_edges": priority_edges,
        "objective_weights": {
            "inventory_gap": 1.0,
            "node_health": 80.0,
            "edge_health": 55.0,
            "retailer_gap": 2.5,
            "action_budget": 65.0,
        },
    }
    candidates = [("native", action) for action in _candidate_interventions(network, rng, config.base)]
    candidates.extend(("codex_priority", action) for action in _admin_guided_candidates(network, plan, rng, config.base))
    selected, _ = _select_monte_carlo_action(
        network,
        candidates,
        config.base,
        rng,
        max_evaluations=max(samples, samples + len(priority_nodes) + len(priority_edges)),
        score_plan=plan,
    )
    return selected


def _qualitative_candidate_score(
    network: SupplyChainNetwork,
    action: Intervention,
    human: HumanLayerState,
    config: QualitativeConfig,
) -> float:
    score = _candidate_score(network, action, config.base)
    if action.expedite_edge:
        cooperation = human.carrier_cooperation.get(action.expedite_edge, 0.75)
        target = action.expedite_edge[1]
        score += 240.0 * max(0.0, 0.65 - cooperation)
        score += 90.0 * human.rumor_pressure.get(target, 0.0)
    if action.buffer_node in human.trust:
        score -= 70.0 * human.equity_pressure[action.buffer_node]
        score -= 45.0 * human.rumor_pressure[action.buffer_node]
    if action.reinforce_edge:
        cooperation = human.carrier_cooperation.get(action.reinforce_edge, 0.75)
        score -= 60.0 * max(0.0, 0.90 - cooperation)
    return float(score)


def _keyword_social_action(
    network: SupplyChainNetwork,
    reports: Sequence[str],
) -> QualitativeAction:
    text = " ".join(reports).lower()
    if any(word in text for word in ["driver", "dispatch", "reluctant", "carrier"]):
        edge = _first_edge_in_text(network, text) or _lowest_health_edge(network)
        return QualitativeAction(
            logistics=Intervention(),
            social_action="carrier_negotiation",
            social_edge=edge,
            social_intensity=1.0,
        )
    if any(word in text for word in ["fatigue", "crew", "loading errors", "shift"]):
        node = _first_node_in_text(network, text, kinds={"factory", "warehouse"}) or _most_loaded_work_node(network)
        return QualitativeAction(
            logistics=Intervention(),
            social_action="staff_rotation",
            social_target=node,
            social_intensity=1.0,
        )
    if any(word in text for word in ["rumor", "claims", "diverted", "social post"]):
        target = _first_node_in_text(network, text, kinds={"retailer"}) or _lowest_inventory_retailer(network)
        return QualitativeAction(
            logistics=Intervention(),
            social_action="public_message",
            social_target=target,
            social_intensity=1.0,
        )
    if any(word in text for word in ["unfair", "doubt", "allocation", "leaders"]):
        target = _first_node_in_text(network, text, kinds={"retailer"}) or _lowest_inventory_retailer(network)
        return QualitativeAction(
            logistics=Intervention(),
            social_action="community_liaison",
            social_target=target,
            social_intensity=1.0,
        )
    return QualitativeAction(logistics=Intervention())


def _structured_human_state_social_action(
    network: SupplyChainNetwork,
    human: HumanLayerState,
) -> QualitativeAction:
    worst_rumor = max(human.rumor_pressure, key=human.rumor_pressure.get)
    worst_equity = max(human.equity_pressure, key=human.equity_pressure.get)
    worst_trust = min(human.trust, key=human.trust.get)
    worst_fatigue = max(human.workforce_fatigue, key=human.workforce_fatigue.get)
    worst_edge = min(human.carrier_cooperation, key=human.carrier_cooperation.get)
    candidates = [
        (human.rumor_pressure[worst_rumor], "public_message", worst_rumor, None),
        (human.equity_pressure[worst_equity], "equity_rebalance", worst_equity, None),
        (1.0 - human.trust[worst_trust], "community_liaison", worst_trust, None),
        (human.workforce_fatigue[worst_fatigue], "staff_rotation", worst_fatigue, None),
        (1.0 - human.carrier_cooperation[worst_edge], "carrier_negotiation", None, worst_edge),
    ]
    _, social_action, target, edge = max(candidates, key=lambda item: item[0])
    return QualitativeAction(
        logistics=Intervention(),
        social_action=social_action,
        social_target=target,
        social_edge=edge,
        social_intensity=1.0,
    )


def _qualitative_action_from_payload(
    payload: Dict[str, Any],
    network: SupplyChainNetwork,
    config: QualitativeConfig,
) -> Tuple[QualitativeAction, List[str]]:
    intervention, _, logistics_errors = _intervention_from_codex_payload(
        payload,
        network,
        config.base,
    )
    social_payload = payload.get("social_action", {})
    errors = list(logistics_errors)
    if not isinstance(social_payload, dict):
        return QualitativeAction(logistics=intervention), errors + ["social_action must be an object"]
    social_type = _normalize_social_action(str(social_payload.get("type", "")))
    social_target = str(social_payload.get("target", "")).strip() or None
    social_edge = _parse_edge(social_payload)
    intensity = _safe_float(social_payload.get("intensity"), 0.0)
    if social_type is None and intensity > 0.0:
        errors.append("unknown social_action type")
    return (
        QualitativeAction(
            logistics=intervention,
            social_action=social_type,
            social_target=social_target,
            social_edge=social_edge,
            social_intensity=intensity,
        ),
        errors,
    )


def _codex_qualitative_prompt(
    network: SupplyChainNetwork,
    history: List[Dict[str, Any]],
    reports: List[str],
    config: QualitativeConfig,
    admin_mode: bool,
) -> str:
    payload = {
        "system": "sociotechnical supply-chain resilience control",
        "objective": (
            "Minimize unmet demand, economic loss, and human-organizational fragility. "
            "Reports are noisy observations of hidden quantitative trust, compliance, rumor, "
            "fatigue, carrier cooperation, and equity states."
        ),
        "mode": "monte_carlo_admin" if admin_mode else "direct_action",
        "network_context": _network_context(network),
        "recent_history": history[-8:],
        "reports": reports,
        "action_budget": config.base.action_budget,
        "social_actions": [
            "public_message",
            "community_liaison",
            "carrier_negotiation",
            "police_escort",
            "staff_rotation",
            "mutual_aid_request",
            "rationing_policy",
            "equity_rebalance",
        ],
        "nodes": [
            {
                "name": node.name,
                "kind": node.kind,
                "inventory": round(node.inventory, 2),
                "max_inventory": round(node.max_inventory, 2),
                "demand": round(node.demand, 2),
                "health": round(node.health, 3),
            }
            for node in network.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "capacity": round(edge.capacity, 2),
                "health": round(edge.health, 3),
            }
            for edge in sorted(network.edges, key=lambda edge: edge.capacity * edge.health)[:10]
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
    if admin_mode:
        schema = (
            "{\"social_action\":{\"type\":\"public_message|community_liaison|carrier_negotiation|"
            "police_escort|staff_rotation|mutual_aid_request|rationing_policy|equity_rebalance\","
            "\"target\":\"NODE_NAME\",\"edge\":[\"SOURCE\",\"TARGET\"],\"intensity\":0.0},"
            "\"priority_nodes\":[\"NODE_NAME\"],\"priority_edges\":[[\"SOURCE\",\"TARGET\"]],"
            "\"latent_estimates\":{\"short qualitative estimates\":\"values\"},"
            "\"reason\":\"short reason\"}"
        )
        instruction = (
            "You are a qualitative administrator for a Monte Carlo logistics controller. "
            "Interpret the reports as evidence about hidden quantitative human states. "
            "Choose one social intervention and logistics priority nodes/edges. "
            "The simulator will use your priorities to generate logistics candidates."
        )
    else:
        schema = (
            "{\"actions\":[{\"type\":\"buffer\",\"node\":\"NODE_NAME\",\"amount\":0.0},"
            "{\"type\":\"reinforce\",\"edge\":[\"SOURCE\",\"TARGET\"],\"amount\":0.0},"
            "{\"type\":\"expedite\",\"edge\":[\"SOURCE\",\"TARGET\"],\"priority\":0.0}],"
            "\"social_action\":{\"type\":\"public_message|community_liaison|carrier_negotiation|"
            "police_escort|staff_rotation|mutual_aid_request|rationing_policy|equity_rebalance\","
            "\"target\":\"NODE_NAME\",\"edge\":[\"SOURCE\",\"TARGET\"],\"intensity\":0.0},"
            "\"latent_estimates\":{\"short qualitative estimates\":\"values\"},"
            "\"reason\":\"short reason\"}"
        )
        instruction = (
            "You are a sociotechnical supply-chain controller. Infer latent human and "
            "organizational risks from the reports, then choose at most one logistics "
            "intervention of each type and one social intervention."
        )
    return (
        f"{instruction} Use only node and edge names from the payload. "
        "Do not claim access to hidden state; reason from reports and telemetry. "
        "Return only JSON with this shape:\n"
        f"{schema}\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _qualitative_run_metrics(
    run: QualitativeRunResult,
    config: QualitativeConfig,
) -> Dict[str, float]:
    total_unmet = float(np.sum(run.unmet_demand))
    total_loss = float(np.sum(run.economic_loss))
    service = float(np.mean(run.service_level))
    terminal_inventory = float(run.total_inventory[-1])
    budget = float(np.mean(run.action_budget_used))
    physical_score = (
        total_unmet / 2500.0
        + total_loss / 25000.0
        + 1.4 * (1.0 - service)
        + 0.15 * (1.0 / max(terminal_inventory / 500.0, 0.1))
        + config.base.action_budget_score_weight * budget
    )
    human_score = (
        1.15 * (1.0 - float(np.mean(run.mean_trust)))
        + 0.95 * (1.0 - float(np.mean(run.mean_compliance)))
        + 1.20 * float(np.mean(run.mean_rumor_pressure))
        + 0.95 * float(np.mean(run.mean_workforce_fatigue))
        + 0.85 * (1.0 - float(np.mean(run.mean_carrier_cooperation)))
        + 1.10 * float(np.mean(run.mean_equity_pressure))
    )
    return {
        "total_unmet_demand": total_unmet,
        "economic_loss": total_loss,
        "service_level": service,
        "terminal_inventory": terminal_inventory,
        "mean_action_budget_used": budget,
        "mean_social_budget_used": float(np.mean(run.social_budget_used)),
        "mean_trust": float(np.mean(run.mean_trust)),
        "mean_compliance": float(np.mean(run.mean_compliance)),
        "mean_rumor_pressure": float(np.mean(run.mean_rumor_pressure)),
        "mean_workforce_fatigue": float(np.mean(run.mean_workforce_fatigue)),
        "mean_carrier_cooperation": float(np.mean(run.mean_carrier_cooperation)),
        "mean_equity_pressure": float(np.mean(run.mean_equity_pressure)),
        "physical_score_lower_is_better": float(physical_score),
        "human_score_lower_is_better": float(human_score),
        "qualitative_resilience_score_lower_is_better": float(
            physical_score + config.human_score_weight * human_score
        ),
    }


def _paired_qualitative_comparisons(
    per_run_metrics: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Any]:
    deployable_non_codex = [
        name
        for name in per_run_metrics
        if not name.startswith("q_codex") and name != "q_baseline"
        and name != "q_structured_human_state_monte_carlo"
    ]
    structured_state = "q_structured_human_state_monte_carlo"
    best_deployable = min(
        deployable_non_codex,
        key=lambda name: np.mean(
            [item["qualitative_resilience_score_lower_is_better"] for item in per_run_metrics[name]]
        ),
    ) if deployable_non_codex else None
    return {
        "best_deployable_non_codex_policy": best_deployable,
        "structured_human_state_policy": (
            structured_state if structured_state in per_run_metrics else None
        ),
        "vs_best_deployable_non_codex": {
            name: _paired_score_delta(
                items,
                per_run_metrics[best_deployable],
                "qualitative_resilience_score_lower_is_better",
            )
            for name, items in per_run_metrics.items()
            if best_deployable and name != best_deployable
        },
        "vs_structured_human_state": {
            name: _paired_score_delta(
                items,
                per_run_metrics[structured_state],
                "qualitative_resilience_score_lower_is_better",
            )
            for name, items in per_run_metrics.items()
            if structured_state in per_run_metrics and name != structured_state
        },
    }


def _paired_score_delta(
    policy_metrics: List[Dict[str, float]],
    reference_metrics: List[Dict[str, float]],
    key: str,
) -> Dict[str, Optional[float]]:
    count = min(len(policy_metrics), len(reference_metrics))
    if count <= 0:
        return {
            "paired_runs": 0,
            "mean_delta_policy_minus_reference": None,
            "win_rate_lower_score": None,
        }
    policy_scores = np.array([item[key] for item in policy_metrics[:count]], dtype=float)
    reference_scores = np.array([item[key] for item in reference_metrics[:count]], dtype=float)
    delta = policy_scores - reference_scores
    interval = _mean_interval(delta)
    return {
        "paired_runs": count,
        "mean_delta_policy_minus_reference": interval["mean"],
        "delta_ci95_low": interval["ci95_low"],
        "delta_ci95_high": interval["ci95_high"],
        "win_rate_lower_score": float(np.mean(policy_scores < reference_scores)),
    }


def _mean_interval(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array)) if array.size else 0.0
    if array.size <= 1:
        return {"mean": mean, "std": 0.0, "sem": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(array, ddof=1))
    sem = std / float(np.sqrt(array.size))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
    }


def _aggregate_policy_stats(runs: Sequence[QualitativeRunResult]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for run in runs:
        for key, value in run.policy_stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _qualitative_action_to_dict(action: QualitativeAction) -> Dict[str, Any]:
    return {
        "logistics": _intervention_to_dict(action.logistics),
        "social_action": action.social_action,
        "social_target": action.social_target,
        "social_edge": list(action.social_edge) if action.social_edge else None,
        "social_intensity": action.social_intensity,
    }


def _normalize_social_action(action: Optional[str]) -> Optional[str]:
    if action is None:
        return None
    normalized = str(action).strip().lower()
    aliases = {
        "message": "public_message",
        "public messaging": "public_message",
        "liaison": "community_liaison",
        "negotiation": "carrier_negotiation",
        "escort": "police_escort",
        "rotation": "staff_rotation",
        "mutual_aid": "mutual_aid_request",
        "rationing": "rationing_policy",
        "equity": "equity_rebalance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {
        "public_message",
        "community_liaison",
        "carrier_negotiation",
        "police_escort",
        "staff_rotation",
        "mutual_aid_request",
        "rationing_policy",
        "equity_rebalance",
    }:
        return normalized
    return None


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _first_node_in_text(
    network: SupplyChainNetwork,
    text: str,
    kinds: Optional[set] = None,
) -> Optional[str]:
    for name, node in network.nodes.items():
        if kinds is not None and node.kind not in kinds:
            continue
        if name.lower() in text:
            return name
    return None


def _first_edge_in_text(
    network: SupplyChainNetwork,
    text: str,
) -> Optional[Tuple[str, str]]:
    for edge in network.edges:
        if edge.source.lower() in text and edge.target.lower() in text:
            return edge.source, edge.target
    return None


def _lowest_health_edge(network: SupplyChainNetwork) -> Optional[Tuple[str, str]]:
    edge = min(network.edges, key=lambda item: item.health, default=None)
    return (edge.source, edge.target) if edge else None


def _lowest_inventory_retailer(network: SupplyChainNetwork) -> Optional[str]:
    node = min(
        network.retailers(),
        key=lambda item: item.inventory / max(item.demand, 1.0),
        default=None,
    )
    return node.name if node else None


def _most_loaded_work_node(network: SupplyChainNetwork) -> Optional[str]:
    candidates = network.factories() + network.warehouses()
    if not candidates:
        return None
    return min(candidates, key=lambda node: node.inventory / max(node.max_inventory, 1.0)).name


def _mean_dict(values: Dict[Any, float]) -> float:
    return float(np.mean(list(values.values()))) if values else 0.0
