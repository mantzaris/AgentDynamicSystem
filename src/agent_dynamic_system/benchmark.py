from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class GenericScenario:
    name: str
    simulator: Callable[[int], "GenericRun"]


@dataclass
class GenericRun:
    time: np.ndarray
    variables: Dict[str, np.ndarray]
    actions: Dict[str, np.ndarray]


@dataclass
class GenericMonteCarloResult:
    system_name: str
    scenario_name: str
    seeds: Sequence[int]
    time: np.ndarray
    variables: Dict[str, np.ndarray]
    actions: Dict[str, np.ndarray]


@dataclass(frozen=True)
class MetricSpec:
    target: Dict[str, float]
    safety_min: Dict[str, float]
    safety_max: Dict[str, float]
    cost_weight: float = 0.05


def run_generic_scenario(
    system_name: str,
    scenario: GenericScenario,
    runs: int,
    base_seed: int,
) -> GenericMonteCarloResult:
    seeds = [base_seed + run_index for run_index in range(runs)]
    simulations: List[GenericRun] = [scenario.simulator(seed) for seed in seeds]
    variable_names = simulations[0].variables.keys()
    action_names = simulations[0].actions.keys()
    return GenericMonteCarloResult(
        system_name=system_name,
        scenario_name=scenario.name,
        seeds=seeds,
        time=simulations[0].time,
        variables={
            name: np.vstack([run.variables[name] for run in simulations])
            for name in variable_names
        },
        actions={
            name: np.vstack([run.actions[name] for run in simulations])
            for name in action_names
        },
    )


def summarize_generic_results(
    results: Sequence[GenericMonteCarloResult],
    metric_spec: MetricSpec,
) -> Dict[str, object]:
    metrics = {
        result.scenario_name: generic_instability(result, metric_spec)
        for result in results
    }
    baseline_score = metrics.get("baseline", {}).get("stability_score")
    ranking = []
    for scenario_name, scenario_metrics in metrics.items():
        score = scenario_metrics["stability_score"]
        relative_to_baseline = None
        if baseline_score and baseline_score > 0.0:
            relative_to_baseline = score / baseline_score
        ranking.append(
            {
                "scenario": scenario_name,
                "stability_score": score,
                "relative_to_baseline": relative_to_baseline,
            }
        )
    ranking.sort(key=lambda item: item["stability_score"])
    return {
        "metrics": metrics,
        "ranking_lower_is_better": ranking,
    }


def generic_instability(
    result: GenericMonteCarloResult,
    metric_spec: MetricSpec,
) -> Dict[str, float]:
    parts: Dict[str, float] = {}
    scores = []
    for name, values in result.variables.items():
        target = metric_spec.target.get(name)
        if target is None:
            continue
        target = max(float(target), 1.0e-9)
        temporal_mean = np.mean(values, axis=1)
        temporal_std = np.std(values, axis=1)
        cv = float(np.mean(temporal_std / np.maximum(np.abs(temporal_mean), 1.0)))
        deviation = float(np.mean(np.abs(values - target)) / target)
        change = float(np.mean(np.abs(np.diff(values, axis=1))) / target)
        low_breach = _breach_low(values, metric_spec.safety_min.get(name))
        high_breach = _breach_high(values, metric_spec.safety_max.get(name))
        variable_score = (
            0.25 * cv
            + 0.35 * deviation
            + 0.25 * change
            + 0.15 * (low_breach + high_breach)
        )
        parts[f"{name}_temporal_cv"] = cv
        parts[f"{name}_target_deviation"] = deviation
        parts[f"{name}_mean_abs_step_change"] = change
        parts[f"{name}_safety_breach"] = float(low_breach + high_breach)
        scores.append(variable_score)

    action_cost = 0.0
    if result.actions:
        action_cost = float(
            np.mean([np.mean(np.abs(values)) for values in result.actions.values()])
        )
    stability_score = float(np.mean(scores) if scores else 0.0)
    stability_score += metric_spec.cost_weight * action_cost
    parts["mean_action_cost"] = action_cost
    parts["stability_score"] = stability_score
    return parts


def _breach_low(values: np.ndarray, threshold: Optional[float]) -> float:
    if threshold is None:
        return 0.0
    return float(np.mean(values < float(threshold)))


def _breach_high(values: np.ndarray, threshold: Optional[float]) -> float:
    if threshold is None:
        return 0.0
    return float(np.mean(values > float(threshold)))
