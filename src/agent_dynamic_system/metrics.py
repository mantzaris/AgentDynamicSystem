from typing import Dict, Iterable, List

import numpy as np

from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.experiment import MonteCarloResult


def stability_metrics(
    result: MonteCarloResult,
    config: SimulationConfig,
) -> Dict[str, float]:
    """Return lower-is-better system instability metrics for ranking later."""

    rabbit_cv = _mean_temporal_cv(result.rabbits)
    fox_cv = _mean_temporal_cv(result.foxes)
    rabbit_deviation_rmse = _normalized_rmse(result.rabbits, config.initial_rabbits)
    fox_deviation_rmse = _normalized_rmse(result.foxes, config.initial_foxes)
    rabbit_percent_change = _mean_abs_percent_change(
        result.rabbits,
        config.initial_rabbits,
    )
    fox_percent_change = _mean_abs_percent_change(
        result.foxes,
        config.initial_foxes,
    )
    rabbit_safety_breach = _safety_breach_fraction(
        result.rabbits,
        config.rabbit_safety_threshold,
    )
    fox_safety_breach = _safety_breach_fraction(
        result.foxes,
        config.fox_safety_threshold,
    )
    extinction_rate = float(
        np.mean((result.rabbits[:, -1] <= 0.0) | (result.foxes[:, -1] <= 0.0))
    )
    mean_cut = float(np.mean(result.cut_fraction))
    mean_fertilizer = float(np.mean(result.fertilizer_fraction))

    stability_score = (
        0.15 * rabbit_cv
        + 0.15 * fox_cv
        + 0.18 * rabbit_percent_change
        + 0.18 * fox_percent_change
        + 0.13 * rabbit_deviation_rmse
        + 0.13 * fox_deviation_rmse
        + 0.03 * rabbit_safety_breach
        + 0.03 * fox_safety_breach
        + 0.02 * extinction_rate
    )

    return {
        "rabbit_temporal_cv": float(rabbit_cv),
        "fox_temporal_cv": float(fox_cv),
        "rabbit_mean_abs_percent_change_from_initial": float(rabbit_percent_change),
        "fox_mean_abs_percent_change_from_initial": float(fox_percent_change),
        "rabbit_initial_deviation_rmse": float(rabbit_deviation_rmse),
        "fox_initial_deviation_rmse": float(fox_deviation_rmse),
        "rabbit_safety_breach_fraction": float(rabbit_safety_breach),
        "fox_safety_breach_fraction": float(fox_safety_breach),
        "terminal_extinction_rate": extinction_rate,
        "mean_cut_fraction": mean_cut,
        "mean_fertilizer_fraction": mean_fertilizer,
        "stability_score": float(stability_score),
    }


def summarize_results(
    results: Iterable[MonteCarloResult],
    config: SimulationConfig,
) -> Dict[str, object]:
    metrics = {
        result.scenario_name: stability_metrics(result, config)
        for result in results
    }
    baseline_score = metrics.get("baseline", {}).get("stability_score")
    ranked = []

    for scenario_name, scenario_metrics in metrics.items():
        score = scenario_metrics["stability_score"]
        relative_to_baseline = None
        if baseline_score and baseline_score > 0.0:
            relative_to_baseline = score / baseline_score
        ranked.append(
            {
                "scenario": scenario_name,
                "stability_score": score,
                "relative_to_baseline": relative_to_baseline,
            }
        )

    ranked.sort(key=lambda item: item["stability_score"])
    return {
        "metrics": metrics,
        "ranking_lower_is_better": ranked,
    }


def _mean_temporal_cv(values: np.ndarray) -> float:
    temporal_mean = np.mean(values, axis=1)
    temporal_std = np.std(values, axis=1, ddof=0)
    return float(np.mean(temporal_std / np.maximum(temporal_mean, 1.0)))


def _normalized_rmse(values: np.ndarray, target: float) -> float:
    error = values - float(target)
    return float(np.sqrt(np.mean(error * error)) / max(float(target), 1.0))


def _mean_abs_percent_change(values: np.ndarray, initial: float) -> float:
    return float(np.mean(np.abs(values - float(initial))) / max(float(initial), 1.0))


def _safety_breach_fraction(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values < float(threshold)))
