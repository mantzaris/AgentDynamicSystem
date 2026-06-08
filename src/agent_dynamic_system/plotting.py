from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.experiment import MonteCarloResult


SERIES = (
    ("grass", "Grass biomass", "#3C8D3F"),
    ("rabbits", "Rabbits", "#C57A20"),
    ("foxes", "Foxes", "#A84632"),
)


def save_scenario_plot(
    result: MonteCarloResult,
    config: SimulationConfig,
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.5, 13.2),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(f"{_title(result.scenario_name)} aggregate trajectories", fontsize=15)

    for axis, (attribute, label, color) in zip(axes[:3], SERIES):
        values = getattr(result, attribute)
        mean, sem = _mean_sem(values)
        axis.plot(result.time, mean, color=color, linewidth=2.0, label="Mean")
        axis.fill_between(
            result.time,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.20,
            linewidth=0,
            label="Mean +/- SEM",
        )
        if attribute == "rabbits":
            _plot_safety_line(axis, config.rabbit_safety_threshold, "Safety floor")
        if attribute == "foxes":
            _plot_safety_line(axis, config.fox_safety_threshold, "Safety floor")
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    _plot_single_action(
        axes[3],
        result.time,
        result.cut_fraction,
        color="#51606F",
        title="Cut choice: standing grass removed",
        ylabel="Fraction removed",
    )
    _plot_single_action(
        axes[4],
        result.time,
        result.fertilizer_fraction,
        color="#2F7D58",
        title="Fertilizer choice: capacity gap filled",
        ylabel="Fraction filled",
    )
    axes[4].set_xlabel("Simulation step")
    filename = _safe_name(result.scenario_name)
    return _save_png_pdf(fig, output_dir / filename)


def save_dashboard(
    results: Iterable[MonteCarloResult],
    config: SimulationConfig,
    summary: Dict[str, object],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_list = list(results)

    fig = plt.figure(figsize=(14, 9.5))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(2.25, 1.0),
        wspace=0.24,
        hspace=0.42,
        left=0.07,
        right=0.98,
        top=0.91,
        bottom=0.10,
    )
    fig.suptitle("Grass/rabbit/fox simulation dashboard", fontsize=16)

    scenario_colors = _scenario_colors(result_list)
    for row, (attribute, label, _) in enumerate(SERIES):
        axis = fig.add_subplot(grid[row, 0])
        for result in result_list:
            values = getattr(result, attribute)
            mean, sem = _mean_sem(values)
            color = scenario_colors[result.scenario_name]
            axis.plot(result.time, mean, color=color, linewidth=2.0, label=_title(result.scenario_name))
            axis.fill_between(
                result.time,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        if attribute == "rabbits":
            _plot_safety_line(axis, config.rabbit_safety_threshold, "Safety floor")
        if attribute == "foxes":
            _plot_safety_line(axis, config.fox_safety_threshold, "Safety floor")
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0.0)
        if row == len(SERIES) - 1:
            axis.set_xlabel("Simulation step")
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    action_grid = grid[0, 1].subgridspec(2, 1, hspace=0.55)
    cut_axis = fig.add_subplot(action_grid[0, 0])
    fertilizer_axis = fig.add_subplot(action_grid[1, 0], sharex=cut_axis)
    _plot_dashboard_action(
        cut_axis,
        result_list,
        scenario_colors,
        attribute="cut_fraction",
        title="Cut choice: standing grass removed",
        ylabel="Fraction removed",
    )
    _plot_dashboard_action(
        fertilizer_axis,
        result_list,
        scenario_colors,
        attribute="fertilizer_fraction",
        title="Fertilizer choice: capacity gap filled",
        ylabel="Fraction filled",
    )
    fertilizer_axis.set_xlabel("Simulation step")

    score_axis = fig.add_subplot(grid[1:3, 1])
    ranking = summary.get("ranking_lower_is_better", [])
    labels = [_title(item["scenario"]) for item in ranking]
    scores = [float(item["stability_score"]) for item in ranking]
    colors = [scenario_colors[item["scenario"]] for item in ranking]
    bars = score_axis.bar(labels, scores, color=colors, alpha=0.86)
    score_axis.set_title("System instability\n(lower is better)", fontsize=11)
    score_axis.set_ylabel("Variability / instability score")
    score_axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)
    score_axis.tick_params(axis="x", rotation=25)
    for bar, score in zip(bars, scores):
        score_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.text(
        0.07,
        0.035,
        (
            "Matched seeds; bands show Monte Carlo SE where repeated; one "
            "controller choice per step; files overwrite."
        ),
        fontsize=8,
        color="#4E5964",
    )
    return _save_png_pdf(fig, output_dir / "dashboard")


def _mean_sem(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, sem


def _plot_single_action(
    axis: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    color: str,
    title: str,
    ylabel: str,
) -> None:
    mean, sem = _mean_sem(values)
    axis.plot(time, mean, color=color, linewidth=1.8, label="Mean")
    axis.fill_between(
        time,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.16,
        linewidth=0,
        label="Mean +/- SEM",
    )
    axis.set_title(title, fontsize=10)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, _action_ylim(mean + sem))
    axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
    axis.legend(loc="upper right", frameon=False, fontsize=8)


def _plot_safety_line(axis: plt.Axes, threshold: float, label: str) -> None:
    axis.axhline(
        threshold,
        color="#B23A48",
        linewidth=1.1,
        linestyle=":",
        alpha=0.95,
        label=label,
    )


def _plot_dashboard_action(
    axis: plt.Axes,
    results: List[MonteCarloResult],
    scenario_colors: Dict[str, str],
    attribute: str,
    title: str,
    ylabel: str,
) -> None:
    for result in results:
        values = getattr(result, attribute)
        if result.scenario_name == "baseline" and not np.any(values):
            continue
        mean, _ = _mean_sem(values)
        axis.plot(
            result.time,
            mean,
            color=scenario_colors[result.scenario_name],
            linewidth=1.8,
            label=_title(result.scenario_name),
        )
    axis.set_title(title, fontsize=10)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, _action_ylim_from_results(results, attribute))
    axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)


def _save_png_pdf(fig: plt.Figure, path_without_suffix: Path) -> List[Path]:
    png_path = path_without_suffix.with_suffix(".png")
    pdf_path = path_without_suffix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _action_ylim(values: np.ndarray) -> float:
    return max(0.05, float(np.max(values)) * 1.15)


def _action_ylim_from_results(
    results: List[MonteCarloResult],
    attribute: str,
) -> float:
    maxima = []
    for result in results:
        values = getattr(result, attribute)
        mean, _ = _mean_sem(values)
        maxima.append(float(np.max(mean)))
    return max(0.05, max(maxima, default=0.0) * 1.15)


def _scenario_colors(results: List[MonteCarloResult]) -> Dict[str, str]:
    palette = ["#4F5D75", "#2B8C67", "#9C5A2E", "#6C5B7B", "#B23A48", "#3F7CAC"]
    return {
        result.scenario_name: palette[index % len(palette)]
        for index, result in enumerate(results)
    }


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _title(name: str) -> str:
    return name.replace("_", " ").title()
