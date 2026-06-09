from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from agent_dynamic_system.benchmark import GenericMonteCarloResult


PALETTE = ["#4F5D75", "#2B8C67", "#9C5A2E", "#6C5B7B", "#B23A48", "#3F7CAC"]
VARIABLE_COLORS = ["#3F7CAC", "#B23A48", "#2B8C67", "#9C5A2E", "#6C5B7B"]


def save_generic_scenario_plot(
    result: GenericMonteCarloResult,
    variable_labels: Dict[str, str],
    action_labels: Dict[str, str],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variables = list(variable_labels.keys())
    actions = list(action_labels.keys())
    rows = len(variables) + len(actions)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(10.5, max(8.0, rows * 2.35)),
        sharex=True,
        constrained_layout=True,
    )
    if rows == 1:
        axes = [axes]
    fig.suptitle(
        f"{_title(result.system_name)}: {_title(result.scenario_name)} trajectories",
        fontsize=15,
    )

    for index, name in enumerate(variables):
        axis = axes[index]
        values = result.variables[name]
        mean, sem = _mean_sem(values)
        color = VARIABLE_COLORS[index % len(VARIABLE_COLORS)]
        axis.plot(result.time, mean, color=color, linewidth=2.0, label="Mean")
        axis.fill_between(result.time, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)
        axis.set_ylabel(variable_labels[name])
        axis.set_ylim(bottom=0.0)
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    for offset, name in enumerate(actions):
        axis = axes[len(variables) + offset]
        values = result.actions[name]
        mean, sem = _mean_sem(values)
        color = PALETTE[offset % len(PALETTE)]
        axis.plot(result.time, mean, color=color, linewidth=1.8, label="Mean")
        axis.fill_between(result.time, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
        axis.set_title(action_labels[name], fontsize=10)
        axis.set_ylabel("Amount")
        axis.set_ylim(0.0, max(0.05, float(np.max(mean + sem)) * 1.15))
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    axes[-1].set_xlabel("Simulation step")
    return _save_png_pdf(fig, output_dir / _safe_name(result.scenario_name))


def save_generic_dashboard(
    results: Sequence[GenericMonteCarloResult],
    variable_labels: Dict[str, str],
    action_labels: Dict[str, str],
    summary: Dict[str, object],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variables = list(variable_labels.keys())[:4]
    actions = list(action_labels.keys())
    fig = plt.figure(figsize=(15.0, 12.8))
    grid = fig.add_gridspec(
        4,
        2,
        width_ratios=(2.25, 1.0),
        wspace=0.24,
        hspace=0.42,
        left=0.07,
        right=0.98,
        top=0.91,
        bottom=0.10,
    )
    fig.suptitle(f"{_title(results[0].system_name)} benchmark dashboard", fontsize=16)
    colors = _scenario_colors(results)

    for row, name in enumerate(variables):
        axis = fig.add_subplot(grid[row, 0])
        for result in results:
            mean, sem = _mean_sem(result.variables[name])
            color = colors[result.scenario_name]
            axis.plot(result.time, mean, color=color, linewidth=2.0, label=_title(result.scenario_name))
            axis.fill_between(result.time, mean - sem, mean + sem, color=color, alpha=0.14, linewidth=0)
        axis.set_ylabel(variable_labels[name])
        axis.set_ylim(bottom=0.0)
        if row == len(variables) - 1:
            axis.set_xlabel("Simulation step")
        axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
        axis.legend(loc="upper right", frameon=False, fontsize=8)

    action_grid = grid[:3, 1].subgridspec(len(actions), 1, hspace=0.52)
    for row, action_name in enumerate(actions):
        action_axis = fig.add_subplot(action_grid[row, 0])
        _plot_dashboard_action(
            action_axis,
            results,
            colors,
            action_name,
            action_labels[action_name],
        )

    score_axis = fig.add_subplot(grid[3, 1])
    ranking = summary.get("ranking_lower_is_better", [])
    labels = [_title(item["scenario"]) for item in ranking]
    scores = [float(item["stability_score"]) for item in ranking]
    bar_colors = [colors[item["scenario"]] for item in ranking]
    bars = score_axis.bar(labels, scores, color=bar_colors, alpha=0.86)
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
        "Matched seeds; bands show Monte Carlo SE; files overwrite.",
        fontsize=8,
        color="#4E5964",
    )
    return _save_png_pdf(fig, output_dir / "dashboard")


def save_cross_system_dashboard(
    summaries: Dict[str, Dict[str, object]],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    systems = list(summaries.keys())
    controllers = sorted(
        {
            item["scenario"]
            for summary in summaries.values()
            for item in summary["ranking_lower_is_better"]
        }
    )
    values = np.full((len(systems), len(controllers)), np.nan)
    for row, system_name in enumerate(systems):
        metric_by_scenario = summaries[system_name]["metrics"]
        for col, controller in enumerate(controllers):
            if controller in metric_by_scenario:
                baseline = metric_by_scenario.get("baseline", {}).get("stability_score")
                score = metric_by_scenario[controller]["stability_score"]
                values[row, col] = score / baseline if baseline else score

    fig, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    x = np.arange(len(systems))
    width = 0.78 / max(len(controllers), 1)
    for col, controller in enumerate(controllers):
        offset = (col - (len(controllers) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values[:, col],
            width=width,
            label=_title(controller),
            color=PALETTE[col % len(PALETTE)],
            alpha=0.88,
        )
    axis.axhline(1.0, color="#B23A48", linestyle=":", linewidth=1.1)
    axis.set_title("Cross-system instability relative to baseline\n(lower is better)")
    axis.set_ylabel("Relative instability")
    axis.set_xticks(x)
    axis.set_xticklabels([_title(name) for name in systems])
    axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    return _save_png_pdf(fig, output_dir / "cross_system_dashboard")


def _plot_action_totals(
    axis: plt.Axes,
    results: Sequence[GenericMonteCarloResult],
    colors: Dict[str, str],
) -> None:
    labels = []
    totals = []
    bar_colors = []
    for result in results:
        total = 0.0
        for values in result.actions.values():
            total += float(np.mean(np.abs(values)))
        labels.append(_title(result.scenario_name))
        totals.append(total)
        bar_colors.append(colors[result.scenario_name])
    axis.bar(labels, totals, color=bar_colors, alpha=0.86)
    axis.set_title("Mean intervention amount", fontsize=10)
    axis.set_ylabel("Action amount")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.8)


def _plot_dashboard_action(
    axis: plt.Axes,
    results: Sequence[GenericMonteCarloResult],
    colors: Dict[str, str],
    action_name: str,
    label: str,
) -> None:
    maximum = 0.0
    for result in results:
        if result.scenario_name == "baseline":
            continue
        mean, _ = _mean_sem(result.actions[action_name])
        maximum = max(maximum, float(np.max(mean)))
        axis.plot(
            result.time,
            mean,
            color=colors[result.scenario_name],
            linewidth=1.7,
            label=_title(result.scenario_name),
        )
    axis.set_title(f"Intervention: {label}", fontsize=9)
    axis.set_ylabel("Amount")
    axis.set_ylim(0.0, max(0.05, maximum * 1.15))
    axis.grid(True, color="#D8DEE6", linewidth=0.8, alpha=0.8)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right", frameon=False, fontsize=7)


def _mean_sem(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    return mean, np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _scenario_colors(results: Sequence[GenericMonteCarloResult]) -> Dict[str, str]:
    return {
        result.scenario_name: PALETTE[index % len(PALETTE)]
        for index, result in enumerate(results)
    }


def _save_png_pdf(fig: plt.Figure, path_without_suffix: Path) -> List[Path]:
    png_path = path_without_suffix.with_suffix(".png")
    pdf_path = path_without_suffix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _title(name: str) -> str:
    return name.replace("_", " ").title()
