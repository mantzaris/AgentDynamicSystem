#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_dynamic_system.benchmark import run_generic_scenario, summarize_generic_results
from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.controllers import (
    CodexAgentInLoopController,
    LookAheadMiniSimulationController,
    NoControl,
    PIGrassController,
    RuleBasedStabilityController,
)
from agent_dynamic_system.experiment import Scenario, run_scenario
from agent_dynamic_system.forest_fire import (
    ForestFireConfig,
    build_forest_fire_scenarios,
    forest_fire_labels,
    forest_fire_metric_spec,
)
from agent_dynamic_system.epidemic_city import (
    EpidemicCityConfig,
    build_epidemic_city_scenarios,
    epidemic_city_labels,
    epidemic_city_metric_spec,
)
from agent_dynamic_system.generic_plotting import (
    save_cross_system_dashboard,
    save_generic_dashboard,
    save_generic_scenario_plot,
)
from agent_dynamic_system.metrics import summarize_results
from agent_dynamic_system.plotting import save_dashboard, save_scenario_plot
from agent_dynamic_system.smart_grid import (
    SmartGridConfig,
    build_smart_grid_scenarios,
    smart_grid_labels,
    smart_grid_metric_spec,
)
from agent_dynamic_system.supply_chain import (
    SupplyChainConfig,
    build_supply_chain_scenarios,
    supply_chain_labels,
    supply_chain_metric_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic-system Monte Carlo benchmark comparisons."
    )
    parser.add_argument("--runs", type=int, default=30, help="Monte Carlo repeats.")
    parser.add_argument("--steps", type=int, default=500, help="Steps per run.")
    parser.add_argument("--seed", type=int, default=20260607, help="Base random seed.")
    parser.add_argument(
        "--system",
        choices=(
            "grass_rabbit_fox",
            "forest_fire",
            "supply_chain",
            "epidemic_city",
            "smart_grid",
            "all",
        ),
        default="grass_rabbit_fox",
        help="Dynamic system benchmark to run.",
    )
    parser.add_argument(
        "--include-agent-in-loop",
        action="store_true",
        help="Include the Codex CLI agent-in-the-loop controller for grass/rabbit/fox.",
    )
    parser.add_argument(
        "--agent-decision-interval",
        type=int,
        default=10,
        help="Steps between Codex agent-in-the-loop consultations.",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=120,
        help="Seconds to wait for each Codex agent-in-the-loop decision.",
    )
    parser.add_argument(
        "--agent-codex-command",
        default="codex",
        help="Codex CLI command used by the agent-in-the-loop controller.",
    )
    parser.add_argument(
        "--agent-runs",
        type=int,
        default=1,
        help=(
            "Number of simulation runs for the Codex agent-in-the-loop scenario. "
            "Defaults to 1 so a 500-step run with interval 10 makes 50 Codex calls."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for overwritten aggregate outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {}
    written = []

    if args.system in ("grass_rabbit_fox", "all"):
        summary, paths = run_grass_rabbit_fox(args)
        summaries["grass_rabbit_fox"] = summary
        written.extend(paths)
    if args.system in ("forest_fire", "all"):
        summary, paths = run_forest_fire(args)
        summaries["forest_fire"] = summary
        written.extend(paths)
    if args.system in ("supply_chain", "all"):
        summary, paths = run_supply_chain(args)
        summaries["supply_chain"] = summary
        written.extend(paths)
    if args.system in ("epidemic_city", "all"):
        summary, paths = run_epidemic_city(args)
        summaries["epidemic_city"] = summary
        written.extend(paths)
    if args.system in ("smart_grid", "all"):
        summary, paths = run_smart_grid(args)
        summaries["smart_grid"] = summary
        written.extend(paths)

    if args.system == "all":
        output_dir = _output_dir(args.output_dir)
        written.extend(save_cross_system_dashboard(summaries, output_dir))
        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
        written.append(summary_path)

    print("Completed simulation comparison.")
    print(f"System: {args.system}")
    print(f"Monte Carlo runs: {args.runs}")
    print(f"Steps per run: {args.steps}")
    print("Wrote:")
    for path in written:
        print(f"- {path.relative_to(ROOT)}")


def run_grass_rabbit_fox(args: argparse.Namespace) -> tuple:
    config = SimulationConfig(steps=args.steps)
    scenarios = [
        Scenario("baseline", lambda config: NoControl()),
        Scenario("control_theory", lambda config: PIGrassController()),
        Scenario("rule_based", lambda config: RuleBasedStabilityController()),
        Scenario("look_ahead", lambda config: LookAheadMiniSimulationController()),
    ]
    if args.include_agent_in_loop:
        scenarios.append(
            Scenario(
                "agent_in_loop",
                lambda config: CodexAgentInLoopController(
                    decision_interval=args.agent_decision_interval,
                    timeout_seconds=args.agent_timeout,
                    codex_command=args.agent_codex_command,
                ),
            )
        )

    scenario_runs = {}
    results = []
    for scenario in scenarios:
        runs = args.agent_runs if scenario.name == "agent_in_loop" else args.runs
        runs = max(1, runs)
        scenario_runs[scenario.name] = runs
        results.append(
            run_scenario(
                config=config,
                scenario=scenario,
                runs=runs,
                base_seed=args.seed,
            )
        )
    summary = summarize_results(results, config)

    output_dir = _system_output_dir(args.output_dir, "grass_rabbit_fox", args.system)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for result in results:
        written.extend(save_scenario_plot(result, config, output_dir))
    written.extend(save_dashboard(results, config, summary, output_dir))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "runs": args.runs,
                "scenario_runs": scenario_runs,
                "steps": args.steps,
                "base_seed": args.seed,
                "system": "grass_rabbit_fox",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(summary_path)
    return summary, written


def run_forest_fire(args: argparse.Namespace) -> tuple:
    config = ForestFireConfig(steps=args.steps)
    scenarios = build_forest_fire_scenarios(config)
    results = [
        run_generic_scenario(
            system_name="forest_fire",
            scenario=scenario,
            runs=args.runs,
            base_seed=args.seed,
        )
        for scenario in scenarios
    ]
    summary = summarize_generic_results(results, forest_fire_metric_spec(config))
    output_dir = _system_output_dir(args.output_dir, "forest_fire", args.system)
    variable_labels, action_labels = forest_fire_labels()
    written = []
    for result in results:
        written.extend(
            save_generic_scenario_plot(result, variable_labels, action_labels, output_dir)
        )
    written.extend(
        save_generic_dashboard(results, variable_labels, action_labels, summary, output_dir)
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "runs": args.runs,
                "steps": args.steps,
                "base_seed": args.seed,
                "system": "forest_fire",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(summary_path)
    return summary, written


def run_supply_chain(args: argparse.Namespace) -> tuple:
    config = SupplyChainConfig(steps=args.steps)
    scenarios = build_supply_chain_scenarios(config)
    results = [
        run_generic_scenario(
            system_name="supply_chain",
            scenario=scenario,
            runs=args.runs,
            base_seed=args.seed,
        )
        for scenario in scenarios
    ]
    summary = summarize_generic_results(results, supply_chain_metric_spec(config))
    output_dir = _system_output_dir(args.output_dir, "supply_chain", args.system)
    variable_labels, action_labels = supply_chain_labels()
    written = []
    for result in results:
        written.extend(
            save_generic_scenario_plot(result, variable_labels, action_labels, output_dir)
        )
    written.extend(
        save_generic_dashboard(results, variable_labels, action_labels, summary, output_dir)
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "runs": args.runs,
                "steps": args.steps,
                "base_seed": args.seed,
                "system": "supply_chain",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(summary_path)
    return summary, written


def run_epidemic_city(args: argparse.Namespace) -> tuple:
    config = EpidemicCityConfig(steps=args.steps)
    scenarios = build_epidemic_city_scenarios(config)
    results = [
        run_generic_scenario(
            system_name="epidemic_city",
            scenario=scenario,
            runs=args.runs,
            base_seed=args.seed,
        )
        for scenario in scenarios
    ]
    summary = summarize_generic_results(results, epidemic_city_metric_spec(config))
    output_dir = _system_output_dir(args.output_dir, "epidemic_city", args.system)
    variable_labels, action_labels = epidemic_city_labels()
    written = []
    for result in results:
        written.extend(
            save_generic_scenario_plot(result, variable_labels, action_labels, output_dir)
        )
    written.extend(
        save_generic_dashboard(results, variable_labels, action_labels, summary, output_dir)
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "runs": args.runs,
                "steps": args.steps,
                "base_seed": args.seed,
                "system": "epidemic_city",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(summary_path)
    return summary, written


def run_smart_grid(args: argparse.Namespace) -> tuple:
    config = SmartGridConfig(steps=args.steps)
    scenarios = build_smart_grid_scenarios(config)
    results = [
        run_generic_scenario(
            system_name="smart_grid",
            scenario=scenario,
            runs=args.runs,
            base_seed=args.seed,
        )
        for scenario in scenarios
    ]
    summary = summarize_generic_results(results, smart_grid_metric_spec(config))
    output_dir = _system_output_dir(args.output_dir, "smart_grid", args.system)
    variable_labels, action_labels = smart_grid_labels()
    written = []
    for result in results:
        written.extend(
            save_generic_scenario_plot(result, variable_labels, action_labels, output_dir)
        )
    written.extend(
        save_generic_dashboard(results, variable_labels, action_labels, summary, output_dir)
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "runs": args.runs,
                "steps": args.steps,
                "base_seed": args.seed,
                "system": "smart_grid",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(summary_path)
    return summary, written


def _output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return ROOT / output_dir


def _system_output_dir(output_dir: Path, system_name: str, selected_system: str) -> Path:
    output_root = _output_dir(output_dir)
    if selected_system == "all":
        return output_root / system_name
    return output_root


if __name__ == "__main__":
    main()
