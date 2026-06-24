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

from agent_dynamic_system.benchmark import GenericScenario, run_generic_scenario, summarize_generic_results
from agent_dynamic_system.codex_guidance import CodexGenericPolicy
from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.controllers import (
    CodexControlAdvisedGrassController,
    CodexGuardianGrassController,
    CodexSteadyGrassController,
    NoControl,
)
from agent_dynamic_system.epidemic_city import (
    EpidemicAction,
    EpidemicCityConfig,
    epidemic_city_labels,
    epidemic_city_metric_spec,
    epidemic_rule_based,
    no_epidemic_control,
    simulate_epidemic_city,
)
from agent_dynamic_system.experiment import Scenario, run_scenario
from agent_dynamic_system.forest_fire import (
    FireAction,
    ForestFireConfig,
    fire_rule_based,
    forest_fire_labels,
    forest_fire_metric_spec,
    no_fire_control,
    simulate_forest_fire,
)
from agent_dynamic_system.generic_plotting import (
    save_cross_system_dashboard,
    save_generic_dashboard,
    save_generic_scenario_plot,
)
from agent_dynamic_system.metrics import summarize_results
from agent_dynamic_system.plotting import save_dashboard, save_scenario_plot
from agent_dynamic_system.smart_grid import (
    GridAction,
    SmartGridConfig,
    grid_rule_based,
    no_grid_control,
    simulate_smart_grid,
    smart_grid_labels,
    smart_grid_metric_spec,
)
from agent_dynamic_system.supply_chain import (
    SupplyAction,
    SupplyChainConfig,
    no_supply_control,
    simulate_supply_chain,
    supply_chain_labels,
    supply_chain_metric_spec,
    supply_rule_based,
)


CODEX_STRATEGIES = ("codex_steady", "codex_guardian", "codex_control_advised")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex-in-the-loop benchmark comparisons."
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per Codex scenario.")
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
        default="all",
        help="Dynamic system Codex benchmark to run.",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=20,
        help="Steps between steady/control-advised Codex consultations.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for each Codex decision.",
    )
    parser.add_argument(
        "--codex-command",
        default="codex",
        help="Codex CLI command used by Codex-in-the-loop strategies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results_codex",
        help="Directory for overwritten Codex benchmark outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {}
    written = []

    if args.system in ("grass_rabbit_fox", "all"):
        summary, paths = run_grass_codex(args)
        summaries["grass_rabbit_fox"] = summary
        written.extend(paths)
    if args.system in ("forest_fire", "all"):
        summary, paths = run_forest_fire_codex(args)
        summaries["forest_fire"] = summary
        written.extend(paths)
    if args.system in ("supply_chain", "all"):
        summary, paths = run_supply_chain_codex(args)
        summaries["supply_chain"] = summary
        written.extend(paths)
    if args.system in ("epidemic_city", "all"):
        summary, paths = run_epidemic_city_codex(args)
        summaries["epidemic_city"] = summary
        written.extend(paths)
    if args.system in ("smart_grid", "all"):
        summary, paths = run_smart_grid_codex(args)
        summaries["smart_grid"] = summary
        written.extend(paths)

    if args.system == "all":
        output_dir = _output_dir(args.output_dir)
        written.extend(save_cross_system_dashboard(summaries, output_dir))
        summary_path = output_dir / "codex_benchmark_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
        written.append(summary_path)

    print("Completed Codex-in-the-loop benchmark.")
    print(f"System: {args.system}")
    print(f"Runs per scenario: {args.runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Decision interval: {args.decision_interval}")
    print("Wrote:")
    for path in written:
        print(f"- {_display_path(path, ROOT)}")


def run_grass_codex(args: argparse.Namespace) -> tuple:
    config = SimulationConfig(steps=args.steps)
    scenarios = [
        Scenario("baseline", lambda config: NoControl()),
        Scenario(
            "codex_steady",
            lambda config: CodexSteadyGrassController(
                decision_interval=args.decision_interval,
                timeout_seconds=args.timeout,
                codex_command=args.codex_command,
            ),
        ),
        Scenario(
            "codex_guardian",
            lambda config: CodexGuardianGrassController(
                decision_interval=args.decision_interval,
                timeout_seconds=args.timeout,
                codex_command=args.codex_command,
            ),
        ),
        Scenario(
            "codex_control_advised",
            lambda config: CodexControlAdvisedGrassController(
                decision_interval=args.decision_interval,
                timeout_seconds=args.timeout,
                codex_command=args.codex_command,
            ),
        ),
    ]
    results = [
        run_scenario(config=config, scenario=scenario, runs=args.runs, base_seed=args.seed)
        for scenario in scenarios
    ]
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
                "steps": args.steps,
                "base_seed": args.seed,
                "decision_interval": args.decision_interval,
                "system": "grass_rabbit_fox",
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(summary_path)
    return summary, written


def run_forest_fire_codex(args: argparse.Namespace) -> tuple:
    config = ForestFireConfig(steps=args.steps)
    return _run_generic_codex(
        args=args,
        system_name="forest_fire",
        config=config,
        labels=forest_fire_labels(),
        metric_spec=forest_fire_metric_spec(config),
        action_type=FireAction,
        action_limits={
            "water_fraction": config.max_water_fraction,
            "firebreak_fraction": config.max_firebreak_fraction,
            "controlled_burn_fraction": config.max_controlled_burn_fraction,
        },
        simulator=simulate_forest_fire,
        baseline_policy=no_fire_control,
        fallback_policy=fire_rule_based,
    )


def run_supply_chain_codex(args: argparse.Namespace) -> tuple:
    config = SupplyChainConfig(steps=args.steps)
    return _run_generic_codex(
        args=args,
        system_name="supply_chain",
        config=config,
        labels=supply_chain_labels(),
        metric_spec=supply_chain_metric_spec(config),
        action_type=SupplyAction,
        action_limits={
            "inventory_release": config.max_inventory_release,
            "production_boost": config.max_production_boost,
            "rationing": config.max_rationing,
            "supplier_subsidy": config.max_supplier_subsidy,
        },
        simulator=simulate_supply_chain,
        baseline_policy=no_supply_control,
        fallback_policy=supply_rule_based,
    )


def run_epidemic_city_codex(args: argparse.Namespace) -> tuple:
    config = EpidemicCityConfig(steps=args.steps)
    return _run_generic_codex(
        args=args,
        system_name="epidemic_city",
        config=config,
        labels=epidemic_city_labels(),
        metric_spec=epidemic_city_metric_spec(config),
        action_type=EpidemicAction,
        action_limits={
            "vaccination": config.max_vaccination,
            "testing_isolation": config.max_testing_isolation,
            "mobility_reduction": config.max_mobility_reduction,
            "hospital_surge": config.max_hospital_surge,
        },
        simulator=simulate_epidemic_city,
        baseline_policy=no_epidemic_control,
        fallback_policy=epidemic_rule_based,
    )


def run_smart_grid_codex(args: argparse.Namespace) -> tuple:
    config = SmartGridConfig(steps=args.steps)
    return _run_generic_codex(
        args=args,
        system_name="smart_grid",
        config=config,
        labels=smart_grid_labels(),
        metric_spec=smart_grid_metric_spec(config),
        action_type=GridAction,
        action_limits={
            "demand_response": config.max_demand_response,
            "battery_dispatch": config.max_battery_dispatch,
            "backup_generation": config.max_backup_generation,
            "renewable_curtailment": config.max_renewable_curtailment,
        },
        simulator=simulate_smart_grid,
        baseline_policy=no_grid_control,
        fallback_policy=grid_rule_based,
    )


def _run_generic_codex(
    args: argparse.Namespace,
    system_name: str,
    config: object,
    labels: tuple,
    metric_spec: object,
    action_type: object,
    action_limits: dict,
    simulator: object,
    baseline_policy: object,
    fallback_policy: object,
) -> tuple:
    variable_labels, action_labels = labels
    scenarios = [
        GenericScenario("baseline", lambda seed: simulator(config, seed, baseline_policy)),
    ]
    for strategy in CODEX_STRATEGIES:
        scenarios.append(
            GenericScenario(
                strategy,
                lambda seed, strategy=strategy: simulator(
                    config,
                    seed,
                    CodexGenericPolicy(
                        system_name=system_name,
                        strategy_name=strategy,
                        action_type=action_type,
                        action_limits=action_limits,
                        metric_spec=metric_spec,
                        fallback_policy=fallback_policy,
                        decision_interval=args.decision_interval,
                        timeout_seconds=args.timeout,
                        codex_command=args.codex_command,
                    ),
                ),
            )
        )

    results = [
        run_generic_scenario(
            system_name=system_name,
            scenario=scenario,
            runs=args.runs,
            base_seed=args.seed,
        )
        for scenario in scenarios
    ]
    summary = summarize_generic_results(results, metric_spec)
    output_dir = _system_output_dir(args.output_dir, system_name, args.system)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "decision_interval": args.decision_interval,
                "system": system_name,
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


def _display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
