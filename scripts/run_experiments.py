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

from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.controllers import (
    CodexAgentInLoopController,
    LookAheadMiniSimulationController,
    NoControl,
    PIGrassController,
    RuleBasedStabilityController,
)
from agent_dynamic_system.experiment import Scenario, run_scenario
from agent_dynamic_system.metrics import summarize_results
from agent_dynamic_system.plotting import save_dashboard, save_scenario_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grass/rabbit/fox Monte Carlo simulation comparisons."
    )
    parser.add_argument("--runs", type=int, default=30, help="Monte Carlo repeats.")
    parser.add_argument("--steps", type=int, default=500, help="Steps per run.")
    parser.add_argument("--seed", type=int, default=20260607, help="Base random seed.")
    parser.add_argument(
        "--include-agent-in-loop",
        action="store_true",
        help="Include the Codex CLI agent-in-the-loop controller.",
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

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for result in results:
        written.extend(save_scenario_plot(result, config, output_dir))
    written.extend(save_dashboard(results, config, summary, output_dir))

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "config": asdict(config),
        "runs": args.runs,
        "scenario_runs": scenario_runs,
        "steps": args.steps,
        "base_seed": args.seed,
        **summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    written.append(summary_path)

    print("Completed simulation comparison.")
    print(f"Monte Carlo runs: {args.runs}")
    print(f"Steps per run: {args.steps}")
    print("Scenario runs:")
    for scenario_name, runs in scenario_runs.items():
        print(f"- {scenario_name}: {runs}")
    print("Wrote:")
    for path in written:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
