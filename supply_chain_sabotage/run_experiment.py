#!/usr/bin/env python3
import argparse
from pathlib import Path

from sabotage_simulation import (
    BaselinePolicy,
    CodexSupplyChainPolicy,
    MonteCarloPolicy,
    RuleBasedPolicy,
    SupplyChainConfig,
    run_repeated,
    save_outputs,
    summarize,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the directed-graph supply-chain sabotage response simulation."
    )
    parser.add_argument("--runs", type=int, default=20, help="Repeated runs per policy.")
    parser.add_argument("--steps", type=int, default=180, help="Simulation steps per run.")
    parser.add_argument("--seed", type=int, default=20260624, help="Base random seed.")
    parser.add_argument(
        "--include-codex",
        action="store_true",
        dest="include_codex",
        help="Include Codex tactic-selection policies. This is now the default.",
    )
    parser.add_argument(
        "--no-codex",
        action="store_false",
        dest="include_codex",
        help="Exclude Codex policies for a non-Codex-only run.",
    )
    parser.add_argument(
        "--codex-command",
        default="codex",
        help="Codex CLI command used by Codex policies.",
    )
    parser.add_argument(
        "--codex-interval",
        type=int,
        default=25,
        help="Steps between steady Codex intervention consultations.",
    )
    parser.add_argument(
        "--codex-timeout",
        type=int,
        default=45,
        help="Seconds to wait for each Codex decision.",
    )
    parser.add_argument(
        "--codex-runs",
        type=int,
        default=1,
        help=(
            "Repeated runs for each Codex policy. Defaults to 1 so real "
            "Codex calls stay practical during exploratory runs."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="Print progress every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "supply_chain_sabotage",
        help="Directory for overwritten outputs.",
    )
    parser.set_defaults(include_codex=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    config = SupplyChainConfig(steps=args.steps)
    policies = [
        BaselinePolicy(),
        RuleBasedPolicy(),
        MonteCarloPolicy(),
    ]
    if args.include_codex:
        policies.extend(
            [
                CodexSupplyChainPolicy(
                    name="codex_steady",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                ),
                CodexSupplyChainPolicy(
                    name="codex_guardian",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                ),
            ]
        )

    runs_by_policy = {policy.name: args.runs for policy in policies}
    if args.include_codex:
        runs_by_policy["codex_steady"] = args.codex_runs
        runs_by_policy["codex_guardian"] = args.codex_runs

    results = run_repeated(
        config=config,
        policies=policies,
        runs=runs_by_policy,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )
    print("Simulation runs complete. Saving figures...", flush=True)
    written = save_outputs(results, output_dir)
    summary = summarize(results)

    print("Completed supply-chain sabotage simulation.")
    print(f"Runs per non-Codex policy: {args.runs}")
    if args.include_codex:
        print(f"Runs per Codex policy: {args.codex_runs}")
    print(f"Steps per run: {args.steps}")
    print("Ranking lower-is-better:")
    for item in summary["ranking_lower_is_better"]:
        print(
            f"- {item['policy']}: "
            f"{item['sabotage_impact_score_lower_is_better']:.3f}"
        )
    print("Wrote:")
    for path in written:
        print(f"- {_display_path(path, REPO_ROOT)}")


def _display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
