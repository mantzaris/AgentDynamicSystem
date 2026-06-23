#!/usr/bin/env python3
import argparse
from pathlib import Path

from urban_response import (
    BaselinePatrolPolicy,
    CodexResponsePolicy,
    DEFAULT_MAP,
    MonteCarloResponsePolicy,
    RuleBasedResponsePolicy,
    UrbanResponseConfig,
    run_repeated,
    save_outputs,
)


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the urban road-network hostile-contagion response simulation."
    )
    parser.add_argument("--runs", type=int, default=5, help="Repeated runs per policy.")
    parser.add_argument("--steps", type=int, default=180, help="Simulation steps per run.")
    parser.add_argument("--seed", type=int, default=20260609, help="Base random seed.")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="Road-network JSON path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "zombies",
        help="Directory for overwritten outputs.",
    )
    parser.add_argument(
        "--include-codex",
        action="store_true",
        help="Include Codex tactic-selection policies.",
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
        help="Steps between steady Codex tactic consultations.",
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
        help="Print one simulation progress line every N steps. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    config = UrbanResponseConfig(steps=args.steps)
    policies = [
        BaselinePatrolPolicy(),
        RuleBasedResponsePolicy(),
        MonteCarloResponsePolicy(),
    ]
    if args.include_codex:
        policies.extend(
            [
                CodexResponsePolicy(
                    name="codex_steady",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                ),
                CodexResponsePolicy(
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
        map_path=args.map,
        progress_interval=args.progress_interval,
    )
    print("Simulation runs complete. Saving figures...", flush=True)
    written = save_outputs(results, output_dir=output_dir, map_path=args.map)
    print("Completed urban response simulation.")
    print(f"Runs per non-Codex policy: {args.runs}")
    if args.include_codex:
        print(f"Runs per Codex policy: {args.codex_runs}")
    print(f"Steps per run: {args.steps}")
    print("Wrote:")
    for path in written:
        print(f"- {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
