#!/usr/bin/env python3
import argparse
from pathlib import Path

from qualitative_simulation import (
    CodexUrbanQualitativePolicy,
    UrbanKeywordMonteCarloPolicy,
    UrbanQualitativeBaselinePolicy,
    UrbanQualitativeConfig,
    UrbanStructuredHumanStateMonteCarloPolicy,
    UrbanTacticalMonteCarloPolicy,
    print_qualitative_summary,
    run_qualitative_repeated,
    save_qualitative_outputs,
    summarize_qualitative,
)
from urban_response import (
    BaselinePatrolPolicy,
    CodexMonteCarloAdminResponsePolicy,
    CodexMonteCarloJudgeResponsePolicy,
    CodexMonteCarloResponsePolicy,
    CodexResponsePolicy,
    DEFAULT_MAP,
    MonteCarloResponsePolicy,
    RuleBasedResponsePolicy,
    UrbanResponseConfig,
    run_repeated,
    save_outputs,
    summarize,
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
        "--scenario",
        choices=["default", "social_complex_large"],
        default="default",
        help=(
            "Urban scenario profile. 'default' preserves the existing road-map "
            "benchmark; 'social_complex_large' uses a generated larger city "
            "network with facility roles, chokepoints, hazardous corridors, "
            "role-biased spawning, and stronger sociotechnical coupling."
        ),
    )
    parser.add_argument(
        "--study",
        choices=["numeric", "qualitative", "both"],
        default="both",
        help=(
            "Which urban study to run. 'numeric' preserves the tactical-only "
            "benchmark; 'qualitative' adds the sociotechnical human-state "
            "benchmark; 'both' runs both from one command."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "urban_response",
        help="Directory for overwritten outputs.",
    )
    parser.add_argument(
        "--control-update-interval",
        type=int,
        default=None,
        help=(
            "Steps between policy replanning decisions. Defaults to --codex-interval "
            "when Codex policies are included, otherwise 1."
        ),
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=24,
        help="Randomized candidate target assignments evaluated by Monte Carlo policies.",
    )
    parser.add_argument(
        "--judge-override-tolerance",
        type=float,
        default=0.10,
        help=(
            "Maximum current-state score penalty tolerated when "
            "codex_monte_carlo_judge overrides the default Monte Carlo winner."
        ),
    )
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
        "--equal-codex-runs",
        action="store_true",
        help="Use --runs for Codex policies too. Recommended for final evaluation.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="Print one simulation progress line every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--qualitative-report-mode",
        choices=["full", "removed", "shuffled", "explicit"],
        default="full",
        help=(
            "Qualitative report ablation mode. 'full' uses natural reports; "
            "'removed' gives no reports; 'shuffled' mismatches report content "
            "to latent district state; 'explicit' gives direct semantic labels."
        ),
    )
    parser.add_argument(
        "--qualitative-report-noise",
        type=float,
        default=0.15,
        help="Probability of adding a noisy qualitative report.",
    )
    parser.set_defaults(include_codex=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if args.control_update_interval is not None:
        control_update_interval = args.control_update_interval
    elif args.include_codex:
        control_update_interval = args.codex_interval
    else:
        control_update_interval = 1
    config = UrbanResponseConfig(
        steps=args.steps,
        scenario=args.scenario,
        control_update_interval=control_update_interval,
        monte_carlo_samples=args.monte_carlo_samples,
        **_scenario_overrides(args.scenario),
    )
    codex_runs = args.runs if args.equal_codex_runs else args.codex_runs

    if args.study in {"numeric", "both"}:
        numeric_output_dir = output_dir / "numeric_only" if args.study == "both" else output_dir
        policies = _policies(args)
        runs_by_policy = {policy.name: args.runs for policy in policies}
        if args.include_codex:
            for policy_name in [
                "codex_steady",
                "codex_guardian",
                "codex_monte_carlo",
                "codex_monte_carlo_admin",
                "codex_monte_carlo_judge",
            ]:
                runs_by_policy[policy_name] = codex_runs

        results = run_repeated(
            config=config,
            policies=policies,
            runs=runs_by_policy,
            seed=args.seed,
            map_path=args.map,
            progress_interval=args.progress_interval,
        )
        print("Numeric simulation runs complete. Saving figures...", flush=True)
        written = save_outputs(results, output_dir=numeric_output_dir, map_path=args.map, config=config)
        summary = summarize(results)
        _print_numeric_footer(
            summary=summary,
            written=written,
            args=args,
            codex_runs=codex_runs,
            control_update_interval=control_update_interval,
        )

    if args.study in {"qualitative", "both"}:
        qualitative_output_dir = (
            output_dir / "qualitative_response" if args.study == "both" else output_dir
        )
        qualitative_config = UrbanQualitativeConfig(
            base=config,
            report_mode=args.qualitative_report_mode,
            report_noise_probability=args.qualitative_report_noise,
        )
        qualitative_policies = _qualitative_policies(args)
        qualitative_runs_by_policy = {
            policy.name: args.runs for policy in qualitative_policies
        }
        if args.include_codex:
            qualitative_runs_by_policy["q_codex_qualitative"] = codex_runs
            qualitative_runs_by_policy["q_codex_monte_carlo_qualitative_admin"] = codex_runs

        qualitative_results = run_qualitative_repeated(
            config=qualitative_config,
            policies=qualitative_policies,
            runs=qualitative_runs_by_policy,
            seed=args.seed + 120_000,
            map_path=args.map,
            progress_interval=args.progress_interval,
        )
        print("Qualitative simulation runs complete. Saving summary...", flush=True)
        qualitative_written = save_qualitative_outputs(
            qualitative_results,
            qualitative_output_dir,
            config=qualitative_config,
        )
        qualitative_summary = summarize_qualitative(qualitative_results, qualitative_config)
        _print_qualitative_footer(
            summary=qualitative_summary,
            written=qualitative_written,
            args=args,
            codex_runs=codex_runs,
            control_update_interval=control_update_interval,
        )


def _policies(args: argparse.Namespace):
    policies = [
        BaselinePatrolPolicy(),
        RuleBasedResponsePolicy(),
        MonteCarloResponsePolicy(samples=args.monte_carlo_samples),
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
                CodexMonteCarloResponsePolicy(
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                ),
                CodexMonteCarloAdminResponsePolicy(
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                ),
                CodexMonteCarloJudgeResponsePolicy(
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    override_tolerance=args.judge_override_tolerance,
                ),
            ]
        )
    return policies


def _scenario_overrides(scenario: str) -> dict:
    if scenario != "social_complex_large":
        return {}
    return {
        "initial_zombies": 220,
        "initial_civilians": 520,
        "initial_defenders": 44,
        "civilian_speed": 16.0,
        "zombie_speed": 32.0,
        "defender_speed": 31.0,
        "defender_visibility_m": 225.0,
        "zombie_detection_m": 260.0,
        "bite_radius_m": 50.0,
        "engagement_radius_m": 29.0,
        "neutralization_probability": 0.44,
        "defender_casualty_probability": 0.68,
        "noise_probability": 0.06,
    }


def _qualitative_policies(args: argparse.Namespace):
    policies = [
        UrbanQualitativeBaselinePolicy(),
        UrbanTacticalMonteCarloPolicy(),
        UrbanKeywordMonteCarloPolicy(),
        UrbanStructuredHumanStateMonteCarloPolicy(),
    ]
    if args.include_codex:
        policies.extend(
            [
                CodexUrbanQualitativePolicy(
                    name="q_codex_qualitative",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    admin_mode=False,
                ),
                CodexUrbanQualitativePolicy(
                    name="q_codex_monte_carlo_qualitative_admin",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    admin_mode=True,
                ),
            ]
        )
    return policies


def _print_numeric_footer(
    summary,
    written,
    args: argparse.Namespace,
    codex_runs: int,
    control_update_interval: int,
) -> None:
    print("Completed urban hostile-contagion response simulation.")
    print(f"Runs per non-Codex policy: {args.runs}")
    if args.include_codex:
        print(f"Runs per Codex policy: {codex_runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Scenario: {args.scenario}")
    print(f"Control update interval: {control_update_interval}")
    print(f"Monte Carlo samples: {args.monte_carlo_samples}")
    print(f"Judge override tolerance: {args.judge_override_tolerance}")
    print("Ranking lower-is-better:")
    for item in summary["ranking_lower_is_better"]:
        print(
            f"- {item['policy']}: "
            f"{item['elimination_score_lower_is_better']:.3f}"
        )
    print("Wrote:")
    for path in written:
        print(f"- {_display_path(path, REPO_ROOT)}")


def _print_qualitative_footer(
    summary,
    written,
    args: argparse.Namespace,
    codex_runs: int,
    control_update_interval: int,
) -> None:
    print("Completed qualitative sociotechnical urban response simulation.")
    print(f"Runs per non-Codex policy: {args.runs}")
    if args.include_codex:
        print(f"Runs per Codex policy: {codex_runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Scenario: {args.scenario}")
    print(f"Control update interval: {control_update_interval}")
    print_qualitative_summary(summary)
    print("Qualitative outputs wrote:")
    for path in written:
        print(f"- {_display_path(path, REPO_ROOT)}")


def _display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
