#!/usr/bin/env python3
import argparse
from pathlib import Path

from sabotage_simulation import (
    BaselinePolicy,
    CodexMonteCarloAdminPolicy,
    CodexMonteCarloJudgePolicy,
    CodexMonteCarloPolicy,
    CodexSupplyChainPolicy,
    MonteCarloPolicy,
    RuleBasedPolicy,
    SupplyChainConfig,
    run_repeated,
    save_outputs,
    summarize,
)
from qualitative_simulation import (
    CodexQualitativePolicy,
    KeywordMonteCarloPolicy,
    LogisticsMonteCarloPolicy,
    QualitativeBaselinePolicy,
    QualitativeConfig,
    StructuredHumanStateMonteCarloPolicy,
    print_qualitative_summary,
    run_qualitative_repeated,
    save_qualitative_outputs,
    summarize_qualitative,
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
        "--study",
        choices=["numeric", "qualitative", "both"],
        default="both",
        help=(
            "Which supply-chain study to run. 'numeric' preserves the physical-only "
            "benchmark; 'qualitative' adds the human-organizational latent-state "
            "benchmark; 'both' runs both from one command."
        ),
    )
    parser.add_argument(
        "--action-budget",
        type=float,
        default=1.0,
        help="Normalized per-step intervention budget shared by all policies.",
    )
    parser.add_argument(
        "--action-budget-score-weight",
        type=float,
        default=0.35,
        help="Score penalty for using the full normalized action budget on average.",
    )
    parser.add_argument(
        "--observation-update-interval",
        type=int,
        default=3,
        help="Steps between shared telemetry updates visible to controllers.",
    )
    parser.add_argument(
        "--inventory-report-noise",
        type=float,
        default=0.08,
        help="Relative noise applied to reported inventory in controller observations.",
    )
    parser.add_argument(
        "--health-report-noise",
        type=float,
        default=0.035,
        help="Additive noise applied to reported node/edge health in controller observations.",
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
        help="Candidate evaluations for native Monte Carlo decisions.",
    )
    parser.add_argument(
        "--hybrid-codex-candidates",
        type=int,
        default=3,
        help="Maximum Codex-proposed candidate actions for codex_monte_carlo.",
    )
    parser.add_argument(
        "--admin-max-sample-multiplier",
        type=float,
        default=2.0,
        help="Maximum evaluation multiplier Codex can request for codex_monte_carlo_admin.",
    )
    parser.add_argument(
        "--judge-shortlist-size",
        type=int,
        default=8,
        help="Number of evaluated candidates shown to codex_monte_carlo_judge.",
    )
    parser.add_argument(
        "--judge-override-tolerance",
        type=float,
        default=0.12,
        help=(
            "Maximum default-score penalty tolerated when codex_monte_carlo_judge "
            "overrides the default Monte Carlo winner."
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
        "--equal-codex-runs",
        action="store_true",
        help="Use --runs for Codex policies too. Recommended for final evaluation.",
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

    if args.control_update_interval is not None:
        control_update_interval = args.control_update_interval
    elif args.include_codex:
        control_update_interval = args.codex_interval
    else:
        control_update_interval = 1
    config = SupplyChainConfig(
        steps=args.steps,
        action_budget=args.action_budget,
        action_budget_score_weight=args.action_budget_score_weight,
        observation_update_interval=args.observation_update_interval,
        inventory_report_noise=args.inventory_report_noise,
        health_report_noise=args.health_report_noise,
        control_update_interval=control_update_interval,
        monte_carlo_samples=args.monte_carlo_samples,
        hybrid_codex_candidates=args.hybrid_codex_candidates,
        admin_max_sample_multiplier=args.admin_max_sample_multiplier,
        judge_shortlist_size=args.judge_shortlist_size,
        judge_override_tolerance=args.judge_override_tolerance,
    )
    codex_runs = args.runs if args.equal_codex_runs else args.codex_runs
    ran_any = False

    if args.study in {"numeric", "both"}:
        ran_any = True
        numeric_output_dir = output_dir / "numeric_only" if args.study == "both" else output_dir
        policies = _numeric_policies(args)
        runs_by_policy = {policy.name: args.runs for policy in policies}
        if args.include_codex:
            runs_by_policy["codex_steady"] = codex_runs
            runs_by_policy["codex_guardian"] = codex_runs
            runs_by_policy["codex_monte_carlo"] = codex_runs
            runs_by_policy["codex_monte_carlo_admin"] = codex_runs
            runs_by_policy["codex_monte_carlo_judge"] = codex_runs

        results = run_repeated(
            config=config,
            policies=policies,
            runs=runs_by_policy,
            seed=args.seed,
            progress_interval=args.progress_interval,
        )
        print("Numeric-only simulation runs complete. Saving figures...", flush=True)
        written = save_outputs(results, numeric_output_dir, config=config)
        summary = summarize(results, config=config)
        _print_numeric_footer(
            summary=summary,
            written=written,
            args=args,
            codex_runs=codex_runs,
            control_update_interval=control_update_interval,
        )

    if args.study in {"qualitative", "both"}:
        ran_any = True
        qualitative_output_dir = (
            output_dir / "qualitative_resilience" if args.study == "both" else output_dir
        )
        qualitative_config = QualitativeConfig(base=config)
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

    if not ran_any:
        raise SystemExit("No study selected.")


def _numeric_policies(args: argparse.Namespace):
    policies = [
        BaselinePolicy(),
        RuleBasedPolicy(),
        MonteCarloPolicy(samples=args.monte_carlo_samples),
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
                CodexMonteCarloPolicy(
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    samples=args.monte_carlo_samples,
                    max_codex_candidates=args.hybrid_codex_candidates,
                ),
                CodexMonteCarloAdminPolicy(
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    base_samples=args.monte_carlo_samples,
                    max_sample_multiplier=args.admin_max_sample_multiplier,
                ),
                CodexMonteCarloJudgePolicy(
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    base_samples=args.monte_carlo_samples,
                    shortlist_size=args.judge_shortlist_size,
                    override_tolerance=args.judge_override_tolerance,
                ),
            ]
        )
    return policies


def _qualitative_policies(args: argparse.Namespace):
    policies = [
        QualitativeBaselinePolicy(),
        LogisticsMonteCarloPolicy(samples=args.monte_carlo_samples),
        KeywordMonteCarloPolicy(samples=args.monte_carlo_samples),
        StructuredHumanStateMonteCarloPolicy(samples=args.monte_carlo_samples),
    ]
    if args.include_codex:
        policies.extend(
            [
                CodexQualitativePolicy(
                    name="q_codex_qualitative",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    samples=args.monte_carlo_samples,
                    admin_mode=False,
                ),
                CodexQualitativePolicy(
                    name="q_codex_monte_carlo_qualitative_admin",
                    codex_command=args.codex_command,
                    decision_interval=args.codex_interval,
                    timeout_seconds=args.codex_timeout,
                    samples=args.monte_carlo_samples,
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
    print("Completed numeric-only supply-chain sabotage simulation.")
    print(f"Runs per non-Codex policy: {args.runs}")
    if args.include_codex:
        print(f"Runs per Codex policy: {codex_runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Action budget: {args.action_budget}")
    print(f"Action budget score weight: {args.action_budget_score_weight}")
    print(f"Control update interval: {control_update_interval}")
    print(f"Monte Carlo samples: {args.monte_carlo_samples}")
    print(f"Hybrid Codex candidates: {args.hybrid_codex_candidates}")
    print(f"Admin max sample multiplier: {args.admin_max_sample_multiplier}")
    print(f"Judge shortlist size: {args.judge_shortlist_size}")
    print(f"Judge override tolerance: {args.judge_override_tolerance}")
    print(
        "Observation: "
        f"update_interval={args.observation_update_interval}, "
        f"inventory_noise={args.inventory_report_noise}, "
        f"health_noise={args.health_report_noise}"
    )
    print("Ranking lower-is-better:")
    for item in summary["ranking_lower_is_better"]:
        print(
            f"- {item['policy']}: "
            f"{item['sabotage_impact_score_lower_is_better']:.3f}"
        )
    print("Numeric outputs wrote:")
    for path in written:
        print(f"- {_display_path(path, REPO_ROOT)}")


def _print_qualitative_footer(
    summary,
    written,
    args: argparse.Namespace,
    codex_runs: int,
    control_update_interval: int,
) -> None:
    print("Completed qualitative sociotechnical supply-chain simulation.")
    print(f"Runs per non-Codex policy: {args.runs}")
    if args.include_codex:
        print(f"Runs per Codex policy: {codex_runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Control update interval: {control_update_interval}")
    print(f"Monte Carlo samples: {args.monte_carlo_samples}")
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
