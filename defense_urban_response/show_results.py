#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a concise urban hostile-contagion response result table."
    )
    parser.add_argument(
        "summary",
        nargs="?",
        type=Path,
        default=Path("results/urban_response/summary.json"),
        help="Path to summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        print(f"Summary file not found: {args.summary}", file=sys.stderr)
        candidates = sorted(Path("results").glob("**/summary.json"))
        if candidates:
            print("Available summaries:", file=sys.stderr)
            for path in candidates:
                print(f"- {path}", file=sys.stderr)
        raise SystemExit(2)

    data = json.loads(args.summary.read_text())
    if data.get("study") == "urban_qualitative_response":
        _print_qualitative_summary(args.summary, data)
        return
    metrics: Dict[str, Dict[str, Any]] = data["metrics"]
    policy_width = max(18, *(len(policy) for policy in metrics))

    print(f"Summary: {args.summary}")
    print()
    print("Ranking lower-is-better")
    print(
        f"{'policy':{policy_width}s} {'score':>8s} {'95% CI':>21s} "
        f"{'victory':>8s} {'civ surv':>8s} {'def surv':>8s} {'zombies':>8s}"
    )
    for item in data["ranking_lower_is_better"]:
        policy = item["policy"]
        row = metrics[policy]
        ci = row.get("elimination_score_ci95", {})
        print(
            f"{policy:{policy_width}s} "
            f"{row['elimination_score_lower_is_better']:8.3f} "
            f"{_ci_text(ci):>21s} "
            f"{row['zombie_victory_rate']:8.3f} "
            f"{row['mean_civilian_survival_fraction']:8.3f} "
            f"{row['mean_defender_survival_fraction']:8.3f} "
            f"{row['mean_final_zombies']:8.1f}"
        )

    comparisons = data.get("comparisons", {})
    best_non_codex = comparisons.get("best_non_codex_policy")
    if best_non_codex:
        print()
        print(f"Best non-Codex policy: {best_non_codex}")

    if comparisons.get("vs_baseline"):
        print()
        print("Paired comparison vs baseline")
        _print_comparison_table(comparisons["vs_baseline"])

    if comparisons.get("vs_best_non_codex"):
        print()
        print("Paired comparison vs best non-Codex")
        _print_comparison_table(comparisons["vs_best_non_codex"])

    codex_rows = {
        policy: row.get("decision_stats", {})
        for policy, row in metrics.items()
        if policy.startswith("codex_")
    }
    if codex_rows:
        print()
        print("Codex decision reliability")
        print(
            f"{'policy':{policy_width}s} {'calls':>7s} {'valid':>7s} "
            f"{'invalid':>8s} {'failures':>8s} {'noop':>7s}"
        )
        for policy, stats in codex_rows.items():
            print(
                f"{policy:{policy_width}s} "
                f"{stats.get('codex_calls', 0.0):7.0f} "
                f"{stats.get('codex_valid_responses', 0.0):7.0f} "
                f"{stats.get('codex_invalid_responses', 0.0):8.0f} "
                f"{stats.get('codex_failures', 0.0):8.0f} "
                f"{stats.get('codex_noop_responses', 0.0):7.0f}"
            )


def _ci_text(interval: Dict[str, Any]) -> str:
    if not interval:
        return "n/a"
    return f"[{interval['ci95_low']:.3f}, {interval['ci95_high']:.3f}]"


def _print_comparison_table(comparisons: Dict[str, Dict[str, Any]]) -> None:
    policy_width = max(18, *(len(policy) for policy in comparisons))
    print(f"{'policy':{policy_width}s} {'n':>4s} {'delta':>9s} {'95% CI':>21s} {'win':>7s}")
    for policy, row in comparisons.items():
        delta = row.get("mean_delta_policy_minus_reference")
        if delta is None:
            continue
        ci = {
            "ci95_low": row.get("delta_ci95_low"),
            "ci95_high": row.get("delta_ci95_high"),
        }
        print(
            f"{policy:{policy_width}s} "
            f"{int(row.get('paired_runs', 0)):4d} "
            f"{delta:9.3f} "
            f"{_ci_text(ci):>21s} "
            f"{row.get('win_rate_lower_score', 0.0):7.2f}"
        )


def _print_qualitative_summary(summary_path: Path, data: Dict[str, Any]) -> None:
    metrics: Dict[str, Dict[str, Any]] = data["metrics"]
    policy_width = max(28, *(len(policy) for policy in metrics))
    print(f"Summary: {summary_path}")
    print()
    print("Qualitative sociotechnical urban ranking lower-is-better")
    print(
        f"{'policy':{policy_width}s} {'score':>9s} {'95% CI':>21s} "
        f"{'physical':>9s} {'human':>9s} {'victory':>8s} "
        f"{'civ':>8s} {'trust':>8s} {'route':>8s} {'panic':>8s} {'budget':>8s}"
    )
    for item in data["ranking_lower_is_better"]:
        policy = item["policy"]
        row = metrics[policy]
        score = row["qualitative_response_score_lower_is_better"]
        ci = row.get("qualitative_response_score_ci95", {})
        print(
            f"{policy:{policy_width}s} {score:9.3f} "
            f"{_ci_text(ci):>21s} "
            f"{_mean(row, 'physical_score_lower_is_better'):9.3f} "
            f"{_mean(row, 'human_score_lower_is_better'):9.3f} "
            f"{_mean(row, 'zombie_victory_rate'):8.3f} "
            f"{_mean(row, 'civilian_survival_fraction'):8.3f} "
            f"{_mean(row, 'mean_trust'):8.3f} "
            f"{_mean(row, 'mean_route_clarity'):8.3f} "
            f"{_mean(row, 'mean_panic'):8.3f} "
            f"{_mean(row, 'mean_action_budget_used'):8.3f}"
        )

    comparisons = data.get("comparisons", {})
    best_deployable = comparisons.get("best_deployable_non_codex_policy")
    structured_state = comparisons.get("structured_human_state_policy")
    if best_deployable:
        print()
        print(f"Best deployable non-Codex qualitative policy: {best_deployable}")
    if structured_state:
        print(f"Structured human-state heuristic baseline: {structured_state}")

    if comparisons.get("vs_best_deployable_non_codex"):
        print()
        print("Paired comparison vs best deployable non-Codex")
        _print_comparison_table(comparisons["vs_best_deployable_non_codex"])

    if comparisons.get("vs_structured_human_state"):
        print()
        print("Paired comparison vs structured human-state baseline")
        _print_comparison_table(comparisons["vs_structured_human_state"])

    codex_rows = {
        policy: row.get("decision_stats", {})
        for policy, row in metrics.items()
        if policy.startswith("q_codex_")
    }
    if codex_rows:
        print()
        print("Codex decision reliability")
        print(
            f"{'policy':{policy_width}s} {'calls':>7s} {'valid':>7s} "
            f"{'invalid':>8s} {'failures':>8s} {'noop':>7s}"
        )
        for policy, stats in codex_rows.items():
            print(
                f"{policy:{policy_width}s} "
                f"{stats.get('codex_calls', 0.0):7.0f} "
                f"{stats.get('codex_valid_responses', 0.0):7.0f} "
                f"{stats.get('codex_invalid_responses', 0.0):8.0f} "
                f"{stats.get('codex_failures', 0.0):8.0f} "
                f"{stats.get('codex_noop_responses', 0.0):7.0f}"
            )


def _mean(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, dict):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


if __name__ == "__main__":
    main()
