# Supply-Chain Sabotage Paper Results

This directory contains the paper-ready supply-chain sabotage results.

Source run:

- copied from `results/supply_chain_sabotage_combined_final`;
- `30` paired runs per policy;
- `180` simulation steps per run;
- shared `25`-step control cadence;
- shared noisy telemetry and normalized action budget.

Naming correction:

- The completed qualitative run originally used
  `q_oracle_human_state_monte_carlo`.
- The paper-ready copy renames that policy to
  `q_structured_human_state_monte_carlo`.
- No numeric result values were changed.
- The rename is methodological: this controller has structured hidden
  human-state access plus a hand-coded heuristic, but it is not a strict
  oracle or mathematical upper bound.

## Files

- `numeric_only/summary.json`: physical/logistics-only supply-chain sabotage
  study.
- `numeric_only/*.png` and `numeric_only/*.pdf`: numeric study plots.
- `qualitative_resilience/summary.json`: sociotechnical qualitative resilience
  study with corrected policy naming.

Print tables:

```bash
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_paper/numeric_only/summary.json
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_paper/qualitative_resilience/summary.json
```

## Numeric-Only Study

Lower score is better.

| Policy | Score | 95% CI | Service | Unmet demand | Budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| `monte_carlo` | 13.275 | [12.954, 13.597] | 0.714 | 15,949.4 | 0.448 |
| `codex_monte_carlo` | 13.564 | [13.194, 13.934] | 0.707 | 16,332.1 | 0.448 |
| `codex_monte_carlo_admin` | 13.745 | [13.413, 14.078] | 0.703 | 16,506.2 | 0.565 |
| `codex_monte_carlo_judge` | 14.044 | [13.694, 14.394] | 0.693 | 17,007.0 | 0.458 |
| `codex_steady` | 15.400 | [15.047, 15.754] | 0.665 | 18,570.8 | 0.710 |
| `rule_based` | 15.734 | [15.429, 16.038] | 0.651 | 19,406.5 | 0.237 |
| `baseline` | 18.867 | [18.549, 19.184] | 0.571 | 23,790.4 | 0.000 |
| `codex_guardian` | 18.867 | [18.549, 19.184] | 0.571 | 23,790.4 | 0.000 |

Key paired comparison:

- `codex_monte_carlo` vs `monte_carlo`: mean score delta `+0.289`, 95% CI
  `[0.164, 0.414]`, win rate `0.03`.

Interpretation:

The physical/logistics-only experiment supports a boundary result. When the
state, action space, and objective are compact, numeric, and simulator-aligned,
Monte Carlo model-based search is the strongest controller. The Codex variants
improve substantially over baseline, but they do not replace Monte Carlo in the
pure numerical setting.

Suggested paper claim:

> In the fully specified numerical supply-chain sabotage benchmark, the
> conventional Monte Carlo controller achieved the best resilience score. This
> provides a negative boundary condition: agentic language-model control should
> not be expected to dominate model-based search when the relevant state and
> objective are already numerically specified.

## Qualitative Sociotechnical Study

Lower score is better.

| Policy | Score | 95% CI | Physical | Human | Service | Trust | Rumor | Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_codex_monte_carlo_qualitative_admin` | 54.703 | [54.385, 55.021] | 46.261 | 3.752 | 0.307 | 0.173 | 0.895 | 0.722 |
| `q_structured_human_state_monte_carlo` | 55.889 | [55.610, 56.167] | 47.792 | 3.599 | 0.260 | 0.153 | 0.802 | 0.679 |
| `q_keyword_monte_carlo` | 59.840 | [59.604, 60.076] | 50.922 | 3.963 | 0.259 | 0.061 | 0.929 | 0.746 |
| `q_monte_carlo_logistics` | 60.506 | [60.283, 60.728] | 51.564 | 3.974 | 0.246 | 0.061 | 0.929 | 0.448 |
| `q_codex_qualitative` | 60.762 | [60.474, 61.050] | 51.983 | 3.902 | 0.230 | 0.086 | 0.912 | 0.419 |
| `q_baseline` | 62.885 | [62.681, 63.089] | 53.998 | 3.950 | 0.206 | 0.051 | 0.941 | 0.000 |

Key paired comparisons:

- `q_codex_monte_carlo_qualitative_admin` vs `q_keyword_monte_carlo`: mean
  score delta `-5.137`, 95% CI `[-5.389, -4.885]`, win rate `1.00`.
- `q_codex_monte_carlo_qualitative_admin` vs
  `q_structured_human_state_monte_carlo`: mean score delta `-1.186`, 95% CI
  `[-1.536, -0.835]`, win rate `0.90`.

Interpretation:

The sociotechnical experiment changes the conclusion. The strongest policy is
not direct Codex control. It is a hybrid controller where Codex interprets the
qualitative human-organizational context and sets priorities for Monte Carlo
logistics search. This supports the hypothesis that Codex is useful as a
semantic supervisory layer, not as a replacement for conventional numerical
search.

The structured human-state baseline is useful but should be described
carefully. It receives true latent human-state variables and uses a hand-coded
heuristic, so it is non-deployable and diagnostic. It is not a strict oracle.
Codex beating it is plausible because the Codex-admin policy can combine
qualitative interpretation with richer Monte Carlo candidate administration.

Suggested paper claim:

> In the sociotechnical supply-chain benchmark, where latent quantitative
> human-organizational variables affect demand, compliance, route cooperation,
> and intervention effectiveness, the Codex-Monte-Carlo administrative hybrid
> outperformed both text-blind logistics Monte Carlo and a keyword-based
> qualitative baseline. This suggests that the agent's value lies in semantic
> state interpretation and supervisory framing of conventional search, rather
> than in replacing the search controller itself.

## Combined Interpretation

The two studies should be reported together:

1. Pure numerical supply-chain risk: Monte Carlo wins.
2. Sociotechnical supply-chain risk: Codex-guided Monte Carlo administration
   wins among deployable controllers.

This supports a bounded claim:

> Agent-in-the-loop control is not uniformly superior to conventional methods.
> Its advantage appears when the control problem includes partially observed
> human, institutional, or qualitative state that must be interpreted before
> numerical control can be applied effectively.

## Caveats For The Paper

- The qualitative scenario is intentionally severe; trust is low and rumor
  pressure is high across all policies.
- `q_codex_qualitative` underperforms the hybrid, showing that direct agent
  control is not the right architecture here.
- The structured human-state baseline is a diagnostic heuristic, not an oracle.
- A future ablation should remove reports, shuffle reports, or disable the
  human layer to confirm that the qualitative advantage depends on semantic
  evidence rather than incidental tuning.
