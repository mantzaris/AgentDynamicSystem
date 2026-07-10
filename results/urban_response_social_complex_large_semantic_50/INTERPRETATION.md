# Semantic Urban Response Result

Run directory:

```text
results/urban_response_social_complex_large_semantic_50/
```

This is the current paper-facing expanded urban-defense run. It uses the
`social_complex_large` scenario with 50 paired runs, Codex included for all
Codex-family policies, 180 simulation steps, shared 25-step control cadence,
and 32 Monte Carlo samples.

## Research Interpretation

The result supports the current paper thesis:

- Pure numeric/tactical urban control does not require Codex to dominate.
- Codex is most useful when it acts as a semantic administrator over qualitative
  sociotechnical reports.
- The deployable Codex qualitative-admin controller nearly matches the
  privileged structured-human-state controller, while decisively beating
  deployable non-Codex qualitative baselines.

## Numeric Tactical Branch

Source:

```text
numeric_only/summary.json
```

Lower-is-better ranking:

| Policy | Score | 95% CI | Victory Rate |
|---|---:|---:|---:|
| `codex_guardian` | 2.481 | [2.226, 2.736] | 0.440 |
| `codex_steady` | 2.524 | [2.277, 2.772] | 0.480 |
| `rule_based` | 2.663 | [2.418, 2.907] | 0.560 |
| `codex_monte_carlo_judge` | 2.936 | [2.702, 3.171] | 0.700 |
| `monte_carlo` | 2.969 | [2.739, 3.199] | 0.720 |
| `codex_monte_carlo` | 2.970 | [2.740, 3.200] | 0.720 |
| `codex_monte_carlo_admin` | 2.973 | [2.743, 3.203] | 0.720 |
| `baseline` | 3.984 | [3.978, 3.990] | 1.000 |

All active controllers beat baseline. The best non-Codex policy is
`rule_based`. `codex_guardian` ranks first but does not significantly beat
`rule_based`: mean delta `-0.182`, 95% CI `[-0.532, 0.169]`, win rate `0.50`.
This branch should therefore be interpreted as Codex-competitive, not as a
decisive Codex numeric-control victory.

Codex numeric reliability was high: all numeric Codex-family policies returned
valid decisions for all calls.

## Qualitative Semantic Branch

Source:

```text
qualitative_response/summary.json
```

Lower-is-better ranking:

| Policy | Score | 95% CI | Physical | Human | Victory Rate |
|---|---:|---:|---:|---:|---:|
| `q_codex_monte_carlo_qualitative_admin` | 73.539 | [71.560, 75.518] | 4.084 | 8.160 | 0.940 |
| `q_structured_human_state_monte_carlo` | 73.739 | [72.072, 75.407] | 4.150 | 8.094 | 0.960 |
| `q_codex_qualitative` | 75.642 | [74.095, 77.188] | 4.199 | 8.398 | 0.980 |
| `q_baseline` | 82.153 | [81.468, 82.839] | 4.527 | 9.187 | 1.000 |
| `q_keyword_monte_carlo` | 85.720 | [83.154, 88.287] | 4.031 | 10.693 | 0.920 |
| `q_monte_carlo_tactical` | 86.649 | [84.153, 89.146] | 4.002 | 10.927 | 0.920 |

The key positive result is the Codex qualitative-admin controller:

- versus best deployable non-Codex qualitative policy
  (`q_keyword_monte_carlo`): mean delta `-12.181`, 95% CI
  `[-15.534, -8.829]`, win rate `0.92`;
- versus baseline: mean delta `-8.614`, 95% CI `[-10.594, -6.634]`, win rate
  `0.96`;
- versus privileged structured-human-state reference: mean delta `-0.200`,
  95% CI `[-2.739, 2.338]`, win rate `0.46`.

This means the deployable Codex-admin controller is decisively better than
deployable non-Codex qualitative baselines and statistically comparable to the
privileged structured controller that receives latent human-state information.

The result is not a no-op artifact. `q_codex_monte_carlo_qualitative_admin`
made 180 Codex calls, with 178 valid responses, 2 invalid JSON responses, and
0 failures. It selected nonzero social actions in 178 calls with mean nonzero
social intensity approximately `0.831`. The most common actions were
`community_liaison`, `responder_rotation`, `shelter_opening`, and
`evacuation_guidance`.

## Paper Use

Use this result as the main expanded urban-defense qualitative result. The
older `urban_response_social_complexity_admin_fixed_50` result should be
treated as superseded by this semantic-complex run.

Recommended wording:

> In the expanded urban-defense case, conventional tactical policies remain
> competitive on the numeric branch, but Codex used as a qualitative
> Monte-Carlo administrator decisively outperforms deployable non-Codex
> qualitative baselines and performs comparably to a privileged structured
> human-state reference controller.
