# Supply-Chain Sabotage Cyclic-Large Result

Result directory:

```text
results/supply_chain_sabotage_cyclic_large_30/
```

Configuration:

- topology: `cyclic_large`;
- graph size: 37 nodes, 127 directed shipment edges;
- runs: 30 paired runs per policy;
- steps: 180;
- shared control cadence: 25 steps;
- Monte Carlo samples: 24;
- Codex and non-Codex policies run on equal footing.

The cyclic-large topology adds supplier diversity, factory rework loops,
warehouse cycles, cross-cycle links, and retailer mutual-aid routes. It is a
topology-complexity sensitivity study for the supply-chain sabotage scenario.

## Numeric-Only Study

Summary file:

```text
results/supply_chain_sabotage_cyclic_large_30/numeric_only/summary.json
```

Lower score is better:

| Policy | Impact score | 95% CI | Mean service | Mean unmet demand | Mean budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| `monte_carlo` | 15.976 | [15.563, 16.389] | 0.767 | 19,976.0 | 0.448 |
| `codex_monte_carlo_admin` | 16.227 | [15.815, 16.640] | 0.763 | 20,301.6 | 0.494 |
| `codex_monte_carlo` | 16.412 | [16.005, 16.818] | 0.760 | 20,581.2 | 0.437 |
| `codex_monte_carlo_judge` | 16.451 | [16.021, 16.881] | 0.760 | 20,564.0 | 0.533 |
| `codex_steady` | 17.643 | [17.086, 18.200] | 0.744 | 21,958.4 | 0.780 |
| `rule_based` | 18.573 | [18.163, 18.984] | 0.724 | 23,678.0 | 0.241 |
| `baseline` | 21.738 | [21.328, 22.148] | 0.671 | 28,181.8 | 0.000 |
| `codex_guardian` | 21.738 | [21.328, 22.148] | 0.671 | 28,181.8 | 0.000 |

Interpretation:

The numeric-only result reinforces the earlier boundary finding. Plain Monte
Carlo remains the strongest controller when the state, action space, and
objective are fully numeric and simulator-aligned. The closest Codex hybrid is
`codex_monte_carlo_admin`, but it is still statistically worse than Monte
Carlo: paired delta `+0.251`, 95% CI `[0.191, 0.311]`, win rate `1/30`.

The Codex hybrids are nevertheless useful relative to baseline and rule-based
control. `codex_monte_carlo_admin` improves over baseline by `-5.510` and over
rule-based by about `-2.346` score units. The correct conclusion is not that
Codex beats Monte Carlo in numeric control. The conclusion is that Codex-guided
search is competitive and useful, but conventional Monte Carlo remains the
best method when the problem is fully specified numerically.

`codex_guardian` made `0` Codex calls in this run and exactly matches baseline,
so it should not be treated as an active controller result.

## Qualitative Sociotechnical Study

Summary file:

```text
results/supply_chain_sabotage_cyclic_large_30/qualitative_resilience/summary.json
```

Lower score is better:

| Policy | Score | 95% CI | Physical | Human | Service | Trust | Rumor | Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_structured_human_state_monte_carlo` | 78.251 | [77.840, 78.661] | 69.998 | 3.668 | 0.292 | 0.145 | 0.828 | 0.716 |
| `q_codex_monte_carlo_qualitative_admin` | 79.086 | [78.703, 79.469] | 70.619 | 3.763 | 0.304 | 0.151 | 0.905 | 0.736 |
| `q_keyword_monte_carlo` | 83.980 | [83.634, 84.327] | 75.094 | 3.949 | 0.278 | 0.065 | 0.924 | 0.746 |
| `q_monte_carlo_logistics` | 84.684 | [84.356, 85.011] | 75.790 | 3.953 | 0.269 | 0.065 | 0.924 | 0.448 |
| `q_codex_qualitative` | 86.989 | [86.464, 87.513] | 78.227 | 3.894 | 0.237 | 0.085 | 0.925 | 0.321 |
| `q_baseline` | 88.108 | [87.808, 88.408] | 79.277 | 3.925 | 0.232 | 0.057 | 0.933 | 0.000 |

Interpretation:

The qualitative study strongly supports the Codex-guidance claim. The best
deployable non-Codex method is `q_keyword_monte_carlo`; the
Codex-Monte-Carlo qualitative administrator beats it by paired delta `-4.895`,
95% CI `[-5.166, -4.623]`, with win rate `30/30`.

The structured human-state policy remains the best overall reference:
`q_structured_human_state_monte_carlo` beats Codex-admin by `0.835`, 95% CI
`[0.443, 1.227]`. That should be treated as a strong reference point, not a
deployable conventional baseline. The paper-facing claim should be that Codex
guidance improves deployable qualitative interpretation over conventional
non-Codex baselines, not that it beats a structured human-state oracle.

The direct Codex qualitative controller underperforms (`86.989`), so the
winning architecture is not "ask Codex for an action." The winning architecture
is Codex as a semantic administrator over Monte Carlo: Codex interprets noisy
human-organizational reports and converts them into bounded search priorities.

Decision reliability supports this interpretation:

- `q_codex_monte_carlo_qualitative_admin`: `240/240` valid Codex decisions;
- `q_codex_qualitative`: `239/240` valid Codex decisions;
- numeric `codex_monte_carlo_admin`: `239/240` valid Codex decisions and
  selected guided candidates `179` times, but still did not beat plain Monte
  Carlo.

## Manuscript Use

This result should be presented as cross-checking the main supply-chain
conclusion:

1. On fully numeric control, Monte Carlo remains the correct strongest
   conventional method.
2. On qualitative sociotechnical control, Codex-guided Monte Carlo is the best
   deployable method.
3. Codex adds value when semantic interpretation is needed to configure
   structured search.
4. Codex should not be framed as a universal replacement for Monte Carlo.

Recommended manuscript sentence:

```text
In the larger cyclic supply-chain topology, the numeric-only benchmark again
favored plain Monte Carlo, but the sociotechnical benchmark favored Codex as a
high-level administrator over Monte Carlo, with the Codex-guided administrator
beating the best deployable non-Codex qualitative controller in all 30 paired
runs.
```

## Reproduce

Run:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --network-topology cyclic_large --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --hybrid-codex-candidates 3 --admin-max-sample-multiplier 2.0 --judge-shortlist-size 8 --judge-override-tolerance 0.12 --progress-interval 20 --codex-timeout 45 --output-dir results/supply_chain_sabotage_cyclic_large_30
```

View:

```bash
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_cyclic_large_30/numeric_only/summary.json
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_cyclic_large_30/qualitative_resilience/summary.json
```
