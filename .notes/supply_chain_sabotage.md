# Supply-Chain Sabotage Notes

## Purpose

`supply_chain_sabotage/` models a directed supply-chain network under adversary
attacks and random failures. The goal is to compare intervention controllers by
how well they reduce unmet demand and economic loss under disruption.

This is separate from the generic aggregate supply-chain system in
`src/agent_dynamic_system/supply_chain.py`.

## Network

The default graph contains:

- 6 suppliers;
- 4 factories;
- 6 warehouses, including a reserve depot;
- 8 retailer and mission demand points;
- 52 directed shipment edges, including cross-warehouse transfer routes.

Nodes have inventory, capacity, health, storage limits, and demand for
retailers. Edges have shipment capacity, health, and reinforcement state.

## Larger Cyclic Topology

A larger network-control scenario is implemented behind
`--network-topology cyclic_large`. It stays in `supply_chain_sabotage/` rather
than a separate directory so it can reuse the same controllers, qualitative
extension, telemetry assumptions, metrics, and plotting code.

The cyclic-large graph contains:

- 9 suppliers;
- 6 factories;
- 10 warehouses;
- 12 retailer and mission demand points;
- 127 directed shipment edges.

The extension adds:

- factory transfer and rework loops;
- a bidirectional warehouse cycle;
- cross-cycle warehouse chords;
- multiple downstream retailer options;
- retailer mutual-aid links.

Purpose: this is the next escalation after the compact supply-chain result.
The original numeric-only result shows that Monte Carlo wins when the problem
is small, fully numeric, and simulator-aligned. The cyclic-large topology tests
whether that conclusion holds when network structure has more cycles,
redundant paths, and routing tradeoffs. It should be interpreted as a topology
complexity sensitivity study, not as a replacement for the completed paper
result.

## Cyclic-Large Result

The completed cyclic-large run is stored in:

```text
results/supply_chain_sabotage_cyclic_large_30/
```

It uses `30` paired runs, `180` steps, `25`-step shared control cadence,
`24` Monte Carlo samples, equal Codex/non-Codex run counts, and
`--network-topology cyclic_large`.

Numeric-only ranking, lower is better:

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

Numeric interpretation: the larger cyclic graph reinforces the boundary
result. Plain Monte Carlo remains the strongest fully numeric controller.
`codex_monte_carlo_admin` is close but statistically worse than Monte Carlo:
paired delta `+0.251`, 95% CI `[0.191, 0.311]`, win rate `1/30`.
The Codex-Monte-Carlo variants are still useful relative to baseline and
rule-based control, but they do not justify claiming that agent guidance beats
model-based search on a simulator-aligned numeric objective. `codex_guardian`
made `0` calls in this run and should not be interpreted as an active
controller result.

Qualitative sociotechnical ranking, lower is better:

| Policy | Score | 95% CI | Physical | Human | Service | Trust | Rumor | Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_structured_human_state_monte_carlo` | 78.251 | [77.840, 78.661] | 69.998 | 3.668 | 0.292 | 0.145 | 0.828 | 0.716 |
| `q_codex_monte_carlo_qualitative_admin` | 79.086 | [78.703, 79.469] | 70.619 | 3.763 | 0.304 | 0.151 | 0.905 | 0.736 |
| `q_keyword_monte_carlo` | 83.980 | [83.634, 84.327] | 75.094 | 3.949 | 0.278 | 0.065 | 0.924 | 0.746 |
| `q_monte_carlo_logistics` | 84.684 | [84.356, 85.011] | 75.790 | 3.953 | 0.269 | 0.065 | 0.924 | 0.448 |
| `q_codex_qualitative` | 86.989 | [86.464, 87.513] | 78.227 | 3.894 | 0.237 | 0.085 | 0.925 | 0.321 |
| `q_baseline` | 88.108 | [87.808, 88.408] | 79.277 | 3.925 | 0.232 | 0.057 | 0.933 | 0.000 |

Qualitative interpretation: the cyclic-large result strongly reinforces the
positive Codex-guidance claim for sociotechnical control. The best deployable
non-Codex policy is `q_keyword_monte_carlo`; the Codex-Monte-Carlo qualitative
administrator beats it by paired delta `-4.895`, 95% CI
`[-5.166, -4.623]`, win rate `30/30`. The stronger
`q_structured_human_state_monte_carlo` reference still ranks first, beating
Codex-admin by `0.835`, 95% CI `[0.443, 1.227]`, so the honest claim is not
that Codex beats a structured human-state oracle. The claim is that Codex
guidance adds clear value over deployable non-Codex methods when qualitative
reports must be interpreted into bounded Monte Carlo search priorities.

Decision reliability supports the qualitative interpretation:

- `q_codex_monte_carlo_qualitative_admin`: `240/240` valid Codex decisions.
- `q_codex_qualitative`: `239/240` valid Codex decisions, but worse score.
- Numeric `codex_monte_carlo_admin`: `239/240` valid Codex decisions and
  selected guided candidates `179` times, but still did not beat plain Monte
  Carlo.

Manuscript conclusion from this result: Codex should be framed as a semantic
administrator over structured search, not as a universal replacement for
Monte Carlo. When the objective is fully numeric and simulator-aligned,
Monte Carlo wins. When the controller must convert qualitative, noisy,
human-organizational reports into search priorities, Codex-guided Monte Carlo
is the strongest deployable controller in this supply-chain scenario.

## Disruption Model

At each step:

- adversarial attacks can hit nodes or edges;
- attack bursts can hit multiple assets in a single step;
- node attacks destroy inventory and reduce capacity health;
- edge attacks reduce shipment route health;
- random failures can degrade node health;
- downstream demand surges can temporarily increase mission demand;
- damaged nodes and edges recover slowly.

Attack targeting is weighted toward high-value inventory, factories,
warehouses, and high-capacity routes.

## High-Pressure Stress-Test Tuning

The supply-chain sabotage case was made deliberately harder after the first
dashboard showed only a modest gap between baseline and controlled policies.
The current configuration is intended to make baseline degradation clear while
preserving enough intervention leverage for controllers to matter.

Current stress settings:

- attack probability: `0.78`;
- burst-attack probability: `0.28`;
- random failure probability: `0.34`;
- node recovery rate: `0.024`;
- edge recovery rate: `0.030`;
- normal shipment fraction: `0.56`;
- demand noise: `0.28`;
- demand surge probability: `0.16`;
- demand surge multiplier range: `1.35` to `2.15`.

Attack damage was increased:

- node attacks destroy more inventory;
- node health can fall as low as `0.10`;
- edge health can fall as low as `0.08`;
- random failures can push node health down to `0.22`;
- recovery is slower, so damage persists.

Intervention leverage was also increased:

- buffer actions add up to `46` inventory units;
- protected stock can reach `55%` of node storage;
- route reinforcement adds up to `0.60` reinforcement and `0.45` route health;
- expedited routes carry `2.0x` effective route capacity.

## Controllers

- `baseline`: no defensive action.
- `rule_based`: generic SOP-style controller using reported fill ratios, route
  health, and recent service drops. It does not use scenario-specific node
  names or hand-picked bottlenecks.
- `monte_carlo`: sample candidate buffer/reinforce/expedite actions and choose
  the lowest short-horizon risk option.
- `codex_steady`: call the local Codex CLI at a fixed interval and ask it to
  choose concrete buffer, reinforce, and expedite actions under the same action
  budget.
- `codex_guardian`: call the local Codex CLI only when recent unmet demand,
  service level, or economic loss is worsening.
- `codex_monte_carlo`: ask Codex to propose extra candidates, then let the
  default Monte Carlo verifier choose from native plus Codex candidates.
- `codex_monte_carlo_admin`: ask Codex to configure the Monte Carlo search
  mix, risk mode, priorities, and bounded sample multiplier.
- `codex_monte_carlo_judge`: score a native plus generic expanded Monte Carlo
  shortlist, then ask Codex to select one candidate subject to a guardrail.

Intervention actions:

- build buffers;
- reinforce routes;
- expedite or redirect shipments.

All controllers see the same delayed, noisy, quantized telemetry snapshot. The
Codex prompts do not expose hidden simulator state. All policies share the same
per-step action budget and the same replanning cadence in the final comparison.
Policy-internal sampling uses a separate random stream from exogenous attacks,
failures, and demand noise, so evaluating more candidates does not change the
disturbance sequence in paired runs.

## Metrics

The lower-is-better sabotage impact score combines:

- total unmet demand;
- total economic loss;
- average service-level failure;
- low terminal inventory;
- average normalized action budget used.

## Final Numeric-Only Result

The paper-ready output is in `results/supply_chain_sabotage_paper/`. It is
copied from the completed `results/supply_chain_sabotage_combined_final/` run
and includes both the numeric-only study and qualitative sociotechnical study.
The copied qualitative summary uses the corrected
`q_structured_human_state_monte_carlo` policy name directly.

The current numeric-only paper result in
`results/supply_chain_sabotage_paper/numeric_only/summary.json` uses `30`
paired runs for every policy, `180` steps, shared `25`-step control cadence,
noisy shared telemetry, and the same normalized action budget.

Lower impact score is better:

| Policy | Impact score | Mean service level | Mean unmet demand | Mean budget |
| --- | ---: | ---: | ---: | ---: |
| `monte_carlo` | 13.275 | 0.714 | 15,949.4 | 0.448 |
| `codex_monte_carlo` | 13.564 | 0.707 | 16,332.1 | 0.448 |
| `codex_monte_carlo_admin` | 13.745 | 0.703 | 16,506.2 | 0.565 |
| `codex_monte_carlo_judge` | 14.044 | 0.693 | 17,007.0 | 0.458 |
| `codex_steady` | 15.400 | 0.665 | 18,570.8 | 0.710 |
| `rule_based` | 15.734 | 0.651 | 19,406.5 | 0.237 |
| `baseline` | 18.867 | 0.571 | 23,790.4 | 0.000 |
| `codex_guardian` | 18.867 | 0.571 | 23,790.4 | 0.000 |

Paired comparison against `monte_carlo`:

- `codex_monte_carlo`: mean score delta `+0.289`; win rate `1/30`.
- `codex_monte_carlo_admin`: mean score delta `+0.470`; win rate `0/30`.
- `codex_monte_carlo_judge`: mean score delta `+0.769`; win rate `0/30`.
- `rule_based`: mean score delta `+2.458`; win rate `0/30`.

Decision diagnostics:

- `codex_monte_carlo` selected Codex-proposed candidates `14` times and native
  Monte Carlo candidates `66` times.
- `codex_monte_carlo_admin` selected guided candidates `41` times and native
  candidates `39` times.
- `codex_monte_carlo_judge` selected expanded-slate candidates `67` times and
  native/default candidates `13` times.

Interpretation: Monte Carlo is the best controller in the current numeric-only
supply-chain formulation. This is not a failure of the benchmark. It is a
boundary result: when the state, action space, and objective are numeric,
compact, and simulator-aligned, a conventional model-based search controller is
expected to outperform a language-model controller. The Codex variants improve
substantially over baseline, but they do not beat Monte Carlo.

The `codex_monte_carlo_judge` result is especially informative. Codex often
selected candidates from the expanded slate, and those selections had lower
internal proxy scores in aggregate, but the final closed-loop simulation score
was worse. That means the expanded slate was exploiting the short-horizon proxy
rather than improving the true long-run system objective.

This result should be reported honestly as evidence that agent-in-the-loop
control is not automatically superior to conventional search on fully specified
numeric control problems. The next scientific question is where qualitative
interpretation, institutional constraints, and human behavior make the control
problem partially semantic rather than purely numeric.

See `.notes/supply_chain_qualitative_extension.md` for the qualitative
sociotechnical scenario design. That design is now implemented as the
`qualitative` study in `supply_chain_sabotage/run_experiment.py`. The
paper-ready interpretation is documented beside the copied result artifacts in
`results/supply_chain_sabotage_paper/README.md`.

## Main Command

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --hybrid-codex-candidates 3 --admin-max-sample-multiplier 2.0 --judge-shortlist-size 8 --judge-override-tolerance 0.12 --progress-interval 0 --output-dir results/supply_chain_sabotage_final
```

Combined numeric plus qualitative command:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --hybrid-codex-candidates 3 --admin-max-sample-multiplier 2.0 --judge-shortlist-size 8 --judge-override-tolerance 0.12 --progress-interval 0 --output-dir results/supply_chain_sabotage_combined_final
```

Cyclic-large numeric plus qualitative command:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --network-topology cyclic_large --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --hybrid-codex-candidates 3 --admin-max-sample-multiplier 2.0 --judge-shortlist-size 8 --judge-override-tolerance 0.12 --progress-interval 20 --codex-timeout 45 --output-dir results/supply_chain_sabotage_cyclic_large_30
```

View cyclic-large summaries:

```bash
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_cyclic_large_30/numeric_only/summary.json
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_cyclic_large_30/qualitative_resilience/summary.json
```

Codex policies are included by default. Use `--no-codex` only for a
deliberately non-Codex run:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --no-codex --runs 20 --steps 180 --progress-interval 20
```

Outputs overwrite:

- `results/supply_chain_sabotage/dashboard.png`
- `results/supply_chain_sabotage/dashboard.pdf`
- `results/supply_chain_sabotage/network_map.png`
- `results/supply_chain_sabotage/network_map.pdf`
- `results/supply_chain_sabotage/summary.json`
