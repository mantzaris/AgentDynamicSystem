# Supply-Chain Sabotage Notes

## Purpose

`supply_chain_sabotage/` models a directed supply-chain network under adversary
attacks and random failures. The goal is to compare intervention controllers by
how well they reduce unmet demand and economic loss under disruption.

This is separate from the generic aggregate supply-chain system in
`src/agent_dynamic_system/supply_chain.py`.

## Network

The graph contains:

- 6 suppliers;
- 4 factories;
- 6 warehouses, including a reserve depot;
- 8 retailer and mission demand points;
- 52 directed shipment edges, including cross-warehouse transfer routes.

Nodes have inventory, capacity, health, storage limits, and demand for
retailers. Edges have shipment capacity, health, and reinforcement state.

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
