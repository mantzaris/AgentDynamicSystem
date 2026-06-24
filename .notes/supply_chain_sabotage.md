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
- `rule_based`: protect the most exposed node and reinforce/expedite a critical
  route.
- `monte_carlo`: sample candidate buffer/reinforce/expedite actions and choose
  the lowest short-horizon risk option.
- `codex_steady`: call the local Codex CLI at a fixed interval and ask it to
  choose one high-level tactic from the constrained supply-chain action
  vocabulary.
- `codex_guardian`: call the local Codex CLI only when recent unmet demand,
  service level, or economic loss is worsening.

Intervention actions:

- build buffers;
- reinforce routes;
- expedite or redirect shipments.

Codex tactics are deliberately constrained to the same action space as the
non-Codex controllers:

- `buffer_critical`;
- `reinforce_bottleneck`;
- `expedite_shortage`;
- `combined_response`.

## Metrics

The lower-is-better sabotage impact score combines:

- total unmet demand;
- total economic loss;
- service-level failure;
- low terminal inventory.

## Latest Stress-Test Result

The latest regenerated output in `results/supply_chain_sabotage/summary.json`
uses `10` non-Codex runs, `1` Codex run per Codex policy, and `180` steps.

Lower impact score is better:

| Policy | Impact score | Mean service level | Mean unmet demand |
| --- | ---: | ---: | ---: |
| `codex_guardian` | 12.467 | 0.735 | 14,880.8 |
| `rule_based` | 13.037 | 0.719 | 15,662.5 |
| `monte_carlo` | 13.821 | 0.701 | 16,706.8 |
| `codex_steady` | 15.891 | 0.650 | 19,629.1 |
| `baseline` | 19.061 | 0.568 | 24,058.6 |

Interpretation: the baseline now experiences visibly worse service degradation
and much higher unmet demand, while controllers reduce impact by choosing
buffers, route reinforcement, and expedited shipments.

## Main Command

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --runs 30 --codex-runs 1 --steps 180 --progress-interval 20
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
