# Supply-Chain Sabotage Case Study

This directory contains a directed-graph supply-chain disruption case study.
It is separate from the existing aggregate supply-chain stability model in
`src/agent_dynamic_system/supply_chain.py`.

## Scenario

The system is a directed multi-region graph:

- 6 suppliers provide constrained raw inputs;
- 4 factories convert input into product;
- 6 warehouses buffer product, including a reserve depot and cross-region
  transfers;
- 8 retailer/demand points represent hospitals, airfields, command posts,
  city districts, ports, shelters, and remote outposts;
- 52 directed edges carry shipments with finite capacity and health.

Nodes track:

- inventory;
- production capacity;
- maximum storage;
- demand, for retailers;
- health, representing degraded capacity after attacks or failures;
- protected stock, representing buffer hardening.

Edges track:

- shipment capacity;
- health;
- reinforcement level.

## Disruptions

Each step may include:

- adversarial node attacks that destroy inventory and reduce node health;
- adversarial edge attacks that reduce route health;
- burst attacks where multiple assets are hit in the same step;
- random non-adversarial failures.
- stochastic demand surges at downstream mission nodes.

Attack targeting is weighted toward high-inventory and operationally important
nodes or high-capacity routes.

The current scenario is intentionally configured as a high-pressure stress
test. Compared with the first supply-chain version, it uses:

- higher attack probability: `0.78`;
- burst-attack probability: `0.28`;
- higher random failure probability: `0.34`;
- slower node and edge recovery;
- larger node inventory destruction and deeper node-health damage;
- stronger route-health degradation;
- stochastic downstream demand surges;
- lower normal shipment throughput.

The intervention actions were also strengthened so the comparison is not just
uniform collapse:

- buffers add more emergency inventory and protected stock;
- reinforced routes gain more health and resist more attack damage;
- expedited routes carry more surge capacity.

## Controllers

- `baseline`: takes no defensive action.
- `rule_based`: buffers the most exposed node, reinforces a critical inbound
  route, and expedites a shipment path toward a low-stock retailer.
- `monte_carlo`: samples candidate interventions and chooses the one with the
  lowest short-horizon risk score.
- `codex_steady`: Codex advisor called at a fixed interval. It chooses
  a high-level tactic that is mapped onto the same buffer, reinforce, and
  expedite action space.
- `codex_guardian`: Codex advisor called only when recent unmet
  demand, service level, or economic loss is worsening.

Available intervention choices:

- build inventory buffers at vulnerable warehouses or retailers;
- reinforce shipment routes;
- expedite or redirect shipments on important edges.

## Metrics

The lower-is-better sabotage impact score combines:

- total unmet demand;
- total economic loss;
- average service-level failure;
- low terminal inventory.

Dashboards show:

- unmet demand over time;
- service level over time;
- total economic loss by policy;
- lower-is-better sabotage impact ranking.

Latest regenerated stress-test ranking from `results/supply_chain_sabotage`:

| Policy | Impact score | Mean service level | Mean unmet demand |
| --- | ---: | ---: | ---: |
| `codex_guardian` | 12.467 | 0.735 | 14,880.8 |
| `rule_based` | 13.037 | 0.719 | 15,662.5 |
| `monte_carlo` | 13.821 | 0.701 | 16,706.8 |
| `codex_steady` | 15.891 | 0.650 | 19,629.1 |
| `baseline` | 19.061 | 0.568 | 24,058.6 |

This version was tuned so baseline degradation is visually and numerically
clear while controller interventions still have enough leverage to improve the
outcome.

## Run

From the repository root:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py
```

Codex policies are included by default. The default uses `20` repeats per
non-Codex policy and `1` repeat per Codex policy.

Larger comparison:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --runs 30 --codex-runs 1 --steps 180 --progress-interval 20
```

Use `--no-codex` only for a deliberately non-Codex run:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --no-codex --runs 20 --steps 180 --progress-interval 20
```

The Codex policies use separate run counts because each consultation launches a
Codex CLI subprocess. `codex_steady` consults at `--codex-interval`; 
`codex_guardian` consults when the recent trajectory deteriorates.

Outputs overwrite:

- `results/supply_chain_sabotage/dashboard.png`
- `results/supply_chain_sabotage/dashboard.pdf`
- `results/supply_chain_sabotage/network_map.png`
- `results/supply_chain_sabotage/network_map.pdf`
- `results/supply_chain_sabotage/summary.json`
- one PNG/PDF plot per policy.
