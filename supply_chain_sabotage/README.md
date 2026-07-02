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

The default topology is preserved for the paper result already documented
below. A larger cyclic topology is also available with
`--network-topology cyclic_large`. It uses the same controllers, metrics,
telemetry model, and output pipeline, but expands the physical graph to:

- 9 suppliers;
- 6 factories;
- 10 warehouses;
- 12 retailer/demand points;
- 127 directed edges.

The larger graph adds factory rework loops, a bidirectional warehouse cycle,
cross-cycle warehouse chords, and retailer mutual-aid links. This makes the
next supply-chain experiment less like a compact acyclic dispatch problem and
more like a resilient network-control problem with redundant paths, cycles,
and second-order routing choices.

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
- `rule_based`: generic SOP-style controller. It uses reported fill ratios,
  route health, and service drops to choose one modest buffer, reinforcement,
  or expedite action per decision cycle, with target cooldowns. It does not
  use scenario-specific node names or hand-picked bottlenecks.
- `monte_carlo`: samples candidate interventions and chooses the one with the
  lowest short-horizon risk score.
- `codex_steady`: Codex advisor called at a fixed interval. It chooses
  concrete buffer, reinforce, and expedite targets with magnitudes under the
  same normalized action budget as the non-Codex controllers.
- `codex_guardian`: Codex advisor called only when recent unmet
  demand, service level, or economic loss is worsening, then returns the same
  concrete intervention JSON schema.
- `codex_monte_carlo`: hybrid controller. Codex proposes a small set of
  candidate interventions from the same reported telemetry; the full native
  Monte Carlo candidate set is evaluated first, Codex candidates are added as
  explicit extra opportunities, and the default Monte Carlo verifier executes
  the lowest-risk action.
- `codex_monte_carlo_admin`: supervisory hybrid. Codex configures the Monte
  Carlo search itself by choosing candidate mix, risk mode, priority nodes and
  edges, objective weights, and a bounded sample multiplier. This is an
  explicit extra-deliberation controller; summary logs report the requested
  multiplier and evaluated candidate counts. The full native Monte Carlo
  candidate set is evaluated first, then extra Codex-guided candidates are
  added under the requested multiplier, with the default Monte Carlo risk score
  as the final verifier.
- `codex_monte_carlo_judge`: judge hybrid. Monte Carlo evaluates the native
  candidate set plus a generic service/tail-risk expansion, then Codex chooses
  from a scored shortlist. It cannot invent actions; a guardrail rejects
  overrides whose default Monte Carlo score is more than the configured
  tolerance worse than the default winner.

Available intervention choices:

- build inventory buffers at vulnerable warehouses or retailers;
- reinforce shipment routes;
- expedite or redirect shipments on important edges.

All policies are normalized to the same per-step action budget. A full buffer,
full reinforcement, and full expedite action consume weighted shares of that
budget; over-budget actions are scaled down before they affect the simulation.

All policies also receive the same controller-visible telemetry snapshot rather
than hidden ground truth. The true simulator state is reported through delayed,
noisy, quantized inventory and health observations before decisions are made.
Codex receives that same reported state in its prompt. Invalid Codex output is
logged and applies no intervention rather than falling back to a strong
heuristic.

In Codex-inclusive runs, all controllers use the same replanning cadence by
default: the non-Codex controllers update on `--control-update-interval`, which
defaults to `--codex-interval` when Codex policies are included. The selected
intervention is then held until the next control update.

Policy-internal sampling uses a separate random stream from exogenous attacks,
failures, and demand noise, so controllers that evaluate more candidates do not
change the disturbance sequence in paired runs.

## Metrics

The lower-is-better sabotage impact score combines:

- total unmet demand;
- total economic loss;
- average service-level failure;
- low terminal inventory.
- average normalized action budget used.

Dashboards show:

- unmet demand over time;
- service level over time;
- total economic loss by policy;
- lower-is-better sabotage impact ranking.

The JSON summary also includes per-run metrics, score confidence intervals,
paired comparisons against baseline and the best non-Codex controller, and
Codex decision reliability counts.

Generic-rule conventional-controller pilot from
`results/supply_chain_sabotage_generic_rule_pilot`, using the same 25-step
control cadence as Codex-inclusive runs:

| Policy | Impact score | Mean service level | Mean unmet demand | Mean budget |
| --- | ---: | ---: | ---: | ---: |
| `monte_carlo` | 14.048 | 0.697 | 16,962.0 | 0.448 |
| `rule_based` | 15.931 | 0.647 | 19,670.7 | 0.237 |
| `baseline` | 19.061 | 0.568 | 24,058.6 | 0.000 |

Codex comparisons should be regenerated after rule-policy changes; stale
pre-change Codex sanity numbers should not be used as evidence.

The intended strongest comparison is `monte_carlo` versus
`codex_monte_carlo`: pure simulator-native candidate generation against
Codex-guided extra candidate generation with the same Monte Carlo verifier.
The summary logs how often hybrid selections come from Codex candidates versus
native Monte Carlo candidates, plus native and extra evaluation counts.

`codex_monte_carlo_admin` is a separate compute-control tradeoff experiment:
Codex acts as a Monte Carlo administrator rather than as an action selector.
Use its results to discuss whether extra agent-guided deliberation buys
resilience, not as a same-compute comparison unless the logged evaluation counts
are normalized. The summary also records `controller_decisions` for every
policy so shared replanning cadence can be audited directly.

`codex_monte_carlo_judge` is another extra-deliberation experiment: Codex acts
as a selector over evaluated Monte Carlo outcomes. Its value is in testing
whether an agent can arbitrate among numeric search outputs using robustness
signals, not whether it can beat plain Monte Carlo at identical compute.

Current final equal-run result in `results/supply_chain_sabotage_final`:

| Policy | Impact score | Mean service level | Mean unmet demand | Mean budget |
| --- | ---: | ---: | ---: | ---: |
| `monte_carlo` | 13.513 | 0.709 | 16,267.8 | 0.448 |
| `codex_monte_carlo` | 13.858 | 0.700 | 16,739.6 | 0.446 |
| `codex_monte_carlo_admin` | 13.993 | 0.699 | 16,832.2 | 0.566 |
| `codex_monte_carlo_judge` | 14.276 | 0.690 | 17,291.0 | 0.451 |
| `codex_steady` | 15.413 | 0.667 | 18,537.1 | 0.787 |
| `rule_based` | 15.931 | 0.647 | 19,670.7 | 0.237 |
| `baseline` | 19.061 | 0.568 | 24,058.6 | 0.000 |

Interpretation: `monte_carlo` is best in this numeric-only formulation. This
is a useful boundary result, not a failure to hide. When state, action space,
and objective are compact and numeric, model-based search is the appropriate
controller. The next supply-chain research step should add a
human-organizational layer where qualitative reports, trust, compliance,
workforce fatigue, carrier cooperation, and legitimacy constraints affect the
physical dynamics. That design is documented in
`.notes/supply_chain_qualitative_extension.md`.

## Run

From the repository root:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py
```

Codex policies are included by default. The default now runs both studies:

- `numeric_only`: the original physical/logistics-only benchmark where Monte
  Carlo is expected to be strongest.
- `qualitative_resilience`: the sociotechnical benchmark with latent
  quantitative human/organizational state observed through qualitative reports.

The default uses `20` repeats per non-Codex policy and `1` repeat per Codex
policy.

Larger comparison:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --study both --runs 30 --codex-runs 1 --steps 180 --progress-interval 20 --output-dir results/supply_chain_sabotage_combined
```

Final-style equal-run comparison:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --hybrid-codex-candidates 3 --admin-max-sample-multiplier 2.0 --judge-shortlist-size 8 --judge-override-tolerance 0.12 --progress-interval 0 --output-dir results/supply_chain_sabotage_combined_final
```

Larger cyclic-network comparison:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --network-topology cyclic_large --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --hybrid-codex-candidates 3 --admin-max-sample-multiplier 2.0 --judge-shortlist-size 8 --judge-override-tolerance 0.12 --progress-interval 20 --codex-timeout 45 --output-dir results/supply_chain_sabotage_cyclic_large_30
```

The default telemetry model updates observations every `3` steps with `8%`
inventory noise and `0.035` health noise. Adjust these with
`--observation-update-interval`, `--inventory-report-noise`, and
`--health-report-noise`. The default score includes a `0.35` secondary penalty
for using the full normalized action budget on average; direct intervention
cost is already included in economic loss. Adjust the regularizer with
`--action-budget-score-weight` for sensitivity analysis. Use
`--control-update-interval` to set the shared policy replanning cadence. Use
`--monte-carlo-samples`, `--hybrid-codex-candidates`, and
`--admin-max-sample-multiplier` to configure the Monte Carlo and reinforced
Codex-admin comparisons. Use `--judge-shortlist-size` and
`--judge-override-tolerance` to configure the Codex judge comparison.

Print the summary table:

```bash
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_combined_final/numeric_only/summary.json
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_combined_final/qualitative_resilience/summary.json
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_cyclic_large_30/numeric_only/summary.json
.venv/bin/python supply_chain_sabotage/show_results.py results/supply_chain_sabotage_cyclic_large_30/qualitative_resilience/summary.json
```

Use `--no-codex` only for a deliberately non-Codex run:

```bash
.venv/bin/python supply_chain_sabotage/run_experiment.py --study both --no-codex --runs 20 --steps 180 --progress-interval 20
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
