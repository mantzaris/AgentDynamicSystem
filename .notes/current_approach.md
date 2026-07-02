# Dynamic Agent System: Current Approach

## Goal

Explore how different intervention models perform across multiple dynamic
systems, not just one ecological example, and use a separate defense-oriented
urban case to test whether more intelligent controllers outperform weak
baseline behavior under pressure.

The project is now a benchmark suite with five systems:

- `grass_rabbit_fox`: predator-prey-resource ecosystem dynamics.
- `forest_fire`: spatial contagion and containment dynamics.
- `supply_chain`: consumer/supplier primary-goods stability dynamics.
- `epidemic_city`: urban epidemic and healthcare-load dynamics.
- `smart_grid`: renewable power-grid balancing and outage-risk dynamics.

A separate defense case study also exists:

- `defense_urban_response`: an urban road-network hostile-contagion response
  benchmark with zombies, civilians, and defenders.
- `supply_chain_sabotage`: a directed-graph supply-chain disruption benchmark
  with suppliers, factories, warehouses, retailers, shipment routes, adversary
  attacks, and random failures.

Each system has an uncontrolled baseline plus controller-driven interventions.
The goal is to compare which controller families reduce instability across
different kinds of dynamics.

For the urban defense case, the goal is different: compare which intervention
policies eliminate zombies faster under a deliberately harsh pressure regime,
with baseline intended to fail and stronger controllers expected to preserve
force and reduce clearance time.

For the supply-chain sabotage case, the goal is to minimize unmet demand and
economic loss after attacks by choosing where to build buffers, which routes to
reinforce, and where to expedite or redirect shipments. It now uses the same
controller comparison family as the zombie scenario: baseline, rule-based,
Monte Carlo, `codex_steady`, `codex_guardian`, and Codex-Monte-Carlo hybrids.
The current version is a hardened stress test with burst attacks, demand
surges, slower recovery, and stronger intervention leverage so baseline
degradation is more clearly distinguishable from controlled response.

The current final supply-chain result is a boundary result: Monte Carlo is the
best controller when the scenario is numeric-only and the action space is
compact. The Codex variants improve over baseline but do not beat Monte Carlo.
This should be framed honestly rather than tuned away. The next proposed
supply-chain step is a qualitative resilience extension with human trust,
compliance, workforce fatigue, carrier cooperation, unstructured incident
reports, and legitimacy constraints. That design is documented in
`.notes/supply_chain_qualitative_extension.md`.

Paper-ready supply-chain results are stored in
`results/supply_chain_sabotage_paper/`. The accompanying README in that
directory contains the interpretation to use in the paper: Monte Carlo wins the
numeric-only study, while the Codex-Monte-Carlo qualitative admin hybrid wins
the sociotechnical study among deployable controllers.

The next supply-chain escalation is implemented as
`--network-topology cyclic_large` inside `supply_chain_sabotage/`. It keeps the
same controllers and scoring but expands the graph from 24 nodes/52 edges to
37 nodes/127 edges with supplier diversity, factory rework loops, a
bidirectional warehouse cycle, warehouse cross-links, and retailer mutual-aid
edges. This should be used as a topology-complexity sensitivity study: it tests
whether the compact-network conclusion still holds when the logistics graph has
more cycles and redundant routing choices.

The completed cyclic-large result is in
`results/supply_chain_sabotage_cyclic_large_30/`. It reinforces the same
methodological conclusion. In the numeric-only branch, plain Monte Carlo still
wins (`15.976`) and `codex_monte_carlo_admin` is close but significantly worse
(`16.227`; paired delta `+0.251`, 95% CI `[0.191, 0.311]`). In the qualitative
sociotechnical branch, the Codex-Monte-Carlo qualitative administrator is the
best deployable controller (`79.086`) and beats the best deployable non-Codex
controller, `q_keyword_monte_carlo` (`83.980`), by paired delta `-4.895` with
95% CI `[-5.166, -4.623]` and win rate `30/30`. The stronger
`q_structured_human_state_monte_carlo` reference remains best overall
(`78.251`), so the claim should be that Codex guidance is valuable as a
semantic administrator over Monte Carlo, not that Codex replaces a structured
human-state oracle or beats Monte Carlo on purely numeric search.

The analysis is intended to be uniform: each benchmark system must have an
explicit intervention action space, each non-baseline controller must choose
from that action space, and dashboards must show both state trajectories and
intervention trajectories.

## Current Controller Families

The common comparison set is:

- `baseline`: no intervention.
- `control_theory`: aggregate feedback controller.
- `rule_based`: fixed interpretable threshold policy.
- `look_ahead`: short-horizon rollout or aggregate forecast policy.

The default `grass_rabbit_fox` run also includes:

- `agent_in_loop`: consults the local Codex CLI every 10 simulation steps and
  asks it to choose do nothing, cut, or fertilize from the current state. The
  controller starts a Codex session on the first consultation and resumes that
  same session for later updates when a session id is available.

The Codex executable defaults to `codex` and can be changed with
`--agent-codex-command`.

Unlike the other scenarios, `agent_in_loop` is not repeated across the Monte
Carlo batch by default. Its default run count is `1`, so a `500`-step simulation
with a `10`-step consultation interval makes `50` Codex calls total. This can be
overridden with `--agent-runs`, but it is intentionally separate from `--runs`.

## Codex-In-The-Loop Benchmark

A separate script runs the Codex-focused experiment:

```bash
.venv/bin/python scripts/run_codex_experiments.py --system all
```

This benchmark compares only:

- `baseline`: no intervention.
- `codex_steady`: Codex is called on a fixed interval, default `20` steps.
- `codex_guardian`: Codex is called when recent instability is worsening or a
  safety boundary is being approached.
- `codex_control_advised`: Codex is called on a fixed interval, but the prompt
  includes a local control-theory advisory.

Outputs are written to `results_codex/`, with one subdirectory per system plus
`codex_benchmark_summary.json` and a cross-system dashboard. This keeps the
Codex-recipe comparison separate from the classical controller benchmark.

The default Codex benchmark uses `--runs 1` to avoid exploding the number of
Codex calls. For 500 steps and a 20-step interval, the steady and
control-advised methods each make roughly 25 calls per system; guardian calls
are event-triggered and typically fewer.

## Systems

### Grass/Rabbit/Fox

The original model lives in `src/agent_dynamic_system/simulation.py`.

It uses individual rabbits and foxes on a toroidal grass grid. Interventions:

- `cut_fraction`: removes a fraction of grass biomass.
- `fertilizer_fraction`: increases grass biomass toward carrying capacity.

Default animal initialization:

- rabbits: `300`
- foxes: `20`

Safety floors:

- rabbits: `50`
- foxes: `10`

### Forest Fire

The forest-fire model lives in `src/agent_dynamic_system/forest_fire.py`.

It uses a spatial grid with fuel, burning cells, burned cells, stochastic
ignition, neighbor spread, and an eastward wind bias. Interventions:

- water drop: extinguishes a fraction of currently burning cells.
- firebreak: removes fuel from unburned cells.
- controlled burn: intentionally removes fuel and marks low-value burn area.

Tracked variables:

- burning area fraction;
- cumulative burned fraction;
- available fuel fraction.

### Supply Chain

The supply-chain model lives in `src/agent_dynamic_system/supply_chain.py`.

It models demand shocks, inventory, supplier health, production capacity, price,
and unmet demand. Interventions:

- inventory release;
- production boost;
- demand rationing;
- supplier subsidy.

Tracked variables:

- inventory;
- unmet demand fraction;
- price index;
- supplier health.

### Epidemic City

The epidemic-city model lives in `src/agent_dynamic_system/epidemic_city.py`.

It models district-level disease spread, recovery, waning immunity, mobility,
event shocks, and hospital load. Interventions:

- vaccination;
- testing/isolation;
- mobility reduction;
- hospital surge capacity.

Tracked variables:

- infected fraction;
- hospital load;
- susceptible fraction;
- recovered fraction.

### Smart Grid

The smart-grid model lives in `src/agent_dynamic_system/smart_grid.py`.

It models electricity load, renewable generation variability, dispatchable
generation, battery storage, price, and outage risk. Interventions:

- demand response;
- battery dispatch;
- backup generation;
- renewable curtailment.

Tracked variables:

- supply-demand imbalance;
- battery charge;
- price index;
- outage fraction.

### Defense Urban Response

The defense case lives in `defense_urban_response/`.

It uses an organic road-network map rather than a grid, with neighborhood-like
clusters and route alternatives. Agents:

- zombies: mobile hostile agents that pursue civilians and defenders, convert
  civilians on contact, and can cause zombie-win collapse if the force loses.
- civilians: vulnerable noncombatants that flee when they can but are not
  guaranteed to escape.
- defenders: response agents that move on roads and actively engage zombies,
  but can be depleted under pressure.

Policies:

Numeric tactical policies:

- `baseline`: weak patrol with reduced mobility.
- `rule_based`: direct visible-threat pursuit.
- `monte_carlo`: randomized candidate target-assignment search.
- `codex_steady`: Codex consulted at fixed intervals.
- `codex_guardian`: Codex consulted when the outbreak is worsening.
- `codex_monte_carlo`: Codex proposes extra tactic candidates for local Monte
  Carlo verification.
- `codex_monte_carlo_admin`: Codex configures the Monte Carlo risk mode and
  tactic priorities.
- `codex_monte_carlo_judge`: Codex judges a scored Monte Carlo tactic slate
  under a bounded override tolerance.

Codex-inclusive urban runs use a shared control-update interval, so non-Codex
controllers do not replan more frequently than Codex-family controllers.

Sociotechnical qualitative policies:

- `q_baseline`: unmanaged patrol and no social intervention.
- `q_monte_carlo_tactical`: physical Monte Carlo only.
- `q_keyword_monte_carlo`: weak lexical trigger for explicitly labeled reports.
- `q_structured_human_state_monte_carlo`: strong structured human-state
  heuristic reference.
- `q_codex_qualitative`: Codex interprets reports and chooses tactical/social
  action.
- `q_codex_monte_carlo_qualitative_admin`: Codex configures social action and
  tactical priorities for local Monte Carlo.

The qualitative urban layer tracks trust, compliance, panic, rumor pressure,
route clarity, repeated-message fatigue, responder fatigue, and institutional
friction. It is the direct parallel to the supply-chain human-organizational
study. Its scoring is caveat-checked so a policy cannot credibly win only by
improving physical clearance while worsening panic, rumor pressure, route
confusion, message fatigue, compliance, trust, or responder fatigue. Physical
and human components are reported separately, and social actions have local
spillover effects on the target district and nearby districts.

The social layer now has conditional effects and backfire modes: public
messaging is not always useful, guidance depends on route clarity and trust,
liaison is slower but more durable, and responder rotation trades short-term
capacity for future reliability. District archetypes and adjacent-district
diffusion make this a semantic/causal interpretation problem rather than a
single report-to-action lookup.

The report generator no longer hands the keyword baseline direct labels for
normal reports. Reports describe observable behavior and coordination symptoms,
so the Codex qualitative controller is tested on semantic interpretation in the
same spirit as the supply-chain qualitative study.

Current urban qualitative result:

- Result file: `results/urban_response_social_complexity_admin_fixed_50/summary.json`
- Interpretation file:
  `results/urban_response_social_complexity_admin_fixed_50/INTERPRETATION.md`
- `q_structured_human_state_monte_carlo`: `49.339`
- `q_codex_monte_carlo_qualitative_admin`: `50.359`
- `q_codex_qualitative`: `50.748`
- `q_monte_carlo_tactical`: `51.487`
- `q_keyword_monte_carlo`: `51.487`
- `q_baseline`: `51.865`

This reinforces the supply-chain conclusion directionally but not decisively.
Codex-admin improves over the best deployable non-Codex baseline by `-1.128`
mean score with 95% CI `[-2.862, 0.607]`; direct Codex improves by `-0.739`
with 95% CI `[-2.508, 1.030]`. The CIs cross zero, so this is cross-domain
support rather than the primary positive result. The strongest positive result
remains the supply-chain qualitative study.

Current pressure regime:

- zombies start at `150` and move quickly;
- civilians start at `300`;
- defenders start at `30`;
- zombies are directional road movers: they prefer continuing forward through
  the road graph and branch at intersections rather than acting as pure random
  walkers;
- civilians flee nearby zombies but are slow and vulnerable;
- defenders move faster than civilians, but the baseline is an unmanaged patrol
  with reduced mobility and no targeting;
- non-baseline defenders are directed by their controller policy;
- zombies can convert or kill civilians on close contact;
- zombies can kill defenders through probabilistic, density-driven swarm
  pressure;
- defender casualty risk increases with local zombie density and, for the
  baseline, with broad unmanaged swarm pressure;
- a zombie-victory condition ends the run when civilians or defenders are
  effectively wiped out.

Primary metric:

- zombie elimination speed is the target, but the score is failure-adjusted.
- zombie victory is an explicit failure mode and is penalized separately from
  zombie clearance time so early baseline collapse cannot be mistaken for fast
  successful clearance.
- the dashboard now emphasizes zombie-victory rate and the failure-adjusted
  elimination score; lower is better.

## Monte Carlo Design

The runner is `scripts/run_experiments.py`.

Default experiment length is `500` simulation steps. Default repeated runs are
`30` unless overridden with `--runs`.

Run the original default system:

```bash
.venv/bin/python scripts/run_experiments.py
```

Run all systems:

```bash
.venv/bin/python scripts/run_experiments.py --system all
```

Run individual systems:

```bash
.venv/bin/python scripts/run_experiments.py --system forest_fire
.venv/bin/python scripts/run_experiments.py --system supply_chain
.venv/bin/python scripts/run_experiments.py --system epidemic_city
.venv/bin/python scripts/run_experiments.py --system smart_grid
```

Each scenario uses the same seed list within a system. For a given run index,
baseline and controlled simulations are initialized from the same seed, giving
matched initial conditions while still letting each scenario evolve
independently after interventions change the system.

Individual run files are not saved. Only aggregate comparison outputs are
written, and they overwrite previous results.

## Outputs

Default single-system outputs are written to `results/`.

With `--system all`, outputs are grouped by system:

- `results/grass_rabbit_fox/`
- `results/forest_fire/`
- `results/supply_chain/`
- `results/epidemic_city/`
- `results/smart_grid/`

The separate defense-oriented case studies write to:

- `results/urban_response/`
- `results/supply_chain_sabotage/`

The urban response runner now writes two subdirectories when `--study both` is
used:

- `results/urban_response/numeric_only/`
- `results/urban_response/qualitative_response/`

Each system directory contains:

- one PNG/PDF trajectory plot per controller;
- `dashboard.png`;
- `dashboard.pdf`;
- `summary.json`.

Dashboards show system-state trajectories, separate intervention-action
trajectories, and the lower-is-better instability ranking.

The all-system run also writes:

- `results/cross_system_dashboard.png`;
- `results/cross_system_dashboard.pdf`;
- `results/benchmark_summary.json`.

## Ranking Extension Point

The original ecosystem metrics are in `src/agent_dynamic_system/metrics.py`.

The new generic metrics are in `src/agent_dynamic_system/benchmark.py`.

All metrics are lower-is-better instability scores. Each system uses
domain-specific tracked variables and safety thresholds, but the generic score
combines:

- temporal variability;
- deviation from target values;
- step-to-step change;
- safety-bound breaches;
- mean intervention amount.

This makes controller ranking possible within each system and provides a
cross-system relative-to-baseline dashboard.
