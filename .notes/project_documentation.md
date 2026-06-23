# Project Documentation

## Purpose

This project explores whether dynamic agent-based systems can be stabilized
through limited interventions, and how different controller families compare
across multiple simulation domains.

The project now includes a benchmark suite with:

- `grass_rabbit_fox`: predator-prey-resource ecosystem dynamics.
- `forest_fire`: spatial contagion and containment dynamics.
- `supply_chain`: consumer/supplier primary-goods stability dynamics.
- `epidemic_city`: urban epidemic and healthcare-load dynamics.
- `smart_grid`: renewable power-grid balancing and outage-risk dynamics.

The original grass/rabbit/fox model remains the default system, but the runner
can now execute all systems with `--system all`.

There is also a separate defense-oriented case study in
`defense_urban_response/` that is not part of the classical instability
benchmark, but serves as a harsh survival / clearance comparison for rule-based
controllers and local Codex-in-the-loop controllers.

The system compares:

- `baseline`: the ecosystem runs with no intervention.
- `control_theory`: a feedback controller can choose one action per step:
  do nothing, cut grass, or apply fertilizer.
- `rule_based`: an interpretable threshold policy that tries to keep rabbit and
  fox populations close to their initial values while avoiding safety floors.
- `look_ahead`: a random shooting policy that samples possible action
  sequences, simulates their aggregate effects 10 steps ahead, and applies the
  first action from the lowest-instability sequence.
- `agent_in_loop`: an optional local Codex CLI policy that is consulted every
  10 simulation steps and returns a grass action for the current state. It
  starts a Codex session on the first consultation and resumes the same session
  for later state updates when the CLI exposes a resumable session id.

The goal is to reduce system instability while keeping rabbit and fox
populations close to their initial conditions and away from unsafe low
population levels.

For the defense case study, the goal is different: maximize zombie elimination
pressure, make the baseline fail under stress, and compare which controllers
can hold up when civilians and defenders are both at risk.

For the multi-system benchmark, the analysis is uniform: every system has an
uncontrolled baseline, explicit intervention actions, controller scenarios that
choose from those actions, state-trajectory plots, intervention-trajectory
plots, and lower-is-better instability rankings.

## Current Run Command

Run the full experiment from the repository root:

```bash
.venv/bin/python scripts/run_experiments.py
```

Run the full multi-system benchmark:

```bash
.venv/bin/python scripts/run_experiments.py --system all
```

Run the separate Codex-in-the-loop benchmark:

```bash
.venv/bin/python scripts/run_codex_experiments.py --system all
```

Run the defense urban-response benchmark:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --include-codex --runs 10 --codex-runs 10 --steps 180 --progress-interval 20 --codex-timeout 45
```

Run a single additional system:

```bash
.venv/bin/python scripts/run_experiments.py --system forest_fire
.venv/bin/python scripts/run_experiments.py --system supply_chain
.venv/bin/python scripts/run_experiments.py --system epidemic_city
.venv/bin/python scripts/run_experiments.py --system smart_grid
```

The default run uses:

- `30` Monte Carlo repeats
- `500` simulation steps
- base seed `20260607`
- matched seed list across baseline and controlled scenarios

Optional overrides:

```bash
.venv/bin/python scripts/run_experiments.py --runs 10 --steps 300 --seed 123
```

Include the optional Codex agent-in-the-loop method:

```bash
.venv/bin/python scripts/run_experiments.py --include-agent-in-loop
```

If needed, choose a different Codex executable name or path with:

```bash
.venv/bin/python scripts/run_experiments.py --include-agent-in-loop --agent-codex-command /path/to/codex
```

The agent-in-loop method can make many local Codex calls during Monte Carlo
runs, so it is intentionally not tied to `--runs`. By default it runs once.
For the default `500` steps and a 10-step consultation interval, it creates
`50` Codex consultations total.

The agent-in-loop run count can be changed separately:

```bash
.venv/bin/python scripts/run_experiments.py --include-agent-in-loop --agent-runs 2
```

## Local Python Environment

A local Python virtual environment was created at:

```text
.venv/
```

Dependencies are listed in:

```text
requirements.txt
```

Installed packages:

- `numpy`
- `matplotlib`

Matplotlib cache is routed into:

```text
.cache/matplotlib/
```

That avoids sandbox/home-directory cache warnings during plot generation.

## Code Structure

```text
src/agent_dynamic_system/config.py
```

Holds all simulation defaults and thresholds.

```text
src/agent_dynamic_system/simulation.py
```

Runs the agent-based simulation. It handles grass regrowth, rabbit movement,
rabbit eating/reproduction/death, fox movement, fox hunting/reproduction/death,
and application of one controller action per step.

```text
src/agent_dynamic_system/controllers.py
```

Defines the controller interface, the no-control baseline, the current PI-style
grass controller, the rule-based stability controller, the look-ahead
mini-simulation controller, and the optional Codex agent-in-loop controller.

```text
src/agent_dynamic_system/experiment.py
```

Runs repeated simulations for each scenario using matched seeds. The runner can
assign a different repeat count to a scenario; this is used so the optional
Codex agent-in-loop method defaults to one run instead of the full Monte Carlo
batch.

```text
src/agent_dynamic_system/benchmark.py
```

Defines generic run/result containers, repeated-simulation execution, and
lower-is-better instability metrics for non-ecosystem benchmark systems.

```text
src/agent_dynamic_system/forest_fire.py
```

Defines the forest-fire spatial contagion model, its controller policies, labels,
and metric targets.

```text
src/agent_dynamic_system/supply_chain.py
```

Defines the supply-chain/resource-market model, its controller policies, labels,
and metric targets.

```text
src/agent_dynamic_system/epidemic_city.py
```

Defines the district-level epidemic response model, its controller policies,
labels, and metric targets.

```text
src/agent_dynamic_system/smart_grid.py
```

Defines the renewable smart-grid balancing model, its controller policies,
labels, and metric targets.

```text
src/agent_dynamic_system/generic_plotting.py
```

Writes scenario plots, system dashboards, and the cross-system dashboard for the
generic benchmark systems.

```text
src/agent_dynamic_system/metrics.py
```

Computes lower-is-better system instability metrics and scenario ranking.

```text
src/agent_dynamic_system/plotting.py
```

Writes scenario plots and the comparison dashboard as PNG and PDF.

```text
scripts/run_experiments.py
```

Main command-line entry point.

```text
scripts/run_codex_experiments.py
```

Separate command-line entry point for the Codex-in-the-loop benchmark. It keeps
the Codex recipe comparison separate from the classical controller comparison.

```text
src/agent_dynamic_system/codex_guidance.py
```

Reusable Codex guidance policy for the generic benchmark systems. It implements
fixed-interval, guardian-triggered, and control-advised prompt recipes.

## Current Model Defaults

Grid:

- width: `45`
- height: `45`
- topology: toroidal wrapping

Simulation horizon:

- steps: `500`

Initial animal populations:

- rabbits: `300`
- foxes: `20`

Safety floors:

- rabbits: `50`
- foxes: `10`

Maximum animal populations:

- rabbits: `900`
- foxes: `250`

Grass:

- per-cell capacity: `6.0`
- initial grass range: `45%` to `100%` of capacity
- regrowth rate: `0.035`
- regrowth noise: `0.012`

Controller action caps:

- max cut fraction: `0.28`
- max fertilizer fraction: `0.28`

## Defense Urban Response Case

Location:

- `defense_urban_response/`

Core files:

- `defense_urban_response/run_experiment.py`
- `defense_urban_response/urban_response.py`
- `defense_urban_response/data/orlando_convention_roads.json`

The defense case is separate from the classical stability benchmark. It is a
road-network hostile-contagion response scenario intended to test whether
guided intervention policies outperform weak unmanaged baseline patrol under
severe outbreak pressure.

Current default counts:

- zombies: `150`
- civilians: `300`
- defenders: `30`
- steps: `180`

Current dynamics:

- zombies move on roads and are faster than civilians;
- zombies use directional road movement, preferring to continue forward and
  branching at intersections instead of behaving as pure random walkers;
- civilians flee nearby zombies but are vulnerable to contact;
- zombie contact with civilians kills or converts them;
- zombie contact and broad swarm pressure can kill defenders
  probabilistically;
- defender casualty risk rises with local zombie density;
- baseline patrol has no targeting and reduced mobility;
- non-baseline defenders are guided by their controller policy.

Policies:

- `baseline`: unmanaged patrol.
- `rule_based`: visible-threat and civilian-threat targeting.
- `monte_carlo`: short risk-heuristic targeting.
- `codex_steady`: Codex tactic advisor at fixed intervals.
- `codex_guardian`: Codex tactic advisor when the outbreak worsens.

Metrics and interpretation:

- the objective is zombie elimination speed, not stability;
- zombie victory is an explicit failure mode when defenders or civilians are
  wiped out;
- zombie victory is penalized separately from zombie clearance time;
- the dashboard reports zombie-victory rate and a failure-adjusted elimination
  score where lower is better;
- early baseline collapse should increase the score, not improve it.

Primary run command:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --include-codex --runs 10 --codex-runs 10 --steps 180 --progress-interval 20 --codex-timeout 45
```

## Simulation Step Order

For each simulation step:

1. The controller observes aggregate grass biomass, rabbit count, and fox
   count.
2. The controller returns a candidate intervention.
3. The simulation enforces a single action choice:
   do nothing, cut, or fertilize.
4. Grass regrows toward carrying capacity.
5. Rabbits move, spend energy, eat grass, reproduce, or die.
6. Foxes move, spend energy, hunt rabbits in the same cell, reproduce, or die.
7. Aggregate grass, rabbit, fox, cut, and fertilizer trajectories are recorded.

Individual run files are not saved.

## Controller Behavior

The current feedback controller is named:

```text
control_theory
```

It is implemented by `PIGrassController`.

The controller computes support and suppression signals from:

- rabbit deficit or surplus relative to the target;
- fox deficit or surplus relative to the target;
- short-term change in the rabbit and fox errors;
- grass shortfall relative to a conservative grass floor.

The controller scores two independent non-negative action channels:

- cut grass;
- apply fertilizer.

The simulation then chooses at most one non-zero action per step:

- do nothing if both action amounts are zero;
- cut if the cut amount is greater than or equal to the fertilizer amount;
- fertilize if the fertilizer amount is greater than the cut amount.

This prevents representing cutting and fertilizer as positive and negative
directions of one signed action.

## Rule-Based Controller

The rule-based controller is named:

```text
rule_based
```

It uses fixed thresholds around the initial rabbit and fox populations. The
policy is focused on minimizing movement away from the initial ideal state:

- do nothing when rabbits and foxes are both inside a narrow ideal band;
- cut grass when rabbits are above the ideal band, especially above the outer
  band;
- cut grass more aggressively when foxes are far above target and rabbits are
  not near the rabbit safety floor;
- avoid cutting when rabbits or foxes are near safety floors;
- fertilize when rabbits or foxes are below the ideal band and grass is
  resource-limited;
- apply stronger support if either population hits its safety floor.

This is intended as an interpretable comparison method, not as a classical
control-theory policy.

## Look-Ahead Controller

The look-ahead controller is named:

```text
look_ahead
```

It uses a random shooting method:

1. Generate candidate action sequences.
2. Simulate each sequence for `10` future steps with a fast aggregate
   grass/rabbit/fox approximation.
3. Score each simulated trajectory by deviation from the initial population
   targets, short-term population change, and safety-floor risk.
4. Choose the first action from the best-scoring sequence.

The implementation uses aggregate mini-simulations rather than cloning the full
individual ABM for each candidate. Full ABM rollouts were too slow for the
default Monte Carlo workload. The state-aware controller hook remains in place,
so a future version can use full-state rollouts or a more accurate learned
surrogate model.

## Codex Agent-In-The-Loop Controller

The optional agent-in-loop controller is named:

```text
agent_in_loop
```

It consults the local Codex CLI through `codex exec` every `10` simulation
steps by default. The first consultation starts a session; later consultations
use `codex exec resume` when the CLI output provides a resumable session id.

Each consultation sends the latest aggregate state containing:

- current simulation step;
- current grass biomass and grass fraction;
- current rabbit and fox counts;
- initial target populations;
- safety floors;
- action limits;
- recent aggregate state history;
- the current instability metric definition.

Codex is instructed to return only JSON:

```json
{"action": "none|cut|fertilize", "amount": 0.0, "reason": "short reason"}
```

The controller parses that JSON and applies one action once at the consultation
step. Non-consultation steps return no intervention.

If Codex is unavailable, times out, or returns invalid JSON, the controller
falls back to the rule-based policy for that consultation rather than crashing
the simulation.

This method is opt-in and defaults to one simulation run because it delegates
decisions to a local Codex session. With the default settings it makes `50`
Codex consultations, not one consultation per Monte Carlo repeat.

## Forest-Fire Benchmark

The forest-fire benchmark is named:

```text
forest_fire
```

It is a spatial contagion model with:

- continuous fuel per grid cell;
- burning cells;
- burned cells;
- stochastic lightning ignition;
- neighbor fire spread;
- eastward wind bias;
- slow fuel regrowth.

The controller chooses one intervention per step:

- do nothing;
- water drop: extinguish a fraction of burning cells;
- firebreak: remove fuel from unburned cells;
- controlled burn: intentionally remove fuel from selected cells.

Tracked variables:

- burning area fraction;
- cumulative burned fraction;
- available fuel fraction.

The lower-is-better score penalizes burning-area variability, cumulative burn
damage, fuel depletion, safety-bound breaches, and intervention amount.

## Supply-Chain Benchmark

The supply-chain benchmark is named:

```text
supply_chain
```

It is an aggregate consumer/supplier primary-goods model with:

- stochastic demand shocks;
- inventory;
- production capacity;
- supplier health;
- unmet demand;
- price feedback from scarcity and surplus.

The controller chooses one intervention per step:

- do nothing;
- inventory release;
- production boost;
- demand rationing;
- supplier subsidy.

Tracked variables:

- inventory;
- unmet demand fraction;
- price index;
- supplier health.

The lower-is-better score penalizes volatility, deviation from target inventory
and price, unmet demand, supplier-health decline, safety-bound breaches, and
intervention amount.

## Epidemic-City Benchmark

The epidemic-city benchmark is named:

```text
epidemic_city
```

It is a district-level epidemic response model with:

- susceptible, infected, and recovered populations;
- district mobility mixing;
- stochastic event shocks;
- recovery and waning immunity;
- hospital load driven by infection prevalence.

The controller chooses one intervention per step:

- do nothing;
- vaccination;
- testing/isolation;
- mobility reduction;
- hospital surge capacity.

Tracked variables:

- infected fraction;
- hospital load;
- susceptible fraction;
- recovered fraction.

The lower-is-better score penalizes infection instability, high hospital load,
large target deviations, safety-bound breaches, and intervention amount.

## Smart-Grid Benchmark

The smart-grid benchmark is named:

```text
smart_grid
```

It is a renewable electric-grid balancing model with:

- time-varying electricity load;
- renewable generation variability;
- weather shocks;
- dispatchable generation;
- battery storage;
- price feedback;
- outage risk from shortage.

The controller chooses one intervention per step:

- do nothing;
- demand response;
- battery dispatch;
- backup generation;
- renewable curtailment.

Tracked variables:

- supply-demand imbalance;
- battery charge;
- price index;
- outage fraction.

The lower-is-better score penalizes imbalance, battery depletion, price
instability, outage risk, safety-bound breaches, and intervention amount.

## Codex-In-The-Loop Benchmark

The Codex benchmark is separate from the classical benchmark and writes to:

```text
results_codex/
```

It compares the same five systems, but only these scenarios:

- `baseline`: no intervention.
- `codex_steady`: Codex is consulted on a fixed interval.
- `codex_guardian`: Codex is consulted only when recent instability worsens or
  a safety threshold is approached.
- `codex_control_advised`: Codex is consulted on a fixed interval after a local
  control-theory advisory is computed and included in the prompt.

The default command is:

```bash
.venv/bin/python scripts/run_codex_experiments.py --system all
```

Defaults:

- runs per scenario: `1`
- steps: `500`
- decision interval: `20`
- output directory: `results_codex/`

For `500` steps and a `20`-step interval, `codex_steady` and
`codex_control_advised` each make roughly `25` Codex calls per system.
`codex_guardian` is event-triggered and should usually make fewer calls.

If Codex is unavailable, times out, or returns invalid JSON, the Codex strategy
falls back to the system's rule-based policy for that decision. This keeps long
benchmark runs from crashing but should be noted when interpreting results.

## Action Meaning

Cut action:

- stored as `cut_fraction`
- means the fraction of standing grass biomass removed

Fertilizer action:

- stored as `fertilizer_fraction`
- means the fraction of the remaining gap to grass carrying capacity filled

The action plots are separated so it is clear which action was chosen.

## Outputs

Outputs are written to:

```text
results/
```

They are overwritten on each run. For a single grass/rabbit/fox run, the files
are:

- `results/baseline.png`
- `results/baseline.pdf`
- `results/control_theory.png`
- `results/control_theory.pdf`
- `results/rule_based.png`
- `results/rule_based.pdf`
- `results/look_ahead.png`
- `results/look_ahead.pdf`
- `results/agent_in_loop.png` and `results/agent_in_loop.pdf` when
  `--include-agent-in-loop` is used.
- `results/dashboard.png`
- `results/dashboard.pdf`
- `results/summary.json`

When `--system all` is used, each system writes its own subdirectory:

- `results/grass_rabbit_fox/`
- `results/forest_fire/`
- `results/supply_chain/`
- `results/epidemic_city/`
- `results/smart_grid/`

Each system subdirectory contains per-controller PNG/PDF trajectory plots,
`dashboard.png`, `dashboard.pdf`, and `summary.json`.

Each dashboard shows both system-state trajectories and separate intervention
action trajectories so the controller mechanism is visible for every system.

The all-system run also writes:

- `results/cross_system_dashboard.png`
- `results/cross_system_dashboard.pdf`
- `results/benchmark_summary.json`

The dashboard shows:

- grass biomass trajectory;
- rabbit trajectory;
- fox trajectory;
- cut action amount;
- fertilizer action amount;
- system instability score comparison.

Rabbit and fox plots include red dotted safety-floor lines.

Population trajectory y-axes start at zero so the zero tick is visible.

## Current Instability Score

The score is lower-is-better and is shown as:

```text
System instability (lower is better)
```

The vertical axis label is:

```text
Variability / instability score
```

The current score treats the initial rabbit and fox populations as the optimal
targets. It combines:

- rabbit temporal coefficient of variation;
- fox temporal coefficient of variation;
- rabbit mean absolute percentage change from initial population;
- fox mean absolute percentage change from initial population;
- rabbit RMSE deviation from initial population;
- fox RMSE deviation from initial population;
- rabbit safety-floor breach fraction;
- fox safety-floor breach fraction;
- terminal extinction penalty.

Current weights:

- rabbit temporal CV: `0.15`
- fox temporal CV: `0.15`
- rabbit percent change from initial: `0.18`
- fox percent change from initial: `0.18`
- rabbit initial-deviation RMSE: `0.13`
- fox initial-deviation RMSE: `0.13`
- rabbit safety breach fraction: `0.03`
- fox safety breach fraction: `0.03`
- terminal extinction rate: `0.02`

## Latest Generated Result

The latest generated result used:

- runs: `30`
- steps: `500`
- base seed: `20260607`
- default scenarios: `baseline`, `control_theory`, `rule_based`, `look_ahead`
- optional `agent_in_loop`: not included in the latest saved full result
- initial rabbits: `300`
- initial foxes: `20`
- rabbit safety floor: `50`
- fox safety floor: `10`

Latest lower-is-better instability scores:

- `rule_based`: `0.9977330744053617`
- `control_theory`: `1.049903374825909`
- `look_ahead`: `1.1605075099149809`
- `baseline`: `2.6466502026074314`

Relative to baseline:

- `rule_based`: `0.37697957721137965`
- `control_theory`: `0.39669140024305566`
- `look_ahead`: `0.4384816356810847`
- `baseline`: `1.0`

The latest run ranks `rule_based` slightly better than `control_theory`, with
`look_ahead` also substantially better than baseline. The look-ahead policy
keeps rabbits close to the initial target but allows foxes to remain farther
from the 20-fox ideal than the two best managed scenarios.

## Extension Plan

Future controller types should be added as new `Controller` implementations in:

```text
src/agent_dynamic_system/controllers.py
```

Then add a new `Scenario` in:

```text
scripts/run_experiments.py
```

This lets future controller types be compared against baseline and
`control_theory` without changing the simulation core.

Future ranking changes should be made in:

```text
src/agent_dynamic_system/metrics.py
```

The current metrics already produce a sorted `ranking_lower_is_better` list in
`results/summary.json`.

## Verification Commands Used

Syntax check:

```bash
.venv/bin/python -m compileall src scripts
```

Full experiment:

```bash
.venv/bin/python scripts/run_experiments.py
```

Inspect summary:

```bash
.venv/bin/python -m json.tool results/summary.json
```

## Notes on Current Limitations

The controller currently often prefers cutting over fertilizer because the
current ecosystem dynamics tend to produce animal surpluses and oscillations
where reducing grass is the stronger stabilizing action.

The fertilizer action channel remains implemented and plotted separately, and
future controller designs can choose it more often if their control law favors
supporting grass biomass under low-population or low-resource conditions.

The current score is a practical first instability metric. It is not yet a
formal control-theoretic objective function, and its weights may need tuning as
new controller types are added.
