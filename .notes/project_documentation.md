# Project Documentation

## Purpose

This project explores whether a dynamic agent-based grass/rabbit/fox ecosystem
can be stabilized through limited interventions.

The system compares:

- `baseline`: the ecosystem runs with no intervention.
- `control_theory`: a feedback controller can choose one action per step:
  do nothing, cut grass, or apply fertilizer.
- `rule_based`: an interpretable threshold policy that tries to keep rabbit and
  fox populations close to their initial values while avoiding safety floors.

The goal is to reduce system instability while keeping rabbit and fox
populations close to their initial conditions and away from unsafe low
population levels.

## Current Run Command

Run the full experiment from the repository root:

```bash
.venv/bin/python scripts/run_experiments.py
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
grass controller, and the rule-based stability controller.

```text
src/agent_dynamic_system/experiment.py
```

Runs Monte Carlo repeats for each scenario using matched seeds.

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

They are overwritten on each run:

- `results/baseline.png`
- `results/baseline.pdf`
- `results/control_theory.png`
- `results/control_theory.pdf`
- `results/rule_based.png`
- `results/rule_based.pdf`
- `results/dashboard.png`
- `results/dashboard.pdf`
- `results/summary.json`

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
- initial rabbits: `300`
- initial foxes: `20`
- rabbit safety floor: `50`
- fox safety floor: `10`

Latest lower-is-better instability scores:

- `rule_based`: `0.9977330744053617`
- `control_theory`: `1.049903374825909`
- `baseline`: `2.6466502026074314`

Relative to baseline:

- `rule_based`: `0.37697957721137965`
- `control_theory`: `0.39669140024305566`
- `baseline`: `1.0`

The latest run ranks `rule_based` slightly better than `control_theory` under
the current instability metric. This is because the rule-based policy now cuts
more aggressively when foxes are far above the initial ideal and rabbits are
not close to the rabbit safety floor. Both managed scenarios remain
substantially lower instability than baseline.

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
