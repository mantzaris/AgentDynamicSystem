# Dynamic Agent System: Current Approach

## Goal

Explore whether a dynamic grass/rabbit/fox agent-based system can be made more
stable through a limited set of interventions.

The baseline is the same system running without intervention. The first
controlled comparison uses a classic control-system-style feedback controller
whose goal is to reduce rabbit and fox population variance over simulation time.

## Current Comparison

Four simulation types run by default:

- `baseline`: no grass cutting and no fertilizer.
- `control_theory`: PI-style feedback controller using two independent
  non-negative action channels.
- `rule_based`: threshold and safety-floor policy that chooses do nothing, cut,
  or fertilize based on population bands around the initial targets.
- `look_ahead`: random shooting policy that evaluates candidate action
  sequences with a 10-step aggregate mini-simulation.

An optional fifth method can be enabled with `--include-agent-in-loop`:

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

The two intervention channels are intentionally modeled as separate actions:

- `cut_fraction`: removes a fraction of grass biomass.
- `fertilizer_fraction`: increases grass biomass toward carrying capacity.

They are not encoded as positive and negative values of one action. The
simulation enforces a single intervention choice at each step: do nothing, cut,
or fertilize.

Default animal initialization is:

- rabbits: `300`
- foxes: `20`

Safety floors are shown as red dotted lines:

- rabbits: `50`
- foxes: `10`

## Agent-Based Model

The model lives in `src/agent_dynamic_system/simulation.py`.

The environment is a toroidal grid with continuous grass biomass in each cell.
Rabbits and foxes are individual agents with position and energy.

At each step:

1. The controller observes aggregate grass, rabbit count, and fox count.
2. One intervention is applied: do nothing, cut, or fertilize.
3. Grass regrows toward local carrying capacity.
4. Rabbits move, spend energy, eat grass, reproduce, or die.
5. Foxes move, spend energy, hunt rabbits in the same cell, reproduce, or die.
6. Aggregate grass biomass, rabbit count, and fox count are recorded.

## Monte Carlo Design

The runner is `scripts/run_experiments.py`.

The default experiment length is 500 simulation steps.

Each scenario uses the same seed list. For a given run index, baseline and
controlled simulations are initialized from the same seed, which gives matched
initial conditions while still letting each scenario evolve independently after
its interventions change the system.

The optional Codex `agent_in_loop` scenario uses only the first matched seed by
default. It is evaluated as a single agent-guided trajectory rather than as a
Monte Carlo repeated scenario.

Individual run files are not saved. Only aggregate comparison outputs are
written, and they overwrite previous results.

Default outputs:

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

## Controller Extension Point

Controllers live in `src/agent_dynamic_system/controllers.py`.

Future controller types should implement:

- `reset(config, seed)`
- `act(observation) -> ControlAction`

Then add a new `Scenario` in `scripts/run_experiments.py`.

This keeps simulation dynamics separate from intervention policy logic.

## Ranking Extension Point

`src/agent_dynamic_system/metrics.py` computes lower-is-better system
instability metrics and a preliminary ranking.

The current control law uses one-sided population deficit and surplus terms,
plus a conservative grass-floor support term. Cutting is the main surplus
suppression actuator. Fertilizer is mainly a resource-floor/support actuator.
Both are exposed as separate non-negative action channels rather than one
signed control value, but only one action can be applied during a simulation
step.

The current system instability score treats the initial populations as the
optimal targets and combines:

- rabbit temporal coefficient of variation;
- fox temporal coefficient of variation;
- rabbit mean absolute percentage change from the initial population;
- fox mean absolute percentage change from the initial population;
- rabbit RMSE deviation from the initial population;
- fox RMSE deviation from the initial population;
- rabbit safety-floor breach fraction;
- fox safety-floor breach fraction;
- terminal extinction penalty.

For later phases, additional controllers can be ranked against the baseline
using the same `summarize_results` function, or the score weights can be
changed as the definition of stability becomes more specific.

## Local Python Environment

A local virtual environment has been created at `.venv`.

Installed packages are listed in `requirements.txt`.

Run the current experiment with:

```bash
.venv/bin/python scripts/run_experiments.py
```
