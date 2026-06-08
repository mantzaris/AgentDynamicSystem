# Agent Dynamic System

Agent-based grass/rabbit/fox simulation with Monte Carlo comparisons between
an uncontrolled baseline and controller-driven interventions.

## Run

Use the local virtual environment:

```bash
.venv/bin/python scripts/run_experiments.py
```

Default outputs are overwritten in `results/`:

- `baseline.png` / `baseline.pdf`
- `control_theory.png` / `control_theory.pdf`
- `rule_based.png` / `rule_based.pdf`
- `look_ahead.png` / `look_ahead.pdf`
- `dashboard.png` / `dashboard.pdf`
- `summary.json`

## Current Scope

The first comparison includes:

- `baseline`: no intervention.
- `control_theory`: a PI-style feedback controller with two independent
  non-negative actions, grass cutting and fertilizer application.
- `rule_based`: an interpretable threshold policy focused on staying close to
  the initial rabbit and fox populations while avoiding safety-floor breaches.
- `look_ahead`: a random shooting policy that simulates candidate action
  sequences 10 steps ahead with a fast aggregate mini-simulation, then chooses
  the first action from the lowest-instability plan.
- `agent_in_loop`: an optional local Codex CLI policy that asks Codex for a
  grass action every 10 simulation steps, reusing an active Codex session when
  the CLI exposes a resumable session id.

The code is structured so future controller types can be added as new
scenarios without changing the simulation core.

The default experiment runs 500 simulation steps. The controller chooses one
intervention per step: do nothing, cut grass, or apply fertilizer.

Default animal initialization is 300 rabbits and 20 foxes. Rabbit and fox plots
include red dotted safety floors at 50 rabbits and 10 foxes. The system
instability score treats the initial populations as the optimum and penalizes
percentage change, deviation from that initial condition, and safety-floor
breaches.

Run the optional Codex agent-in-the-loop comparison with:

```bash
.venv/bin/python scripts/run_experiments.py --include-agent-in-loop
```

That mode is not repeated across the Monte Carlo batch by default. With the
default `500` steps and `10`-step decision interval, it makes `50` Codex
consultations total.

If Codex is installed under a different command name, pass it with
`--agent-codex-command`.
