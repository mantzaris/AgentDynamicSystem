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
- `dashboard.png` / `dashboard.pdf`
- `summary.json`

## Current Scope

The first comparison includes:

- `baseline`: no intervention.
- `control_theory`: a PI-style feedback controller with two independent
  non-negative actions, grass cutting and fertilizer application.
- `rule_based`: an interpretable threshold policy focused on staying close to
  the initial rabbit and fox populations while avoiding safety-floor breaches.

The code is structured so future controller types can be added as new
scenarios without changing the simulation core.

The default experiment runs 500 simulation steps. The controller chooses one
intervention per step: do nothing, cut grass, or apply fertilizer.

Default animal initialization is 300 rabbits and 20 foxes. Rabbit and fox plots
include red dotted safety floors at 50 rabbits and 10 foxes. The system
instability score treats the initial populations as the optimum and penalizes
percentage change, deviation from that initial condition, and safety-floor
breaches.
