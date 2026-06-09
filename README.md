# Agent Dynamic System

Benchmark suite for comparing intervention models across dynamic simulations.

The current systems are:

- `grass_rabbit_fox`: predator-prey-resource ecosystem control.
- `forest_fire`: spatial contagion and containment control.
- `supply_chain`: consumer/supplier primary-goods stability control.
- `epidemic_city`: urban epidemic response and healthcare-load control.
- `smart_grid`: renewable electric-grid balancing and outage-risk control.

Each system compares an uncontrolled baseline against controller-driven
interventions using matched random seeds and lower-is-better instability scores.
Every non-baseline controller uses an explicit intervention action space, and
the dashboards show those intervention choices over time.

## Run

Use the local virtual environment:

```bash
.venv/bin/python scripts/run_experiments.py
```

The default command runs the original grass/rabbit/fox benchmark and writes
over `results/`.

Run all benchmark systems:

```bash
.venv/bin/python scripts/run_experiments.py --system all
```

Run one new system:

```bash
.venv/bin/python scripts/run_experiments.py --system forest_fire
.venv/bin/python scripts/run_experiments.py --system supply_chain
.venv/bin/python scripts/run_experiments.py --system epidemic_city
.venv/bin/python scripts/run_experiments.py --system smart_grid
```

When `--system all` is used, outputs are written under:

- `results/grass_rabbit_fox/`
- `results/forest_fire/`
- `results/supply_chain/`
- `results/epidemic_city/`
- `results/smart_grid/`
- `results/cross_system_dashboard.png`
- `results/cross_system_dashboard.pdf`
- `results/benchmark_summary.json`

## Controller Families

The benchmark currently compares:

- `baseline`: no intervention.
- `control_theory`: aggregate feedback controller.
- `rule_based`: interpretable threshold policy.
- `look_ahead`: short-horizon rollout policy.
- `agent_in_loop`: optional Codex CLI controller for `grass_rabbit_fox`.

Run the optional Codex agent-in-the-loop comparison with:

```bash
.venv/bin/python scripts/run_experiments.py --include-agent-in-loop
```

That mode is not repeated across the Monte Carlo batch by default. With the
default `500` steps and `10`-step decision interval, it makes `50` Codex
consultations total.

If Codex is installed under a different command name, pass it with
`--agent-codex-command`.

## Codex-In-The-Loop Benchmark

The Codex-focused benchmark is separate from the classical controller
benchmark, so the plots stay readable. It compares:

- `baseline`: no intervention.
- `codex_steady`: Codex is consulted on a fixed interval.
- `codex_guardian`: Codex is consulted only when recent instability worsens or
  safety thresholds are approached.
- `codex_control_advised`: Codex is consulted on a fixed interval after a local
  control-theory advisory is computed and included in the prompt.

Run the Codex benchmark over all five systems:

```bash
.venv/bin/python scripts/run_codex_experiments.py --system all
```

Defaults are intentionally conservative: `--runs 1`, `--steps 500`, and
`--decision-interval 20`. Outputs are written under `results_codex/`.

For a no-Codex smoke test that exercises fallback behavior:

```bash
.venv/bin/python scripts/run_codex_experiments.py --system all --runs 1 --steps 20 --codex-command not-a-real-codex-command --output-dir results_codex_smoke
```

## Dynamic Systems

`grass_rabbit_fox` models individual rabbits and foxes on a grass grid.
Interventions are grass cutting and fertilizer application. Only one action is
applied per simulation step.

`forest_fire` models stochastic fire spread across a fuel grid. Interventions
are water drops, firebreaks, and controlled burns. Only one action is applied
per simulation step.

`supply_chain` models demand shocks, inventory, supplier health, price, and
unmet demand. Interventions are inventory release, production boost, rationing,
and supplier subsidy. Only one action is applied per simulation step.

`epidemic_city` models district-level disease spread, immunity, mobility, and
hospital load. Interventions are vaccination, testing/isolation, mobility
reduction, and hospital surge capacity. Only one action is applied per
simulation step.

`smart_grid` models electricity load, renewable variability, battery storage,
price, and outage risk. Interventions are demand response, battery dispatch,
backup generation, and renewable curtailment. Only one action is applied per
simulation step.

Each system has domain-specific metrics that are normalized into a common
lower-is-better instability score for ranking controllers within and across
systems.
