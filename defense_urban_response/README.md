# Defense Urban Response Case Study

This directory is a separate defense-oriented case study for JDMS/IITSEC-style
framing. It models an urban hostile-contagion response on a road network rather
than on a flat grid.

The current map is a hand-curated Orlando metro neighborhood road network with
clustered districts, loops, and alternate routes. It uses approximate WGS84
coordinates and real public road names as a lightweight development scaffold.
For publication-grade geospatial fidelity, replace
`data/orlando_convention_roads.json` with an OpenStreetMap-derived road extract.

## Agents

- `zombie`: hostile contagious agent; directional road mover that tends to
  continue forward through the road graph, branches at intersections, pursues
  nearby civilians/defenders, and can kill or convert on contact.
- `civilian`: innocent bystander; attempts to move away from nearby zombies but
  is slow and vulnerable.
- `defender`: response force; moves on roads, has limited visibility, and can
  neutralize zombies within engagement range, but can be depleted by
  density-driven zombie pressure.

## Policies

- `baseline`: unmanaged defender patrol with weak mobility and no targeting.
- `rule_based`: defenders move toward visible or civilian-threatening zombies.
- `monte_carlo`: defenders choose targets using a short risk heuristic over
  candidate threats.
- `codex_steady`: optional Codex tactic advisor called at fixed intervals.
- `codex_guardian`: optional Codex tactic advisor called when the outbreak is
  worsening.

The paper-facing framing should use terms like hostile contagion, urban response
force, civilian protection, and road-network agent-based simulation rather than
leaning on zombie-themed language.

## Run

From the repository root:

```bash
.venv/bin/python defense_urban_response/run_experiment.py
```

The default run is `5` repeats per non-Codex policy and `180` steps per run.
The current hard-pressure defaults start with `150` zombies, `300` civilians,
and `30` defenders. Use `--runs` and `--steps` to scale up for final figures.

Include Codex policies:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --include-codex
```

When Codex is enabled, real Codex policies default to `1` repeat each because
they launch subprocess consultations during the simulation. Use `--codex-runs`
to scale them up later, and `--codex-timeout` to cap each decision call.
Progress prints every 20 simulation steps by default; change this with
`--progress-interval`:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --include-codex --codex-runs 1 --codex-timeout 45 --progress-interval 20
```

No-Codex smoke test for fallback behavior:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --runs 1 --steps 40 --include-codex --codex-command not-a-real-codex-command --codex-timeout 1 --output-dir results_smoke
```

Outputs are overwritten under repo-level `results/zombies/` by default:

- per-policy trajectory plots;
- per-policy final populated road maps;
- `dashboard.png` / `dashboard.pdf`;
- `summary.json`.

The simulation itself is lightweight; generating both PNG and PDF plots is the
slowest part of short test runs.

## Metrics

The primary outcome is zombie elimination speed under survival pressure. The
lower-is-better failure-adjusted elimination score combines:

- time to zombie clearance;
- zombie-victory failures, which are penalized rather than treated as fast
  clearance;
- civilian losses;
- defender losses;
- remaining zombies at the terminal step.

The dashboard also reports zombie-victory rate. In the hard-pressure regime,
many policies may fail to clear zombies; in that case the victory rate and
failure-adjusted score are more informative than raw mean clearance time.
