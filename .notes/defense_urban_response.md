# Defense Urban Response Notes

## Purpose

`defense_urban_response/` is a separate defense-oriented case study. It is not
the same stability problem as the grass/rabbit/fox or multi-system benchmark.
The goal is to compare intervention policies under a hostile urban contagion
scenario where weak baseline behavior should fail.

## Current Scenario

- Map: `defense_urban_response/data/orlando_convention_roads.json`
- Initial zombies: `150`
- Initial civilians: `300`
- Initial defenders: `30`
- Horizon: `180` steps

The map is a hand-curated Orlando metro neighborhood road network with loops,
spurs, and clustered districts. It is a development scaffold, not a
publication-grade GIS extraction.

## Agents

- Zombies move on roads, prefer continuing forward, branch at intersections,
  pursue nearby civilians/defenders, and can kill or convert on contact.
- Civilians move away from nearby zombies but remain vulnerable.
- Defenders move on roads and may neutralize zombies, but can be depleted by
  direct contact and density-based swarm pressure.

## Policies

- `baseline`: unmanaged patrol with weak mobility and no targeting.
- `rule_based`: targets visible zombies or civilian-threatening zombies.
- `monte_carlo`: chooses targets using a short risk heuristic.
- `codex_steady`: asks local Codex for a tactic on a fixed interval.
- `codex_guardian`: asks local Codex when the outbreak worsens.

Codex policies currently choose among:

- `nearest_threat`
- `protect_civilians`
- `contain_hotspot`

## Scoring

The key metric is a lower-is-better failure-adjusted elimination score.

Zombie clearance and zombie victory are separated:

- `clearance_time`: successful zombie elimination time.
- `zombie_victory`: true when defenders or civilians are wiped out.
- `zombie_victory_time`: time of failure.

Zombie victory is penalized and must not be interpreted as fast clearance.

Score components:

- clearance or failure-adjusted victory time;
- civilian loss;
- defender loss;
- remaining zombies;
- zombie victory rate.

The dashboard should show:

- zombies remaining over time;
- civilians alive over time;
- zombie victory rate;
- failure-adjusted elimination score.

## Main Command

```bash
.venv/bin/python defense_urban_response/run_experiment.py --include-codex --runs 10 --codex-runs 10 --steps 180 --progress-interval 20 --codex-timeout 45
```

Outputs overwrite:

- `results/zombies/dashboard.png`
- `results/zombies/dashboard.pdf`
- `results/zombies/summary.json`
- per-policy plots and final maps.
