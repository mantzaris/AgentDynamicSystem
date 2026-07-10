# Defense Urban Response Case Study

This directory is a separate defense-oriented case study for JDMS/IITSEC-style
framing. It models an urban hostile-contagion response on a road network rather
than on a flat grid.

The current map is a hand-curated Orlando metro neighborhood road network with
clustered districts, loops, and alternate routes. It uses approximate WGS84
coordinates and real public road names as a lightweight development scaffold.
For publication-grade geospatial fidelity, replace
`data/orlando_convention_roads.json` with an OpenStreetMap-derived road extract.

A generated extended scenario is available with
`--scenario social_complex_large`. It keeps the same controller families and
metrics but replaces the physical substrate with a larger synthetic urban
network:

- 72 road nodes;
- 156 undirected road edges;
- eight districts: downtown, hospital, airport, industrial port, university,
  residential shelter, responder reserve, and river-island bridgehead;
- role-labeled critical facilities including command, hospital, shelters,
  airfield, port, staging base, and bridgehead;
- bridges, tunnels, causeways, freight routes, evacuation corridors, and
  staging routes with different travel multipliers and route hazards;
- role-biased initial placement for hostiles, civilians, and defenders.

The extended scenario is intentionally more complex than a larger social layer:
route hazards affect travel time, civilian conversion risk, defender casualty
risk, and tactical scoring. Civilians in the qualitative branch can move toward
shelters, hospitals, or evacuation hubs when route clarity and compliance are
high enough. Codex prompts receive map context such as facility roles and
high-hazard corridors, but not hidden future outcomes or latent human state.

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

The urban benchmark now has two study layers.

Numeric tactical study:

- `baseline`: unmanaged defender patrol with weak mobility and no targeting.
- `rule_based`: defenders move toward visible or civilian-threatening zombies.
- `monte_carlo`: defenders evaluate randomized target assignments from the
  same observed state and choose the lowest current-risk assignment.
- `codex_steady`: Codex tactic advisor called at fixed intervals.
- `codex_guardian`: Codex tactic advisor called when the outbreak is
  worsening.
- `codex_monte_carlo`: Codex proposes extra tactic candidates; the local Monte
  Carlo verifier selects among native Monte Carlo and valid Codex candidates.
- `codex_monte_carlo_admin`: Codex configures the Monte Carlo risk mode and
  tactical priorities; local scoring keeps the final choice bounded.
- `codex_monte_carlo_judge`: Monte Carlo scores the native tactic slate, then
  Codex may choose among scored candidates subject to an override tolerance.

All policies replan at the same `control_update_interval` in Codex-inclusive
runs. Between replans, defenders continue toward the last assigned targets.
This keeps the comparison on the same footing: no controller receives faster
state updates simply because it is not using Codex.

Sociotechnical qualitative study:

- `q_baseline`: unmanaged patrol and no social intervention.
- `q_monte_carlo_tactical`: physical Monte Carlo only.
- `q_keyword_monte_carlo`: physical Monte Carlo plus a weak lexical trigger
  for explicitly labeled reports. It is intentionally not a hand-coded decoder
  for every report phrase.
- `q_structured_human_state_monte_carlo`: structured human-state heuristic that
  sees the latent trust, compliance, panic, and rumor state. This is a strong
  comparison baseline, not a deployable text-only controller.
- `q_codex_qualitative`: Codex reads telemetry plus qualitative reports and
  chooses bounded tactical and social interventions.
- `q_codex_monte_carlo_qualitative_admin`: Codex interprets reports and
  configures social action plus tactical priorities for local Monte Carlo.

The qualitative layer adds neighborhood trust, compliance, panic, rumor
pressure, route clarity, repeated-message fatigue, responder fatigue, and
institutional friction. These states affect civilian movement, conversion risk,
defender effectiveness, and the final score. Reports expose these states only
qualitatively, so a controller must interpret human context rather than only
optimize visible counts.

The qualitative score is deliberately caveat-checked: a controller should not
rank first by clearing hostiles while leaving worse panic, rumor pressure,
route confusion, message fatigue, compliance, trust, or responder fatigue. The
score reports physical and human components separately, weights the human
component materially, and social actions affect the target district plus nearby
districts so the human layer is decision-relevant rather than cosmetic.
When a qualitative run ends early through clearance or zombie-victory collapse,
the terminal human state is carried forward instead of leaving zero-filled
arrays. Zombie victory and survival loss also add explicit human-outcome
penalties, so early collapse cannot look socially benign merely because the run
ended before panic, rumor, or fatigue accumulated.

Social actions have conditional effects rather than fixed gains. Public
messages can backfire in low-trust or message-saturated districts; evacuation
guidance helps when routes are unclear and trust is adequate; community liaison
is slower but repairs legitimacy and message fatigue; shelter and triage help
when panic prevents movement; and responder rotation trades short-term pressure
for future responder quality. District archetypes modulate these effects.

To match the supply-chain qualitative study, normal reports now describe
observable behavior rather than giving direct labels such as `panic`, `rumor`,
or `distrust`. The keyword baseline only responds to explicit labels; Codex is
tested on whether it can infer the appropriate bounded social intervention from
the report language.

For `--scenario social_complex_large`, the qualitative branch is deliberately
set up as a semantic-administration experiment rather than a tactical-search
contest. Non-baseline qualitative controllers share a Monte Carlo tactical
base; they differ in whether and how they interpret qualitative reports into
social interventions. The complex branch also starts with stronger latent
district crises, applies unmanaged social drift when no social intervention is
active, strengthens correct social-action effects, and discounts post-collapse
civilian survival after zombie victory. The default/simple scenario is left
unchanged.

The paper-facing framing should use terms like hostile contagion, urban response
force, civilian protection, and road-network agent-based simulation rather than
leaning on zombie-themed language.

## Run

From the repository root:

```bash
.venv/bin/python defense_urban_response/run_experiment.py
```

The default run includes Codex policies. It uses `5` repeats per non-Codex
policy, `1` repeat per Codex policy, and `180` steps per run. It now runs both
the numeric tactical study and the qualitative sociotechnical study.
The current hard-pressure defaults start with `150` zombies, `300` civilians,
and `30` defenders. Use `--runs` and `--steps` to scale up for final figures.

Default Codex-inclusive run:

```bash
.venv/bin/python defense_urban_response/run_experiment.py
```

Real Codex policies default to `1` repeat each because they launch subprocess
consultations during the simulation. Use `--equal-codex-runs` for final
evaluation so Codex and non-Codex policies have the same number of paired runs,
and `--codex-timeout` to cap each decision call.
Progress prints every 20 simulation steps by default; change this with
`--progress-interval`:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --codex-runs 1 --codex-timeout 45 --progress-interval 20
```

Paper-facing paired run:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --study both --runs 10 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --progress-interval 20 --codex-timeout 45 --output-dir results/urban_response_final
```

Extended social-complex-large paired run:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --scenario social_complex_large --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --progress-interval 20 --codex-timeout 45 --output-dir results/urban_response_social_complex_large_30
```

Current semantic-complex paper run:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --scenario social_complex_large --study both --runs 50 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 32 --progress-interval 20 --codex-timeout 45 --output-dir results/urban_response_social_complex_large_semantic_50
```

The extended scenario is computationally heavier than the default map because
it starts with `220` hostiles, `520` civilians, and `44` defenders on the larger
hazard-weighted graph. For a quick non-Codex validation run:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --scenario social_complex_large --study both --no-codex --runs 1 --steps 8 --monte-carlo-samples 2 --progress-interval 0 --output-dir /tmp/urban_social_complex_large_smoke
```

Use `--no-codex` only for a deliberately non-Codex run.

Codex fallback smoke test:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --runs 1 --steps 40 --codex-command not-a-real-codex-command --codex-timeout 1 --output-dir results_smoke
```

With `--study both`, outputs are written under:

- `results/urban_response/numeric_only/` by default, or
  `results/urban_response_final/numeric_only/` for the paper command.
- `results/urban_response/qualitative_response/` by default, or
  `results/urban_response_final/qualitative_response/` for the paper command.

Numeric outputs include:

- per-policy trajectory plots;
- per-policy final populated road maps;
- `dashboard.png` / `dashboard.pdf`;
- `summary.json`.

Qualitative outputs currently include:

- `summary.json` with physical and human scores, paired comparisons, Codex
  reliability, and decision logs.

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

`summary.json` also stores per-run metrics, paired comparisons versus baseline
and the best non-Codex controller, aggregate Codex decision statistics, and
decision logs for Codex-family policies.

View numeric results:

```bash
.venv/bin/python defense_urban_response/show_results.py results/urban_response_final/numeric_only/summary.json
.venv/bin/python defense_urban_response/show_results.py results/urban_response_social_complex_large_30/numeric_only/summary.json
.venv/bin/python defense_urban_response/show_results.py results/urban_response_social_complex_large_semantic_50/numeric_only/summary.json
```

View qualitative results:

```bash
.venv/bin/python defense_urban_response/show_results.py results/urban_response_final/qualitative_response/summary.json
.venv/bin/python defense_urban_response/show_results.py results/urban_response_social_complex_large_30/qualitative_response/summary.json
.venv/bin/python defense_urban_response/show_results.py results/urban_response_social_complex_large_semantic_50/qualitative_response/summary.json
```

## Current Semantic Result

The current expanded urban result is:

```text
results/urban_response_social_complex_large_semantic_50/
results/urban_response_social_complex_large_semantic_50/INTERPRETATION.md
```

Numeric tactical branch: `codex_guardian` ranks first at `2.481`, but it does
not significantly beat the best non-Codex policy, `rule_based` (`-0.182`, 95%
CI `[-0.532, 0.169]`). Treat this branch as Codex-competitive rather than as a
decisive Codex numeric-control win.

Qualitative semantic branch: `q_codex_monte_carlo_qualitative_admin` ranks
first at `73.539`, beats `q_keyword_monte_carlo` by `-12.181` mean score with
95% CI `[-15.534, -8.829]`, and is statistically comparable to the privileged
structured-human-state reference (`-0.200`, 95% CI `[-2.739, 2.338]`). This is
the main urban result supporting Codex as a semantic administrator over
qualitative reports.
