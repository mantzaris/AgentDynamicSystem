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

## Social Complex Large Scenario

An extended scenario profile is implemented with:

```text
--scenario social_complex_large
```

This is the urban analogue of the extended cyclic supply-chain topology. It is
implemented inside the same `defense_urban_response/` code path so it reuses
the same policy families, metrics, paired seeds, Codex cadence, and result
viewer. The default scenario remains unchanged for reproducibility.

The generated social-complex-large network contains:

- 72 road nodes;
- 156 undirected road edges;
- eight districts: downtown core, hospital/medical, airport gate, industrial
  port, university campus, suburb shelter, north reserve, and river-island
  bridgehead;
- critical facility roles: command, hospital, shelters, airfield, port,
  staging base, bridgehead, evacuation hub, and reserve depot;
- bridges, tunnels, causeways, freight roads, evacuation corridors, and
  responder staging routes;
- route travel multipliers and route-hazard values;
- role-biased spawning for hostiles, civilians, and defenders.

The extension changes the physical control problem, not only the social layer:

- bridge/tunnel/causeway hazards increase travel time;
- hazardous routes increase civilian conversion risk and responder casualty
  risk;
- tactical scoring accounts for critical-facility pressure and route-hazard
  exposure;
- civilians in the qualitative branch can route toward shelters, hospitals, or
  evacuation hubs when route clarity and compliance are high;
- Codex prompts include network context such as roles and high-hazard
  corridors, but not hidden future outcomes or latent human state.

Scenario-specific pressure defaults:

- initial zombies: `220`;
- initial civilians: `520`;
- initial defenders: `44`;
- steps: inherited from `--steps`, normally `180`.

This should be interpreted as a cross-domain complexity sensitivity study. The
expected honest result mirrors the supply-chain extension: Monte Carlo may
remain strongest on the numeric tactical branch, while Codex guidance is most
scientifically relevant in the qualitative branch where reports and social
constraints must be translated into bounded tactical/social priorities.

## Agents

- Zombies move on roads, prefer continuing forward, branch at intersections,
  pursue nearby civilians/defenders, and can kill or convert on contact.
- Civilians move away from nearby zombies but remain vulnerable.
- Defenders move on roads and may neutralize zombies, but can be depleted by
  direct contact and density-based swarm pressure.

## Policies

The urban response study now mirrors the supply-chain sabotage structure with
two layers:

1. a numeric/tactical road-network response benchmark;
2. a sociotechnical qualitative benchmark with latent human state and reports.

Numeric tactical policies:

- `baseline`: unmanaged patrol with weak mobility and no targeting.
- `rule_based`: targets visible zombies or civilian-threatening zombies.
- `monte_carlo`: samples target assignments from the same observed state and
  selects the lowest current-risk assignment.
- `codex_steady`: asks local Codex for a tactic on a fixed interval.
- `codex_guardian`: asks local Codex when the outbreak worsens.
- `codex_monte_carlo`: asks Codex for additional tactic candidates, then uses
  local Monte Carlo scoring to choose among native Monte Carlo and valid Codex
  candidates.
- `codex_monte_carlo_admin`: asks Codex to configure Monte Carlo risk mode and
  tactic priorities; local scoring makes the final bounded choice.
- `codex_monte_carlo_judge`: scores the native tactic slate first, then lets
  Codex choose among scored candidates subject to a bounded override tolerance.

Codex policies currently choose among:

- `nearest_threat`
- `protect_civilians`
- `contain_hotspot`
- `monte_carlo` for hybrid policies.

Sociotechnical qualitative policies:

- `q_baseline`: unmanaged patrol and no social intervention.
- `q_monte_carlo_tactical`: physical Monte Carlo only; no human-layer action.
- `q_keyword_monte_carlo`: physical Monte Carlo plus a weak lexical trigger for
  explicitly labeled report text. It is not a hand-coded decoder for every
  report phrase.
- `q_structured_human_state_monte_carlo`: strong structured human-state
  heuristic with access to latent trust, compliance, panic, and rumor state.
  This is an upper comparison baseline, not a deployable text-only controller.
- `q_codex_qualitative`: Codex interprets telemetry plus qualitative reports and
  chooses bounded tactical/social action.
- `q_codex_monte_carlo_qualitative_admin`: Codex interprets qualitative reports,
  configures social action and tactical priorities, then local Monte Carlo keeps
  final tactical choice bounded.

The qualitative layer attaches human state to road-network districts:

- trust;
- evacuation/shelter compliance;
- panic;
- rumor pressure;
- route clarity;
- repeated-message fatigue;
- responder fatigue;
- institutional friction.

These variables affect civilian movement, conversion risk, defender mobility
and engagement quality, and the final qualitative score. Reports expose the
state in natural-language fragments rather than as exact controller telemetry.

The qualitative scoring now explicitly guards against the earlier caveat where
a controller could rank first mainly through physical clearance while producing
worse human-state outcomes. The final qualitative score reports physical and
human components separately, gives the human component material weight, and
charges social-action budget. Social interventions now affect the target
district plus adjacent districts, so report interpretation changes both the
human score and the physical dynamics.

The social-complex-large qualitative metric also fixes the early-termination
artifact found in the first extended run. When a run ends early through
clearance or zombie-victory collapse, terminal human-state values are carried
forward across the remaining horizon instead of leaving zero-filled human
arrays. The human score also includes explicit `human_collapse_penalty` and
`human_survival_penalty` terms, so early collapse cannot look socially benign
because there was less simulated time for panic, rumor, or fatigue to
accumulate.

The qualitative layer now includes conditional sociotechnical effects. Public
messages can backfire under low trust or repeated-message fatigue; evacuation
guidance is useful when route clarity is low and trust is adequate; community
liaison repairs legitimacy more durably; shelter opening and medical triage
address immobilizing panic; responder rotation lowers future fatigue at a
short-term coordination cost. District archetypes modulate these effects, and
rumor, panic, and route confusion diffuse through adjacent road districts.

The report generator now follows the supply-chain qualitative pattern more
closely: normal reports describe observed behavior and coordination symptoms
rather than direct hidden-state labels. The weak keyword baseline only fires on
explicit labels such as `rumor`, `distrust`, `fatigue`, or `rotation`, while
Codex receives the same reports and must infer the appropriate social action.

The expanded `social_complex_large` qualitative branch is now intentionally
capability-aligned with the research question. The simple/default urban
scenario is unchanged. In the expanded branch, non-baseline qualitative
controllers share the same Monte Carlo tactical base so the qualitative study
does not ask Codex to beat Monte Carlo at short-horizon target assignment.
The comparison is instead over the social/semantic layer:

- `q_monte_carlo_tactical`: Monte Carlo tactical response with no social
  interpretation.
- `q_keyword_monte_carlo`: Monte Carlo plus a weak deployable keyword trigger.
- `q_structured_human_state_monte_carlo`: Monte Carlo plus privileged
  structured human state, used as an upper reference.
- `q_codex_qualitative` and `q_codex_monte_carlo_qualitative_admin`: Monte
  Carlo tactical execution plus Codex interpretation of qualitative reports.

The expanded branch now starts with stronger latent social crises across
district types: route confusion in bridge/airport/port districts, low trust in
industrial and bridgehead districts, panic around medical/shelter districts,
and higher responder fatigue/friction in the command-and-staging layer. These
conditions are exposed through natural-language reports rather than direct
labels. No-social policies now suffer unmanaged crisis drift when civilians are
exposed and no social intervention is active; correct social actions have
stronger causal effects in this branch. This is deliberate: the purpose is to
explore when a higher-level semantic controller has a legitimate advantage over
pure numeric control.

The collapse metric was also refined for `social_complex_large`: after zombie
victory, remaining civilians are not fully credited as saved, because a
defender or civilian wipeout represents loss of urban control. The score now
uses a projected survival discount and a stronger human collapse penalty in
that expanded scenario, while preserving the terminal-state carry-forward fix
that prevents early termination from zeroing human-state arrays.

## Fairness Controls

The urban response study now mirrors the supply-chain comparison principle:
all controllers replan at the same `control_update_interval` in Codex-inclusive
runs. Default Codex-inclusive runs use the Codex interval as the shared control
interval. Non-Codex-only runs default to per-step replanning unless
`--control-update-interval` is supplied.

All policies observe the same live road-network state at replanning time. Codex
policies do not receive hidden state, future outcomes, or privileged simulator
access. Hybrid Codex-Monte-Carlo policies can only add or configure bounded
tactic candidates; local scoring and guardrails still choose the executable
targeting behavior.

`codex_guardian` now consults at step zero and at the shared decision interval
when the observed outbreak worsens or defender pressure becomes severe. The
previous schedule could miss every shared replanning step and silently collapse
to the rule-based fallback.

The `monte_carlo` policy is now a randomized candidate-assignment search rather
than a deterministic hand-coded minimum. This makes the urban comparison closer
to the supply-chain numeric study, where Monte Carlo is expected to be a strong
purely quantitative controller.

For the qualitative study, Codex is not expected to beat Monte Carlo by doing
more brute-force search. The hypothesis is that it can interpret neighborhood
reports and choose a better social/tactical administration policy than a
keyword baseline when latent human variables are decision-relevant.

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
.venv/bin/python defense_urban_response/run_experiment.py --study both --runs 10 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --progress-interval 20 --codex-timeout 45 --output-dir results/urban_response_final
```

Social-complex-large command:

```bash
.venv/bin/python defense_urban_response/run_experiment.py --scenario social_complex_large --study both --runs 30 --equal-codex-runs --steps 180 --codex-interval 25 --control-update-interval 25 --monte-carlo-samples 24 --progress-interval 20 --codex-timeout 45 --output-dir results/urban_response_social_complex_large_30
```

Codex policies are included by default. Use `--no-codex` only for a
deliberately non-Codex run.

Outputs overwrite:

- `results/urban_response/numeric_only/summary.json` by default, or
  `results/urban_response_final/numeric_only/summary.json` for the main command.
- `results/urban_response/qualitative_response/summary.json` by default, or
  `results/urban_response_final/qualitative_response/summary.json` for the main
  command.
- numeric per-policy plots, final maps, and dashboard files under the numeric
  output directory.

The runner prints the lower-is-better ranking by default at the end of the run.
The summary file also stores per-run metrics, paired comparisons against
baseline and the best non-Codex controller, Codex decision statistics, and
Codex-family decision logs.

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

## Expected Interpretation

This case study should be treated as a second-domain transfer check, not as a
tuned proof that Codex must win every scenario. A credible outcome is:

- Monte Carlo remains strong when the problem is mostly short-horizon numeric
  target assignment.
- Direct Codex control may be competitive but should not be expected to
  dominate a well-matched Monte Carlo controller.
- Codex hybrids are valuable if they improve robustness, avoid pathological
  tactic choices, or produce useful tactical-administration logs without
  relying on privileged information.
- If plain Monte Carlo wins, the result supports the same negative finding as
  the supply-chain numeric study: agent-in-the-loop control is not automatically
  superior for purely quantitative short-horizon control.
- The qualitative study is the parallel to the supply-chain human layer. A
  meaningful Codex result should be assessed against `q_keyword_monte_carlo`
  and the strong `q_structured_human_state_monte_carlo` reference, not only
  against the tactical numeric policies.

## Current Paper Result

Current expanded urban semantic run:

```text
results/urban_response_social_complex_large_semantic_50/
results/urban_response_social_complex_large_semantic_50/INTERPRETATION.md
```

This supersedes the older `urban_response_social_complexity_admin_fixed_50`
qualitative result. The current run uses the larger `social_complex_large`
network, 50 paired runs, Codex included for Codex-family policies, 180 steps,
a shared 25-step decision cadence, and 32 Monte Carlo samples.

Numeric tactical branch, lower is better:

| Policy | Score | 95% CI | Victory Rate |
|---|---:|---:|---:|
| `codex_guardian` | 2.481 | [2.226, 2.736] | 0.440 |
| `codex_steady` | 2.524 | [2.277, 2.772] | 0.480 |
| `rule_based` | 2.663 | [2.418, 2.907] | 0.560 |
| `codex_monte_carlo_judge` | 2.936 | [2.702, 3.171] | 0.700 |
| `monte_carlo` | 2.969 | [2.739, 3.199] | 0.720 |
| `codex_monte_carlo` | 2.970 | [2.740, 3.200] | 0.720 |
| `codex_monte_carlo_admin` | 2.973 | [2.743, 3.203] | 0.720 |
| `baseline` | 3.984 | [3.978, 3.990] | 1.000 |

`codex_guardian` ranks first in the numeric branch, but it does not
significantly beat the best non-Codex controller, `rule_based`: mean delta
`-0.182`, 95% CI `[-0.532, 0.169]`, win rate `0.50`. This should be framed as
Codex being competitive in numeric tactical control, not as a decisive numeric
Codex victory.

Qualitative semantic branch, lower is better:

| Policy | Score | 95% CI | Physical | Human | Victory Rate |
|---|---:|---:|---:|---:|---:|
| `q_codex_monte_carlo_qualitative_admin` | 73.539 | [71.560, 75.518] | 4.084 | 8.160 | 0.940 |
| `q_structured_human_state_monte_carlo` | 73.739 | [72.072, 75.407] | 4.150 | 8.094 | 0.960 |
| `q_codex_qualitative` | 75.642 | [74.095, 77.188] | 4.199 | 8.398 | 0.980 |
| `q_baseline` | 82.153 | [81.468, 82.839] | 4.527 | 9.187 | 1.000 |
| `q_keyword_monte_carlo` | 85.720 | [83.154, 88.287] | 4.031 | 10.693 | 0.920 |
| `q_monte_carlo_tactical` | 86.649 | [84.153, 89.146] | 4.002 | 10.927 | 0.920 |

The main positive finding is the qualitative semantic branch:

- `q_codex_monte_carlo_qualitative_admin` beats the best deployable non-Codex
  qualitative policy, `q_keyword_monte_carlo`, by `-12.181` mean score with
  95% CI `[-15.534, -8.829]` and win rate `0.92`.
- It beats `q_baseline` by `-8.614` mean score with 95% CI
  `[-10.594, -6.634]` and win rate `0.96`.
- It is statistically comparable to the privileged
  `q_structured_human_state_monte_carlo` reference: mean delta `-0.200`, 95% CI
  `[-2.739, 2.338]`, win rate `0.46`.

The result is not a no-op artifact. Codex-admin made 180 qualitative calls, 178
valid responses, 2 invalid JSON responses, and 0 failures. It selected nonzero
social actions in 178 calls, with mean nonzero social intensity approximately
`0.831`. The most common actions were `community_liaison`,
`responder_rotation`, `shelter_opening`, and `evacuation_guidance`.

Paper interpretation: the expanded urban-defense case supports the main thesis
cleanly. Conventional tactical policies remain competitive on the numeric
branch, but Codex used as a qualitative Monte Carlo administrator decisively
outperforms deployable non-Codex qualitative baselines and performs comparably
to a privileged structured human-state reference controller.
