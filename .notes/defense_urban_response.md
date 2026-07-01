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
```

View qualitative results:

```bash
.venv/bin/python defense_urban_response/show_results.py results/urban_response_final/qualitative_response/summary.json
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

Latest qualitative-only run:

```text
results/urban_response_social_complexity_admin_fixed_50/summary.json
```

This run fixes the earlier Codex-admin no-op caveat where selected social
actions could have `social_intensity: 0.0`. After the fix, Codex-admin selected
265 social actions, zero selected social actions had zero intensity, and mean
admin social intensity was approximately `0.826`.

Lower-is-better ranking:

| Policy | Score | Physical | Human |
|---|---:|---:|---:|
| `q_structured_human_state_monte_carlo` | 49.339 | 3.317 | 4.536 |
| `q_codex_monte_carlo_qualitative_admin` | 50.359 | 3.448 | 4.532 |
| `q_codex_qualitative` | 50.748 | 3.491 | 4.543 |
| `q_monte_carlo_tactical` | 51.487 | 3.235 | 5.121 |
| `q_keyword_monte_carlo` | 51.487 | 3.235 | 5.121 |
| `q_baseline` | 51.865 | 3.696 | 4.460 |

The urban result supports the same conclusion as the supply-chain sabotage
study, but more weakly. Codex-admin improves over the best deployable non-Codex
qualitative baseline by `-1.128` mean score with a 95% CI of
`[-2.862, 0.607]` and win rate `0.72`. Direct Codex also improves over the
deployable baseline by `-0.739` mean score with a 95% CI of `[-2.508, 1.030]`.

Because the confidence intervals cross zero, this should be framed as
cross-domain supporting evidence rather than a decisive standalone win. The
primary positive evidence remains the supply-chain qualitative study, where the
Codex-Monte-Carlo qualitative admin hybrid wins decisively. The urban result is
valuable because it reproduces the same direction under a different dynamic
system: Codex is most useful when qualitative sociotechnical reports affect the
physical dynamics and must be interpreted into bounded interventions.
