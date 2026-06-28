# Supply-Chain Qualitative Resilience Extension

## Purpose

The current supply-chain sabotage case is primarily a numeric control problem.
That makes it a strong test of whether an agent can replace conventional
model-based search. The answer from the current final run is no: Monte Carlo is
the best controller when the state, action space, and score are compact,
numeric, and simulator-aligned.

The next scenario should test a different and more defensible research
question:

> Can an agent-in-the-loop controller improve resilience when the system
> includes human behavior, institutional constraints, ambiguous text reports,
> and qualitative tradeoffs that are not fully captured by numeric telemetry?

This is not a way to weaken Monte Carlo artificially. Monte Carlo should remain
the correct baseline for physical logistics. The extension should add a
human-organizational layer where qualitative interpretation affects the
physical supply-chain dynamics.

Implementation status: this extension is implemented in
`supply_chain_sabotage/qualitative_simulation.py` and is run from the same
entry point as the numeric study with `supply_chain_sabotage/run_experiment.py
--study both`.

Paper-ready results are stored in
`results/supply_chain_sabotage_paper/qualitative_resilience/summary.json`, with
interpretation notes in `results/supply_chain_sabotage_paper/README.md`.

## Current Boundary Result

Final numeric-only result from
`results/supply_chain_sabotage_final/summary.json`:

| Policy | Impact score | Mean service level | Mean unmet demand | Mean budget |
| --- | ---: | ---: | ---: | ---: |
| `monte_carlo` | 13.513 | 0.709 | 16,267.8 | 0.448 |
| `codex_monte_carlo` | 13.858 | 0.700 | 16,739.6 | 0.446 |
| `codex_monte_carlo_admin` | 13.993 | 0.699 | 16,832.2 | 0.566 |
| `codex_monte_carlo_judge` | 14.276 | 0.690 | 17,291.0 | 0.451 |
| `codex_steady` | 15.413 | 0.667 | 18,537.1 | 0.787 |
| `rule_based` | 15.931 | 0.647 | 19,670.7 | 0.237 |
| `baseline` | 19.061 | 0.568 | 24,058.6 | 0.000 |
| `codex_guardian` | 19.061 | 0.568 | 24,058.6 | 0.000 |

Interpretation:

- Monte Carlo is best in all `10` paired runs.
- Codex hybrids are useful relative to baseline but not superior to Monte
  Carlo in the numeric-only setup.
- `codex_monte_carlo_judge` frequently chose expanded-slate candidates, but
  those choices worsened the true closed-loop objective despite lower proxy
  scores.
- The result is a boundary condition: language-model agency is not a substitute
  for conventional search when the problem is fully specified as numeric
  control.

## Current Qualitative Result

The paper-ready qualitative output in
`results/supply_chain_sabotage_paper/qualitative_resilience/summary.json`
uses corrected policy naming. The structured-state diagnostic baseline is named
`q_structured_human_state_monte_carlo`; it should not be called an oracle.

Lower score is better:

| Policy | Score | Physical | Human | Service | Trust | Rumor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_codex_monte_carlo_qualitative_admin` | 54.703 | 46.261 | 3.752 | 0.307 | 0.173 | 0.895 |
| `q_structured_human_state_monte_carlo` | 55.889 | 47.792 | 3.599 | 0.260 | 0.153 | 0.802 |
| `q_keyword_monte_carlo` | 59.840 | 50.922 | 3.963 | 0.259 | 0.061 | 0.929 |
| `q_monte_carlo_logistics` | 60.506 | 51.564 | 3.974 | 0.246 | 0.061 | 0.929 |
| `q_codex_qualitative` | 60.762 | 51.983 | 3.902 | 0.230 | 0.086 | 0.912 |
| `q_baseline` | 62.885 | 53.998 | 3.950 | 0.206 | 0.051 | 0.941 |

Key interpretation:

- Codex qualitative admin beats the best deployable non-Codex baseline,
  `q_keyword_monte_carlo`, by `-5.137` score with 95% CI
  `[-5.389, -4.885]`.
- Direct Codex control is not the right architecture; `q_codex_qualitative`
  underperforms the hybrid.
- The useful agent role is semantic administration of a conventional Monte
  Carlo search process.

## Scientific Reframing

The supply-chain case should become a two-layer resilience problem:

- Physical layer: factories, warehouses, retailers, inventories, routes,
  attacks, failures, and demand.
- Human-organizational layer: trust, compliance, workforce fatigue, carrier
  cooperation, rumor propagation, political legitimacy, and fairness concerns.

Monte Carlo should be expected to remain strong on the physical layer. The
agent should be tested on whether it can interpret qualitative reports and
choose social or institutional interventions that modify the physical layer.

The claim to test should be:

> Agents are not generally better than model-based search, but they can be
> useful as semantic controllers when resilience depends on human and
> organizational context.

## Human-Organizational State

Add district-level and network-level state variables that are only partially
visible through reports:

- `public_trust[district]`: willingness to follow official guidance.
- `compliance[district]`: actual compliance with rationing, shelter delivery
  windows, evacuation routing, and anti-hoarding requests.
- `rumor_pressure[district]`: strength of misinformation or panic narratives.
- `workforce_fatigue[node_or_route]`: staff fatigue at warehouses, carriers,
  shelters, and high-pressure route crews.
- `carrier_cooperation[route_or_region]`: willingness of private carriers or
  drivers to accept risky routes.
- `equity_pressure[district]`: accumulated perception that a district is being
  neglected relative to others.
- `institutional_friction`: delays from permits, legal disputes, jurisdictional
  conflict, or coordination failure.

These variables should affect physical outcomes:

- Low trust increases demand surges, hoarding, and noncompliance.
- High rumor pressure increases panic demand and may reduce route access.
- Workforce fatigue reduces production, shipment capacity, and intervention
  effectiveness.
- Low carrier cooperation reduces edge capacity or makes expedite actions less
  reliable.
- High equity pressure increases legitimacy penalty and can reduce future
  compliance.
- Institutional friction delays or attenuates selected interventions.

## Qualitative Reports

Generate short text reports each control interval from the hidden
human-organizational state and recent physical events. Reports should be noisy,
partial, and sometimes ambiguous.

Example reports:

- "Shelter coordinators report residents are refusing the eastern delivery
  window after a rumor that supplies are being diverted to the port."
- "Two carrier dispatchers say drivers will avoid the coastal bridge unless
  police escorts are confirmed before nightfall."
- "Hospital procurement is asking for a public explanation before accepting
  rationing; staff believe the previous allocation was unfair."
- "Warehouse night crew completed the last surge but supervisors warn that
  another emergency shift will likely increase loading errors."
- "Local radio is amplifying a claim that the reserve depot has hidden stock;
  district demand may spike unless officials respond."

The reports should not reveal numeric hidden state directly. They should
describe cues that a qualitative controller can map to likely operational
risks.

## Extended Action Space

Keep the existing logistics actions:

- buffer inventory;
- reinforce routes;
- expedite shipments.

Add social/institutional interventions with a separate or explicitly weighted
budget:

- `public_message`: clarify allocation, counter rumors, request compliance.
- `community_liaison`: target a district with local coordination support.
- `carrier_negotiation`: improve route cooperation or secure private capacity.
- `police_escort`: improve willingness to use a risky route but may harm trust
  if overused.
- `staff_rotation`: reduce workforce fatigue at a warehouse or route.
- `mutual_aid_request`: improve supply or staff capacity after a delay.
- `rationing_policy`: reduce demand pressure but may damage trust if poorly
  justified.
- `equity_rebalance`: prioritize a neglected district to reduce legitimacy
  pressure.

The key design requirement is that these actions should not be decorative. They
must change future demand, capacity, compliance, intervention effectiveness, or
score penalties.

## Controller Comparisons

All controllers should receive the same numeric telemetry and the same text
reports. Codex must not receive hidden variables or privileged labels.

Recommended comparison set:

- `baseline`: no action.
- `monte_carlo_logistics`: current numeric Monte Carlo over physical logistics
  actions only.
- `rule_based_social`: generic thresholds plus keyword matching over reports.
- `keyword_monte_carlo`: conventional baseline with a hand-coded text feature
  extractor feeding Monte Carlo priorities.
- `structured_human_state_monte_carlo`: non-deployable diagnostic baseline
  that receives the true structured human-organizational state and uses a
  hand-coded heuristic to act on it. This is not a strict oracle or upper
  bound; it tests whether structured latent-state access is useful.
- `codex_qualitative`: Codex reads telemetry plus reports and chooses social
  and logistics actions directly.
- `codex_monte_carlo_qualitative_admin`: Codex interprets reports, sets
  priorities and constraints, and Monte Carlo chooses logistics actions under
  those priorities.
- `codex_monte_carlo_qualitative_judge`: Monte Carlo provides numeric candidate
  outcomes; Codex selects among them using report-aware risks such as trust,
  fatigue, and legitimacy.

This comparison prevents an unfair setup where Codex simply receives more
information than the conventional controllers. The deployable conventional
baseline gets the same reports but only simple keyword extraction. The
structured-state baseline shows whether direct access to latent human-state
variables helps, but it should not be interpreted as a mathematical upper
bound.

## Metrics

Keep physical resilience metrics:

- total unmet demand;
- economic loss;
- service level;
- terminal inventory;
- action budget use.

Add human-resilience metrics:

- mean public trust;
- compliance failure rate;
- workforce fatigue;
- carrier refusal rate;
- rumor pressure;
- equity gap across districts;
- worst-district or CVaR service level;
- legitimacy penalty from coercive or unfair interventions.

The final score should combine physical and human terms, but the dashboard
should report them separately. Otherwise a controller can win by improving
short-term delivery while destroying trust or fairness.

## Experimental Hypotheses

H1: In the physical-only scenario, Monte Carlo remains the best controller.

H2: When human-organizational dynamics materially affect physical outcomes,
text-blind Monte Carlo degrades because it cannot anticipate trust, compliance,
fatigue, and cooperation effects.

H3: Keyword-based conventional parsing improves over text-blind Monte Carlo but
is brittle to paraphrase, ambiguity, and conflicting reports.

H4: Codex qualitative hybrids improve over keyword baselines when reports
contain actionable semantic information that is not reliably captured by fixed
rules.

H5: If Codex does not improve over keyword or structured-state heuristic
baselines, that is still a valuable negative result showing that qualitative
interpretation was not the limiting factor or that the prompt/controller design
was insufficient.

## Fairness And Anti-Gaming Rules

The extension must avoid making Codex win by construction:

- Do not weaken Monte Carlo on the physical logistics task.
- Do not give Codex hidden state or labels unavailable to other controllers.
- Do not hand-write report templates that contain obvious policy names.
- Use held-out report phrasings for evaluation after prompt development.
- Keep social-action budget costs explicit.
- Log Codex failures, invalid outputs, selected action budgets, and decision
  latency.
- Use paired seeds and the separate policy RNG already added to the numeric
  supply-chain case.
- Report each controller's physical score, human score, and combined score.
- Include a structured-state heuristic baseline to expose whether the
  qualitative layer is actually decision-relevant.

## Implementation Plan

Phase 1: Add human state.

- Add a `HumanLayerState` object with trust, compliance, rumor pressure,
  workforce fatigue, carrier cooperation, equity pressure, and institutional
  friction.
- Update the simulator step so human variables affect demand surges, route
  capacity, expedite reliability, and intervention effectiveness.
- Add human-state recovery and degradation dynamics.

Phase 2: Add report generation.

- Generate reports from latent human state and recent events at each control
  interval.
- Store reports in the controller-visible history.
- Keep reports noisy and partial rather than direct labels.

Phase 3: Add social actions.

- Extend `Intervention` with social-action fields.
- Normalize logistics and social actions under explicit budgets.
- Apply social actions to future trust, compliance, fatigue, rumor, and carrier
  cooperation dynamics.

Phase 4: Add controllers.

- Add `rule_based_social`.
- Add `keyword_monte_carlo`.
- Add `structured_human_state_monte_carlo` as a diagnostic baseline.
- Add `codex_qualitative`.
- Add qualitative admin/judge hybrids that combine Codex semantic
  interpretation with Monte Carlo logistics search.

Phase 5: Add outputs and ablations.

- Add human-state plots to the dashboard.
- Add score decomposition into physical, human, and combined terms.
- Run ablations with human layer disabled, reports removed, reports shuffled,
  and structured human-state access enabled.

## Publication Framing

The intended paper narrative should be:

1. Conventional search is superior in fully specified numeric control.
2. Agent-in-the-loop control should not be sold as a generic replacement for
   control or Monte Carlo search.
3. The agent's plausible contribution is semantic control: interpreting
   qualitative human/institutional context and translating it into constraints,
   priorities, or interventions.
4. The supply-chain qualitative extension tests that claim directly.

This framing is stronger than trying to force Codex to beat Monte Carlo on a
task where Monte Carlo is naturally well matched.
