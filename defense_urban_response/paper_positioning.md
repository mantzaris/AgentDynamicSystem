# Paper Positioning Notes

Candidate paper framing:

> LLM-guided intervention policies for defense-relevant urban agent-based
> simulations on road-network maps.

The scenario can be described as an abstract hostile-contagion response problem:
an urban road network contains hostile contagious walkers, civilians, and a
limited defender force with partial observability. The objective is rapid
neutralization while preserving civilians and response-force capacity.

Why this is more defense-relevant than the ecological benchmark:

- road-network movement rather than a flat cellular grid;
- civilians, hostile agents, and defenders are distinct agent classes;
- defender visibility and engagement radius model partial observability;
- outputs include populated map states that resemble operational simulation
  products;
- objective is response effectiveness, not ecological stability.

Useful venue framing:

- JDMS: modeling and simulation method for defense-relevant intervention policy
  comparison.
- I/ITSEC: training/simulation decision-support benchmark with LLM-guided
  tactic selection.

Current result to cite:

- `results/urban_response_social_complex_large_semantic_50/`
- Numeric tactical branch: `codex_guardian` ranks first but does not
  significantly beat the best non-Codex controller, `rule_based`.
- Qualitative semantic branch:
  `q_codex_monte_carlo_qualitative_admin` ranks first, decisively beats the
  best deployable non-Codex qualitative baseline, and is statistically
  comparable to the privileged structured-human-state controller.

Paper framing should emphasize this split: Codex is not a universal numeric
controller, but it is valuable as a semantic administrator when qualitative
reports must be interpreted into bounded interventions.

Important caveat:

The bundled road file is a development scaffold with approximate coordinates.
For submission, use a proper OSM extraction and cite OpenStreetMap contributors
if the map remains central to the paper.
