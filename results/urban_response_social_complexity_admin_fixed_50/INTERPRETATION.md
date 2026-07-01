# Urban Response Social-Complexity Result

Generated: 2026-07-01

This run is the qualitative-only urban sociotechnical study after fixing the
Codex-admin zero-intensity social-action issue. It uses 50 paired runs, 180
steps, shared 25-step control updates, and 24 Monte Carlo samples.

## Ranking

Lower is better.

| Policy | Score | Physical | Human | Interpretation |
|---|---:|---:|---:|---|
| `q_structured_human_state_monte_carlo` | 49.339 | 3.317 | 4.536 | Strong reference with privileged latent human-state access. |
| `q_codex_monte_carlo_qualitative_admin` | 50.359 | 3.448 | 4.532 | Best deployable Codex architecture in this urban run. |
| `q_codex_qualitative` | 50.748 | 3.491 | 4.543 | Direct Codex control; improves over deployable non-Codex baselines. |
| `q_monte_carlo_tactical` | 51.487 | 3.235 | 5.121 | Physical Monte Carlo only; better physical score but worse human state. |
| `q_keyword_monte_carlo` | 51.487 | 3.235 | 5.121 | Weak lexical baseline; no benefit over physical Monte Carlo. |
| `q_baseline` | 51.865 | 3.696 | 4.460 | No social intervention; poor physical outcome. |

## Key Comparisons

- `q_codex_monte_carlo_qualitative_admin` vs best deployable non-Codex:
  mean delta `-1.128`, 95% CI `[-2.862, 0.607]`, win rate `0.72`.
- `q_codex_qualitative` vs best deployable non-Codex:
  mean delta `-0.739`, 95% CI `[-2.508, 1.030]`, win rate `0.70`.
- `q_structured_human_state_monte_carlo` vs best deployable non-Codex:
  mean delta `-2.148`, 95% CI `[-4.342, 0.046]`, win rate `0.70`.

The confidence intervals still cross zero for the Codex-vs-deployable
comparisons, so this is supportive rather than decisive statistical evidence.

## Admin Fix Check

The previous caveat was that Codex-admin often selected a plausible social
action with `social_intensity: 0.0`, turning it into a no-op. This run fixes
that issue.

- `q_codex_monte_carlo_qualitative_admin` selected 265 social actions.
- Zero selected social actions had zero intensity.
- Mean admin social intensity was approximately `0.826`.
- Codex-admin now ranks above direct Codex.

## Paper Interpretation

This result reinforces the supply-chain sabotage conclusion directionally:

> Codex is not expected to dominate pure numeric control, but it becomes useful
> when qualitative sociotechnical reports must be interpreted and translated
> into bounded interventions.

For the paper, treat this urban result as cross-domain support, not the primary
positive result. The supply-chain qualitative result remains the strongest
evidence because `q_codex_monte_carlo_qualitative_admin` wins decisively there.
The urban result shows the same pattern more weakly: Codex-admin and direct
Codex improve over deployable non-Codex qualitative baselines, while the
privileged structured human-state baseline remains the upper reference.

