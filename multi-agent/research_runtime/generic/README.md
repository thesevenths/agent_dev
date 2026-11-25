# Generic research infrastructure — fixing ARR-07's two problems generically

## The two problems, stated domain-independently

ARR-07 found: (1) a benchmark can register two experiments that claim different
mechanisms but apply the same intervention, making the mechanism label
unfalsifiable; (2) after reaching a preregistered improvement, the agent chased
residual error and degraded the result.

Neither problem is specific to Reacher. Both were fixed as plain-dict / plain-number
modules with **no dependency on any benchmark, domain, or ARR version**.

## What was built

- `identifiability.py` — given any `(task_id, base_config)` pairs and any
  `RegisteredExperiment(id, claimed_mechanism, overrides)`, deep-merges each
  override and hashes the effective config. Experiments producing identical
  configs on a task are reported as a collision; their mechanism labels are
  marked UNFALSIFIABLE. Evaluators should then accept the whole equivalence
  class, not demand one arbitrary label.
- `stopping.py` — frozen `GoalCriterion(target, max_iterations, min_consecutive)`.
  `GoalGate.observe()` returns stop/continue from measured numbers only; once
  stopped it is sticky (no residual chasing). Optional `mechanism_hit_required`.

## Cross-domain validation (same code, zero changes)

| Domain | Contexts | Experiments | Collisions found |
|---|---:|---:|---|
| Reacher world model | 15 | 5 | 12 (EXP_ACT ≡ EXP_DIST) — the real ARR-07 bug |
| JEPA-like representation | 6 | 4 | 0 — experiments genuinely distinguishable |
| AI4S-like molecular | 6 | 4 | 2 (EXP_NORM ≡ EXP_SPLIT) — same bug, new domain |

Stop rule on ARR-07's actual numbers `0.13 → 0.91 → 0.16 → −0.03`: stops at
iteration 1 (0.91 ≥ 0.3). The original loop continued into 0.16/−0.03.

## Honest boundary

The generic modules DETECT non-identifiability and ENFORCE stopping. They do
not invent a physics-specific new experiment that separates two mechanisms —
that still needs domain knowledge (e.g. for JEPA: reweight training toward the
held-out target distribution; for AI4S: a scaffold-split + normalized-feature
combination). The fix is therefore reusable as a guardrail across domains, but
the creation of new discriminating experiments remains domain-informed.

No fabricated outcomes: the cross-domain validation is structural (config
level). The only measured numbers quoted (0.8274 MSE etc.) come from the real
Reacher probe already committed in ARR-07.
