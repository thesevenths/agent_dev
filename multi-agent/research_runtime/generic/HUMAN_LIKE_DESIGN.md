# How humans design experiments and stop — and what is now implemented

## Q1: How does a human design a NEW experiment, and why couldn't the agent?

Human procedure (abstracted, from actual practice):
1. Hold two competing hypotheses H1 and H2 about a failure.
2. Ask: "what intervention would give DIFFERENT observable results under H1 vs H2?"
3. Run that intervention; the outcome eliminates one hypothesis.

The key property is DISCRIMINATIVE POWER, not "pick the best-looking experiment".

Why the old agent could not do it: in ARR-07 `benchmark.EXPERIMENTS` is a
HARDCODED menu of 5 experiments. The LLM can only choose one entry; there is no
operator composition and no predicted-signature model. Worse, two menu entries
(EXP_ACT / EXP_DIST) applied an identical intervention on 12 tasks — the menu
literally could not separate the mechanisms it claimed to test.

What was added (`experiment_design.py`, domain-agnostic):
- `Hypothesis(id, statement, predicted_signature)` — what each mechanism predicts.
- `Operator(id, apply, parameter_space)` — available manipulations (domain input).
- `design_discriminative(h1, h2, operators)` — finds a manipulation whose predicted
  signature differs between hypotheses and returns a `DesignedExperiment`.
- `check_unfalsifiable(hypotheses)` — flags pairs no manipulation can distinguish.

Demonstrated on the REAL ARR-07 collision: identifiability audit finds
`EXP_ACT ≡ EXP_DIST`; the two hypotheses predict different `train_loss` and
`ood_error` directions; the module designs
`SEP_action_conditioning_vs_distribution_shift` with separation 0.67.

HONEST BOUNDARY: the search and the falsifiability check are generic. Supplying
the predicted signature and a physically meaningful operator still requires
domain knowledge. The module cannot invent physics out of nothing; it can tell
you whether your knowledge is sufficient to separate two explanations.

## Q2: Why not borrow human/tool-call stopping?

We now did. `stopping.py` (frozen GoalCriterion + sticky GoalGate) was previously
written but NOT WIRED into the loop — a real integration gap. It is now wired
into `arr07/loop.py`: after each measured experiment the loop consults the gate;
it stops on preregistered improvement OR iteration budget (the tool-call limit
analogue). The LLM cannot override this.

Measured effect on the real failing case (distribution_shift v30):
- before: ran 4 iterations, final improvement −0.026 (degraded from 0.914)
- after: stopped at iteration 2 with 0.914

Across the 9 tasks that reached the goal, immediate stopping was better in 1
(the degradation case) and neutral in 8 — so the rule prevents harm without
changing most outcomes. This is an honest effect size, not a universal win.

## Summary

| Capability | Before | Now |
|---|---|---|
| Design a new discriminating experiment | impossible (fixed menu) | generic designer + falsifiability check |
| Detect two labels that cannot be separated | manual post-hoc probe | automatic identifiability audit |
| Stop at goal / tool budget | only a bare for-loop; no early stop | frozen sticky gate wired into loop |
| Domain-independence | n/a | all modules in `research_runtime/generic/`, zero benchmark imports |
