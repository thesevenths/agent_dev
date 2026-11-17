# M7.4 Recursive Improvement Benchmark Specification

## Method-change model

Each round changes exactly one component: `scientific_judgment_instruction`. Candidate instructions add evidence-gate or adversarial-counterexample requirements. Model, tools, token budget, cases and scoring remain fixed.

## Splits

Split assignment is frozen before results are inspected:

- Development: KNOWN / PRIOR_ART / UNFALSIFIABLE cases.
- Holdout: OPEN / COMPETING cases.
- OOD: CONTRADICTORY / INSUFFICIENT cases.

Case IDs, not seeds, define family membership. Holdout is never used for Candidate selection.

## Preregistered decision

- Primary outcome: evidence-sufficiency accuracy.
- MME: 0.10 absolute paired improvement.
- Minimum families: 8.
- Test: paired family sign test; Holm correction for secondary outcomes.
- Parse failures score zero and are retained.
- No early stopping.

## Promotion states

`PROPOSED → EXPERIMENTED → HOLDOUT_PASSED → APPROVED → ADOPTED → POST_ADOPTION_VERIFIED` or `REJECTED / INCONCLUSIVE / ROLLED_BACK`.

Human approval is never automated. If no candidate is adopted, post-adoption is `NOT_RUN` and rollback is `NOT_APPLICABLE_NO_ADOPTION`.
