# ARR-07 Final Report — Autonomous Research Loop Validation

## Executive Summary

**Overall: NOT VALIDATED** · Terminal: COMPLETED_WITH_FAILURES

| Validation | Verdict | Key data |
|---|---|---|
| V1 closed-loop convergence | PARTIAL | 7/10 converged, 4 spurious |
| V2 blind vs leaky | PARTIAL | blind 6/10 vs leaky 7/10 |
| V3 memory gate repair | PASS | abstain instead of placebo |

## Architecture

LLM does COGNITION only (observe -> hypothesise -> choose experiment -> replan from measured data). A FROZEN evaluator judges goal attainment from measured improvement AND mechanism correctness; the LLM never self-certifies success. Blind mode exposes ONLY experimental measurements.

## The decisive control (V2)

Blind 60% vs leaky 70%; gap 10%. The loop works mainly from experimental data, NOT from config leakage — but reliability is still below the bar and the verdict is therefore PARTIAL, not PASS.

## Spurious convergence (V1)

4 of 7 converged runs reached the improvement target with the WRONG mechanism. Generic improvements (e.g. more data) can masquerade as scientific discovery. The frozen evaluator now catches this by requiring mechanism consistency.

## Honest conclusion

The loop RUNS end-to-end with real LLM planning, real experiments, and data-driven replanning. It is NOT yet a reliable autonomous researcher: convergence is partial, spurious convergence is frequent, and the hardest mechanisms (distribution shift) are not solved. NOT VALIDATED — do not upgrade to a research-system claim.

- Canonical SHA-256: fa6e71fd69e882512ba416d442740a4838fbe7ef35ea8f33285ce2d3f90f0d02
