# ARR Capability Final Report

## Executive Summary

ARR now has auditable benchmark infrastructure for Scientific Judgment, Research Memory and Recursive Improvement, and the pilots ran with real LLM calls and frozen raw evidence. The pilots do **not** validate the three capabilities. The appropriate overall result is `INCONCLUSIVE`.

No confirmed new scientific finding was produced. No Candidate was adopted. No production RSI success is claimed.

## Source Audit

M7.1 source audit found: field-presence value scores, default EIG, title-only gap generation, hardcoded `experiences=[]`, missing LLM decision ledger, Jaccard-driven novelty, empty pilot power, and a zero-family REJECTED bug. These were fixed with fail-closed semantics, but formal production Dispatcher integration remains partial.

## Runtime Integrity

Runtime Journal and budget reservations are wired and reservations now reach `consumed`. A formal HarnessLoader seam can be injected and is tested. The default path still bypasses `ResearchService._run_campaign` / `CampaignRunner`; therefore production runtime integration is `PARTIAL`, not full.

## Scientific Judgment Results

50-case corpus with isolated ground truth was built. A 14-case real-LLM pilot scored 14/14. Novelty accuracy was 1.000 but only within a narrow label distribution; evidence-sufficiency accuracy was 8/14 and abstention F1 was 0.75. Independent expert blind review was not performed. Verdict: `INCONCLUSIVE`.

## Research Memory Results

Four arms (OFF / ON / SHUFFLED / EQUAL_TOKEN) ran 40 decisions with matched context length. ON used relevant memories in 10/10 cases; SHUFFLED caused harmful memory use in 3/10. The `repeat_failure` metric had a construct-validity defect. Verdict: `INCONCLUSIVE`.

## Recursive Improvement Results

Two development rounds and one holdout round ran with a single changed instruction component. Development deltas were -0.333 and 0.000; the holdout delta for the second candidate was +0.250, but using it after development failure would violate the preregistered workflow. No candidate was approved or adopted. Verdict: `INCONCLUSIVE`.

## Integrated Capability Results

A/B/C/D groups used the same 7 cases and same model. Evidence accuracy: A 0.571, B 0.571, C 0.714, D 0.714. RSI-style instruction added +0.143; memory added no primary-outcome gain. Token cost increased for memory arms. n=7 is below the required 8 families; expert review absent. Verdict: `INCONCLUSIVE`.

## Verification

Final full suite: **797 passed, 12 skipped, 37 subtests passed**. Stage pilots executed real LLM calls with frozen raw results and trajectory SHAs.

## Statistical Analysis

All pilots are underpowered for formal validation. MME was pre-registered at 0.10 and was not lowered. Negative and mixed outcomes were retained. No p-value or CI should be interpreted as decision-grade at these sample sizes.

## Known Limitations

- Independent expert blind review not performed.
- Production Dispatcher/Campaign lifecycle not fully integrated.
- Memory benchmark primary metric needs repair.
- Only 7–14 pilot cases per benchmark; full 50-case scale-up not run.
- Full-text literature access not available; novelty remains corpus-limited.
- No confirmed scientific finding and no new M7 method adoption.

## Final Verdict

See `ARR_CAPABILITY_VERDICT.json`.
