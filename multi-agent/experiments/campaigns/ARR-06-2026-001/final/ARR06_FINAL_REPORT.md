# ARR-06 Final Report — Research Evidence, Verified Memory & Recursive Methodology Validation

## Executive Summary

Overall: **PARTIALLY VALIDATED** · Certification: **L4** · Terminal: COMPLETED_WITH_FAILURES

| Layer | Verdict |
|---|---|
| certification_level | L4 |
| engineering | PASS |
| independent_audit | PASS |
| overall | PARTIALLY VALIDATED |
| reproducibility | PASS |
| rq1_belief_calibration | PASS |
| rq2_research_selection | PASS |
| rq3_memory_transfer | INCONCLUSIVE |
| rq4_methodology_evolution | INCONCLUSIVE |
| rq5_rq_discovery | PASS |
| source_integrity | PASS |

## Source & Determinism

- ARR-05 replay issue: 13/43 matched
- Root cause identified: weight-init order in ARR-03 MLP.fit
- ARR-06 deterministic conditions: 4, all deterministic: True

## Reproducibility Levels (R0-R3)

{
  "R0_artifact_integrity": true,
  "R1_statistical_recalculation": true,
  "R2_experiment_replay": true,
  "R3_full_campaign_reproduction": true,
  "n_records": 457
}

## RQ1 Belief Update & Calibration

- Cases: 40; accuracy 0.9000; ECE 0.0821; IG 0.7929
- LLM confidence used: False (frozen Bayesian rule)

## RQ2 Information-Gain Research Selection

- Units: 30; ARR vs Random 0.7023, sig True

## RQ3 Verified Memory & Applicability Transfer

- Gated vs Fresh -0.4257; negative transfer 0.0000

## RQ4 Recursive Methodology

- Promotions: 0; curve {'heldout': [0.9333333333333333, 0.9333333333333333, 0.9333333333333333], 'replication': [0.9333333333333333, 0.9333333333333333, 0.9333333333333333]}

## RQ5 Research Question Discovery

- ARR vs Random 0.7004; falsifiability 0.9400; novelty UNVERIFIED

## Independent Audit

{
  "claims": true,
  "integrity": true,
  "replay": true,
  "statistics": true
}

## Budget

{
  "formal": 405,
  "llm": 60,
  "max_formal": 700,
  "max_llm": 600,
  "max_preflight": 80,
  "max_replay": 120,
  "preflight": 12,
  "replay": 80
}

- Canonical SHA-256: d6daf9fb21e8ee107e8210a10ff4e937d5d80926a4769f32fe527c18e8fdba21
