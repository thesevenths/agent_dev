# M4.3 Pilot Report -- m4_3_pilot

Generated: 2026-11-16T12:17:44  
Protocol SHA-256: `2b206e999d6abab5bb5449c5ee18fcaff2ceded64f060f543fbf5a191f8075b6`  
Metric contract: `3138acff1ef629bcc95802109ae43bb60da6b6076c231282aeac0465567e5ae4`  
Blocks: **4** (eligible for pairing: 3) / Runs: **12**

> This is a PILOT report. It computes descriptive statistics and does not claim significance. The confirmatory value conclusion belongs to `M4_3_VALUE_REPORT.md`.

## 1. Completeness and real-environment evidence

| check | value |
|---|---|
| gate_pass_rate | 0.75 |
| external_approval_rate | 0.75 |
| report_rate | 0.75 |
| experience_rate | 0.75 |
| campaign_rate | 0.75 |
| finalized_rate | 0.833333 |
| failure_rate | 0.0 |
| interruption_rate | 0.166667 |
| invalid_rate | 0.083333 |
| inconclusive_rate | 0.0 |
| approval_bypass_count | 0 |
| outcome_counts | {'valid_finding': 0, 'negative_finding': 9, 'inconclusive': 0, 'invalid': 1, 'failed': 0, 'interrupted': 2} |

## 2. All 16 metrics per arm

| metric | A | B | C |
|---|---|---|---|
| independently_reviewed_objective_quality | 0.0 | 0.0 | 0.0 |
| valid_research_findings_per_compute_budget | 0.0 | 0.0 | 0.0 |
| evidence_grounding_rate | 1.0 | 1.0 | 0.75 |
| objective_novelty_rate | 0.0 | 0.0 | 0.0 |
| hypothesis_feasibility_rate | 0.0 | 0.0 | 0.0 |
| plan_validity_rate | 0.75 | 0.75 | 0.75 |
| objective_acceptance_rate | 0.75 | 0.75 | 0.75 |
| duplicate_research_rate | 0.0 | 0.0 | 0.0 |
| invalid_experiment_rate | 0.0 | 0.0 | 0.25 |
| time_to_valid_finding | n/a | n/a | n/a |
| compute_per_valid_finding | n/a | n/a | n/a |
| unsupported_evidence_claim_rate | 0.0 | 0.0 | 0.0 |
| false_objective_rate | 0.0 | 0.0 | 0.0 |
| frozen_split_violation_count | 0 | 0 | 0 |
| approval_bypass_count | 0 | 0 | 0 |
| duplicate_campaign_count | 0 | 0 | 0 |

Runs per arm: A=4, B=4, C=4

## 3. Paired contrasts (block-level, descriptive)

### B-A (B vs A)

| metric | n | mean diff | sd | variance | bootstrap 95% CI |
|---|---|---|---|---|---|
| independently_reviewed_objective_quality | 0 | None | None | None | n/a (n<2) |
| valid_research_findings_per_compute_budget | 3 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] |

### C-B (C vs B)

| metric | n | mean diff | sd | variance | bootstrap 95% CI |
|---|---|---|---|---|---|
| independently_reviewed_objective_quality | 0 | None | None | None | n/a (n<2) |
| valid_research_findings_per_compute_budget | 3 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] |

### C-A (C vs A)

| metric | n | mean diff | sd | variance | bootstrap 95% CI |
|---|---|---|---|---|---|
| independently_reviewed_objective_quality | 0 | None | None | None | n/a (n<2) |
| valid_research_findings_per_compute_budget | 3 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] |

## 4. Objective quality (human blind review)

Review status: **pending_or_partial** -- human blind reviews recorded: 0, automated scores (kept separate): 0

| arm | n reviewed | mean | sd | distribution |
|---|---|---|---|---|
| A | 0 | None | None | [] |
| B | 0 | None | None | [] |
| C | 0 | None | None | [] |

## 5. Blocks

| block | snapshot | seed | A | B | C | gate ok |
|---|---|---|---|---|---|---|
| pilot_block_1 | arr_m34_001::block_1 | 42 | negative_finding | negative_finding | negative_finding | True |
| pilot_block_2 | arr_m34_001::block_2 | 43 | negative_finding | negative_finding | negative_finding | True |
| pilot_block_3 | arr_m34_001::block_3 | 44 | negative_finding | negative_finding | negative_finding | True |
| pilot_block_4 | arr_m34_001::block_4 | 45 | interrupted | interrupted | invalid | False |

Excluded blocks: [{"block_id": "pilot_block_4", "reasons": ["not all arms finalized"]}]

## 6. Economics

- compute units: 9.0 ({'A': 3.0, 'B': 3.0, 'C': 3.0})
- wall time (s): 2019.87 ({'A': 642.8, 'B': 646.97, 'C': 730.1})
- valid findings: 0

## 7. Sample-size recommendation (confirmatory planning)

- status: **evidence_insufficient**
- blocks recommended: None
- alpha per contrast: 0.01666667 (power 0.8, inflation 1.25)
- NOT estimated (insufficient pilot data): ['B-A', 'C-B', 'C-A']

## 8. Limitations

- only 3 eligible block(s): descriptive, not inferential
- objective-quality reviews are not complete: at least two independent HUMAN reviewers must submit before that metric (and any contrast on it) can be reported
- A/B objectives were selected by the recorded, pre-registered delegated-operator protocol, not by independent human researchers; this weakens any claim about human-vs-automation selection

## 9. Conclusion

**evidence_insufficient** -- the primary objective-quality metric has no complete human blind review, so no value claim can be made

Significance claimed: **False**

