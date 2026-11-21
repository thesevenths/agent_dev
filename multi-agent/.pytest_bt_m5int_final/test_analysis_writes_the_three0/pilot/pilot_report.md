# M4.3 Pilot Report -- m4_3_pilot

Generated: 2026-11-16T17:36:38  
Protocol SHA-256: `8a6d8a3704a02aa1b3516d42ac76ad34aa77f1fbefbddd26196470964d80925f`  
Metric contract: `3138acff1ef629bcc95802109ae43bb60da6b6076c231282aeac0465567e5ae4`  
Blocks: **3** (eligible for pairing: 3) / Runs: **9**

> This is a PILOT report. It computes descriptive statistics and does not claim significance. The confirmatory value conclusion belongs to `M4_3_VALUE_REPORT.md`.

## 1. Completeness and real-environment evidence

| check | value |
|---|---|
| gate_pass_rate | 1.0 |
| external_approval_rate | 1.0 |
| report_rate | 1.0 |
| experience_rate | 1.0 |
| campaign_rate | 1.0 |
| finalized_rate | 1.0 |
| failure_rate | 0.0 |
| interruption_rate | 0.0 |
| invalid_rate | 0.0 |
| inconclusive_rate | 0.0 |
| approval_bypass_count | 0 |
| outcome_counts | {'valid_finding': 0, 'negative_finding': 9, 'inconclusive': 0, 'invalid': 0, 'failed': 0, 'interrupted': 0, 'stopped': 0} |

## 2. All 16 metrics per arm

| metric | A | B | C |
|---|---|---|---|
| independently_reviewed_objective_quality | 0.0 | 0.0 | 0.0 |
| valid_research_findings_per_compute_budget | 0.0 | 0.0 | 0.0 |
| evidence_grounding_rate | 1.0 | 1.0 | 1.0 |
| objective_novelty_rate | 0.0 | 0.0 | 0.0 |
| hypothesis_feasibility_rate | 0.0 | 0.0 | 0.0 |
| plan_validity_rate | 1.0 | 1.0 | 1.0 |
| objective_acceptance_rate | 1.0 | 1.0 | 1.0 |
| duplicate_research_rate | 0.0 | 0.0 | 0.0 |
| invalid_experiment_rate | 0.0 | 0.0 | 0.0 |
| time_to_valid_finding | n/a | n/a | n/a |
| compute_per_valid_finding | n/a | n/a | n/a |
| unsupported_evidence_claim_rate | 0.0 | 0.0 | 0.0 |
| false_objective_rate | 0.0 | 0.0 | 0.0 |
| frozen_split_violation_count | 0 | 0 | 0 |
| approval_bypass_count | 0 | 0 | 0 |
| duplicate_campaign_count | 0 | 0 | 0 |

Runs per arm: A=3, B=3, C=3

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
| b1 | snap_1 | 42 | negative_finding | negative_finding | negative_finding | True |
| b2 | snap_2 | 43 | negative_finding | negative_finding | negative_finding | True |
| b3 | snap_3 | 44 | negative_finding | negative_finding | negative_finding | True |


## 6. Economics

- compute units: 9.0 ({'A': 3.0, 'B': 3.0, 'C': 3.0})
- wall time (s): 90.0 ({'A': 30.0, 'B': 30.0, 'C': 30.0})
- valid findings: 0

## 7. Sample-size recommendation (confirmatory planning)

- status: **evidence_insufficient**
- blocks recommended (most conservative across BOTH primary metrics): None
- recommendation rule: max over all required primary metrics
- alpha per contrast: 0.01666667 (power 0.8, inflation 1.25)
- **independently_reviewed_objective_quality** (pre-registered SEOI=0.5): status=evidence_insufficient, variance_source=none, n=None
    - B-A: variance_status=no_human_review, n=None, variance=None
    - C-B: variance_status=no_human_review, n=None, variance=None
    - C-A: variance_status=no_human_review, n=None, variance=None
- **valid_research_findings_per_compute_budget** (pre-registered SEOI=0.01): status=evidence_insufficient, variance_source=none, n=None
    - B-A: variance_status=zero_variance, n=None, variance=None
    - C-B: variance_status=zero_variance, n=None, variance=None
    - C-A: variance_status=zero_variance, n=None, variance=None
- NOT estimated (metric::contrast / zero variance / no human review / missing pairs): ['independently_reviewed_objective_quality::B-A', 'independently_reviewed_objective_quality::C-B', 'independently_reviewed_objective_quality::C-A', 'valid_research_findings_per_compute_budget::B-A', 'valid_research_findings_per_compute_budget::C-B', 'valid_research_findings_per_compute_budget::C-A']

## 8. Limitations

- only 3 eligible block(s): descriptive, not inferential
- objective-quality reviews are not complete: at least two independent HUMAN reviewers must submit before that metric (and any contrast on it) can be reported
- A/B objectives were selected by the recorded, pre-registered delegated-operator protocol, not by independent human researchers; this weakens any claim about human-vs-automation selection

## 9. Conclusion

**evidence_insufficient** -- the primary objective-quality metric has no complete human blind review, so no value claim can be made

Significance claimed: **False**

