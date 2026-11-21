# M4.3-04 PILOT Value Report -- m4_3_pilot

Generated: 2026-11-17T18:31:29  
Stage: **pilot**  
Protocol SHA-256: ``

> **PILOT ARTEFACT.** The value conclusion below is deliberately `evidence insufficient`; pilots do not declare winners. The frozen confirmatory analysis produces `M4_3_VALUE_REPORT.md`.

## A. Experiment design

- blocks: 0; snapshots: 0
- arm A: Current-evidence delegated baseline: the pre-registered delegated-operator rule selects an Objective from CURRENT evidence only, WITHOUT historical Experience. NOT an independent human researcher (unless a real human selection record is supplied).
- arm B: History-assisted delegated baseline: the pre-registered delegated-operator rule selects an Objective reading the SAME historical Evidence / Experience / Failure context as C. NOT an independent human researcher.
- arm C: Autonomous Objective Discovery: ObjectiveCandidateGenerator + Critic + Ranker select the Objective automatically.
- model / prompt / ranker / policy: `` / `` / `` / ``
- paired seeds: []; budget: {}
- exclusion rules: []
- statistical method:  (correction: holm)
- primary contrast: ['C-B']; secondary: ['B-A', 'C-A']

## B. Data integrity

| check | value |
|---|---|
| n_runs | 0 |
| gate_completeness | 0.0 |
| approval_completeness | 0.0 |
| report_completeness | 0.0 |
| experience_completeness | 0.0 |
| registry_event_chain_valid | True |
| outcome_counts | {'valid_finding': 0, 'negative_finding': 0, 'inconclusive': 0, 'invalid': 0, 'failed': 0, 'interrupted': 0, 'stopped': 0} |

### Approval boundary (authorisation vs human identity)

- record-level check: the 12-step gate verified an approval RECORD that is approved and SHA-bound to the current plan (approval_source label, approval_approved_by identity string)
- approval sources (labels): {}
- approved_by identities: []
- CLI subprocess grants detected: 0
- independent human approval confirmed: **False**
- identity authentication confirmed: **False**
- claim: approval evidence proves a SHA-bound CLI subprocess grant under the identity label(s) listed above; it does NOT prove an independent, authenticated human approver

## C. Primary metrics

Objective quality (HUMAN-BLIND; missing values shown as `missing`, never as a 0 score):

| arm | n reviewed | value | status |
|---|---|---|---|
| A | 0 | missing | missing_no_human_review |
| B | 0 | missing | missing_no_human_review |
| C | 0 | missing | missing_no_human_review |

| metric (raw contract values) | A | B | C |
|---|---|---|---|
| independently_reviewed_objective_quality | 0.0 | 0.0 | 0.0 |
| valid_research_findings_per_compute_budget | 0.0 | 0.0 | 0.0 |

### independently_reviewed_objective_quality

| contrast | n | mean diff | 95% CI | effect (dz) | p | corrected | significant |
|---|---|---|---|---|---|---|---|
| C-B | 0 | None | None | None | None | 0.0 | False |
| B-A | 0 | None | None | None | None | 0.0 | False |
| C-A | 0 | None | None | None | None | 0.0 | False |

### valid_research_findings_per_compute_budget

| contrast | n | mean diff | 95% CI | effect (dz) | p | corrected | significant |
|---|---|---|---|---|---|---|---|
| C-B | 0 | None | None | None | None | 0.0 | False |
| B-A | 0 | None | None | None | None | 0.0 | False |
| C-A | 0 | None | None | None | None | 0.0 | False |

## D. Secondary metrics

| metric | A | B | C |
|---|---|---|---|
| evidence_grounding_rate | 0.0 | 0.0 | 0.0 |
| objective_novelty_rate | 0.0 | 0.0 | 0.0 |
| hypothesis_feasibility_rate | 0.0 | 0.0 | 0.0 |
| plan_validity_rate | 0.0 | 0.0 | 0.0 |
| objective_acceptance_rate | 0.0 | 0.0 | 0.0 |
| duplicate_research_rate | 0.0 | 0.0 | 0.0 |
| invalid_experiment_rate | 0.0 | 0.0 | 0.0 |
| time_to_valid_finding | None | None | None |
| compute_per_valid_finding | None | None | None |

## E. Safety

| metric | A | B | C |
|---|---|---|---|
| unsupported_evidence_claim_rate | 0.0 | 0.0 | 0.0 |
| false_objective_rate | 0.0 | 0.0 | 0.0 |
| frozen_split_violation_count | 0 | 0 | 0 |
| approval_bypass_count | 0 | 0 | 0 |
| duplicate_campaign_count | 0 | 0 | 0 |

Counts: {'unsupported_evidence_claim': 0, 'false_objective': 0, 'frozen_split_violation': 0, 'approval_bypass': 0, 'duplicate_campaign': 0}

## F. Economics

- total compute units: 0 ({'A': 0, 'B': 0, 'C': 0})
- total wall time (s): 0 ({'A': 0, 'B': 0, 'C': 0})
- valid findings: 0; compute per valid finding: None
- compute per objective: {'A': None, 'B': None, 'C': None}

## G. Conclusion

**evidence insufficient**

no persisted runs: nothing to report

Significance claimed: **False**

Evidence chains (block -> campaign -> report -> experience):

- **C-B**
- **B-A**
- **C-A**

