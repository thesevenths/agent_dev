# M4.3-04 PILOT Value Report -- m4_3

Generated: 2026-11-17T15:00:08  
Stage: **pilot**  
Protocol SHA-256: `53a42baa516e12f475cd05519d5bc711fcde100e6a58ce7d65de179101fdd65e`

> **PILOT ARTEFACT.** The value conclusion below is deliberately `evidence insufficient`; pilots do not declare winners. The frozen confirmatory analysis produces `M4_3_VALUE_REPORT.md`.

## A. Experiment design

- blocks: 3; snapshots: 1
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
| n_runs | 9 |
| gate_completeness | 1.0 |
| approval_completeness | 1.0 |
| report_completeness | 1.0 |
| experience_completeness | 1.0 |
| registry_event_chain_valid | True |
| outcome_counts | {'valid_finding': 0, 'negative_finding': 9, 'inconclusive': 0, 'invalid': 0, 'failed': 0, 'interrupted': 0, 'stopped': 0} |

### Approval boundary (authorisation vs human identity)

- record-level check: the 12-step gate verified an approval RECORD that is approved and SHA-bound to the current plan (approval_source label, approval_approved_by identity string)
- approval sources (labels): {'human': 9}
- approved_by identities: ['operator']
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
| C-B | 3 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |
| B-A | 3 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |
| C-A | 3 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |

## D. Secondary metrics

| metric | A | B | C |
|---|---|---|---|
| evidence_grounding_rate | 0.0 | 0.0 | 0.0 |
| objective_novelty_rate | 0.0 | 0.0 | 0.0 |
| hypothesis_feasibility_rate | 0.0 | 0.0 | 0.0 |
| plan_validity_rate | 1.0 | 1.0 | 1.0 |
| objective_acceptance_rate | 1.0 | 1.0 | 1.0 |
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

- total compute units: 9.0 ({'A': 3.0, 'B': 3.0, 'C': 3.0})
- total wall time (s): 90.0 ({'A': 30.0, 'B': 30.0, 'C': 30.0})
- valid findings: 0; compute per valid finding: None
- compute per objective: {'A': 1.0, 'B': 1.0, 'C': 1.0}

## G. Conclusion

**evidence insufficient**

PILOT: this report is a pilot artefact. The pilot never declares a winner; the four-way value conclusion is reserved for the frozen confirmatory analysis.

Significance claimed: **False**

Evidence chains (block -> campaign -> report -> experience):

- **C-B**
  - `b1_B` block=`b1` campaign=`c_b1_B` report=`rep_b1_B` experience=`e_b1_B` outcome=negative_finding gate=12/12
  - `b1_C` block=`b1` campaign=`c_b1_C` report=`rep_b1_C` experience=`e_b1_C` outcome=negative_finding gate=12/12
  - `b2_B` block=`b2` campaign=`c_b2_B` report=`rep_b2_B` experience=`e_b2_B` outcome=negative_finding gate=12/12
  - `b2_C` block=`b2` campaign=`c_b2_C` report=`rep_b2_C` experience=`e_b2_C` outcome=negative_finding gate=12/12
  - `b3_B` block=`b3` campaign=`c_b3_B` report=`rep_b3_B` experience=`e_b3_B` outcome=negative_finding gate=12/12
  - `b3_C` block=`b3` campaign=`c_b3_C` report=`rep_b3_C` experience=`e_b3_C` outcome=negative_finding gate=12/12
- **B-A**
  - `b1_A` block=`b1` campaign=`c_b1_A` report=`rep_b1_A` experience=`e_b1_A` outcome=negative_finding gate=12/12
  - `b1_B` block=`b1` campaign=`c_b1_B` report=`rep_b1_B` experience=`e_b1_B` outcome=negative_finding gate=12/12
  - `b2_A` block=`b2` campaign=`c_b2_A` report=`rep_b2_A` experience=`e_b2_A` outcome=negative_finding gate=12/12
  - `b2_B` block=`b2` campaign=`c_b2_B` report=`rep_b2_B` experience=`e_b2_B` outcome=negative_finding gate=12/12
  - `b3_A` block=`b3` campaign=`c_b3_A` report=`rep_b3_A` experience=`e_b3_A` outcome=negative_finding gate=12/12
  - `b3_B` block=`b3` campaign=`c_b3_B` report=`rep_b3_B` experience=`e_b3_B` outcome=negative_finding gate=12/12
- **C-A**
  - `b1_A` block=`b1` campaign=`c_b1_A` report=`rep_b1_A` experience=`e_b1_A` outcome=negative_finding gate=12/12
  - `b1_C` block=`b1` campaign=`c_b1_C` report=`rep_b1_C` experience=`e_b1_C` outcome=negative_finding gate=12/12
  - `b2_A` block=`b2` campaign=`c_b2_A` report=`rep_b2_A` experience=`e_b2_A` outcome=negative_finding gate=12/12
  - `b2_C` block=`b2` campaign=`c_b2_C` report=`rep_b2_C` experience=`e_b2_C` outcome=negative_finding gate=12/12
  - `b3_A` block=`b3` campaign=`c_b3_A` report=`rep_b3_A` experience=`e_b3_A` outcome=negative_finding gate=12/12
  - `b3_C` block=`b3` campaign=`c_b3_C` report=`rep_b3_C` experience=`e_b3_C` outcome=negative_finding gate=12/12

