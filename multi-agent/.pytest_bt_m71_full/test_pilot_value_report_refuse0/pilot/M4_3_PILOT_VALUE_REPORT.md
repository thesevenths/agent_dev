# M4.3-04 PILOT Value Report -- m4_3

Generated: 2026-11-17T16:44:48  
Stage: **pilot**  
Protocol SHA-256: `8a6d8a3704a02aa1b3516d42ac76ad34aa77f1fbefbddd26196470964d80925f`

> **PILOT ARTEFACT.** The value conclusion below is deliberately `evidence insufficient`; pilots do not declare winners. The frozen confirmatory analysis produces `M4_3_VALUE_REPORT.md`.

## A. Experiment design

- blocks: 4; snapshots: 3
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
| n_runs | 12 |
| gate_completeness | 1.0 |
| approval_completeness | 1.0 |
| report_completeness | 1.0 |
| experience_completeness | 1.0 |
| registry_event_chain_valid | True |
| outcome_counts | {'valid_finding': 12, 'negative_finding': 0, 'inconclusive': 0, 'invalid': 0, 'failed': 0, 'interrupted': 0, 'stopped': 0} |

### Approval boundary (authorisation vs human identity)

- record-level check: the 12-step gate verified an approval RECORD that is approved and SHA-bound to the current plan (approval_source label, approval_approved_by identity string)
- approval sources (labels): {'unknown': 12}
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
| valid_research_findings_per_compute_budget | 1.0 | 1.0 | 1.0 |

### independently_reviewed_objective_quality

| contrast | n | mean diff | 95% CI | effect (dz) | p | corrected | significant |
|---|---|---|---|---|---|---|---|
| C-B | 0 | None | None | None | None | 0.0 | False |
| B-A | 0 | None | None | None | None | 0.0 | False |
| C-A | 0 | None | None | None | None | 0.0 | False |

### valid_research_findings_per_compute_budget

| contrast | n | mean diff | 95% CI | effect (dz) | p | corrected | significant |
|---|---|---|---|---|---|---|---|
| C-B | 4 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |
| B-A | 4 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |
| C-A | 4 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |

## D. Secondary metrics

| metric | A | B | C |
|---|---|---|---|
| evidence_grounding_rate | 1.0 | 1.0 | 1.0 |
| objective_novelty_rate | 0.0 | 0.0 | 0.0 |
| hypothesis_feasibility_rate | 0.0 | 0.0 | 0.0 |
| plan_validity_rate | 1.0 | 1.0 | 1.0 |
| objective_acceptance_rate | 1.0 | 1.0 | 1.0 |
| duplicate_research_rate | 0.0 | 0.0 | 0.0 |
| invalid_experiment_rate | 0.0 | 0.0 | 0.0 |
| time_to_valid_finding | 10.0 | 10.0 | 10.0 |
| compute_per_valid_finding | 1.0 | 1.0 | 1.0 |

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

- total compute units: 12.0 ({'A': 4.0, 'B': 4.0, 'C': 4.0})
- total wall time (s): 120.0 ({'A': 40.0, 'B': 40.0, 'C': 40.0})
- valid findings: 12; compute per valid finding: 1.0
- compute per objective: {'A': 1.0, 'B': 1.0, 'C': 1.0}

## G. Conclusion

**evidence insufficient**

PILOT: this report is a pilot artefact. The pilot never declares a winner; the four-way value conclusion is reserved for the frozen confirmatory analysis.

Significance claimed: **False**

Evidence chains (block -> campaign -> report -> experience):

- **C-B**
  - `b1_B` block=`b1` campaign=`camp_b1_B` report=`rep_b1_B` experience=`exp_b1_B` outcome=valid_finding gate=12/12
  - `b1_C` block=`b1` campaign=`camp_b1_C` report=`rep_b1_C` experience=`exp_b1_C` outcome=valid_finding gate=12/12
  - `b2_B` block=`b2` campaign=`camp_b2_B` report=`rep_b2_B` experience=`exp_b2_B` outcome=valid_finding gate=12/12
  - `b2_C` block=`b2` campaign=`camp_b2_C` report=`rep_b2_C` experience=`exp_b2_C` outcome=valid_finding gate=12/12
  - `b3_B` block=`b3` campaign=`camp_b3_B` report=`rep_b3_B` experience=`exp_b3_B` outcome=valid_finding gate=12/12
  - `b3_C` block=`b3` campaign=`camp_b3_C` report=`rep_b3_C` experience=`exp_b3_C` outcome=valid_finding gate=12/12
  - `b4_B` block=`b4` campaign=`camp_b4_B` report=`rep_b4_B` experience=`exp_b4_B` outcome=valid_finding gate=12/12
  - `b4_C` block=`b4` campaign=`camp_b4_C` report=`rep_b4_C` experience=`exp_b4_C` outcome=valid_finding gate=12/12
- **B-A**
  - `b1_A` block=`b1` campaign=`camp_b1_A` report=`rep_b1_A` experience=`exp_b1_A` outcome=valid_finding gate=12/12
  - `b1_B` block=`b1` campaign=`camp_b1_B` report=`rep_b1_B` experience=`exp_b1_B` outcome=valid_finding gate=12/12
  - `b2_A` block=`b2` campaign=`camp_b2_A` report=`rep_b2_A` experience=`exp_b2_A` outcome=valid_finding gate=12/12
  - `b2_B` block=`b2` campaign=`camp_b2_B` report=`rep_b2_B` experience=`exp_b2_B` outcome=valid_finding gate=12/12
  - `b3_A` block=`b3` campaign=`camp_b3_A` report=`rep_b3_A` experience=`exp_b3_A` outcome=valid_finding gate=12/12
  - `b3_B` block=`b3` campaign=`camp_b3_B` report=`rep_b3_B` experience=`exp_b3_B` outcome=valid_finding gate=12/12
  - `b4_A` block=`b4` campaign=`camp_b4_A` report=`rep_b4_A` experience=`exp_b4_A` outcome=valid_finding gate=12/12
  - `b4_B` block=`b4` campaign=`camp_b4_B` report=`rep_b4_B` experience=`exp_b4_B` outcome=valid_finding gate=12/12
- **C-A**
  - `b1_A` block=`b1` campaign=`camp_b1_A` report=`rep_b1_A` experience=`exp_b1_A` outcome=valid_finding gate=12/12
  - `b1_C` block=`b1` campaign=`camp_b1_C` report=`rep_b1_C` experience=`exp_b1_C` outcome=valid_finding gate=12/12
  - `b2_A` block=`b2` campaign=`camp_b2_A` report=`rep_b2_A` experience=`exp_b2_A` outcome=valid_finding gate=12/12
  - `b2_C` block=`b2` campaign=`camp_b2_C` report=`rep_b2_C` experience=`exp_b2_C` outcome=valid_finding gate=12/12
  - `b3_A` block=`b3` campaign=`camp_b3_A` report=`rep_b3_A` experience=`exp_b3_A` outcome=valid_finding gate=12/12
  - `b3_C` block=`b3` campaign=`camp_b3_C` report=`rep_b3_C` experience=`exp_b3_C` outcome=valid_finding gate=12/12
  - `b4_A` block=`b4` campaign=`camp_b4_A` report=`rep_b4_A` experience=`exp_b4_A` outcome=valid_finding gate=12/12
  - `b4_C` block=`b4` campaign=`camp_b4_C` report=`rep_b4_C` experience=`exp_b4_C` outcome=valid_finding gate=12/12

