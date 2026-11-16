# M4.3-04 PILOT Value Report -- m4_3_pilot

Generated: 2026-11-16T12:17:44  
Stage: **pilot**  
Protocol SHA-256: `2b206e999d6abab5bb5449c5ee18fcaff2ceded64f060f543fbf5a191f8075b6`

> **PILOT ARTEFACT.** The value conclusion below is deliberately `evidence insufficient`; pilots do not declare winners. The frozen confirmatory analysis produces `M4_3_VALUE_REPORT.md`.

## A. Experiment design

- blocks: 4; snapshots: 4
- arms: A manual / B memory-assisted / C autonomous discovery
- model / prompt / ranker / policy: `` / `` / `` / ``
- paired seeds: []; budget: {}
- exclusion rules: []
- statistical method:  (correction: holm)
- primary contrast: ['C-B']; secondary: ['B-A', 'C-A']

## B. Data integrity

| check | value |
|---|---|
| n_runs | 12 |
| gate_completeness | 0.75 |
| approval_completeness | 0.75 |
| report_completeness | 0.75 |
| experience_completeness | 0.75 |
| registry_event_chain_valid | True |
| outcome_counts | {'valid_finding': 0, 'negative_finding': 9, 'inconclusive': 0, 'invalid': 1, 'failed': 0, 'interrupted': 2} |

Missing trusted ids per run:

- pilot_block_4_A (A, interrupted): ['plan', 'approval', 'campaign', 'report', 'experience']
- pilot_block_4_B (B, interrupted): ['plan', 'approval', 'campaign', 'report', 'experience']
- pilot_block_4_C (C, invalid): ['plan', 'approval', 'campaign', 'report', 'experience']

## C. Primary metrics

| metric | A | B | C |
|---|---|---|---|
| independently_reviewed_objective_quality | 0.0 | 0.0 | 0.0 |
| valid_research_findings_per_compute_budget | 0.0 | 0.0 | 0.0 |

### independently_reviewed_objective_quality

| contrast | n | mean diff | 95% CI | effect (dz) | p | corrected | significant |
|---|---|---|---|---|---|---|---|
| B-A | 0 | None | None | None | None | None | False |
| C-B | 0 | None | None | None | None | None | False |
| C-A | 0 | None | None | None | None | None | False |

### valid_research_findings_per_compute_budget

| contrast | n | mean diff | 95% CI | effect (dz) | p | corrected | significant |
|---|---|---|---|---|---|---|---|
| B-A | 4 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |
| C-B | 4 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |
| C-A | 4 | 0.0 | [0.0, 0.0] | None | 1.0 | 0.0 | False |

## D. Secondary metrics

| metric | A | B | C |
|---|---|---|---|
| evidence_grounding_rate | 1.0 | 1.0 | 0.75 |
| objective_novelty_rate | 0.0 | 0.0 | 0.0 |
| hypothesis_feasibility_rate | 0.0 | 0.0 | 0.0 |
| plan_validity_rate | 0.75 | 0.75 | 0.75 |
| objective_acceptance_rate | 0.75 | 0.75 | 0.75 |
| duplicate_research_rate | 0.0 | 0.0 | 0.0 |
| invalid_experiment_rate | 0.0 | 0.0 | 0.25 |
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
- total wall time (s): 2019.87 ({'A': 642.8, 'B': 646.97, 'C': 730.1})
- valid findings: 0; compute per valid finding: None
- compute per objective: {'A': 0.75, 'B': 0.75, 'C': 0.75}

## G. Conclusion

**evidence insufficient**

PILOT: this report is a pilot artefact. The pilot never declares a winner; the four-way value conclusion is reserved for the frozen confirmatory analysis.

Significance claimed: **False**

Evidence chains (block -> campaign -> report -> experience):

- **C-B**
  - `pilot_block_1_B` block=`pilot_block_1` campaign=`campaign_b2194a9a09ae50b7` report=`report_ef3027e866d8` experience=`exp_bfc83e6a37f5` outcome=negative_finding gate=12/12
  - `pilot_block_1_C` block=`pilot_block_1` campaign=`campaign_a1cbe67a6acd574b` report=`report_4662252263b7` experience=`exp_dc49a33cbad9` outcome=negative_finding gate=12/12
  - `pilot_block_2_B` block=`pilot_block_2` campaign=`campaign_62f9c52bfbe255a7` report=`report_1d647286fd1d` experience=`exp_e3eeac65b1d6` outcome=negative_finding gate=12/12
  - `pilot_block_2_C` block=`pilot_block_2` campaign=`campaign_261edb0d7ef95ab5` report=`report_b7c9efd63851` experience=`exp_4847dfd84c33` outcome=negative_finding gate=12/12
  - `pilot_block_3_B` block=`pilot_block_3` campaign=`campaign_3bd725cf5ecf5ead` report=`report_16a4c87a367e` experience=`exp_7aa940df26e3` outcome=negative_finding gate=12/12
  - `pilot_block_3_C` block=`pilot_block_3` campaign=`campaign_599ef73b8add58dd` report=`report_6aeee3d74952` experience=`exp_be6c3ac7233f` outcome=negative_finding gate=12/12
  - `pilot_block_4_B` block=`pilot_block_4` campaign=`` report=`` experience=`` outcome=interrupted gate=0/12
  - `pilot_block_4_C` block=`pilot_block_4` campaign=`` report=`` experience=`` outcome=invalid gate=0/12
- **B-A**
  - `pilot_block_1_A` block=`pilot_block_1` campaign=`campaign_49c4c465ad3d5526` report=`report_b06860829d72` experience=`exp_385336ee0163` outcome=negative_finding gate=12/12
  - `pilot_block_1_B` block=`pilot_block_1` campaign=`campaign_b2194a9a09ae50b7` report=`report_ef3027e866d8` experience=`exp_bfc83e6a37f5` outcome=negative_finding gate=12/12
  - `pilot_block_2_A` block=`pilot_block_2` campaign=`campaign_ffb29548cdd15684` report=`report_4f84bbf482a0` experience=`exp_4adebf421eed` outcome=negative_finding gate=12/12
  - `pilot_block_2_B` block=`pilot_block_2` campaign=`campaign_62f9c52bfbe255a7` report=`report_1d647286fd1d` experience=`exp_e3eeac65b1d6` outcome=negative_finding gate=12/12
  - `pilot_block_3_A` block=`pilot_block_3` campaign=`campaign_2975a6b0e4715dea` report=`report_bb26b6a2c2e7` experience=`exp_62e5a35410d0` outcome=negative_finding gate=12/12
  - `pilot_block_3_B` block=`pilot_block_3` campaign=`campaign_3bd725cf5ecf5ead` report=`report_16a4c87a367e` experience=`exp_7aa940df26e3` outcome=negative_finding gate=12/12
  - `pilot_block_4_A` block=`pilot_block_4` campaign=`` report=`` experience=`` outcome=interrupted gate=0/12
  - `pilot_block_4_B` block=`pilot_block_4` campaign=`` report=`` experience=`` outcome=interrupted gate=0/12
- **C-A**
  - `pilot_block_1_A` block=`pilot_block_1` campaign=`campaign_49c4c465ad3d5526` report=`report_b06860829d72` experience=`exp_385336ee0163` outcome=negative_finding gate=12/12
  - `pilot_block_1_C` block=`pilot_block_1` campaign=`campaign_a1cbe67a6acd574b` report=`report_4662252263b7` experience=`exp_dc49a33cbad9` outcome=negative_finding gate=12/12
  - `pilot_block_2_A` block=`pilot_block_2` campaign=`campaign_ffb29548cdd15684` report=`report_4f84bbf482a0` experience=`exp_4adebf421eed` outcome=negative_finding gate=12/12
  - `pilot_block_2_C` block=`pilot_block_2` campaign=`campaign_261edb0d7ef95ab5` report=`report_b7c9efd63851` experience=`exp_4847dfd84c33` outcome=negative_finding gate=12/12
  - `pilot_block_3_A` block=`pilot_block_3` campaign=`campaign_2975a6b0e4715dea` report=`report_bb26b6a2c2e7` experience=`exp_62e5a35410d0` outcome=negative_finding gate=12/12
  - `pilot_block_3_C` block=`pilot_block_3` campaign=`campaign_599ef73b8add58dd` report=`report_6aeee3d74952` experience=`exp_be6c3ac7233f` outcome=negative_finding gate=12/12
  - `pilot_block_4_A` block=`pilot_block_4` campaign=`` report=`` experience=`` outcome=interrupted gate=0/12
  - `pilot_block_4_C` block=`pilot_block_4` campaign=`` report=`` experience=`` outcome=invalid gate=0/12

