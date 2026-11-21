# M4.3 — ARR Director Value Validation: Experiment Protocol (M4.3-01)

> **Scope of this stage.** This document specifies the A/B/C evaluation
> protocol and records that the harness is implemented. **M4.3-02 Pilot and
> M4.3-03 Confirmatory are NOT started.** No significance conclusion and no
> value judgement about ARR is made here. `M4_3_runs/protocol.yaml` is a
> pre-registration *shell* — all confirmatory fields are intentionally empty.

## 0. Total goal & reuse policy

Validate whether ARR's autonomous Objective Discovery actually improves
**research quality** and **research efficiency**. No new Research Planner,
ResearchService, Evidence Reader, Decision Engine, or second RRSI is created.
Everything reuses the existing `DiscoveryContextProvider`,
`ObjectiveCandidateGenerator`, `ObjectiveCritic`, `ObjectiveRanker`,
`ObjectiveDiscovery`, `ResearchDirector`, `ResearchService`, the trusted stores,
the 12-step fail-closed gate, and the existing Evidence / Report / Experience.

## 1. Groups

| Arm | Name | Historical context | Selector |
|-----|------|--------------------|----------|
| **A** | Manual | **NO** historical Experience / Evidence / Failure; current evidence only | Human |
| **B** | Memory-assisted Manual | the **SAME** historical Evidence + Experience + Failure as C | Human |
| **C** | Autonomous Discovery | the **SAME** historical context as B | `Generator + Critic + Ranker` |

A must not read historical Experience. B and C read an identical context. The
harness fails closed if arm A is handed any historical record, and asserts B/C
history equality per block.

## 2. Control rules (enforced by `EvaluationHarness`)

1. A/B/C use the **same research-opportunity Snapshot** (blocked pairing).
2. A reads no history; B/C read the same history.
3. Model version, Prompt, Ranker version and DomainPolicy are **fixed**.
4. Candidate budget and human-selection time budget are fixed per block.
5. All arms pass the **same existing 12-step fail-closed gate** (recorded via a
   read-only adapter; the harness never re-implements or bypasses it).
6. All arms require **external Approval** (`approval_bypass_count` is a Safety
   metric; see `research_runtime/APPROVAL_BOUNDARY_M4_3.md` for the honest trust
   model — `approved_by` is an unauthenticated string).
7. All arms use the same **Campaign Budget**.
8. Applicable training uses the same **Paired Seeds**.
9. **Every** outcome is retained: valid, negative, INCONCLUSIVE, invalid, failed,
   interrupted. None is dropped from the denominator.
10. The **primary metric is frozen** — no post-hoc change, no exclusion of
    unfavourable experiments.

### 2.1 Evidence & execution boundaries (M4.3-01.1 P1-02, executable)

These are enforced by `evidence_boundary.py` at `add_block()` time — they are
code, not prose. Any breach raises `EvidenceBoundaryViolation` (fail-closed).

* **Arm A** reads ONLY the current evidence of the research opportunity.
* **Arms B/C** read the SAME historical Evidence / Experience / Failure.
* All three arms share ONE research-opportunity Snapshot, one Block and one
  Seed; the Snapshot / Block / Seed hierarchy is preserved in `paired_metrics()`.
* **Current vs historical evidence are separated by EXPLICIT, DISJOINT ID
  sets** (`current_evidence_ids` / `historical_ids`). When not supplied, the
  harness derives stable IDs from the records (explicit id key, else a content
  hash) and still enforces disjointness. An explicit set that disagrees with the
  actual records is rejected (no phantom / omitted evidence).
* **No arm may read a reserved (frozen) final split.** The validator scans both
  current and historical payloads, including nested metadata and split-valued
  keys, against `RESERVED_FINAL_SPLITS` plus the frozen-split aliases in
  `protocol.yaml`. The held-out validation split remains usable.
* **Per-block budgets are fixed**: candidate budget, human-selection time
  budget and Campaign Budget are positive, recorded with the block, and (when
  the protocol freezes a Campaign Budget) must equal it exactly. A block cannot
  be re-added with a different budget.
* **Every block retains all three arms' COMPLETE records**, including failed /
  interrupted / invalid / INCONCLUSIVE outcomes. Nothing is dropped.

### 2.2 What is explicitly NOT in this stage

Pilot (M4.3-02), Confirmatory (M4.3-03) and the Value Report (M4.3-04) are NOT
started. No significance or value conclusion is drawn anywhere in the harness:
`paired_metrics()["significance_conclusion"]` is `None`. Engine-free and
real-engine results are reported separately.

## 3. Metrics (frozen vocabulary)

**Primary**
- `independently_reviewed_objective_quality`
- `valid_research_findings_per_compute_budget`

**Secondary**
`evidence_grounding_rate`, `objective_novelty_rate`,
`hypothesis_feasibility_rate`, `plan_validity_rate`,
`objective_acceptance_rate`, `duplicate_research_rate`,
`invalid_experiment_rate`, `time_to_valid_finding`,
`compute_per_valid_finding`.

**Safety**
`unsupported_evidence_claim_rate`, `false_objective_rate`,
`frozen_split_violation_count`, `approval_bypass_count`,
`duplicate_campaign_count`.

The metric set is hashed by `frozen_metric_contract()`; any change is detectable.

## 4. Independent, blind objective-quality review

- The reviewer **cannot see the arm (A/B/C) or the Ranker score**.
  `blind()` raises if `arm`, `rank_score`, `rank_breakdown`, `ranker_version`,
  `generation_source`, `snapshot_id` or `block_id` is present.
- **The Ranker never scores itself.** There is no code path from a rank score
  to a review score; `IndependentReviewer` only receives a `BlindedObjective`.
- Fixed rubric `m4_3_objective_quality_rubric_v1`, six dimensions scored `[0,4]`:
  `evidence_grounding`, `novelty`, `falsifiability`,
  `intervention_feasibility`, `metric_measurability`,
  `expected_information_gain`.
- A **reason is mandatory** for every review; scores out of range or missing
  dimensions fail closed.

## 5. Stage plan (hard boundaries)

| Stage | Status | What it does |
|-------|--------|--------------|
| M4.3-00 Closure | **complete** | P0-01..P0-06 fixes |
| M4.3-01 Harness | **complete** | this protocol + `research_runtime/m4_3/*` + tests |
| M4.3-02 Pilot | **NOT STARTED** | real SciNet++ + real history; tune flow, budget, statistics, estimate variance; **no final significance conclusion**; power analysis → sample size |
| M4.3-03 Confirmatory | **NOT STARTED** | freeze Snapshot set / primary metric / sample size / Seed / model+Prompt / budget / exclusions / statistical method / stopping condition, THEN run A/B/C with blocked pairing and pre-determined multiple-comparison correction; report A vs B, B vs C, A vs C with effect size, CI, uncertainty |
| M4.3-04 Value Report | **NOT STARTED** | answer the 7 value questions; must allow all four outcomes below |

**Pilot is explicitly not started in this stage.** The harness does not launch
Pilot or Confirmatory runs by itself; `FrozenProtocol.assert_confirmatory_ready()`
refuses a confirmatory run until every pre-registration field is frozen.

## 6. Allowed final conclusions (M4.3-04, NOT made here)

1. ARR Autonomous Discovery has significant incremental value.
2. ARR improves efficiency but not the scientific finding rate.
3. ARR shows no significant difference vs manual.
4. Current evidence is insufficient to conclude.

**Forcing a "C wins" conclusion is forbidden.** No conclusion is drawn before
the confirmatory experiment is executed.

## 7. Engineering discipline

- Branch `feature/arr-0.1-domain-registry`; no merge to main, no PR.
- Each logical change is an independent commit + push.
- No modification of historical Evidence / metrics / verdict.
- No lowering of the statistical gate; no bypass of the frozen split.
- No second ResearchService / Planner / Decision Engine.
- No FakeRunner used to impersonate the real engine; engine-free harness tests
  and real-engine E2E are **reported separately**.
- `M4_3_EXPERIMENT_PROTOCOL.md`, `M4_3_runs/protocol.yaml`,
  `research_runtime/m4_3/{metrics,experiment_registry,evaluation_harness}.py`,
  `research_runtime/tests/test_m4_3_evaluation_harness.py` are the complete
  M4.3-01 deliverable set.

## 8. The seven value questions (to be answered in M4.3-04, unanswered now)

1. Does historical Experience improve manual objective-selection quality?
2. Is Autonomous Discovery better than memory-assisted manual selection?
3. Does Autonomous Discovery reduce duplicate and invalid research?
4. Does it improve valid findings per unit budget?
5. Does it improve efficiency from Objective to valid finding?
6. Are there Objective types where the autonomous system is clearly worse?
7. Is ARR worth advancing to cross-domain M5?
