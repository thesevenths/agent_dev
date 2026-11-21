# M7.1 Source Audit

Audit baseline: commit `729184c4d8a217d121f577243cc4b1f987e7922a` (feature/arr-m7-scientific-discovery). All line references refer to this commit unless stated otherwise.

## Scope

Audited source and artifacts:

- `research_runtime/discovery/questions.py`
- `research_runtime/discovery/novelty/assessor.py`
- `research_runtime/discovery/hypotheses/engine.py`
- `research_runtime/discovery/discovery_coordinator.py`
- `research_runtime/discovery/memory/verified.py`
- `research_runtime/discovery/experiment/designer.py`
- `research_runtime/discovery/domain_packs/embodied_world_model/evaluator.py`
- `research_runtime/harness_rsi/statistics_v2.py`
- `M7_EVIDENCE/run_v4/artifacts/*.json`
- `research_runtime/build_m7_report.py`
- `M7_FINAL_ACCEPTANCE_REPORT.md`

## Confirmed defects at baseline

| ID | Severity | Evidence | Finding |
|---|---|---|---|
| M7.1-D01 | P1 | `discovery/questions.py:134-143` | Research Value scores are mostly field-presence constants. Generated questions always populate those fields, so the scores have no discriminative power. |
| M7.1-D02 | P1 | `discovery/questions.py:144-145`; `discovery_coordinator.py:336-337` | EIG uses default `[0.5,0.5]` → `[0.9,0.1]`; output is constant `0.531` in run_v4. VALUE_ASSESSMENT runs before hypothesis generation, so it cannot be “computed from competing hypotheses”. |
| M7.1-D03 | P1 | `discovery/questions.py:71`; `discovery_coordinator.py:362` | Question titles are generic (`literature gap #N`) and LLM prompts use only the title. Real gap text is not included in the hypothesis prompt. |
| M7.1-D04 | P1 | `discovery_coordinator.py:322-324`; `memory/verified.py:248-282` | `experiences=[]` is hardcoded; VerifiedResearchMemory has retrieve() but discovery never calls it. Historical experience cannot influence question generation. |
| M7.1-D05 | P1 | `discovery_coordinator.py:354-405`; `artifacts/hypotheses.json` | LLM proposal / critic decision / actual execution are not recorded per hypothesis. `llm_used` conflates transport success with adoption. |
| M7.1-D06 | P1 | `novelty/assessor.py:84-145` | Novelty relies on token Jaccard with unvalidated thresholds. Empty relation set plus sufficient coverage becomes `POTENTIALLY_NOVEL`, which overstates evidence from abstract-only retrieval. |
| M7.1-D07 | P2 | `hypotheses/engine.py:88-117`; `build_m7_report.py:91` | M7 uses its own `IndependentCritic`; the M7 report claimed M6 Critic was reused and hardcoded G11 as PASS. Generator/critic separation is structurally valid, but the report claim was false. |
| M7.1-D08 | P0 | `experiment/designer.py:69-73`; `artifacts/experiment_protocol.json` | Pilot power analysis is `n_families=0 / too_few_families`; no real pilot effects are passed. Planned 8 families is a default constant. |
| M7.1-D09 | P0 | `build_m7_report.py:51-93`; `statistics_v2.py:231-236` | Gate PASS labels conflate process completion with scientific confirmation. `analyse_family_confirmation` can return `REJECTED` when `n_fam=0`, contradicting its own fail-closed contract. Family-level confirmation is not wired into M7 replication. |
| M7.1-D10 | P0 | See `M7_1_RUNTIME_INTEGRATION_AUDIT.md` | M7 execution bypasses formal M5 Dispatcher, Campaign Lifecycle and M6 HarnessLoader. Reservations remain `bound`, never consumed. |

## Honest baseline interpretation

The M7 run_v4 evidence supports **engineering loop completion**, not Scientific Judgment, Research Memory or Recursive Improvement validation.

The baseline report’s `NO_CONFIRMED_NEW_FINDING` conclusion remains correct and must not be revised upward.
