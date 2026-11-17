# M7.1 Defect Register

Baseline commit: `729184c4d8a217d121f577243cc4b1f987e7922a`.

| ID | Severity | Source evidence | Fix | Regression test | Status |
|---|---|---|---|---|---|
| M7.1-D01 | P1 | `discovery/questions.py:134-143` field-presence scores | Replaced with evidence-gated dimensions; unsupported dimensions return `NOT_ASSESSED` | `test_m7_1_defect_regressions.py::test_value_does_not_invent_eig_or_priority` | FIXED |
| M7.1-D02 | P1 | `questions.py:144-145` default EIG arrays | Removed defaults; added explicit observation-model KL EIG | `test_eig_requires_explicit_observation_model` | FIXED |
| M7.1-D03 | P1 | `questions.py:71`, coordinator LLM prompt | Questions use abstract snippets in titles/evidence; prompts include motivation and gap | `test_m7_1_defect_regressions.py` + coordinator E2E | FIXED |
| M7.1-D04 | P1 | coordinator `experiences=[]` | Coordinator calls VerifiedMemory.retrieve and records retrieved/used/excluded trace | `test_engine_free_coordinator_end_to_end` | FIXED |
| M7.1-D05 | P1 | coordinator hypothesis artifact | Per-proposal LLM ledger: trajectory ID, prompt SHA, critic status, errors, adopted flag | coordinator E2E assertions | FIXED |
| M7.1-D06 | P1 | `novelty/assessor.py` Jaccard-only novelty | `NOVELTY_UNCERTAIN` for abstract-only no-hit; `LIKELY_NOVEL` requires full text + multi-source | `test_abstract_only_no_match_is_novelty_uncertain`, `test_full_text_multi_source_can_support_likely_novel` | FIXED |
| M7.1-D07 | P2 | M7 Critic and false report claim | Adversarial Critic added counterexample/confounder/circularity/correlation checks; G11 changed to NOT_RUN | `test_stronger_critic_checks_counterexample_and_confounders` | FIXED |
| M7.1-D08 | P0 | `experiment_protocol.json` pilot `n_families=0` | No-pilot power is `NOT_ASSESSED` and blocks scientific confirmation; real pilot effects can be injected | `test_missing_pilot_effects_force_inconclusive_verdict` | FIXED |
| M7.1-D09 | P0 | `statistics_v2.py:231-236` REJECTED with zero families | Early INCONCLUSIVE when families below prereg minimum | `test_family_confirmation_with_no_families_is_inconclusive_not_rejected` | FIXED |
| M7.1-D10a | P1 | reservations remained `bound` | Reservations consumed with real outcomes after execution | `test_formal_loader_binding_and_reservations_are_consumed` | FIXED |
| M7.1-D10b | P0 | formal Dispatcher bypass | Formal Dispatcher integration still NOT established. Added optional HarnessLoader adapter and explicit `PARTIAL_LOCAL_DIRECT_EXECUTOR` mode; no false full-integration claim. | `test_formal_loader_binding_and_reservations_are_consumed` | PARTIAL / BLOCKED FOR PRODUCTION CLAIM |

## Unresolved P0

`M7.1-D10b` remains a production-integration blocker. M7 can now accept a formal Loader and record its binding, but the default execution path still uses the local domain-pack executor and does not route through `ResearchService._run_campaign` / `CampaignRunner`. Final capability reports must mark runtime integration `PARTIAL`, not `FULL`.
