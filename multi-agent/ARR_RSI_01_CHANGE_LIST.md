# ARR-RSI-01 P0 — File-Level Change List (pre-coding)

Audit date: 2026-11-21. Scope: implement the minimum viable ARR↔RSI experimental loop
without creating a parallel multi-agent framework.

## Reused (NO changes)

| Capability | Existing module |
|---|---|
| Research entrypoint | `research_runtime/arr_runtime.py:493 run_research_task` |
| Task/result schema | `research_runtime/research_task.py:47 ResearchTask`, `:117 TaskResult` |
| Diagnosis rules | `research_runtime/failure_diagnoser.py:152 FailureDiagnoser` |
| Experiment runner | `research_runtime/campaign/runner.py`, `experiment_manager.py` |
| Budget | `research_runtime/domain_budget_manager.py`, `loop_budget_ledger.py` |
| Paired statistics / gates | `research_runtime/harness_rsi/evaluation.py:234 analyse_paired` |
| Promotion | `research_runtime/harness_rsi/promotion.py` |
| Experience store | `research_runtime/harness_rsi/experience.py:68 ExperienceBank` |
| Harness versioning | `research_runtime/harness_rsi/harness.py` + `arr_runtime.py:60 ResearchHarness` |
| SciNet++ engine | `/e/LLM/SciNet++/research_api/cli.py` via `adapters/scinet_adapter.py` |
| Contamination guards | `harness_rsi/benchmark_v2.py:424 HoldoutAccessVerifier`, `m4_3/evidence_boundary.py` |

## Genuine gaps (P0 only)

1. No 20-task deterministic scientific benchmark with the 5 required families.
2. No pluggable `Diagnosis` interface returning the 10 required fields.
3. No Placebo / Shuffled-Diagnosis attribution arms.
4. No role-level ground-truth isolation for Research/Diagnosis agents.
5. No independent evaluator producing the 6 required dimensions + 4-valued verdict.
6. No single artifact emitter for the 8 required deliverables.

## New files (only these)

- `research_runtime/arr_rsi_01/__init__.py`
- `research_runtime/arr_rsi_01/tasks.py` — 20 tasks, sealed GT, isolation guard
- `research_runtime/arr_rsi_01/diagnosis.py` — Diagnosis interface + evidence-grounded + shuffled
- `research_runtime/arr_rsi_01/arms.py` — Base/Candidate/Placebo/Shuffled, diagnosis traceability
- `research_runtime/arr_rsi_01/evaluation.py` — independent 6-dimension evaluation + verdict
- `research_runtime/arr_rsi_01/runner.py` — real-runtime orchestration + artifacts
- `research_runtime/arr_rsi_01/cli.py` — run-one / pilot / full entrypoints
- `research_runtime/arr_rsi_01/tests/test_arr_rsi_01.py` — unit tests

## Root-level deliverables (generated, not hand-written source)

- `ARR_RSI_01_REPORT.md`, `ARR_RSI_01_RUN/experiment_manifest.json`,
  `task_manifest.json`, `trajectory.json`, `diagnosis.json`,
  `intervention.json`, `results.json`, `statistical_analysis.json`

## Explicit non-changes

No edits to existing production modules; no new agent roles; no orchestration
framework; no model-weight modification; no unrestricted code modification;
no knowledge graph, dashboard or P1 work.

## Known engineering limitation (disclosed, not hidden)

All 20 P0 tasks execute on the two frozen, already-verified SciNet++ smoke
configs (`pendulum_smoke.yaml`, `pendulum_smoke_ood.yaml`). Family labels
encode distinct research objectives/failure modes, not distinct data
generators. A derived `newton` smoke config was tested against the real engine
and failed (`worker_interrupted`), so no unverified engine source is used.
Consequently any claim of cross-family scientific generalization is BLOCKED;
the verdict will reflect only the evidence actually produced.

## Post-run finding (fixed for subsequent runs, full pilot NOT re-run)

SciNet++ derives job IDs from the research task ID. Re-running the same task in
a later experiment collided with the earlier pilot's job IDs, which contributed
to 13/80 failed arms. `runner.build_research_task` now adds a run-scoped nonce
to the research task ID. The recorded full pilot is preserved unchanged and its
BLOCKED verdict was not retroactively upgraded.
