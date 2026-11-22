# ARR_EMB_RSI_01 — Phase 0 Source Audit

Branch: `feature/arr-embodied-rsi-01` · Base: `4ff27ba63` · Date: 2026-11-22
Scope: read-only audit of `agent_dev/multi-agent`, `LLM/SciNet++`, existing ARR Runtime, Engine, World Model, Dataset Generator, Training/Evaluation Pipeline, Artifact Storage, CLI, Statistics.

## 1. Reusable modules (no modification required)

| Capability | Module | Reuse decision |
|---|---|---|
| Real engine adapter | `research_runtime/adapters/scinet_adapter.py` `SciNetAdapter` | Reuse as-is. Supports `submit/status/result/wait/list/freeze/health_check`, real subprocess JSON CLI, run-id, retry-safe job lookup. |
| Engine spec/result schema | `research_runtime/schemas.py` `ExperimentSpec`, `ExperimentResult` | Reuse as-is. |
| Engine wiring | `research_runtime/settings.py` | Reuse `resolve_engine`, `DEFAULT_SCINET_DIR`, `WAIT_TIMEOUT_S`. Override timeout in new adapter. |
| Statistical primitives | `research_runtime/evolution/statistics.py` | Reuse paired t, CI, Holm; pure stdlib. |
| Paired analysis | `research_runtime/harness_rsi/evaluation.py` | Reuse `analyse_paired`, `PreRegistration`, clustered/permutation/bootstrap helpers where applicable. |
| ARR evidence pattern | `research_runtime/arr_rsi_01/*` | Read only as reference. No P0/P0.1 code or artifact will be modified. |
| Config verification pattern | `research_runtime/candidate_config.py` | Reuse idea; implement new registry in new module to avoid touching historical policy. |

## 2. Can ARR invoke the real SciNet++ engine?

**YES.** `SciNetAdapter` launches `<SCINET_PYTHON> -m research_api.cli ...` in `<SCINET_DIR>`; verified in P0/P0.1 with hundreds of real jobs. Engine CLI stages: `train`, `evaluate`, `analyze`, `symbolic`, `tta`; dataset freeze available.

Engine identity at audit: SciNet++ branch `feature/p2-03a-action-jepa`, HEAD `c6f66c3`; runtime Python `D:\conda_envs\scinet_gpu\python.exe` (torch 2.5.1+cu121 verified earlier).

## 3. Can engine configuration be modified and verified?

**YES, with a new config path.** `research_api.cli submit --config <path>` accepts absolute/relative YAML. Train/evaluate record `dataset_version`, `run_id`, config path, metrics and artifacts. `evaluate.py` reports actual split and dataset version. A new adapter will write per-run configs, apply registered intervention keys, and re-read the actual executed config to verify application.

Existing whitelist in `campaign/scinet.py` only covers six model/training keys; new benchmark needs dataset/coverage/horizon keys. We will add a **new** registry in `arr_emb_rsi_01`, not modify historical code.

## 4. Does the dataset generator support controlled failure injection?

**PARTIAL / mechanism-dependent.**

- Real generators in `data/dataset.py`: `pendulum`, `double_pendulum`, `newton`.
- Frozen split manifest: `pendulum_smoke_v1` (ID, `t_max=8.0`) and `pendulum_smoke_ood_v1` (OOD, `t_max=14.0`) both verified in P0/P0.1.
- `data.frozen.build_frozen_dataset` can materialise immutable train/validation/adapt/test splits with SHA-256 and `dataset_version`.
- `data.dataset.dataset_spec` shows dataset identity depends on generator, sequence/block size, `t_max`, dataset seed and per-split sample counts. Therefore changing `train_samples` mints a new dataset version; to keep the evaluation split fixed, the new dataset must reuse the baseline test artifacts with declared hashes verified.
- Capacity and rollout-horizon failure injection can be done via model/V-JEPA config without changing data identity.
- Distribution shift can use the two existing pendulum regimes (ID train → OOD eval).

**Action-conditioned limitation:** the pendulum smoke data has no action channel. The "Action-Conditioned World Model" research domain cannot be claimed with this phase; only state-based dynamics/world-model diagnosis is testable. Visual JEPA/Reacher remain out of scope. This must be reported as a scope limitation, not silently ignored.

## 5. IID and OOD evaluation

**YES.** `research_api.cli evaluate --split validation|test` and a separate OOD config path are used by existing `domains/scinet.scinet_compute` to evaluate the same checkpoint on a different frozen regime. P0/P0.1 already executed holdout and OOD confirmation paths.

## 6. Multi-step prediction

**YES.** `pendulum_smoke.yaml` sets `jepa.rollout_steps: [1,2,3]`; `evaluate.py` emits `latent_mse@k` and `latent_cos@k`. P0/P0.1 recorded these metrics.

## 7. Real training metrics / model artifacts

**YES.** Engine result includes status, `run_id`, `dataset_version`, metrics, artifacts and checkpoint references. Outputs are persisted under `outputs/<experiment>/<run_id>/`. Artifact hashes can be computed by the new adapter.

## 8. Independent evaluation

**YES, implementable.** Evaluation can be a separate module receiving only sealed evidence and recorded arm results; ground truth sealed outside agent-visible run root. Statistical analysis can reuse existing pure-Python paired/CI/Holm primitives.

## 9. Budget enforcement

**YES, but needs new adapter-level enforcement.** Existing campaign policy has `max_jobs`, `max_seeds`, `max_total_seconds`; new adapter will record per-arm job count, retries, wall time and abort on budget breach. No budget difference between arms will be allowed.

## 10. Scope decisions from audit

- Reuse `SciNetAdapter`; do not refactor ARR Runtime; do not modify P0/P0.1 artifacts.
- New module only: `research_runtime/arr_emb_rsi_01/`.
- Phase 1 smoke must prove real engine execution, config application, metric retrieval, artifact hash, run id, retry and budget.
- If coverage/dataset intervention cannot keep the test split fixed and hash-verified, E02 must be marked BLOCKED rather than faked.
- If Oracle cannot produce a measurable effect for a failure mode, that failure mode is invalid and must be reported BLOCKED.
- Seeds are repeated measurements, never independent scientific units.
