# ARR_EMB_RSI_02 — Source Audit

Branch: `feature/arr-emb-rsi-02-blind-validation` · Base: `41d9309b7` · Date: 2026-11-22
Experiment ID: `ARR_EMB_RSI_02`

## 1. Immutability baseline

ARR_EMB_RSI_01 (including PILOT4 and all failed attempts), P0 and P0.1 branches/artifacts are immutable. V2 adds new modules under `research_runtime/arr_emb_rsi_02/` and new artifacts only. No V1 code, report, preregistration or result was modified.

## 2. Required corrections implemented

| Requirement | Finding / correction |
|---|---|
| Actual ARR Research Agent | `ARR.LLMProposer` is invoked with project-configured `LLM_*` gateway (`qwen3.6-flash` at `192.168.50.46:8888/v1`) via `llm_client.make_generate()`. Preflight produced a schema-valid hypothesis with prompt/response SHA-256. Hardcoded rules remain only in the Rule-Based baseline. |
| Diagnostic shortcuts removed | Agent receives only an opaque regime id, neutral objective, normalized metrics/curves, budget and verified artifact refs. It never sees failure ID, mechanism, generator, dataset path, root-cause label or OOD label. |
| GT isolation | Sealed GT is derived from mechanism and stored outside the agent-visible run root. Agent view is recursively checked against leak keys. |
| Adapter shallow-copy bug | V1 `build_config` used `dict(source_config)` (shallow). V2 adapter subclasses V1 and uses `copy.deepcopy` for both main and OOD snapshots; main config cannot be mutated by OOD config generation. |
| Statistics | V2 uses paired t with small-sample t critical values, preregistered bootstrap/permutation, Holm, clustered/unit-level inference. Seeds are repeated measurements. |
| Confidence evaluation | Replace 3-case ECE shortcut with standard mean-confidence-per-bin ECE, Brier score, reliability diagram and bootstrap CIs; calibration assessed over all diagnosis cases, not 3. |
| Evidence evaluation | Separate artifact integrity, citation validity, evidence relevance and sufficiency. |
| Canonical results | Report and JSON are generated from ONE canonical results object. |
| Same failure baseline | All arms start from `regime.common + regime.failure_override`; only arm intervention overrides differ. |
| Random control | Reproducible `random.Random(seed)` draws a valid intervention from the registered space; no fixed hash producing one deterministic outcome per task. |
| Budget matching | All arms use identical epochs, batch size, jobs and timeout; agent/random/oracle differ only in selected parameter values. |

## 3. Reusable modules

- Real engine adapter: `research_runtime/adapters/scinet_adapter.py` (unmodified).
- ARR LLM proposer: `research_runtime/hypothesis_manager.LLMProposer` (unmodified).
- LLM config: `config.py`, `research_runtime/llm_client.py` (unmodified).
- Engine: SciNet++ `research_api.cli`, generators `pendulum`, `newton`, `double_pendulum`.
- Stats primitives: `research_runtime/evolution/statistics.py`, `harness_rsi/evaluation.py`.

## 4. Generator compatibility

- Pendulum: previously verified in V1.
- Newton: initially failed with `ZeroDivisionError` from inherited full-config `batch_size=1024 > 64` samples. After `train.batch_size=16`, real train/evaluate succeeded (`newton-7e41e961ec97`, mse@1≈0.00394).
- Double pendulum: same fix, real train/evaluate succeeded (`double_pendulum-352db3916b05`, mse@1≈0.01564).
- Conclusion: 3 generator families are available; 12 regimes = 3 generators × 4 mechanisms is valid if Oracle sensitivity passes.

## 5. Actual ARR Agent preflight

- `LLMProposer` with configured model produced a schema-valid hypothesis from normalized evidence.
- First two attempts failed due prompt/schema mismatch (nested intervention object, wrong `predicted_effects` shape); both failures are recorded in `ARR_EMB_RSI_02_AGENT_PREFLIGHT*` artifacts. Prompt was then corrected to the real Hypothesis schema; third attempt succeeded.
- This is evidence the real configured model/runtime path is invoked, not a hardcoded rule.

## 6. Preflight decision rule

Full campaign is authorized ONLY if:
1. actual agent invocation succeeds;
2. GT leak = 0;
3. all 12 regimes have distinct frozen datasets and fixed hidden-test hashes;
4. Oracle produces a measurable effect for every mechanism/regime;
5. configs applied 100%;
6. budget controlled;
7. power analysis supports the frozen MME within 192 runs.

Any failure => STOP and report BLOCKED.
