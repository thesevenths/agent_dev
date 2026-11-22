# ARR_RESEARCH_CAPABILITY_03 — Source Audit & Architecture

Campaign ID: ARR_RESEARCH_CAPABILITY_03
Branch: feature/arr-research-capability-03
Date: 2026-11-22

## 1. Objective

One integrated, preregistered campaign to determine whether ARR can:
- M1: diagnose scientific failures from real experimental evidence;
- M2: select and execute interventions outperforming uninformed research;
- M3: transfer verified research knowledge to a new action-conditioned
  embodied world model (Gymnasium MuJoCo Reacher-v5).

## 2. Immutability (verified by SHA-256 before and after)

P0 / P0.1 / ARR_EMB_RSI_01 (incl. PILOT4) / ARR_EMB_RSI_02 (incl. FULL2)
code, results, reports and preregistrations were NOT modified. Their key
artifact hashes were identical at campaign start and end. All failed
attempts from this campaign are preserved (see §7).

## 3. Modules reused without modification

| Module | Role |
|---|---|
| `research_runtime/arr_emb_rsi_02/adapter.py` | real SciNet++ world-model adapter (deep-copy config fix) |
| `research_runtime/arr_emb_rsi_02/benchmark.py` | 10 validated state-based regimes (V2) |
| `research_runtime/arr_emb_rsi_02/evidence.py` | artifact claims + resolver for M2 |
| `research_runtime/hypothesis_manager.py` | real ARR LLM proposer lineage |
| `research_runtime/llm_client.py` | configured OpenAI-compatible gateway |
| `research_runtime/evolution/statistics.py` | paired t primitives |
| `research_runtime/arr_emb_rsi_01/adapter.py` | engine adapter base |

## 4. New modules (all under `research_runtime/arr_research_capability_03/`)

| File | Role |
|---|---|
| `integrity.py` | stable hashes, ground-truth leak detection, sealed GT store |
| `diagnosis.py` | REAL configured ARR LLM diagnosis with structured output, alternatives, abstention |
| `reacher.py` | real Reacher-v5 simulator trajectory generation + action-conditioned PyTorch MLP |
| `benchmark.py` | unified cause registry, M2 reuse, 6 M3 regimes, intervention registry |
| `evidence.py` / `evaluation.py` | evidence audit, diagnosis metrics (Brier/ECE/reliability/abstention), 6 verdicts |
| `statistics.py` | task-level paired t, permutation, cluster bootstrap, Holm, effect size |
| `knowledge.py` | verified research knowledge with provenance hash |
| `m2_campaign.py` / `m3_campaign.py` | state-based and Reacher campaign runners |
| `campaign.py` / `resume.py` / `finalize.py` | orchestration, resume, offline finalize |
| `prereg.py` / `report.py` / `cli.py` | frozen preregistration, canonical report, CLI |
| `tests/` | 14 engine-free unit tests |

## 5. Critical engineering facts

- The REAL configured LLM (qwen3.6-flash via the project gateway) is invoked
  for the ARR diagnosis arm. Hardcoded rules exist ONLY as a separate
  Rule-Based baseline. Every invocation records prompt/response SHA-256.
- M3 uses the REAL Gymnasium MuJoCo Reacher-v5 simulator and a REAL trained
  PyTorch MLP `f_theta(s_t, a_t) -> s_{t+1}`. No FakeRunner/MockEngine.
- The interpreter matters: M3 requires python with gymnasium
  (`/d/python313/python.exe`). An earlier full run launched with a python
  lacking gymnasium produced 90 infra-failed arms; that evidence is preserved
  and the run was redone with the correct interpreter.
- Ground truth is sealed OUTSIDE the agent-visible run root; zero leakage.

## 6. Validated failure mechanisms (real Oracle sensitivity, scouting)

| Track | Regime | Oracle relative improvement |
|---|---|---|
| M3 | E01 capacity | 0.91 |
| M3 | E02 coverage | 0.94 |
| M3 | E03 action-direction shift | 0.86 |
| M3 | E04 action-amplitude shift | 0.73 |
| M3 | E05 optimisation | 0.90 |
| M3 | E06 action conditioning | 0.94 |
| M2 | 10 V2 state regimes | previously verified in V2 |

The original naive M3 "target location shift" was rejected (Oracle 0.01 —
target coordinates do not alter Reacher dynamics); it was replaced by a real
action-distribution shift. E02 for Newton/Double pendulum remains excluded
(Oracle inversion, inherited from V2).

## 7. Preserved failed attempts (no evidence deleted)

- `ARR_RESEARCH_CAPABILITY_03_FULL`: first full run, stopped after discovering
  an evidence-construction gap (E03/E04 diagnosis); kept.
- `ARR_RESEARCH_CAPABILITY_03_FINAL`: full run with wrong python interpreter
  (M3 90 infra-failed arms: no gymnasium); kept.
- `ARR_RESEARCH_CAPABILITY_03_V2`: correct run; crashed at M3 case 13 due to
  alternatives-format bug; 12 completed M3 cases were reused via `resume.py`,
  only 6 cases re-run. Final canonical rebuilt from all 210 real runs.
- Diagnosis evidence was iteratively improved (IID/held-out gap, lr ratio,
  evaluation-context) — all refinements are derived ONLY from observed
  metrics/config, never from labels.

## 8. Known scientific limitations (honest)

- M2 E01 and E03 share the same failure config (hidden=[8,8]); the only
  difference is the evaluation split, and the held-out metrics did not worsen.
  From evidence alone the diagnosis is information-theoretically ambiguous;
  ARR misdiagnosed E03 as capacity. This is a benchmark limitation, not
  patched with hidden hints.
- Confidence remains over-confident (ECE 0.241); abstention never triggered.
- M3 Transfer vs Fresh shows no positive effect; verified knowledge did not
  improve research on the new domain.

## 9. Reproducible commands

```bash
# Environment: /d/python313/python.exe (has gymnasium 1.2.1 / mujoco 3.3.7 / torch)
cd /e/agent_dev/multi-agent
export PYTHONPATH=/e/agent_dev/multi-agent

# Unit tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  research_runtime/arr_research_capability_03/tests -q

# Freeze preregistration
python -m research_runtime.arr_research_capability_03.cli --stage prereg \
  --out ARR_RESEARCH_CAPABILITY_03_FINAL

# Full campaign (210 runs)
python -m research_runtime.arr_research_capability_03.cli --stage campaign \
  --out ARR_RESEARCH_CAPABILITY_03_V2 --seal-dir ARR_03_SEALED_V2 \
  --concurrency-m2 3 --concurrency-m3 1

# Offline finalize (recompute evidence/verdicts from recorded cases)
python -m research_runtime.arr_research_capability_03.cli --stage finalize \
  --out ARR_RESEARCH_CAPABILITY_03_V2

# Final report
python -m research_runtime.arr_research_capability_03.cli --stage report \
  --out ARR_RESEARCH_CAPABILITY_03_V2
```

## 10. Campaign identity

- Preregistration SHA-256: 6aa6ff7c824e8a75f52d96de68f16810ed963558220469bddf1126b585bc4f28
- Final canonical SHA-256: 293b5411e2a87c22a6433a486eb700d8092cd79acbf5e56a73fdf191c6ccabda
- Final report: `ARR_RESEARCH_CAPABILITY_03_V2/ARR_RESEARCH_CAPABILITY_03_FINAL_REPORT.md`
- Canonical results: `ARR_RESEARCH_CAPABILITY_03_V2/canonical_results.json`
