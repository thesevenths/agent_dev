# ARR-RSI-01 P0.2-01 — Benchmark Independence & Integrity Audit

Experiment: `ARR-RSI-01-P0.2` (draft) · Branch: `feature/arr-rsi-01-p0-2-causal-validation`
Audit scope: current repository + `/e/LLM/SciNet++` at `c6f66c3`.
Status of this document: **AUDIT / NOT A PREREGISTRATION**. It blocks the full experiment until its findings are resolved.

## 1. Immutability baseline (P0 / P0.1)

Measured before any P0.2 work; these files must remain byte-identical:

| Artifact | SHA-256 |
|---|---|
| `multi-agent/ARR_RSI_01_REPORT.md` | `086ef26c243d6eb93b7fb9c0ab9a79fa66afa8ab0397bf5e94a492e157bb452c` |
| `multi-agent/ARR_RSI_01_RUN_FULL/results.json` | `3b28b17ffbe0bb5e7c830f50a490be514b0fcb0fd14eb4668dab57a2f0c7c0c9` |
| `multi-agent/ARR_RSI_01_P0_1_FINAL_REPORT.md` | `b98c59f7b4113ddaa729050f810d8c8012a1e90f7d624a695c98e9cff2ad7455` |
| `multi-agent/ARR_RSI_01_P0_1_FULL/results.json` | `a05179efaa85a4b2401b0e4fcc9f9ad3c8d9d9d8a52e2b481e4a6fd0a57e2f54` |
| `multi-agent/ARR_RSI_01_P0_1_FULL/p0_1_protocol.json` | `b12e6405eeaa4f1f088fbe2b937de6fae60601ca13b02e2011de4f236dd53753` |

No P0/P0.1 file was modified. P0.2 adds a new branch and new files only.

## 2. What P0.1 actually measured

- P0.1 final verdict: **INCONCLUSIVE**.
- Diagnosis root-cause accuracy: **12/20 (0.60)**, below its own 0.8 gate.
- Candidate vs Base: Δ=+0.016885, Holm p=0.013641 (significant, below MME 0.02).
- Candidate vs Placebo: Δ=+0.004517, p=0.1939.
- Candidate vs Shuffled: Δ=+0.000384, p=0.9406.
- Evidence refs resolved: 160/160; infrastructure failures: 0; complete-case: 20/20.

Interpretation: engineering integrity was largely repaired, but the benchmark did **not** establish that evidence-based diagnosis adds information over shuffled diagnosis.

## 3. Benchmark Independence Audit — result

P0.1 declared five "independent units" (`family|dataset`). Source audit shows this is **false independence**:

| Declared unit | Real config | Real frozen dataset | Generator | Independent? |
|---|---|---|---|---|
| pendulum_dynamics | `configs/pendulum_smoke.yaml` | `pendulum_smoke_v1` | pendulum | baseline |
| newtonian_dynamics | `configs/pendulum_smoke.yaml` | `pendulum_smoke_v1` | pendulum | **NO** — label only |
| symbolic_regression | `configs/pendulum_smoke.yaml` | `pendulum_smoke_v1` | pendulum | **NO** — label only |
| robot_dynamics | `configs/pendulum_smoke.yaml` | `pendulum_smoke_v1` | pendulum | **NO** — label only |
| scientific_ml | `configs/pendulum_smoke_ood.yaml` | `pendulum_smoke_ood_v1` | pendulum | distinct regime, same generator |

**Verdict: P0.1's five units collapse to at most two real regimes, both from the same `pendulum` generator (ID `t_max=8.0` vs OOD `t_max=14.0`). Seeds 42–45 are repeated measurements, not independent scientific units. Family names are research-question labels, not generators.**

Consequence: P0.1's "5 independent units" claim is rejected for P0.2. Treating its 20 runs as 20 independent observations would be pseudo-replication and P-hacking.

## 4. Real engine sources available for P0.2

Audited in `/e/LLM/SciNet++` (`research_api/runner.py`, `data/dataset.py`, `configs/`, `datasets/`):

- `research_api.cli` supports stages `train`, `evaluate`, `analyze`, `symbolic`, `tta`.
- `data/dataset.py` registers distinct generators: `pendulum`, `double_pendulum`, `newton`.
- Frozen datasets with train/validation/adapt/test manifests:
  - `datasets/pendulum_smoke_v1` (`pendulum-a0e03386be89`)
  - `datasets/pendulum_smoke_ood_v1` (`pendulum-a884d327551d`)
- Other dataset directories (`controlled_pendulum_pilot`, `controlled_pendulum_v2_pilot`, `pusher_v1`) have no split manifests and are driven by separate stage scripts, not the standard `train`/`evaluate` adapter path. Not eligible without new integration (out of scope).

Candidate independent scientific regimes for P0.2:

| Regime | Generator / data identity | Engine status |
|---|---|---|
| R1 pendulum ID | `pendulum`, `pendulum_smoke_v1` | VERIFIED from P0/P0.1 |
| R2 pendulum OOD | `pendulum`, `pendulum_smoke_ood_v1` | VERIFIED from P0/P0.1 |
| R3 Newton dynamics | `newton` generator, `configs/newton.yaml` | **UNVERIFIED** in this adapter; a derived small config failed in P0 with `worker_interrupted`. Must be smoke-verified before use. |
| R4 double pendulum | `double_pendulum` generator, `configs/double_pendulum.yaml` | **UNVERIFIED** in this adapter; must be smoke-verified before use. |

**Maximum realistic independent generators = 3** (`pendulum`, `newton`, `double_pendulum`), with at most 2 verified pendulum regimes today. If R3/R4 cannot be verified, P0.2 cannot support a PASS-capable full experiment and must stop **BLOCKED**.

## 5. Ground Truth Isolation audit

P0.1 improvements:
- `PublicTask.agent_view()` removes family, failure mode, sealed intervention labels/values, hidden-test ref, engine config, seed.
- Ground truth is sealed in `ground_truth.sealed.json` with SHA verification and read only by `evaluation.py`.
- Tests assert recursive `LEAK_KEYS` checks on agent-facing payloads.

Remaining P0.2 requirements (not yet implemented):
1. Seal file must live **outside** the run root visible to the Research/Diagnosis agent (P0.1 currently writes it inside the run directory).
2. Evidence-based diagnosis must receive only an evidence package (artifact records + hashes + metric snapshots), never the whole task object or path to the seal.
3. Random / Shuffled / Rule-Based / Oracle controllers must each have an explicit access contract; Oracle receives GT by design and is evaluator-side only.
4. Hidden test result must not enter any agent-facing trajectory or diagnosis input.

Status: **PARTIAL — isolation not yet sufficient for P0.2**.

## 6. Artifact-Level Evidence Resolver audit

P0.1 `_resolve_refs` (in `evaluation.py`) currently resolves a ref when:
- the suffix is an existing path, or
- the ref starts with a known prefix (`baseline_id:`, `job_id:`, `tool_call:`, etc.).

This is **not** artifact-level validation. It does not verify file hash, does not bind a claim to an artifact, and cannot detect a fabricated-but-existing file.

P0.2 requires a resolver that, for every evidence claim:
1. resolves to a concrete file (path + canonical run id);
2. recomputes SHA-256 and compares with the recorded hash;
3. binds the claim text/metric to that artifact (claim hash);
4. rejects if any link is missing, tampered, or cross-run.

Status: **GAP — must be implemented before Diagnosis Benchmark**.

## 7. Power / feasibility (draft)

Paired t-test power at α=0.05, two-sided, by number of **independent** units (computed with noncentral t):

| dz | n=3 | n=5 | n=6 | n=8 | n=10 |
|---|---|---|---|---|---|
| 0.8 | 0.13 | 0.28 | 0.36 | 0.50 | 0.62 |
| 1.0 | 0.18 | 0.40 | 0.51 | 0.68 | 0.80 |
| 1.2 | 0.23 | 0.53 | 0.66 | 0.83 | 0.92 |
| 1.5 | 0.32 | 0.71 | 0.83 | 0.95 | 0.99 |

With only 2–3 real independent units, P0.2 has **no statistical power to support PASS** for any causal claim. A PASS-capable design needs at least 6 independent units for a large effect (dz≥1.5) or 10 for dz≈1.0. If the engine cannot supply at least 6 verified independent regimes, the correct outcome is **BLOCKED**, not INCONCLUSIVE-by-low-power or a relaxed criterion.

## 8. Audit verdict

| Check | Result |
|---|---|
| P0/P0.1 immutability | PASS (hashes recorded, untouched) |
| P0.1 five-unit independence claim | **FAIL** — 2 real regimes, same generator |
| Seed as independent unit | **FAIL** in P0.1; must never be repeated in P0.2 |
| Generator independence | UNKNOWN — 3 generators exist, only pendulum verified |
| GT isolation | PARTIAL — must move seal outside agent-visible root |
| Artifact resolver with hash/claim binding | **FAIL — GAP** |
| Power for PASS-capable full experiment | **FAIL at current verified unit count** |

**P0.2 gate:** do not implement the full 150-run experiment and do not claim any scientific capability until (a) independent regimes are verified, (b) GT isolation and artifact resolver gaps are closed, and (c) a frozen preregistration with adequate power is accepted.
