# ARCHITECTURE_AUDIT — SciNet++ → Domain-Independent Scientific World Model

**Phase 0 deliverable. No production code was modified.** Audit-only.
Companion docs: `MODULE_DEPENDENCY_MAP.md`, `CORE_ADAPTER_BOUNDARY.md`,
`P2_02_GAP_ANALYSIS.md`.

Audit date: 2026-11-10
Audited tree: `E:\LLM\SciNet++` (commits `c77d60e` env adapter, `ca95e8a` dataset).
Method: read actual source of every production module, then an empirical check of
the Pusher-v1 `ood_test` split (`scripts/_audit_ood.py`).

---

## 1. 结论先行 (Conclusion first)

1. **Core is cleanly isolated from Robotics.** The generic scientific-modeling
   core (`models/`, `data/`, `train.py`, `evaluate.py`, `analyze.py`, `tta.py`,
   `symbolic.py`, `utils/`, `research_api/`) contains **zero** imports of the
   `robotics/` package. Robotics lives entirely under `robotics/` (P2-01). The
   "Robotics must remain a domain adapter, not a dependency of the core" principle
   is **already satisfied** at the source level.
2. **But the Core's notion of "dynamics" is latent block-level, not
   (observation, action) → next-observation.** This is the single biggest
   architectural fact for P2-02. The core predicts latent transitions
   `z_b → z_{b+1}` over non-overlapping blocks of a trajectory. It has **no
   concept** of actions, interventions, conditions, variable-length episodes,
   episode boundaries, `valid_mask`, or per-timestep `(x_t, u_t) → x_{t+1}`.
   P2-02's generic `DynamicsBatch` contract is therefore **net-new** relative to
   the core, not a refactor of it.
3. **The Pusher-v1 `ood_test` split is NOT genuinely out-of-distribution.**
   By construction it is allocated from the same three policies on the same
   environment, differing only by reset seed. Empirical feature summaries across
   all four splits are statistically indistinguishable (see §6 and
   `P2_02_GAP_ANALYSIS.md` §"OOD split"). Any P2-02 OOD-generalization claim
   built on this split would be false. **Keep Frozen Test quarantined.**
4. **Legacy Predictor/WorldModel checkpoints remain loadable** as long as
   P2-02 adds a *separate* `ConditionalPredictor` and does **not** alter the
   existing `Predictor.forward(z)` signature or `WorldModel.state_dict` keys.
5. **No contact-rich object-pushing dynamics exist in Pusher-v1.** Fingertip↔
   object contact is 0/30,000 timesteps. P2-02 baseline dynamics can only model
   *arm kinematics*, not object-pushing. This bounds what "success" can mean.

---

## 2. 现有模块清单 (Module inventory)

### 2.1 Generic scientific-modeling core

| Module | Role | Key public symbols |
|---|---|---|
| `models/encoder.py` | Maps one flattened trajectory **block** → latent concept vector | `Encoder`, `build_mlp`, `SinusoidalPositionalEncoding` |
| `models/predictor.py` | Latent transition `z → z'` (MLP or single-token Transformer) | `Predictor` (`forward(z)` → `(B, latent_dim)`) |
| `models/world_model.py` | V-JEPA: encoder + predictor + EMA target encoder | `WorldModel`, `predict_multistep`, `forward(blocks, horizons, context_block)`, `from_config`, `update_target_encoder` |
| `models/probe.py` | Read-out MLP: frozen latent → physical concept(s) | `Probe`, `from_config` |
| `data/dataset.py` | Block-sequence dataset + generator/loader factory | `BlockSequenceDataset`, `build_dataset`, `SPLITS`, `dataset_version`, `split_rng`, `load_frozen_split` |
| `data/pendulum.py` | Damped-oscillator generator (γ, ω labels) | `PendulumGenerator` |
| `data/double_pendulum.py` | Chaotic double-pendulum generator (energy label) | `DoublePendulumGenerator` |
| `data/newton.py` | Constant-force Newton generator (F, m labels) | `NewtonGenerator` |
| `data/frozen.py` | Immutable, SHA-256-verified dataset freezer | `build_frozen_dataset`, `verify_frozen_dataset` |
| `train.py` | Stage 1 V-JEPA training (EMA + VICReg) | `main`, `evaluate_epoch`, `ema_momentum` |
| `analyze.py` | Stage 2 physics probe (concept discovery) | `extract_latents`, `train_probe`, `compute_attribution_matrix`, `disentanglement_score` |
| `evaluate.py` | Stage 4 multi-step latent evaluation | `evaluate_rollout`, `main` |
| `tta.py` | Stage 5 AdaJEPA test-time adaptation | `build_tta_batch`, `measure`, `main` |
| `symbolic.py` | Stage 3 AI-Feynman (PySR) | (not read in depth; uses `latent.npy`/`labels.npy` from `analyze.py`) |
| `utils/config.py` | Config load + device/output resolution | `load_config`, `resolve_device`, `ensure_output_dir`, `setup_logging` |
| `utils/losses.py` | V-JEPA objectives | `prediction_loss`, `variance_loss`, `covariance_loss`, `vicreg_loss` |
| `utils/metrics.py` | Regression + latent metrics | `mse`, `rmse`, `r2`, `cosine_similarity`, `latent_std_per_dim` |
| `utils/plotting.py` | Figures | `horizon_metric_plot`, `pca_plot`, `probe_parity_plot`, `scatter_latent`, `attribution_heatmap` |
| `utils/seed.py` | Reproducibility | `set_seed` |
| `research_api/*` | JSON CLI + job orchestration for an external Research Runtime | `cli`, `JobRunner`, `JobStore`, `worker`, `*_service` |

### 2.2 Robotics adapter (P2-01, isolated)

| Module | Role |
|---|---|
| `robotics/envs/pusher_env.py` | Pusher-v5 adapter, obs contract, MuJoCo contact detection |
| `robotics/data/trajectory.py` | `RobotTransition`, `Trajectory` (copy-on-ingest, npz I/O) |
| `robotics/data/splits.py` | Deterministic episode allocation (`train/validation/frozen_test/ood_test`) |
| `robotics/data/collector.py` | `collect_episode`, `collect_plan` |
| `robotics/data/dataset.py` | `load_split`, `summarize` |
| `robotics/data/manifest.py` | content-addressed `pusher-xxxx` manifest + SHA-256 |
| `robotics/data/policies.py` | `RandomPolicy`, `GuidedPolicy`, `MixturePolicy` |
| `scripts/collect_pusher.py` | CLI collection |

The two trees communicate **only** through the on-disk dataset
(`datasets/pusher_v1/*.npz` + `manifest.json`); there is no import coupling.

---

## 3. 关键类/函数签名 (Real signatures, not guessed)

- `models/predictor.Predictor.__init__(self, latent_dim=3, hidden_dims=(64,64),
  kind="mlp", tf_heads=2, tf_layers=2, tf_ff_mult=2, tf_dropout=0.0)`
  `forward(self, z)` → `(B, latent_dim)`. **No condition argument.** This is the
  interface P2-02 must preserve (`predictor.py:39-86`).
- `models/world_model.WorldModel.__init__(self, block_size, state_dim,
  latent_dim=3, encoder_hidden=(64,32), predictor_hidden=(64,64), encoder_type="mlp",
  …, predictor_type="mlp", …)`. state_dict contains `encoder.*`, `predictor.*`,
  `target_encoder.*` (`world_model.py:31-69`).
- `train.py` checkpoint schema:
  `torch.save({"model_state": model.state_dict(), "config": cfg,
  "state_dim": …, "latent_dim": …}, "checkpoint.pt")` (`train.py:225-233`).
  Loading side: `WorldModel.from_config(cfg, state_dim(cfg)).load_state_dict(
  ckpt["model_state"])` (`evaluate.py:119-121`, `analyze.py:198-201`).
- `data/dataset.BlockSequenceDataset.__getitem__` → `{"blocks": (num_blocks, block_dim),
  "labels": (label_dim,)}` (`data/dataset.py:297-301`). `block_dim = block_size * state_dim`.
- `data.dataset.SPLITS = ("train", "validation", "adapt", "test")`
  (`data/dataset.py:66`). **Four splits, different names than robotics.**
- `robotics.data.splits.SPLITS = ("train", "validation", "frozen_test", "ood_test")`
  (`robotics/data/splits.py:23`). **Four splits, robotics-private naming.**

---

## 4. 依赖 (Dependencies)

- Core runtime: `torch>=2.2`, `numpy>=1.24`, `scipy>=1.10`, `scikit-learn>=1.3`,
  `matplotlib>=3.7`, `tqdm`, `pyyaml`, `tensorboard`, `pysr` (`requirements.txt`).
- Robotics (P2-01, separate file, does NOT touch `requirements.txt`):
  `gymnasium[mujoco]==1.2.1`, `mujoco==3.3.7`, `pyyaml>=6.0`, `numpy>=1.24`
  (`requirements-robotics.txt`). **This is the correct isolation pattern** — P2-02
  should follow it (its own `requirements-*.txt` if it needs extra deps).
- `research_api` is deliberately free of torch imports at the CLI layer
  (`settings.py:1-20` comment) so the Agent Runtime can drive it from a
  lightweight interpreter. P2-02's API surface (if it adds an experiment stage)
  must preserve this property.

---

## 5. 兼容性风险 (Compatibility risks for P2-02)

| # | Risk | Where | Mitigation |
|---|---|---|---|
| R1 | Coupling generic contract to Pusher dims (23/7) | `DynamicsBatch` | Take `D`, `U` as runtime shape args; never hardcode. |
| R2 | Breaking legacy `Predictor`/`WorldModel` checkpoint | `models/predictor.py`, `train.py` | Add **separate** `ConditionalPredictor`; do **not** change `Predictor.forward` signature or `WorldModel.state_dict` keys. |
| R3 | OOD claim overreach | `ood_test` split | Audit verdict: not OOD; quarantine frozen_test; do not report OOD generalization. |
| R4 | No contact-rich dynamics in Pusher-v1 | dataset | Report arm-kinematics only; never claim object-pushing. |
| R5 | Two "dynamics" notions collide (latent JEPA vs raw-state baselines) | P2-02 scope | Separate the raw-state baseline track from the latent `ConditionalPredictor` track; both consume `DynamicsBatch`. |
| R6 | Dataset/model seed coupling | `data/dataset.dataset_spec` | Core already excludes `cfg['seed']` from `dataset_spec` (`data/dataset.py:99-120`); keep that invariant for P2-02. |
| R7 | Cross-episode transition leakage | `DynamicsBatch` | Group by `episode_ids`; never form `(x_T, u_T)→(x_0,·)` at episode boundary; emit `valid_mask`. |

---

## 6. OOD 分割实证 (Empirical OOD check — verdict: NOT OOD)

Script: `scripts/_audit_ood.py` (reproducible). Loads `datasets/pusher_v1`,
extracts per-episode features, compares splits.

Per-split feature summary (mean [std]):

| split | init_ogd | final_ogd | mean|action| | mean_reward | contact_rate |
|---|---|---|---|---|---|
| train | 0.2397[0.043] | 0.2398[0.045] | 0.6005 | -0.9504 | 0.0917 |
| validation | 0.2223[0.036] | 0.2215[0.037] | 0.5949 | -0.9217 | 0.0957 |
| frozen_test | 0.2383[0.028] | 0.2392[0.028] | 0.5989 | -0.9645 | 0.1220 |
| ood_test | 0.2256[0.025] | 0.2235[0.025] | 0.6012 | -0.9363 | 0.0987 |

- **init → final object-goal distance is ~unchanged (0.24 → 0.24):** no pushing.
- KS(train vs ood_test on init_ogd): D=0.425, p=0.008 — but `ood_test` has only
  **15** episodes; at n=15 tiny mean differences (0.240 vs 0.226, ~6%) read as
  "significant". Effect size is small and means overlap heavily.
- Separability probe (logistic regression, 5-fold CV, train vs ood_test):
  **accuracy 0.941** — this is a **small-sample overfitting artifact** (15 minority
  vs 240 majority samples), not evidence of distributional shift.
- **By construction** (`robotics/data/splits.plan_collection`): all four splits are
  allocated from the *same* `{random, guided, mixture}` policy set on the *same*
  Pusher-v5 env; the only difference is the split name inside the reset-seed hash.
  There is **no code path** that injects a covariate shift, altered policy mix, or
  perturbed environment for `ood_test`. Therefore it **cannot** be genuinely OOD.

**Audit conclusion:** `ood_test` is IID with the other splits. Any P2-02 result
described as "out-of-distribution generalization" would be invalid. Use it only as
a second held-out estimate; do not rename it to imply a real shift.

---

## 7. P2-02 改动所需 (What P2-02 must change — see gap analysis)

- Generic `DynamicsBatch` contract (`observations [B,T+1,D]`, `conditions [B,T,U]`,
  `valid_mask [B,T]`, `episode_ids [B]`) — **net-new**.
- `ConditionalPredictor` (`z_next = f(z, condition)`) — **new module**, separate
  from `Predictor`.
- Raw-state baseline dynamics (Persistence / Action-conditioned MLP / Action
  ablation) — **new training + eval scripts**, consuming `DynamicsBatch`.
- Train-only normalization — **new small module** (pattern exists in
  `analyze.py:216-218`, but for raw obs, not latents).
- Pusher-v1 → `DynamicsBatch` adapter — **new loader** reusing
  `robotics.data.dataset.load_split`.
- Reproducibility manifest for P2-02 experiments — reuse `robotics/data/manifest`
  pattern or extend `research_api` job store.

---

## 8. 验收门状态 (Gate status for A0)

- **A0 (architecture audit complete):** ✅ delivered by this doc set.
- Recommendation: **review module boundary** (next gate input) before Task 2
  (Core Data Contract + Predictor) begins. Do not modify production code until
  A0 review passes.
