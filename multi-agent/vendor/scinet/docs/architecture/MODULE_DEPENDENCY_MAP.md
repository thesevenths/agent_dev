# MODULE_DEPENDENCY_MAP — SciNet++

**Phase 0 deliverable (audit only; no production code changed).**
Companion: `ARCHITECTURE_AUDIT.md`, `CORE_ADAPTER_BOUNDARY.md`,
`P2_02_GAP_ANALYSIS.md`.

Legend: `A ──▶ B` means "A imports/uses B". Edges are real import relationships
read from source.

---

## 1. 核心依赖图 (Core dependency graph)

```
                 configs/*.yaml
                        │ (yaml.safe_load)
                        ▼
                  utils/config.py  ◀── train.py / evaluate.py / analyze.py / tta.py / symbolic.py
                        │ load_config, resolve_device, ensure_output_dir, setup_logging
                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ data/dataset.py                                                     │
   │   ├─ build_dataset, BlockSequenceDataset, dataset_version, split_rng │
   │   └─ imports: data.pendulum / data.double_pendulum / data.newton    │
   └───────┬───────────────────────────┬─────────────────────────────────┘
           │ X,labels (n,length,state_dim) │ frozen_dir → load_frozen_split
           ▼                               ▼
   train.py / evaluate.py             data/frozen.py (build/verify .npz+manifest)
   analyze.py / tta.py
           │
           ▼
   models/world_model.py ──▶ models/encoder.py (Encoder, build_mlp)
                          └─▶ models/predictor.py (Predictor)
   analyze.py ──▶ models/probe.py (Probe)
   train.py / tta.py ──▶ utils/losses.py (prediction_loss, vicreg_loss)
   train/evaluate/analyze/tta ──▶ utils/metrics.py (mse, rmse, r2, cosine_similarity)
   evaluate.py ──▶ utils/plotting.py (horizon_metric_plot, …)
   all stages ──▶ utils/seed.py (set_seed)
```

Observations:
- `data/dataset.py` is the **single hub** for the data contract; all stages depend
  on it. Generators are behind a registry (`_GENERATORS` at `data/dataset.py:59-63`),
  so adding a generator is additive.
- `models/world_model.py` is the **single hub** for the model; `encoder` and
  `predictor` are leaf nn.Modules with no cross-dependency except `predictor`
  importing `encoder.build_mlp` (`predictor.py:36`).
- `utils/*` is a pure leaf layer (no internal deps beyond torch/numpy).

---

## 2. Research Runtime 依赖图 (research_api)

```
   external Research Runtime (JSON only)
            │  python -m research_api.cli <subcommand>
            ▼
   research_api/cli.py
     ├─ research_api/runner.py  (JobRunner, build_command, make_run_id)
     ├─ research_api/settings.py (PROJECT_DIR, DATA_SPLITS, resolve_config_path)
     ├─ research_api/job_store.py (SQLite JobStore)
     ├─ research_api/result_parser.py (build_result, parse_horizon_metrics)
     ├─ research_api/train_service.py / eval_service.py (command builders)
     └─ research_api/worker.py (detached subprocess running train/evaluate/…)
            │ spawns sys.executable -m research_api.worker
            ▼
   train.py / evaluate.py / analyze.py / symbolic.py / tta.py  (UNCHANGED stage scripts)
            │ produce artifacts: checkpoint.pt, horizon_metrics.npy, latent.npy, tta_metrics.json
            ▼
   result_parser.build_result reads artifacts → result.json
```

Key isolation property: `research_api/settings.py` deliberately avoids torch
(`settings.py:1-20`). `DATA_SPLITS = ("train","validation","adapt","test")` is
**duplicated** here and must stay in sync with `data.dataset.SPLITS` (guarded by
`test_split_isolation.py::test_cli_split_constant_matches_engine`).

---

## 3. Robotics 适配器依赖图 (robotics adapter, P2-01)

```
   configs/robotics/pusher_data.yaml
            │
            ▼
   scripts/collect_pusher.py
     ├─ robotics/data/dataset.py (collect_dataset, load_split, summarize)
     ├─ robotics/data/collector.py (collect_episode, collect_plan)
     ├─ robotics/data/policies.py (Random/Guided/Mixture)
     ├─ robotics/data/splits.py (plan_collection, episode_reset_seed)
     ├─ robotics/data/manifest.py (dataset_version, build_manifest, verify_*)
     ├─ robotics/data/trajectory.py (RobotTransition, Trajectory)
     └─ robotics/envs/pusher_env.py (create_pusher_env, parse_observation,
                                      detect_contact, verify_observation_contract)

   robotics/data/dataset.py ──▶ robotics/envs/pusher_env.ENV_ID
   robotics/data/dataset.py ──▶ robotics/data/collector / policies / splits / trajectory
```

**There is no edge from `robotics/*` to `models/*`, `data/*`, `train.py`, or
`research_api/*`.** The two worlds meet only on disk
(`datasets/pusher_v1/*.npz` + `manifest.json`). This is the boundary that P2-02
must preserve: the generic core must consume the robotics dataset **through a
generic `DynamicsBatch` loader**, never by importing `robotics/`.

---

## 4. 跨模块数据流 (Cross-module data flow)

```
generator.sample(n)            → X (n,length,state_dim), labels (n,label_dim)
BlockSequenceDataset(X,labels) → blocks (n,num_blocks,block_dim), labels
  (train/evaluate/analyze/tta consume {"blocks","labels"})
WorldModel.encode(blocks[:,0,:]) → z (n,latent_dim)            [analyze.py]
WorldModel.predict_multistep(z, horizons) → {k: z_k}           [evaluate.py]
Probe(z) → concept predictions                                 [analyze.py]
```

P2-02 introduces a **parallel** flow that does NOT go through `BlockSequenceDataset`:

```
robotics Trajectory (.npz) → [NEW] DynamicsBatch loader →
   observations (B,T+1,D), conditions (B,T,U), valid_mask (B,T), episode_ids (B)
→ [NEW] Persistence / ActionMLP / ConditionalPredictor
```

The `DynamicsBatch` loader is the only sanctioned bridge between the robotics
dataset and the generic core. It must accept the robotics `.npz` files via
`robotics.data.dataset.load_split` and emit a backend-agnostic batch.

---

## 5. 检查点依赖 (Checkpoint dependency — backward-compat contract)

```
train.py writes:
   checkpoint.pt = {
     "model_state": WorldModel.state_dict(),   # keys: encoder.*, predictor.*, target_encoder.*
     "config": cfg,
     "state_dim": int,
     "latent_dim": int,
   }
evaluate.py / analyze.py / tta.py read:
   WorldModel.from_config(cfg, state_dim(cfg)).load_state_dict(ckpt["model_state"])
```

**Backward-compat invariant for P2-02:** any new module (`ConditionalPredictor`,
raw-action MLP) must serialize under its **own** key/checkpoint file. It must NOT
add keys to `WorldModel.state_dict` nor change `Predictor`'s `forward(z)` signature,
or legacy `load_state_dict` calls break. A separate checkpoint (e.g.
`dynamics_checkpoint.pt` with `{"model_state": <new module>.state_dict(),
"config": …, "obs_dim": …, "action_dim": …}`) keeps legacy paths intact.

---

## 6. 测试依赖图 (Test surface — regression scope for P2-02)

```
test_data.py                         → Pendulum/Double-Pendulum/Newton smoke (CORE regression)
tests/api_tests/test_split_isolation.py → train/val/adapt/test isolation + E2E (CORE regression)
tests/api_tests/test_job_store.py, test_services.py, test_cli_e2e.py, test_data_reproducibility.py → research_api
tests/robotics/* (28 tests)          → P2-01 env adapter + dataset (must stay green)
```

P2-02 must **add** (not modify existing):
- `tests/robotics/test_dynamics_batch.py` (contract tests)
- `tests/robotics/test_conditional_predictor.py` (incl. legacy checkpoint load)
- `tests/robotics/test_pusher_training.py` (3 baselines + eval)
- A "legacy checkpoint loading" test that loads a pre-existing `WorldModel`
  checkpoint and asserts `load_state_dict` succeeds unchanged.
