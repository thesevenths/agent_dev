# CORE_ADAPTER_BOUNDARY — Where the generic core ends and Robotics begins

**Phase 0 deliverable (audit only).** Companion: `ARCHITECTURE_AUDIT.md`,
`MODULE_DEPENDENCY_MAP.md`, `P2_02_GAP_ANALYSIS.md`.

Purpose: pin down **every** place where Robotics assumptions could leak into the
generic scientific-modeling core, and state the rule that keeps Robotics a
domain adapter.

---

## 1. 当前边界状态 (Current boundary state: GOOD)

A grep-level fact: **`robotics/` is never imported by `models/`, `data/`,
`train.py`, `evaluate.py`, `analyze.py`, `tta.py`, `symbolic.py`, `utils/`,
or `research_api/`.** The import graph in `MODULE_DEPENDENCY_MAP.md` §3 confirms
it. Robotics is a self-contained subtree that communicates with the core only
through the on-disk dataset (`datasets/pusher_v1/*.npz` + `manifest.json`).

This satisfies the project directive: *"Robotics must remain a domain adapter,
not become a dependency of the generic scientific modeling core."* ✅

---

## 2. 核心缺失的"动力学"概念 (Dynamics concepts the core does NOT have)

The core's "dynamics" is **latent, block-level, unconditioned**:

| Concept needed by P2-02 | Present in core? | Evidence |
|---|---|---|
| Raw observation `x_t` (e.g. dim 23) as first-class | ❌ | Core trains on `blocks` (flattened `block_size*state_dim`); `state_dim` is generator-specific (1, 2, or 4) |
| Action / intervention `u_t` | ❌ | No variable named action/condition/intervention anywhere in `models/` or `data/` |
| Per-timestep transition `(x_t, u_t) → x_{t+1}` | ❌ | Core predicts `z_b → z_{b+1}` over non-overlapping **blocks** (`world_model.py:82-97`) |
| Variable-length episodes | ❌ | `BlockSequenceDataset` trims to `num_blocks*block_size` and assumes fixed `length` (`data/dataset.py:281-291`) |
| `episode_ids` / episode grouping | ❌ | No episode concept; a "sample" is one whole trajectory |
| `valid_mask` (which transitions are real) | ❌ | Not present; blocks are uniformly valid |
| Conditioning a predictor on `u_t` | ❌ | `Predictor.forward(z)` takes **only** `z` (`predictor.py:80-86`) |
| Train-only normalization of raw obs | ⚠️ partial | `analyze.py` normalizes *probe targets* with train stats (`analyze.py:216-218`), but there is **no** raw-observation normalizer |

**Implication:** P2-02's generic `DynamicsBatch` contract is **additive**, not a
refactor. Nothing in the core must be changed to support it; a new loader + new
models consume it.

---

## 3. 核心已有的可复用资产 (Core assets P2-02 SHOULD reuse)

| Asset | Location | Why reuse |
|---|---|---|
| `build_mlp(dims, activation)` | `models/encoder.py:41-48` | Builds MLPs uniformly; `Predictor` and `Probe` both use it. New action-MLP should too, for consistency. |
| `Probe` (latent→concept MLP) | `models/probe.py` | Template for a small residual read-out; reuse the shape convention. |
| `prediction_loss` / `vicreg_loss` | `utils/losses.py` | Latent losses; relevant only if P2-02 conditions the *latent* JEPA path (`ConditionalPredictor`). |
| `mse` / `rmse` / `r2` / `cosine_similarity` | `utils/metrics.py` | Directly usable for P2-02 one-step / rollout metrics. |
| `set_seed` | `utils/seed.py` | Reproducibility for P2-02 runs. |
| `load_config` / `ensure_output_dir` / `resolve_device` | `utils/config.py` | P2-02 config + output layout should follow `<output_dir>/<experiment>/<run_id>`. |
| `dataset_version` / SHA-256 pattern | `data/dataset.py`, `robotics/data/manifest.py` | Copy the content-addressed + integrity pattern for P2-02 experiment manifests. |
| `research_api` job store + CLI | `research_api/*` | If P2-02 exposes an experiment stage, it should plug in as a new `SUPPORTED_STAGES` entry + command builder, **without modifying** existing stage scripts. |

---

## 4. 边界规则 (Boundary rules for P2-02 — must hold at A1 gate)

1. **No `robotics/` import in `models/`, `data/`, `train.py`, `evaluate.py`,
   `analyze.py`, `tta.py`, `symbolic.py`, `utils/`, `research_api/`.**
   The core consumes Pusher-v1 only via a generic `DynamicsBatch` loader that
   reads the `.npz` files by path. (This loader may live in `robotics/data/` or a
   new `dynamics/` package, but must NOT be imported by the core's training path
   in a way that makes Robotics a hard dependency.)
2. **`DynamicsBatch` carries no Pusher-specific constant.** `D`, `U`, episode
   length are runtime attributes, never hardcoded 23 / 7. It must serve Pendulum
   (D=1, U=0) and Pusher (D=23, U=7) identically.
3. **`ConditionalPredictor` is a new `nn.Module`, separate from `Predictor`.**
   `Predictor.forward(z)` stays unchanged → legacy `WorldModel` checkpoints load
   unchanged. `ConditionalPredictor` may share `build_mlp` but defines its own
   `state_dict` keys.
4. **The core's latent JEPA pipeline (`train.py`/`evaluate.py`/…) is untouched.**
   P2-02 baseline dynamics (Persistence / Action-MLP / Action-ablation) are a
   **parallel** raw-state pipeline; they do not alter `WorldModel`.
5. **Split naming stays split-scoped.** Core `SPLITS =
   (train,validation,adapt,test)`; robotics `SPLITS =
   (train,validation,frozen_test,ood_test)`. P2-02 reuses robotics' splits when
   reading Pusher-v1, but must not rename or merge them into the core's enum.
6. **Frozen Test is quarantined.** `frozen_test` / `ood_test` are read-only
   evidence; P2-02 must never write to or depend on them for model selection
   (mirror `train.py:121-126` which keeps `test` out of training).

---

## 5. 若违反边界的回退信号 (Regression signals that the boundary held)

- `tests/robotics/test_dynamics_batch.py` asserts `DynamicsBatch` works for
  `D=1,U=0` (Pendulum-like) and `D=23,U=7` (Pusher) with identical code path.
- `tests/robotics/test_conditional_predictor.py` asserts legacy
  `WorldModel.from_config(...).load_state_dict(ckpt["model_state"])` still
  succeeds after P2-02 lands (load a real/synthetic legacy checkpoint).
- A CI grep test: `robotics` must not appear in `models/`, `data/`, `train.py`,
  `evaluate.py`, `analyze.py`, `tta.py`, `symbolic.py`, `utils/`, `research_api/`.
- Existing `tests/api_tests/test_split_isolation.py` must remain green, proving
  core split isolation was not disturbed.

---

## 6. 边界小结 (Summary)

The boundary is **structurally clean today** — the only work is *adding* a
generic contract and models that respect it, not *untangling* leakage. The main
discipline required is (a) keeping `ConditionalPredictor` separate from
`Predictor`, (b) never hardcoding Pusher dims into the generic contract, and
(c) consuming Pusher-v1 through a path-agnostic loader. None of these require
modifying existing production code, which is exactly why Phase 0 can stay
read-only.
