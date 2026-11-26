# P1-A Controlled-Pendulum Pilot — Report

**Status:** ✅ Pilot complete and accepted by its own scientific audit (8/8 checks PASS).
**Branch:** `feature/p2-03a-action-jepa`
**Depends on:** P0-fix.1 (commits `d28e007` fix(core), `0e305d6` docs) — both landed before this work.

This document covers the deliverables for **P1-A (Controlled Pendulum)** of the
P2-03A Action-conditioned JEPA thread. It does **not** implement training, the
`ActionConditionedWorldModel`, or MPC (see the STOP list at the end).

---

## 1. What was delivered

| # | Artifact | Path | Notes |
|---|----------|------|-------|
| 1 | Simulator (RK4, float64) | `data/controlled_pendulum.py` | New standalone module; does **not** touch `data/pendulum.py` |
| 2 | Pilot config | `configs/dynamics/controlled_pendulum_jepa.yaml` | 5 scenarios × 4 action profiles |
| 3 | Collection script | `scripts/collect_controlled_pendulum.py` | Generates exactly 20 episodes + content-addressed manifest |
| 4 | Audit script | `scripts/audit_controlled_pendulum.py` | 8-item numerical acceptance checklist |
| 5 | Pilot dataset | `datasets/controlled_pendulum_pilot/` | `pilot_episodes.npz` (float64) + `manifest.json` + `audit_report.json` |
| 6 | Unit tests | `tests/dynamics/test_controlled_pendulum.py` | 11 tests, all passing |
| 7 | This report | `docs/architecture/P1_A_PILOT_REPORT.md` | — |

### Restricted files — confirmed UNMODIFIED
`data/pendulum.py`, `data/__init__.py`, `models/state_dynamics.py` (WorldModel /
Predictor / ConditionalPredictor), `train.py`, `utils/losses.py`, `robotics/`,
`datasets/pusher_v1/`. Verified via `git diff` (empty) and `git status` (only the
6 new files above are added). The simulator imports only `numpy`, `math`,
`dataclasses`, `hashlib` — it has **no `robotics` dependency**.

---

## 2. Simulator specification (P1-A1)

**State** `x = [theta, omega]` (D = 2), `theta` is **unwrapped** (never reduced
mod 2π, so it can wind past ±π without a spurious jump).

**Control** `u = [torque]` (U = 1), bounded `[-5.0, +5.0]` N·m.

**Equation of motion (RK4, dt = 0.02 s, float64):**

```
theta_dot = omega
omega_dot = -(g / L) sin(theta) - b·omega + torque / (m·L²)
```

Integrated with classical **RK4** at fixed step `dt`. All arithmetic is float64.
The moment of inertia is `I = m·L²`.

**Four deterministic action profiles** (each a pure function of `(profile, T, dt, seed)`):

| Profile | Definition |
|---------|------------|
| `random` | `Uniform(-τ_max, +τ_max)`, seeded `RandomState(seed)` |
| `step` | `±magnitude·τ_max`, switching at `fraction` of the horizon |
| `sinusoid` | `amplitude·τ_max · sin(2π f t + phase)` |
| `alternating` | `+magnitude·τ_max / -magnitude·τ_max` every step |

**Five physical scenarios** (initial conditions held fixed at `theta0=2.0,
omega0=0.0` so the *only* varying quantity is the physical parameter set):

| Scenario | g | L | m | b | Interpretation |
|----------|---|---|---|---|----------------|
| nominal | 9.81 | 1.0 | 1.0 | 0.10 | reference |
| short_rod | 9.81 | 0.3 | 1.0 | 0.10 | short pendulum → fast dynamics |
| heavy_bob | 9.81 | 1.0 | 3.0 | 0.10 | large inertia |
| high_damping | 9.81 | 1.0 | 1.0 | 1.00 | overdamped |
| low_gravity | 1.62 | 1.0 | 1.0 | 0.10 | lunar gravity |

**Determinism.** Per-episode seed = `SHA256("ms={master}|s={s_idx}|p={p_idx}")[:4]`,
a 32-bit value (the naive `master*1000 + …` overflows `RandomState`'s range and
was rejected). The whole pilot is byte-reproducible from the config alone.

---

## 3. Dataset (P1-A2)

- **Exactly 20 episodes** = 5 scenarios × 4 profiles (hard requirement enforced in
  the collector; `pilot.require_exactly: true`).
- **Storage dtype:** float64 (per task). Stored tensors:

```
observations  [20, 201, 2]  float64   (theta_unwrapped, omega)
conditions    [20, 200, 1]  float64   (torque)
valid_mask    [20, 200]     bool      (all True)
initial_state [20, 2]       float64
scenario_index[20]          int64
profile_index [20]          int64
meta_str      JSON (dt, T, dims, master_seed, episode_ids, dataset_version, config_hash)
```

- **`manifest.json`** is content-addressed: `dataset_version = cpilot-41bb7d668870`,
  records `npz_sha256`, per-episode records (seed, scenario, profile, action
  stats, final state, max |Δθ| per step), and an integrity summary.
- **PILOT npz SHA-256:** `8d769ff576a5c6e855947c43b2bf697f200f59a04cdb0e19c787ad684af4ed89`

### DynamicsBatch contract alignment
The stored tensors satisfy `dynamics.contracts.DynamicsBatch` with **float64**
tensors (validated inside the audit): `batch_size=20`, `num_steps=200`,
`obs_dim=2`, `cond_dim=1`, `is_conditioned=True`. `validate_dynamics_batch`
accepts float64, so no precision is lost on load.

---

## 4. Scientific acceptance audit (P1-A3) — 8/8 PASS

Run: `python scripts/audit_controlled_pendulum.py --config configs/dynamics/controlled_pendulum_jepa.yaml`

| # | Check | Result | Key measured value |
|---|-------|--------|--------------------|
| 1 | Integrity & schema | ✅ | shapes correct, `manifest_sha256` matches npz, 20 unique ids, `DynamicsBatch(float64)` validates |
| 2 | Deterministic replay | ✅ | 20/20 episodes reproduce stored obs/cond **bitwise**; 0 seed-derivation mismatches |
| 3 | Zero-torque analytic | ✅ | max |θ_RK4 − θ_analytic| = **1.35e-4** < 1e-3 (small-angle, b=0) |
| 4 | Zero-torque energy | ✅ | relative energy drift = **1.56e-7** < 1e-5 (b=0, τ=0) |
| 5 | Action-sensitivity (sign) | ✅ | |θ(+τ) − θ(−τ)| = **22.74 rad** > 0.1 (control authority) |
| 6 | Action-sensitivity (magnitude) | ✅ | deviations {0, 0.716, 1.021, 1.042} — 4 distinct, monotonic-ish, no collapse |
| 7 | RK4 convergence | ✅ | error ratio dt vs dt/2 = **15.86** ≈ 16 (confirms genuine 4th order) |
| 8 | Unwrapped-theta & bounds | ✅ | max |Δθ|/step = **1.03 rad** < π; all torques in bounds; no NaN/Inf |

The full machine-readable report is at
`datasets/controlled_pendulum_pilot/audit_report.json`.

---

## 5. Regression (PHASE 4)

- **Full unit-test regression** (`pytest tests/ --ignore=tests/test_pysr.py`):
  **255 passed, 101 subtests passed, 0 failures** (was 244 before P1-A; the +11
  are the new `test_controlled_pendulum.py` suite).
- **Architecture checks:** no restricted file modified; simulator has no
  `robotics` import; only additive new files; `git status` clean of unrelated
  changes.

---

## 6. Reproduce

```bash
# 1) Collect the 20-episode pilot (float64, content-addressed manifest)
python scripts/collect_controlled_pendulum.py \
    --config configs/dynamics/controlled_pendulum_jepa.yaml

# 2) Run the scientific acceptance audit (writes audit_report.json)
python scripts/audit_controlled_pendulum.py \
    --config configs/dynamics/controlled_pendulum_jepa.yaml \
    --report datasets/controlled_pendulum_pilot/audit_report.json

# 3) Unit tests for the simulator
python -m pytest tests/dynamics/test_controlled_pendulum.py -v
```

---

## 7. STOP list — explicitly NOT done (per P2-03A task book)

- ❌ **Full dataset generation** — only the 20-episode pilot was produced.
- ❌ `trajectory_windows.py` / `ActionConditionedWorldModel` — not implemented.
- ❌ **JEPA training** (E0–E5) — no model was trained.
- ❌ **Eval (E0–E5)** — no evaluation runs.
- ❌ **MPC controller** — out of scope for the pilot.
- ❌ **Modifications to `robotics/`, `datasets/pusher_v1/`, or any restricted file.**
- ❌ **Merge to `main` / auto-PR** — all work stays on `feature/p2-03a-action-jepa`.

---

## 8. Next-step recommendations (for the human review gate)

1. **Inspect the pilot data** before authorizing the full collection. The
   `short_rod` scenario under strong torque produces large unwrapped θ
   (e.g. θ_f ≈ +79 rad for the `step` profile) — physically correct (high
   angular acceleration) but worth confirming this is the regime you want for
   the JEPA pretraining signal. If not, lower `action_max_torque` or shorten
   `horizon_steps` for that scenario.
2. **Confirm the de-duplication / episode-isolation story** for windows: the
   `DynamicsBatch` builder already guarantees no window crosses an episode
   boundary; the pilot stores whole episodes, so this is preserved by
   construction.
3. **Authorize full dataset + JEPA training** only after the 20-episode pilot is
   signed off. The same collector scales by raising `horizon_steps` / generating
   more (scenario, profile, seed) combinations — no code change required beyond
   config.
