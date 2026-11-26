# P2-03A — Phase 0.1 Review Amendments (authoritative change log)

> **Status:** Phase 0 architecture audit (`144a300`) was accepted. This document
> records the **specification corrections and experimental-definition locks**
> demanded by the review, BEFORE any P1 dataset/simulator/model work.
>
> **Scope of this phase:** documentation amendments (Commit A) + narrowly scoped
> validation hardening (Commit B). **No** controlled-pendulum simulator, **no**
> dataset generation, **no** JEPA model, **no** training, **no** MPC, **no**
> robotics changes, **no** merge to `main`, **no** PR.
>
> Historical Phase 0 text in the four audited documents is **preserved**; every
> correction is marked `AMENDED (Phase 0.1)` inline and itemised here. The
> original (incorrect) expressions are quoted verbatim below so the change is
> auditable rather than silently overwritten.

## 0. Review verdict addressed

The review confirmed the Phase 0 delivery on GitHub
(`feature/p2-03a-action-jepa`, ahead 1 / behind 0, 4 new docs, 890 doc lines,
0 production lines, `main` untouched) but required nine specification fixes
before P1. It also required that G2/G3 not be judged on latent loss alone, and
that the E0/E1/OOD definitions be locked to prevent ambiguous experiments.

| # | Review item | Resolution | Where |
|---|---|---|---|
| 1 | multi-step window takes K+1 frames | **FIXED** (formula + assertions + new start-bound finding) | spec §2.2, audit §10.1 |
| 2 | E0 Persistence undefined | **LOCKED** | spec §5.0 |
| 3 | E1/E2 capacity difference | **LOCKED** + reported | spec §3, §5.2 |
| 4 | Parameter OOD changes 3 params at once | **REDESIGNED** into 4 groups | spec §5.1, map §7 |
| 5 | G2/G3 compare latent losses | **CHANGED** to train-only Linear-Probe raw-state metrics | spec §6.1, §7 |
| 6 | collapse criteria not pre-registered | **PRE-REGISTERED** in config | spec §4.4 |
| 7 | physics config incomplete | **COMPLETED** + pilot stability gates | spec §4.4 |
| 8 | incomplete `isinstance(...)` dispatch | **REPLACED** by explicit `predictor_mode` | spec §3 |
| 9 | Q7 stale | **CLOSED** | gap analysis §12 |

---

## 1. Item 1 — Multi-step target window (specification defect)

**Original text (Phase 0, `P2_03_ACTION_JEPA.md` §2.2):**

```
target_k_block = flatten(obs[s+k : s+k+K+1])       # start shifted by k
```

**Defect.** `Encoder.forward` requires `[B, K*D]` (`encoder.py:139-150`). The
slice `obs[s+k : s+k+K+1]` contains **K+1** frames, producing `[B, (K+1)*D]`.
One-step targets would be K frames while multi-step targets would be K+1 —
dimensionally incompatible and, if passed through NumPy, silently truncated.

**Correction (now in spec §2.2):**

```
target_k_block = flatten(obs[s+k : s+k+K])         # EXACTLY K frames -> [B, K*D]
```

With `t = s+K-1`, its last frame is `obs[s+k+K-1] = x_{t+k}`, matching the
k-th autoregressive prediction `rollout(z_context, action_seq)[:, k-1]`.

**Empirical verification (executed, not assumed).** Synthetic `obs[60,2]`,
`cond[59,1]`, `K=10`, `s=3`, `t=12`:

| k | corrected shape | erroneous shape | `action_seq` shape | last frame | expected | k=1 == one-step target |
|---|---|---|---|---|---|---|
| 1 | (1, 20) | (1, 22) | (1,1,1) | 13 | `x_13` | **YES (elementwise)** |
| 2 | (1, 20) | (1, 22) | (1,2,1) | 14 | `x_14` | no |
| 5 | (1, 20) | (1, 22) | (1,5,1) | 17 | `x_17` | no |
| 10 | (1, 20) | (1, 22) | (1,10,1) | 22 | `x_22` | no |
| 20 | (1, 20) | (1, 22) | (1,20,1) | 32 | `x_32` | no |

**Additional defect found during the amendment (not in the review).** Validity
requires `s+k+K <= L` and `s+K-1+k <= L-2`, i.e. **`0 <= s <= L-K-k_max`**.
The Phase 0 single-step bound `s <= L-K-1` is only valid for k=1. With
`K=10, L=60, k_max=20`, `s_max=30` (not 49). P1 must derive all windows with
`k_max`, otherwise horizon-20 windows are silently short. Added as a required
test case. All five horizons now have explicit specification-level shape
assertions plus the h=1 identity assertion, to be implemented as unit tests in
P2 (`tests/dynamics/test_trajectory_windows.py`, **not yet implemented**).

## 2. Item 2 — E0 Persistence locked

**Original:** "E0 | — | Persistence | — | lowest baseline (Q1: define precisely)".

**Locked definition (spec §5.0):**

```
x_hat[t+h] = x[t]        for every h >= 1, in RAW state (unwrapped theta)
```

- No Encoder, no Predictor, no Linear Probe, no training.
- MSE and MAE computed on **exactly the same evaluation episodes/timesteps**
  (same mask, same split) as every learned model.
- `Skill = 1 - MSE_model / MSE_persistence`; **the denominator must be reported
  in every table**. A config-declared `min_denominator` guard yields
  `skill=None` + an explicit degenerate-baseline flag when the denominator is
  near zero (no Inf, no silent clipping).
- `Skill > 0` better than standing still, `=0` equal, `<0` worse.
- E0 is the denominator of G4 and all skill scores; it is NOT a latent
  persistence model with a random encoder.

## 3. Item 3 — E1 unconditioned JEPA and capacity

**Locked (spec §3, §5.2):**

- E1 uses the **existing legacy `Predictor(z)`**, instantiated only — no edits
  to `models/predictor.py`. `condition=None` must **never** be passed to A2's
  `ConditionalPredictor` (it rejects `None` by design,
  `conditional_predictor.py:102-108`).
- E1 and E2 share the identical Encoder, `latent_dim`, `predictor_hidden`,
  epochs, batch, seeds and train/validation protocol.
- Because the first layers have widths `L` (E1) vs `L+U` (E2), **parameter
  counts differ by construction**. `sum(p.numel())` for encoder / predictor /
  total MUST be recorded per variant and the capacity difference reported.
- **E2, E3, E4 MUST be architecturally identical with exactly the same
  parameter count** (only the action tensor differs).
- Therefore `E2 vs E1` alone cannot prove action value; the primary evidence is
  `E2 vs E3` / `E2 vs E4`, with `E2 vs E1` secondary and the capacity caveat
  stated. E4 (zero action tensor) remains distinct from E1 (no condition input).

## 4. Item 4 — Parameter OOD split into four attributable groups

**Original:** one `parameter_ood_test` where mass, length and damping ALL shift
simultaneously (0.8–1.2 → 1.4–1.6, etc.). A degradation could not be attributed.

**Redesign (spec §5.1, map §7):** 300 episodes → four disjoint labelled groups
of 75 each; single-parameter groups keep the other two inside train ranges.

| Group | Mass | Length | Damping | Episodes |
|---|---|---|---|---|
| `mass_ood` | 1.4–1.6 | train 0.8–1.2 | train 0.05–0.15 | 75 |
| `length_ood` | train | 1.4–1.6 | train | 75 |
| `damping_ood` | train | train | 0.20–0.30 | 75 |
| `joint_ood` | 1.4–1.6 | 1.4–1.6 | 0.20–0.30 | 75 |

Per-episode `(m, L, b)`, the four ranges and the group counts MUST be written
into `manifest.json`; the audit script asserts OOD ranges are disjoint from
train (R8). G6 is reported **per group**, so failures are attributable to a
single physical parameter.

## 5. Item 5 — G2/G3 primary metrics are raw-state, not latent

**Reason:** independently trained latent spaces can differ in scale, rotation
and coordinate system; a lower latent MSE does not prove better physical
prediction.

**Changed (spec §6.1, §7):**

| Gate | PRIMARY (decides the gate) | SECONDARY (diagnostic only) |
|---|---|---|
| G2 E2 vs E1 | train-only Linear-Probe **raw-state RMSE** + Frozen-Test raw-state **MAE** | latent prediction MSE |
| G3 E2 vs E3 | train-only Linear-Probe **raw-state RMSE** + Frozen-Test raw-state **MAE** | latent prediction MSE |
| both | paired per-episode errors + 95% episode-level paired bootstrap CI | — |

Protocol constraints: fit **each** variant's probe on TRAIN representations
only; use VALIDATION for probe hyper-parameter selection; NEVER fit/tune on
`frozen_test` or `parameter_ood_test`; evaluate both test splits **without
refitting**; compare E0 on identical episodes and report the denominator.

## 6. Item 6 — Pre-registered collapse criteria

Thresholds MUST be in `configs/dynamics/controlled_pendulum_jepa.yaml` and
frozen **before** any frozen/OOD result is inspected (no post-hoc tuning):

| Criterion | Config key | Pilot threshold |
|---|---|---|
| per-dim latent std | `collapse.min_latent_std` | 0.1 |
| collapsed-dimension fraction | `collapse.max_collapsed_fraction` | 0.25 |
| covariance off-diagonal magnitude | `collapse.max_cov_offdiag` | 0.1 |
| finite latent check | `collapse.require_finite` | true |

Values are unverified starting points; revisions are allowed only from
train/validation evidence and must be logged with the reason.

## 7. Item 7 — Complete physics configuration

All keys are now explicit (spec §4.4): `gravity_g=9.81`, `dt=0.02`,
deterministic RK4, `episode_steps=1000`, **unwrapped theta** (no ±π wrap
discontinuity), `theta0/omega0` ranges, `torque_bounds=[-1,1]`,
`action_hold_steps`, diverse `action_profiles`, mass/length/damping
distributions, OOD ranges, and per-step finite-state guards. All are labelled
**pilot starting values, not validated results**.

P1-A must run **20 pilot episodes** first and pass: RK4 finite everywhere;
zero-torque damped behaviour; distinguishable ± torque trajectories;
exact-action replay within tolerance; no NaN/Inf or blow-up. Only then may P1-B
generate the full dataset. This gate is mandatory.

## 8. Item 8 — Explicit model dispatch

**Original:** `self.is_conditioned = isinstance(predictor, ...)` (incomplete)
plus an ambiguous `predictor or ConditionalPredictor(...)` default.

**Replaced (spec §3):** a single config-driven `predictor_mode` string with
`PREDICTOR_MODES = ("conditional", "unconditioned")`, an explicit constructor
branch, and a `predict_step(z, condition)` that dispatches on the mode:
`conditional` → `ConditionalPredictor(z, condition)` (condition MUST be tensor),
`unconditioned` → `Predictor(z)` (condition MUST be None/ignored, tensor
rejected). `rollout_latent` is valid only in conditional mode. Neither existing
predictor implementation is modified. Required P2 tests: both modes produce
`[B,L]`; wrong-mode inputs rejected; E1/E2 differ only in the predictor child.

## 9. Item 9 — Stale question closed

**Q7** ("push the feature branch for review?") is **CLOSED**: the branch was
pushed at Phase 0, read back from `origin/feature/p2-03a-action-jepa`, and
reviewed. Per the current task book, `main` remains unmerged and no PR is
created. Q1/Q2/Q5 are closed with the decisions above; **Q3, Q4, Q6 remain
explicitly OPEN** scientific/engineering decisions in
`P2_03A_GAP_ANALYSIS.md` §12.

---

## 10. What this phase does NOT do (stop condition)

Not implemented and explicitly deferred until review approval:
`data/controlled_pendulum.py`, any dataset generation (including the 20 pilot
episodes), `models/action_conditioned_world_model.py`, any JEPA training or
evaluation pipeline, Linear Probe code, MPC, robotics changes, P2-02B contact
data. No merge to `main`; no PR.

Production changes in this phase are limited to the two validation-hardening
files described in `P2_03A_P0_FIX_REPORT.md`. Historical P2-02 evidence
(`normalization.json`, 20 checkpoints, metrics, plots) is untouched and must
remain loadable — verified by regression.
