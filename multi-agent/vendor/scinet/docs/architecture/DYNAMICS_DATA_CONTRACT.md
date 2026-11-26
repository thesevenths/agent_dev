# Dynamics Data Contract — `DynamicsBatch`

**Phase:** A1 (Generic Dynamics Data Contract) · **Gate:** A1 must pass before A2.
**Branch:** `feature/a1-dynamics-contract`
**Commit:** `feat(dynamics): introduce domain-independent dynamics contract`

This document specifies the domain-independent dynamics data contract introduced
in A1. It is **additive infrastructure only** — it does not modify the existing
Scientific-Modeling core (`models/`, `data/`, `train.py`, `evaluate.py`,
`analyze.py`, `tta.py`, `utils/`, `research_api/`) and it does **not** import the
`robotics` package.

---

## 1. Why a new contract

The existing core is built around
`data.dataset.BlockSequenceDataset` shaped `(n, num_blocks, block_dim)` +
`labels`. That abstraction:

- assumes a **fixed** sequence length,
- assumes **uniform** validity (every block is real),
- has **no** control input (action / condition),
- has **no** episode grouping.

P2-02 and the broader "conditioned dynamics" roadmap need a transition-level
contract that expresses

    (x_t, u_t) -> x_{t+1}

for *arbitrary* scientific domains. `DynamicsBatch` is that contract. It is a
**net-new** abstraction; nothing in the old pipeline is rewritten.

---

## 2. Field semantics

| Field         | Shape            | Type   | Meaning                                                              |
|---------------|------------------|--------|----------------------------------------------------------------------|
| `observations`| `[B, T+1, D]`    | float  | state sequence `x_0, x_1, ..., x_T` for each episode/row            |
| `conditions`  | `[B, T,   U]`    | float  | control input `u_0, ..., u_{T-1}`; **`None` if unconditioned**      |
| `valid_mask`  | `[B, T]`         | bool   | `True` iff transition `(x_t, u_t) -> x_{t+1}` is a real transition  |
| `episode_ids` | `[B]`            | str    | source-episode identifier for each row (prevents cross-episode joins) |

`B` = batch size, `T` = number of transitions per row, `D` = observation dim,
`U` = condition (action) dim.

### `conditions = None` is NOT "zero condition"

This is the single most important semantic distinction in the contract:

- **Unconditioned** (`conditions = None`): the dynamical system has *no* external
  control variable at all. Example: an uncontrolled pendulum.
- **Conditioned with zero input** (`conditions` is a real tensor of zeros): the
  system *has* a control channel, but the input happens to be zero at every step.

These are different data-generating processes and must not be conflated. The
contract stores `None` for the first and a zero tensor (with `U>=1`) for the
second, and the validator `validate_dynamics_batch` enforces the
`None`/tensor distinction explicitly.

**A1 amendment (single unconditioned representation).** An earlier draft of the
gap analysis suggested `U=0 => unconditioned`. That second convention is
**rejected**: `conditions=None` is the *only* representation of an unconditioned
system. A `conditions` tensor with `U=0` (zero-width) is now rejected by
`validate_dynamics_batch` with a `ValueError`. An all-zero tensor with `U>=1`
remains valid — it means "a control channel exists, but the input is zero",
which is not the same experiment as "no external control variable exists".

Likewise, `episode_ids` is a **sequence of length B of string identifiers**
(or integer IDs normalised to strings), not a `torch.Tensor`. It is sample
identity metadata and never participates in GPU computation.

---

## 3. Temporal alignment (the invariant)

For every row `b` and every step `t` with `valid_mask[b, t] == True`:

    observations[b, t+1] == f(observations[b, t], conditions[b, t])

i.e. `x_{t+1}` is the state reached by applying control `u_t` to state `x_t`.
`training_pairs(batch)` returns `(x_t, u_t, x_{t+1}, mask)` where:

- `x_t      = observations[:, :-1]`   # `[B, T, D]`
- `x_{t+1}  = observations[:, 1:]`     # `[B, T, D]`
- `u_t      = conditions`              # `[B, T, U]` or `None`
- `mask     = valid_mask`              # `[B, T]` bool

Training code must only backprop through `mask == True` transitions.

---

## 4. Episode boundaries & isolation

A `DynamicsBatch` row is **always** a contiguous slice of exactly one episode.
Two builders are provided, both isolation-safe:

### `dynamics_batch_from_episodes(...)`
One row per episode. Variable-length episodes are padded to the longest episode:
- observation padding uses `pad_value` (default `0.0`),
- condition padding uses `0`,
- `valid_mask` is `True` only for real transitions; padded steps are `False`.

### `dynamics_batch_from_windows(..., window_len, stride)`
Generates sliding windows of `window_len` observations **strictly within** each
episode (`start = w*stride`, `end = start + window_len`, never crossing the
episode length). Every window is tagged with its source `episode_id`. This
implements the required discipline:

> Train/Validation/Frozen-Test must first be split **by episode**, then windows
> are generated **inside** each episode. Never generate all sliding windows and
> then randomly split them — adjacent timesteps are highly correlated and that
> causes leakage.

---

## 5. Optional conditions and domain examples

| Domain            | `D` | `conditions`            | `U`  | Notes |
|-------------------|-----|-------------------------|------|-------|
| Pendulum          | 2-3 | `None`                  | —    | unconditioned |
| Double-Pendulum   | 4-6 | `None`                  | —    | unconditioned |
| Newton            | 2   | `None`                  | —    | **force is an *observation* channel (column 1), not a control condition** — see §6 |
| Pusher-v5         | 23  | action tensor           | 7    | conditioned; action range `[-2, 2]` |

The contract never hardcodes `D` or `U`; both are inferred at build time from
the data. A single code path builds the Pendulum (`conditions=None`) and Pusher
(`conditions` with `U=7`) batches, proving the contract is domain-independent
(see `tests/dynamics/test_contracts.py`). Note there is **no** `U=0` row:
Pendulum is unconditioned via `None`, not via a zero-width condition tensor.

---

## 6. Newton condition decision (source-verified)

`data/newton.py` defines the signal as

    sig = stack([x(t), F(t)])          # (length, 2)
    channel 0 = position x(t)
    channel 1 = applied force F(t)      # constant over the window

The applied force is therefore **part of the observation**, not a separate
control input. There is **no external-force time-series condition** in the
current Newton generator. Per the explicit instruction *"如果没有，就继续使用
None，不能凭空构造外力标签"*, Newton maps to `conditions = None` (unconditioned).
We do **not** fabricate a force condition. The contract test
`test_newton_maps_to_unconditioned` locks this decision.

---

## 7. Mask semantics

`valid_mask[b, t] == True` means: "transition `t` of episode `b` is real and
should be used for training/evaluation." Padding and episode-length differences
are expressed purely through the mask; they never silently zero-out real data.

`validate_dynamics_batch` additionally requires **every row to contain at least
one valid transition** — a row with all-`False` mask is rejected (empty
episodes are not allowed in a batch).

---

## 8. Non-finite handling

`validate_dynamics_batch` rejects any batch containing `NaN`/`Inf` in
`observations` or `conditions` (raises `ValueError`). There is no implicit
clipping or replacement — non-finite values are treated as a data-integrity
error, never silently passed downstream.

---

## 9. Serialization

`DynamicsBatch.save(path)` writes a `.npz` containing:

- `observations` (`float32`, `[B, T+1, D]`),
- `conditions` (only if conditioned; `float32`, `[B, T, U]`),
- `valid_mask` (`uint8`, `[B, T]` — restored as `bool` on load),
- `meta_str` — JSON with `episode_ids` and `has_conditions`.

`DynamicsBatch.load(path)` reconstructs the batch exactly. Round-trip is covered
by `tests/dynamics/test_contracts.py::TestSerialization`.

---

## 10. Validation API

`dynamics.validation.validate_dynamics_batch(batch)` performs strict,
non-coercing checks:

- `observations` is `[B, T+1, D]` float with `T+1 >= 2`,
- `valid_mask` is `[B, T]` **bool** (dtype is checked, not cast),
- `conditions` (if present) is `[B, T, U]` float with **`U >= 1`** (`U=0` is
  rejected; use `None` for unconditioned),
- `episode_ids` is a length-`B` sequence of string identifiers (not a Tensor),
- every row has `>= 1` valid transition,
- no `NaN`/`Inf` in `observations` or `conditions`.

`dynamics.validation.summarize_batch(batch)` returns a read-only dict summary
(no mutation). Both are re-exported from the `dynamics` package.

---

## 11. A1 Acceptance Criteria — status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | New `DynamicsBatch` contract | ✅ `dynamics/contracts.py` |
| 2 | `conditions=None` and conditioned inputs | ✅ both paths tested |
| 2a | **`conditions=None` is the ONLY unconditioned representation** (`U=0` rejected; all-zero `U>=1` still conditioned) | ✅ `test_zero_width_conditions_*` / `test_none_vs_all_zero_are_distinct` |
| 2b | `episode_ids` are string metadata, not a Tensor | ✅ `test_episode_ids_are_string_metadata_not_tensor` |
| 3 | Variable-length episodes | ✅ padding + mask |
| 4 | Strict `x_t, u_t -> x_{t+1}` alignment | ✅ `training_pairs` + integration test |
| 5 | Episode-isolation test | ✅ `test_episode_isolation.py` |
| 6 | NaN/Inf and dtype validation | ✅ `test_validation.py` |
| 7 | Pendulum/Double-Pendulum/Newton regression | ✅ `test_data.py` green |
| 8 | Pusher-v1 raw data + manifest zero modification | ✅ read-only npz load; dataset untouched |
| 9 | New interface does not depend on `robotics` | ✅ grep confirms no core→robotics import |
| 10 | Contract doc + Git commit | ✅ this doc + `feat(dynamics): ...` |

**No production model code was modified** (`models/predictor.py`,
`models/encoder.py`, `models/world_model.py`, `data/dataset.py`, `train.py`,
`evaluate.py`, `analyze.py`, `tta.py` untouched). The Pusher-v1 dataset and its
manifest were read but never written.

### Out of scope for A1 (by design)
Implementation of `ConditionalPredictor`, the three P2-02 baselines, and
evaluation are **A2/A3/A4**. This phase is contract infrastructure only.
