# Conditional Predictor — A2 Infrastructure

**Phase:** A2 (Conditional Dynamics Model) · **Gate:** G1–G5
**Branch:** `feature/a2-conditional-predictor`
**Commit:** `feat(models): add independent conditional predictor`

This document specifies the **independent**, domain-independent conditional
latent dynamics predictor added in A2. It is model infrastructure **only**:
no Pusher training, no JEPA integration, no changes to existing model code.

---

## 1. Two deliberately separate model tracks

| Track | Phase | Model | Purpose |
|---|---|---|---|
| A — Latent Dynamics | **A2** | `z_{t+1} = f_theta(z_t, u_t)` | conditional latent transition interface; future Action-conditioned JEPA |
| B — Raw-state Dynamics | P2-02 | `x_{t+1} = x_t + g_theta(x_t, u_t)` | test whether action conditioning actually helps state prediction |

`ConditionalPredictor` implements Track A. **It is NOT inserted into
`WorldModel`.** The existing `Encoder` consumes *flattened blocks*
(`block_size * state_dim`), not Pusher's single-frame 23-D observation, so
`z = encoder(pusher_obs)` is **not** an established interface. Robot
observation encoding / JEPA integration is deferred to P2-03.

**Scientific boundary:** passing G1–G5 unit tests only proves the *interface*
is correct. It does **not** prove the model has learned physical laws. Only a
P2-02 result showing the action-conditioned model beating Persistence with
stable multi-step rollouts can justify P2-03.

---

## 2. Mathematical interface

```
z_next = f_theta([z, condition])
```

`f_theta` is an MLP built with the shared `models.encoder.build_mlp`
(no duplicated MLP utility). Default `hidden_dims=(256, 256)`; all of
`latent_dim`, `condition_dim`, `hidden_dims` are constructor-configurable.

First version is deliberately a plain MLP — no Transformer / FiLM /
cross-attention — because the latent dim is typically small and the immediate
goal is to validate condition effectiveness and keep ablations simple.

### Shapes

| Mode | `z` | `condition` | output |
|---|---|---|---|
| Single-step | `[B, L]` | `[B, U]` | `[B, L]` |
| Batched sequence | `[B, T, L]` | `[B, T, U]` | `[B, T, L]` |

Alignment is elementwise: `z[:, t] + condition[:, t] -> out[:, t]`.
**Leading dimensions must match exactly.** A single condition is never
silently broadcast across a time sequence (rank mismatch is rejected).

### Autoregressive rollout

```
rollout(z0 [B,L], conditions [B,H,U]) -> predictions [B,H,L]
```

Step `t` uses exactly `conditions[:, t]`, and each prediction becomes the next
step's input (no teacher forcing). The computation graph is preserved —
gradients propagate through every rollout step. This is the interface that can
later serve MPC cost evaluation; A2 does **not** implement MPC.

---

## 3. Constructor / signatures

```python
class ConditionalPredictor(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int,
                 hidden_dims: Sequence[int] = (256, 256)) -> None: ...
    def forward(self, z, condition): ...       # [B,L] or [B,T,L]
    def rollout(self, z0, conditions): ...     # -> [B,H,L]
    def extra_repr(self) -> str: ...
```

Module-level helper: `_check_tensor(name, x, min_ndim=2)`.

### Validation (G4)

Rejects with `TypeError`/`ValueError`:

- non-tensor inputs, non-float dtypes, rank < 2;
- `latent_dim` / `condition_dim` mismatch on the last axis;
- rank mismatch and leading-dimension mismatch (batch or sequence length);
- `NaN` / `Inf` in `z` or `condition`;
- invalid constructor dimensions (non-positive, non-int, `bool`);
- rollout shape errors (`z0` not 2-D, `conditions` not 3-D, batch mismatch,
  `H = 0`).

### `condition = None` is rejected

Unconditioned dynamics remain the job of `models.predictor.Predictor(z)`.
`ConditionalPredictor.forward(z, None)` raises `TypeError` instead of silently
substituting zeros. This keeps the A1 rule — `conditions=None` (no external
control variable) is a *different experiment* from a zero-valued condition —
from being undermined by a second implicit convention at the model layer.
A real `[B,T,U>=1]` all-zero tensor is valid and means "control channel
exists, zero input".

---

## 4. Architecture rules (enforced by tests)

- No `robotics` imports, no `pusher` identifiers, no `action_dim` hardcoding.
- `latent_dim` / `condition_dim` are runtime-configurable; no domain constants.
- **Do NOT modify** `models/predictor.py`, `models/world_model.py`,
  `train.py`, `evaluate.py`, or the existing checkpoint schema.
- `ConditionalPredictor` is instantiated independently; it is **not** a member
  of `WorldModel`, so `WorldModel.state_dict()` keys stay exactly
  `encoder.*` / `predictor.*` / `target_encoder.*`.
- No changes to the original JEPA training objective.
- No modifications to the Pusher-v1 dataset.

---

## 5. G5 — Legacy checkpoint verification (CRITICAL)

**Real historical checkpoints are used, not freshly generated fixtures.**
All three below were trained in 2026-09 (pre-A0/A1) and are already tracked in
git as research evidence, so CI can load them too. The test asserts each
SHA-256, so it cannot silently pass against a substituted/retrained file.

| Checkpoint | Size (bytes) | SHA-256 |
|---|---|---|
| `outputs/pendulum/20260918_235258/checkpoint.pt` | 2,196,962 | `fe7b0b412fdff1dc113726a6538bd21800174af0b1b5c8708eb41517b9e1c1f3` |
| `outputs/double_pendulum/20260919_093324/checkpoint.pt` | 2,202,037 | `d42db8e9551d122d3c82ea6d67c66bb673fe70e9f7fa9c658bfc892ce5594baa` |
| `outputs/newton/20260919_201912/checkpoint.pt` | 2,199,157 | `03098584ffd54267060ad5674d51beb6310bbc0f64d3d0d34bc15cde38320792` |

Verification procedure (executed by
`test_legacy_world_model_checkpoint_loading`):

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model = WorldModel.from_config(ckpt["config"], ckpt["state_dim"])
result = model.load_state_dict(ckpt["model_state"], strict=True)
assert result.missing_keys == []
assert result.unexpected_keys == []
assert {k.split(".")[0] for k in ckpt["model_state"]} == {
    "encoder", "predictor", "target_encoder"
}
```

Also verified: `Predictor.forward` signature is still `(self, z)`;
`WorldModel` has no `conditional_predictor` attribute; the new model's
state_dict keys (`net.*`) do not collide with any legacy key.

---

## 6. Gate results

| Gate | Requirement | Evidence | Result |
|---|---|---|---|
| G1 | Single-step `[B,L]` | `TestG1SingleStep` (4 tests) | ✅ |
| G2 | Sequence `[B,T,L]` | `TestG2Sequence` (3 tests) | ✅ |
| G3 | Rollout `[B,H,L]` | `TestG3Rollout` (6 tests: shape, alignment, horizons, gradient, determinism, loop-equivalence) | ✅ |
| G4 | Input validation | `TestG4Validation` (10 tests) | ✅ |
| G5 | Legacy compatibility | `TestG5LegacyCompatibility` (4 tests, real ckpt + SHA-256) | ✅ |
| — | Architecture guards | `TestArchitectureGuards` (2 tests) | ✅ |

Total: **29 tests, all passing**, in `tests/models/test_conditional_predictor.py`.

G3 action alignment is proven exactly, not approximately: replacing `net`
with the deterministic surrogate `z_next = z + u` gives the closed form
`preds[:, t] = z0 + sum_{s<=t} u_s`, which the test asserts elementwise.
This proves step `t` uses `conditions[:, t]` and that predictions are fed back
autoregressively.

---

## 7. Out of scope / unresolved

- **No training.** P2-02 (Persistence / Action-MLP / Action Ablation),
  evaluation metrics and plots are not part of A2.
- **No JEPA integration**, no robot observation encoder, no MPC.
- **`ConditionalPredictor` has not learned any dynamics.** All tests use random
  weights; only the interface is validated.
- **Pusher-v1 limitation stands:** 30,000 transitions with **zero
  fingertip-object contacts**. Even successful arm-state prediction must not be
  presented as a contact-rich object-pushing world model.
- The real historical checkpoints are ~2.2 MB each and are already tracked in
  git; A2 adds no model binary.

---

## 8. Files

| File | Change |
|---|---|
| `models/conditional_predictor.py` | **NEW** — independent MLP predictor |
| `tests/models/__init__.py` | **NEW** — package marker |
| `tests/models/test_conditional_predictor.py` | **NEW** — 29 G1–G5 + guard tests |
| `docs/architecture/CONDITIONAL_PREDICTOR.md` | **NEW** — this document |

No existing production model file was modified.
