"""
tests.models.test_conditional_predictor
=======================================
A2 acceptance suite for :class:`models.conditional_predictor.ConditionalPredictor`.

G1  single-step forward [B,L]
G2  sequence forward   [B,T,L]
G3  autoregressive rollout [B,H,L]
G4  input validation (dims, leading dims, no silent broadcast, NaN/Inf)
G5  legacy compatibility: real historical WorldModel checkpoints load with
    strict=True, no missing/unexpected keys, state_dict schema unchanged.

G5 uses REAL 2026-09 checkpoints already tracked in git (research evidence),
NOT a freshly generated fixture. SHA-256 of each checkpoint is asserted so the
test cannot silently pass against a substituted/retrained file.
"""

from __future__ import annotations

import hashlib
import inspect
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from models.conditional_predictor import ConditionalPredictor
from models.encoder import build_mlp
from models.predictor import Predictor
from models.world_model import WorldModel

# Repo root is SciNet++ (this file lives at <root>/tests/models/).
ROOT = Path(__file__).resolve().parents[2]

# Real historical checkpoints (2026-09), tracked on origin/main, with the
# SHA-256 recorded from the actual files during A2 verification.
LEGACY_CHECKPOINTS = {
    "outputs/pendulum/20260918_235258/checkpoint.pt":
        "fe7b0b412fdff1dc113726a6538bd21800174af0b1b5c8708eb41517b9e1c1f3",
    "outputs/double_pendulum/20260919_093324/checkpoint.pt":
        "d42db8e9551d122d3c82ea6d67c66bb673fe70e9f7fa9c658bfc892ce5594baa",
    "outputs/newton/20260919_201912/checkpoint.pt":
        "03098584ffd54267060ad5674d51beb6310bbc0f64d3d0d34bc15cde38320792",
}


def _model(latent_dim=4, condition_dim=3, hidden_dims=(16, 16), seed=0):
    torch.manual_seed(seed)
    return ConditionalPredictor(latent_dim, condition_dim, hidden_dims)


class _AdditiveNet(nn.Module):
    """Deterministic surrogate: z_next = z + u (requires L == U).

    Replacing ``model.net`` with this lets the rollout tests assert EXACT action
    alignment (``conditions[:, t]`` is used at step ``t``) and that each
    prediction is fed back as the next input.
    """

    def forward(self, x):
        d = x.shape[-1] // 2
        return x[..., :d] + x[..., d:]


# --------------------------------------------------------------------------- #
# G1: single-step forward
# --------------------------------------------------------------------------- #
class TestG1SingleStep(unittest.TestCase):
    def test_single_step_forward(self):
        m = _model(latent_dim=4, condition_dim=7)
        z = torch.randn(32, 4)
        u = torch.randn(32, 7)
        out = m(z, u)
        self.assertEqual(tuple(out.shape), (32, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_configurable_dimensions(self):
        for L, U in ((1, 1), (3, 2), (64, 7), (8, 11)):
            m = _model(latent_dim=L, condition_dim=U)
            z, u = torch.randn(5, L), torch.randn(5, U)
            self.assertEqual(tuple(m(z, u).shape), (5, L))
            self.assertEqual(m.latent_dim, L)
            self.assertEqual(m.condition_dim, U)

    def test_default_hidden_dims(self):
        m = ConditionalPredictor(latent_dim=4, condition_dim=7)
        self.assertEqual(m.hidden_dims, (256, 256))
        self.assertEqual(tuple(m(torch.randn(2, 4), torch.randn(2, 7)).shape), (2, 4))

    def test_reuses_build_mlp(self):
        # The net is the shared MLP builder's Sequential, not a duplicated MLP.
        m = _model(hidden_dims=(8, 8))
        self.assertIsInstance(m.net, nn.Sequential)
        linears = [mod for mod in m.net if isinstance(mod, nn.Linear)]
        self.assertEqual(linears[0].in_features, 4 + 3)
        self.assertEqual(linears[-1].out_features, 4)


# --------------------------------------------------------------------------- #
# G2: sequence forward
# --------------------------------------------------------------------------- #
class TestG2Sequence(unittest.TestCase):
    def test_sequence_forward(self):
        m = _model(latent_dim=4, condition_dim=7)
        z = torch.randn(8, 10, 4)
        u = torch.randn(8, 10, 7)
        out = m(z, u)
        self.assertEqual(tuple(out.shape), (8, 10, 4))

    def test_sequence_step_alignment(self):
        # z[:, t] + u[:, t] -> out[:, t], independently per timestep (MLP is
        # applied on the last axis only). Compare to per-step calls.
        m = _model(latent_dim=4, condition_dim=3)
        z = torch.randn(3, 6, 4)
        u = torch.randn(3, 6, 3)
        seq_out = m(z, u)
        for t in range(6):
            step_out = m(z[:, t], u[:, t])
            self.assertTrue(torch.allclose(seq_out[:, t], step_out))

    def test_single_step_and_sequence_agree_when_T1(self):
        m = _model(latent_dim=4, condition_dim=3)
        z, u = torch.randn(5, 4), torch.randn(5, 3)
        a = m(z, u)
        b = m(z.unsqueeze(1), u.unsqueeze(1))[:, 0]
        self.assertTrue(torch.allclose(a, b))


# --------------------------------------------------------------------------- #
# G3: autoregressive rollout
# --------------------------------------------------------------------------- #
class TestG3Rollout(unittest.TestCase):
    def test_rollout(self):
        m = _model(latent_dim=4, condition_dim=7)
        z0 = torch.randn(6, 4)
        conds = torch.randn(6, 12, 7)
        preds = m.rollout(z0, conds)
        self.assertEqual(tuple(preds.shape), (6, 12, 4))
        self.assertTrue(torch.isfinite(preds).all())

    def test_rollout_multiple_horizons(self):
        m = _model(latent_dim=4, condition_dim=3)
        z0 = torch.randn(4, 4)
        for H in (1, 2, 5, 10, 20):
            preds = m.rollout(z0, torch.randn(4, H, 3))
            # output last dim is latent_dim (4), NOT condition_dim (3)
            self.assertEqual(tuple(preds.shape), (4, H, 4))

    def test_rollout_action_alignment_and_autoregression(self):
        # With z_next = z + u, the exact closed form is
        #   preds[:, t] = z0 + sum_{s=0..t} u_s
        # This proves step t uses exactly conditions[:, t] AND that prediction
        # t-1 is fed back as the input for step t (no teacher forcing).
        m = _model(latent_dim=3, condition_dim=3)
        m.net = _AdditiveNet()
        z0 = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        conds = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        )
        preds = m.rollout(z0, conds)
        self.assertEqual(tuple(preds.shape), (2, 3, 3))
        expected = torch.tensor(
            [
                [[2.0, 2.0, 3.0], [2.0, 4.0, 3.0], [2.0, 4.0, 6.0]],
                [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            ]
        )
        self.assertTrue(torch.allclose(preds, expected))

    def test_rollout_matches_manual_forward_loop(self):
        m = _model(latent_dim=4, condition_dim=3)
        z0, conds = torch.randn(3, 4), torch.randn(3, 7, 3)
        preds = m.rollout(z0, conds)
        z = z0
        manual = []
        for t in range(conds.shape[1]):
            z = m(z, conds[:, t])
            manual.append(z)
        self.assertTrue(torch.allclose(preds, torch.stack(manual, dim=1)))

    def test_rollout_gradient_propagation(self):
        m = _model(latent_dim=4, condition_dim=3)
        z0 = torch.randn(3, 4, requires_grad=True)
        conds = torch.randn(3, 6, 3, requires_grad=True)
        preds = m.rollout(z0, conds)
        preds.sum().backward()
        self.assertIsNotNone(z0.grad)
        self.assertIsNotNone(conds.grad)
        self.assertGreater(float(z0.grad.abs().sum()), 0.0)
        self.assertGreater(float(conds.grad.abs().sum()), 0.0)
        # gradients reach the net parameters through every rollout step
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_rollout_deterministic_inference(self):
        m = _model(latent_dim=4, condition_dim=3).eval()
        z0, conds = torch.randn(2, 4), torch.randn(2, 5, 3)
        with torch.no_grad():
            a = m.rollout(z0, conds)
            b = m.rollout(z0, conds)
        self.assertTrue(torch.equal(a, b))


# --------------------------------------------------------------------------- #
# G4: input validation
# --------------------------------------------------------------------------- #
class TestG4Validation(unittest.TestCase):
    def setUp(self):
        self.m = _model(latent_dim=4, condition_dim=3)

    def test_latent_dimension_validation(self):
        with self.assertRaises(ValueError):
            self.m(torch.randn(3, 5), torch.randn(3, 3))

    def test_condition_dimension_validation(self):
        with self.assertRaises(ValueError):
            self.m(torch.randn(3, 4), torch.randn(3, 7))

    def test_leading_dimension_validation(self):
        # batch mismatch
        with self.assertRaises(ValueError):
            self.m(torch.randn(3, 4), torch.randn(4, 3))
        # sequence-length mismatch
        with self.assertRaises(ValueError):
            self.m(torch.randn(2, 6, 4), torch.randn(2, 5, 3))

    def test_no_silent_broadcast_across_time(self):
        # One condition per row must NOT be silently broadcast over T steps.
        with self.assertRaises(ValueError):
            self.m(torch.randn(2, 6, 4), torch.randn(2, 3))

    def test_rank_and_type_validation(self):
        with self.assertRaises(ValueError):
            self.m(torch.randn(4), torch.randn(3))  # 1-D
        with self.assertRaises(TypeError):
            self.m([[1.0] * 4], [[1.0] * 3])  # not tensors
        with self.assertRaises(TypeError):
            self.m(torch.randn(3, 4), None)  # unconditioned is NOT represented here
        with self.assertRaises(TypeError):
            self.m(torch.ones(3, 4, dtype=torch.int64), torch.randn(3, 3))

    def test_nan_rejection(self):
        z = torch.randn(3, 4)
        z[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.m(z, torch.randn(3, 3))
        u = torch.randn(3, 3)
        u[1, 1] = float("nan")
        with self.assertRaises(ValueError):
            self.m(torch.randn(3, 4), u)

    def test_inf_rejection(self):
        z = torch.randn(3, 4)
        z[2, 3] = float("inf")
        with self.assertRaises(ValueError):
            self.m(z, torch.randn(3, 3))

    def test_rollout_input_validation(self):
        with self.assertRaises(ValueError):
            self.m.rollout(torch.randn(3, 4, 1), torch.randn(3, 5, 3))  # z0 not 2-D
        with self.assertRaises(ValueError):
            self.m.rollout(torch.randn(3, 4), torch.randn(3, 3))  # conds not 3-D
        with self.assertRaises(ValueError):
            self.m.rollout(torch.randn(3, 4), torch.randn(4, 5, 3))  # batch mismatch
        with self.assertRaises(ValueError):
            self.m.rollout(torch.randn(3, 4), torch.randn(3, 0, 3))  # H=0

    def test_constructor_validation(self):
        for L, U in ((0, 3), (-1, 3), (4, 0), (True, 3), (2.5, 3)):
            with self.assertRaises(ValueError):
                ConditionalPredictor(L, U)
        with self.assertRaises(ValueError):
            ConditionalPredictor(4, 3, hidden_dims=(8, -1))
        with self.assertRaises(ValueError):
            ConditionalPredictor(4, 3, hidden_dims=None)

    def test_gradient_flow_single_step(self):
        z = torch.randn(4, 4, requires_grad=True)
        u = torch.randn(4, 3, requires_grad=True)
        out = self.m(z, u)
        out.sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertIsNotNone(u.grad)
        self.assertGreater(float(u.grad.abs().sum()), 0.0)  # condition is used


# --------------------------------------------------------------------------- #
# G5: legacy compatibility (CRITICAL)
# --------------------------------------------------------------------------- #
class TestG5LegacyCompatibility(unittest.TestCase):
    def test_legacy_predictor_unchanged(self):
        # Signature is still forward(self, z) -- no condition argument.
        sig = inspect.signature(Predictor.forward)
        self.assertEqual(list(sig.parameters.keys()), ["self", "z"])
        p = Predictor(latent_dim=4, hidden_dims=(8, 8))
        self.assertEqual(tuple(p(torch.randn(6, 4)).shape), (6, 4))
        self.assertFalse(hasattr(p, "condition_dim"))

    def test_world_model_not_extended_with_conditional_predictor(self):
        wm = WorldModel(block_size=5, state_dim=2, latent_dim=4,
                        encoder_hidden=(8,), predictor_hidden=(8,))
        self.assertFalse(hasattr(wm, "conditional_predictor"))
        prefixes = {k.split(".")[0] for k in wm.state_dict().keys()}
        self.assertEqual(prefixes, {"encoder", "predictor", "target_encoder"})

    def test_legacy_world_model_checkpoint_loading(self):
        """strict=True load of REAL 2026-09 checkpoints, with SHA-256 pinning."""
        loaded = []
        for rel, expected_sha in LEGACY_CHECKPOINTS.items():
            path = ROOT / rel
            self.assertTrue(
                path.is_file(),
                f"REAL legacy checkpoint missing: {path}. A2 G5 forbids "
                f"substituting a freshly generated checkpoint.",
            )
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            self.assertEqual(
                sha, expected_sha,
                f"{rel} SHA-256 changed -- checkpoint is not the real historical one",
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = WorldModel.from_config(ckpt["config"], ckpt["state_dim"])
            result = model.load_state_dict(ckpt["model_state"], strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            prefixes = {k.split(".")[0] for k in ckpt["model_state"].keys()}
            self.assertEqual(prefixes, {"encoder", "predictor", "target_encoder"})
            # inference still works after loading
            with torch.no_grad():
                z = torch.randn(2, ckpt["latent_dim"])
                self.assertEqual(tuple(model.predictor(z).shape), (2, ckpt["latent_dim"]))
            loaded.append((rel, sha, len(ckpt["model_state"])))
        self.assertEqual(len(loaded), 3)

    def test_conditional_predictor_is_independent_state_dict(self):
        # Its keys must not collide with any legacy WorldModel key.
        m = _model(latent_dim=4, condition_dim=7)
        wm = WorldModel(block_size=5, state_dim=2, latent_dim=4,
                        encoder_hidden=(8,), predictor_hidden=(8,))
        self.assertEqual(set(m.state_dict()) & set(wm.state_dict()), set())
        self.assertTrue(all(k.startswith("net.") for k in m.state_dict()))


# --------------------------------------------------------------------------- #
# Architecture guards (no robotics import / no Pusher constants)
# --------------------------------------------------------------------------- #
class TestArchitectureGuards(unittest.TestCase):
    def test_no_robotics_or_pusher_specifics_in_source(self):
        # Strip docstrings first: prose that *documents* the architectural rule
        # ("no Pusher-specific constants") is not a code dependency. Only the
        # executable AST is checked for domain-specific imports / identifiers.
        import ast

        src = (ROOT / "models" / "conditional_predictor.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]
        code = ast.unparse(tree)
        low = code.lower()
        self.assertNotIn("robotics", low)
        self.assertNotIn("pusher", low)
        self.assertNotIn("action_dim", low)
        # and explicitly: no import of any robotics module
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith("robotics"))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith("robotics"))

    def test_build_mlp_import_is_shared(self):
        # The module reuses models.encoder.build_mlp rather than redefining it.
        src = (ROOT / "models" / "conditional_predictor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from .encoder import build_mlp", src)
        self.assertNotIn("def build_mlp", src)


if __name__ == "__main__":
    unittest.main()
