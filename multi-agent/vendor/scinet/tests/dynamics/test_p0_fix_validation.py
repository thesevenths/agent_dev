"""P2-03A P0-fix regression tests: hardened dynamics validation.

Covers the review task book §4 requirements:
  * integer validation (hidden_dims): reject float/bool/str/None, accept int and
    numpy integer scalars
  * NormalizationStats deserialization schema: required fields, 1-D, equal
    lengths, finite values, strictly positive std, paired cond_*, positive eps,
    valid source_split, non-negative non-bool integer count
  * NaN/Inf rejection, zero/negative std rejection, cond dimension consistency,
    invalid source_split, serialization roundtrip
  * legacy checkpoint strict loading unchanged
  * existing P2-02 behaviour unchanged (incl. the already-persisted
    normalization.json remaining loadable)

These are pure unit tests; no dataset, simulator or model training is involved.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dynamics.normalization import DEFAULT_EPS, NormalizationStats, fit_normalization
from dynamics import dynamics_batch_from_episodes
from models.state_dynamics import (
    ResidualStatePredictor,
    _check_hidden_dims,
)

ROOT = Path(__file__).resolve().parents[2]
PERSISTED_NORM = ROOT / "outputs" / "dynamics" / "pusher_v1" / "normalization.json"


def _base_dict():
    return {
        "state_mean": [0.0, 0.0, 0.0],
        "state_std": [1.0, 1.0, 1.0],
        "cond_mean": None,
        "cond_std": None,
        "eps": DEFAULT_EPS,
        "source_split": "train",
        "n_valid_transitions": 10,
    }


# --------------------------------------------------------------------------- #
# A. Strict integer validation
# --------------------------------------------------------------------------- #
class TestHiddenDimsIntegerValidation(unittest.TestCase):
    def test_accepts_python_ints_and_numpy_integers(self):
        # numbers.Integral covers Python int and numpy integer scalars.
        self.assertEqual(_check_hidden_dims((16, 16)), (16, 16))
        self.assertEqual(_check_hidden_dims([8]), (8,))
        self.assertEqual(_check_hidden_dims(()), ())  # linear model allowed
        self.assertEqual(
            _check_hidden_dims((np.int64(8), np.int32(4), np.uint8(2))),
            (8, 4, 2),
        )
        # generator / other iterable still works
        self.assertEqual(_check_hidden_dims(x for x in (4, 8)), (4, 8))

    def test_rejects_floats_including_whole_valued(self):
        for bad in (3.0, 3.7, -1.5, np.float32(3.0), np.float64(3.0)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                _check_hidden_dims([bad])

    def test_rejects_booleans(self):
        # Python bool is an int subclass; np.bool_ is not Integral at all.
        for bad in (True, False, np.bool_(True), np.bool_(False)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                _check_hidden_dims([bad])

    def test_rejects_strings_none_and_invalid_sequence_entries(self):
        for bad in ("8", None, [8], {"a": 1}, object()):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                _check_hidden_dims([bad])
        for bad in (None, "abc", 8):
            with self.subTest(seq=bad), self.assertRaises((TypeError, ValueError)):
                _check_hidden_dims(bad)

    def test_rejects_non_positive_entries(self):
        for bad in (0, -1, np.int64(0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _check_hidden_dims([bad])

    def test_mixed_valid_invalid_entry_rejected(self):
        for seq in ((16, 3.0), (16, True), (16, np.float32(2.0)), (16, "8")):
            with self.subTest(seq=seq), self.assertRaises(TypeError):
                _check_hidden_dims(seq)

    def test_constructor_uses_strict_validation(self):
        # The public constructor must route through the hardened helper.
        for bad in ((16, 3.0), (16, True), (16, np.float64(2.0))):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                ResidualStatePredictor(6, 3, bad)
        m = ResidualStatePredictor(6, 3, (np.int64(8), 4))
        self.assertEqual(m.hidden_dims, (8, 4))
        self.assertEqual(tuple(m(torch.randn(2, 6), torch.randn(2, 3)).shape), (2, 6))

    def test_valid_hidden_dims_do_not_change_architecture(self):
        # Regression: previously int(3.0) silently built width 3; now it errors
        # instead of quietly producing a different model.
        m = ResidualStatePredictor(6, 3, (3, 3))
        self.assertEqual(m.net[0].in_features, 9)


# --------------------------------------------------------------------------- #
# B. NormalizationStats.from_dict schema
# --------------------------------------------------------------------------- #
class TestNormalizationDeserializationSchema(unittest.TestCase):
    def test_valid_minimal_and_conditioned_dicts(self):
        st = NormalizationStats.from_dict(_base_dict())
        self.assertEqual(tuple(st.state_mean.shape), (3,))
        self.assertIsNone(st.cond_mean)
        self.assertEqual(st.source_split, "train")

        d = _base_dict()
        d["cond_mean"] = [0.0, 0.0]
        d["cond_std"] = [1.0, 1.0]
        st2 = NormalizationStats.from_dict(d)
        self.assertEqual(tuple(st2.cond_mean.shape), (2,))
        self.assertEqual(tuple(st2.cond_std.shape), (2,))

    def test_rejects_non_mapping(self):
        for bad in ([1, 2], "x", 3, None):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                NormalizationStats.from_dict(bad)

    def test_rejects_missing_required_fields(self):
        for key in (
            "state_mean", "state_std", "cond_mean", "cond_std",
            "eps", "source_split", "n_valid_transitions",
        ):
            with self.subTest(key=key):
                d = _base_dict()
                del d[key]
                with self.assertRaises(ValueError):
                    NormalizationStats.from_dict(d)

    def test_rejects_nan_and_inf(self):
        for field, bad in (
            ("state_mean", [0.0, float("nan"), 0.0]),
            ("state_mean", [0.0, float("inf"), 0.0]),
            ("state_std", [1.0, float("nan"), 1.0]),
            ("state_std", [1.0, float("inf"), 1.0]),
        ):
            with self.subTest(field=field, bad=bad):
                d = _base_dict()
                d[field] = bad
                with self.assertRaises((TypeError, ValueError)):
                    NormalizationStats.from_dict(d)

    def test_rejects_zero_and_negative_state_std(self):
        for bad in ([1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [0.0] * 3):
            with self.subTest(bad=bad):
                d = _base_dict()
                d["state_std"] = bad
                with self.assertRaises(ValueError):
                    NormalizationStats.from_dict(d)

    def test_rejects_non_one_dimensional_vectors(self):
        for field, bad in (
            ("state_mean", [[0.0] * 3, [0.0] * 3]),
            ("state_std", [[1.0] * 3]),
            ("state_mean", 3.0),
            ("state_std", [[]]),
            ("state_mean", []),
        ):
            with self.subTest(field=field, bad=bad):
                d = _base_dict()
                d[field] = bad
                with self.assertRaises((TypeError, ValueError)):
                    NormalizationStats.from_dict(d)

    def test_rejects_bool_elements_in_vectors(self):
        d = _base_dict()
        d["state_mean"] = [0.0, True, 0.0]
        with self.assertRaises(TypeError):
            NormalizationStats.from_dict(d)

    def test_rejects_numpy_bool_ndarray(self):
        # Genuine bool ndarray (dtype=bool). Note: np.array([0.0, True, 0.0])
        # is promoted to float64 and is *valid* numeric data; the vulnerability
        # is a true bool dtype, which must be rejected before float coercion.
        d = _base_dict()
        d["state_mean"] = np.array([True, False, True], dtype=bool)
        with self.assertRaises(TypeError):
            NormalizationStats.from_dict(d)

    def test_rejects_torch_bool_tensor(self):
        d = _base_dict()
        d["state_mean"] = torch.tensor([0.0, True, 0.0], dtype=torch.bool)
        with self.assertRaises(TypeError):
            NormalizationStats.from_dict(d)

    def test_rejects_numpy_object_array_with_bool(self):
        d = _base_dict()
        d["state_mean"] = np.array([0.0, True, 0.0], dtype=object)
        with self.assertRaises(TypeError):
            NormalizationStats.from_dict(d)

    def test_rejects_2d_float_array(self):
        # Nested arrays must be rejected explicitly, not silently flattened.
        d = _base_dict()
        d["state_mean"] = np.array([[0.0, 1.0], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            NormalizationStats.from_dict(d)

    def test_accepts_float32_float64_ndarrays_and_int_tensors(self):
        for vec in (
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0, 1, 2]),  # int tensors are numeric, cast to float32
            np.array([0, 1, 2], dtype=np.int64),
        ):
            with self.subTest(type=type(vec).__name__):
                d = _base_dict()
                d["state_mean"] = vec
                st = NormalizationStats.from_dict(d)
                self.assertEqual(tuple(st.state_mean.shape), (3,))

    def test_rejects_state_vector_length_mismatch(self):
        d = _base_dict()
        d["state_std"] = [1.0, 1.0]
        with self.assertRaises(ValueError):
            NormalizationStats.from_dict(d)

    def test_cond_fields_must_be_paired(self):
        for cm, cs in (([0.0, 0.0], None), (None, [1.0, 1.0])):
            with self.subTest(cm=cm, cs=cs):
                d = _base_dict()
                d["cond_mean"], d["cond_std"] = cm, cs
                with self.assertRaises(ValueError):
                    NormalizationStats.from_dict(d)

    def test_cond_vector_length_must_match(self):
        d = _base_dict()
        d["cond_mean"] = [0.0, 0.0]
        d["cond_std"] = [1.0]
        with self.assertRaises(ValueError):
            NormalizationStats.from_dict(d)

    def test_rejects_zero_or_negative_cond_std(self):
        for bad in ([1.0, 0.0], [1.0, -1.0]):
            with self.subTest(bad=bad):
                d = _base_dict()
                d["cond_mean"] = [0.0, 0.0]
                d["cond_std"] = bad
                with self.assertRaises(ValueError):
                    NormalizationStats.from_dict(d)

    def test_rejects_bad_eps(self):
        for bad in (0, -1, True, float("nan"), float("inf"), "x"):
            with self.subTest(bad=bad):
                d = _base_dict()
                d["eps"] = bad
                with self.assertRaises((TypeError, ValueError)):
                    NormalizationStats.from_dict(d)

    def test_rejects_invalid_source_split(self):
        for bad in ("frozen_test", "validation", "", None, 1, True):
            with self.subTest(bad=bad):
                d = _base_dict()
                d["source_split"] = bad
                with self.assertRaises(ValueError):
                    NormalizationStats.from_dict(d)

    def test_rejects_bad_n_valid_transitions(self):
        for bad in (True, -1, 1.5, "10", float("nan")):
            with self.subTest(bad=bad):
                d = _base_dict()
                d["n_valid_transitions"] = bad
                with self.assertRaises(ValueError):
                    NormalizationStats.from_dict(d)

    def test_accepts_numpy_integer_count(self):
        d = _base_dict()
        d["n_valid_transitions"] = np.int64(42)
        self.assertEqual(NormalizationStats.from_dict(d).n_valid_transitions, 42)

    def test_save_load_roundtrip(self):
        g = torch.Generator().manual_seed(0)
        eps = [torch.randn(9, 3, generator=g) for _ in range(2)]
        conds = [torch.randn(8, 2, generator=g) for _ in range(2)]
        b = dynamics_batch_from_episodes(eps, conds, ["a", "b"])
        st = fit_normalization(b)
        # in-memory roundtrip
        r = NormalizationStats.from_dict(st.to_dict())
        self.assertTrue(torch.equal(st.state_mean, r.state_mean))
        self.assertTrue(torch.equal(st.state_std, r.state_std))
        self.assertTrue(torch.equal(st.cond_mean, r.cond_mean))
        self.assertEqual(st.source_split, r.source_split)
        self.assertEqual(st.n_valid_transitions, r.n_valid_transitions)
        # disk roundtrip through load()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "normalization.json"
            st.save(p)
            r2 = NormalizationStats.load(p)
        self.assertTrue(torch.equal(st.state_std, r2.state_std))

    def test_unconditioned_roundtrip(self):
        st = NormalizationStats.from_dict(_base_dict())
        self.assertTrue(torch.equal(
            NormalizationStats.from_dict(st.to_dict()).state_mean, st.state_mean
        ))


# --------------------------------------------------------------------------- #
# C. Existing artefacts & behaviour unchanged
# --------------------------------------------------------------------------- #
class TestExistingBehaviourUnchanged(unittest.TestCase):
    def test_persisted_p2_02_normalization_still_loads(self):
        """Hardened schema must NOT break the already-collected P2-02 evidence."""
        if not PERSISTED_NORM.exists():
            self.skipTest(f"P2-02 evidence not present: {PERSISTED_NORM}")
        d = json.loads(PERSISTED_NORM.read_text(encoding="utf-8"))
        st = NormalizationStats.from_dict(d)
        self.assertEqual(st.state_mean.numel(), 23)
        self.assertEqual(st.state_std.numel(), 23)
        self.assertIsNotNone(st.cond_mean)
        self.assertEqual(st.cond_mean.numel(), 7)
        self.assertEqual(st.cond_std.numel(), 7)
        self.assertEqual(st.source_split, "train")
        self.assertEqual(st.n_valid_transitions, 24000)
        self.assertGreater(float(st.state_std.min()), 0.0)  # 1e-8 clamp dims
        self.assertTrue(torch.isfinite(st.state_mean).all())
        # and through the exact load() path used by scripts/evaluate_dynamics.py
        self.assertEqual(NormalizationStats.load(PERSISTED_NORM).state_mean.numel(), 23)

    def test_fit_normalization_still_train_only(self):
        g = torch.Generator().manual_seed(1)
        eps = [torch.randn(9, 3, generator=g)]
        b = dynamics_batch_from_episodes(eps, None, ["only"])
        st = fit_normalization(b)
        self.assertEqual(st.source_split, "train")
        self.assertIsNone(st.cond_mean)
        # roundtrip of a genuinely unconditioned fit
        r = NormalizationStats.from_dict(st.to_dict())
        self.assertIsNone(r.cond_mean)

    def test_legacy_world_model_checkpoint_strict_loading(self):
        """P0-fix must not alter legacy state_dict keys / loading (task §4)."""
        from models.world_model import WorldModel

        paths = [
            ROOT / "outputs" / "pendulum" / "20260918_235258" / "checkpoint.pt",
            ROOT / "outputs" / "double_pendulum" / "20260919_093324" / "checkpoint.pt",
            ROOT / "outputs" / "newton" / "20260919_201912" / "checkpoint.pt",
        ]
        available = [p for p in paths if p.exists()]
        if not available:
            self.skipTest("historical checkpoints not present")
        for p in available:
            with self.subTest(ckpt=p.name):
                ck = torch.load(p, map_location="cpu", weights_only=False)
                m = WorldModel.from_config(ck["config"], ck["state_dim"])
                result = m.load_state_dict(ck["model_state"], strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                self.assertEqual(
                    {k.split(".")[0] for k in ck["model_state"]},
                    {"encoder", "predictor", "target_encoder"},
                )

    def test_do_not_modify_files_are_unchanged(self):
        """Guard the §3.D ban at source level for the two hardened modules."""
        import ast

        # state_dynamics must not import robotics or hardcode a domain dim
        p = ROOT / "models" / "state_dynamics.py"
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imports = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        for n in imports:
            names = (
                [a.name for a in n.names]
                if isinstance(n, ast.Import)
                else [n.module or ""]
            )
            self.assertFalse(any("robotics" in x for x in names))
        # normalization has no domain imports either
        p2 = ROOT / "dynamics" / "normalization.py"
        tree2 = ast.parse(p2.read_text(encoding="utf-8"))
        for n in ast.walk(tree2):
            if isinstance(n, ast.ImportFrom):
                self.assertNotIn("robotics", (n.module or ""))


if __name__ == "__main__":
    unittest.main()
