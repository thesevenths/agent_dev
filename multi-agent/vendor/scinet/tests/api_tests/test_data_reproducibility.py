"""Regression tests for the P0 data-reproducibility fix.

Run from anywhere::

    D:/conda_envs/scinet_gpu/python.exe -m unittest discover -s tests -v

The critical case is :meth:`test_train_then_test_matches_isolated_test`:
before the fix, ``train.py`` generated train first (consuming the global RNG)
and ``evaluate.py`` re-seeded then generated test from a fresh stream, so the
two test sets were different and no baseline/candidate comparison was fair.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make the package importable no matter what the caller's cwd is.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from data.dataset import (
    build_dataset,
    dataset_version,
    derive_split_seed,
    split_rng,
)
from data.frozen import build_frozen_dataset, verify_frozen_dataset
from utils.seed import set_seed


def tiny_config(generator: str = "pendulum") -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "data": {
            "generator": generator,
            "sequence_length": 80,
            "block_size": 20,
            "t_max": 8.0,
            "train_samples": 8,
            "test_samples": 4,
        },
        "jepa": {"rollout_steps": [1, 2, 3]},
        "train": {"batch_size": 4, "epochs": 1, "lr": 0.001, "weight_decay": 0.0},
        "model": {"latent_dim": 4},
    }


class TestSplitDeterminism(unittest.TestCase):
    def test_same_split_is_bitwise_identical(self):
        cfg = tiny_config()
        a = build_dataset(cfg, "test")
        b = build_dataset(cfg, "test")
        self.assertTrue(np.array_equal(a.blocks, b.blocks))
        self.assertTrue(np.array_equal(a.labels, b.labels))

    def test_train_then_test_matches_isolated_test(self):
        """The exact train.py -> evaluate.py discrepancy that P0 fixes."""
        cfg = tiny_config()

        # train.py: set_seed(42) -> build train -> build test
        set_seed(cfg["seed"])
        _train = build_dataset(cfg, "train")
        test_from_train_process = build_dataset(cfg, "test")

        # evaluate.py: set_seed(42) -> build test directly
        set_seed(cfg["seed"])
        test_from_eval_process = build_dataset(cfg, "test")

        self.assertTrue(
            np.array_equal(test_from_train_process.blocks,
                           test_from_eval_process.blocks),
            "test split must be identical across train and evaluate processes",
        )
        self.assertTrue(
            np.array_equal(test_from_train_process.labels,
                           test_from_eval_process.labels)
        )

    def test_train_and_test_are_disjoint(self):
        cfg = tiny_config()
        train = build_dataset(cfg, "train")
        test = build_dataset(cfg, "test")
        n = min(len(train), len(test))
        self.assertFalse(np.array_equal(train.blocks[:n], test.blocks[:n]))

    def test_global_rng_is_not_consumed(self):
        """build_dataset must not perturb the process-global NumPy RNG."""
        cfg = tiny_config()
        np.random.seed(1234)
        expected = np.random.uniform(size=5)
        np.random.seed(1234)
        build_dataset(cfg, "train")
        build_dataset(cfg, "test")
        actual = np.random.uniform(size=5)
        self.assertTrue(np.array_equal(expected, actual))

    def test_seeds_are_stable_and_split_specific(self):
        cfg = tiny_config()
        self.assertEqual(derive_split_seed(cfg, "test"), derive_split_seed(cfg, "test"))
        self.assertNotEqual(derive_split_seed(cfg, "train"), derive_split_seed(cfg, "test"))
        self.assertEqual(split_rng(cfg, "test").randint(0, 10**6),
                         split_rng(cfg, "test").randint(0, 10**6))

    def test_dataset_version_is_stable_and_sensitive(self):
        cfg = tiny_config()
        self.assertEqual(dataset_version(cfg), dataset_version(copy.deepcopy(cfg)))
        changed = copy.deepcopy(cfg)
        changed["data"]["test_samples"] = 5
        self.assertNotEqual(dataset_version(cfg), dataset_version(changed))
        # Training hyper-parameters must NOT change the dataset version.
        hyper = copy.deepcopy(cfg)
        hyper["train"]["epochs"] = 99
        self.assertEqual(dataset_version(cfg), dataset_version(hyper))

    def test_all_generators_are_deterministic(self):
        for name in ("pendulum", "newton", "double_pendulum"):
            with self.subTest(generator=name):
                cfg = tiny_config(name)
                a = build_dataset(cfg, "test")
                b = build_dataset(cfg, "test")
                self.assertTrue(np.array_equal(a.blocks, b.blocks), name)
                self.assertTrue(np.array_equal(a.labels, b.labels), name)


class TestFrozenDataset(unittest.TestCase):
    def _frozen(self, cfg):
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name) / "frozen"
        manifest = build_frozen_dataset(cfg, out, splits=("train", "test"))
        return tmp, out, manifest

    def test_roundtrip_matches_generated_split(self):
        cfg = tiny_config()
        generated = build_dataset(cfg, "test")
        tmp, out, manifest = self._frozen(cfg)
        try:
            frozen_cfg = copy.deepcopy(cfg)
            frozen_cfg["data"]["frozen_dir"] = str(out)
            loaded = build_dataset(frozen_cfg, "test")
            self.assertTrue(np.array_equal(generated.blocks, loaded.blocks))
            self.assertTrue(np.array_equal(generated.labels, loaded.labels))
            self.assertEqual(loaded.dataset_version, manifest["dataset_version"])
            self.assertEqual(loaded.split, "test")
        finally:
            tmp.cleanup()

    def test_manifest_hash_is_stable_across_rebuilds(self):
        cfg = tiny_config()
        tmp1, out1, m1 = self._frozen(cfg)
        tmp2, out2, m2 = self._frozen(cfg)
        try:
            self.assertEqual(m1["dataset_version"], m2["dataset_version"])
            self.assertEqual(m1["splits"]["test"]["sha256"],
                             m2["splits"]["test"]["sha256"])
            self.assertEqual(m1["splits"]["train"]["sha256"],
                             m2["splits"]["train"]["sha256"])
        finally:
            tmp1.cleanup()
            tmp2.cleanup()

    def test_verify_accepts_clean_tree(self):
        cfg = tiny_config()
        tmp, out, _ = self._frozen(cfg)
        try:
            report = verify_frozen_dataset(out)
            self.assertTrue(all(v["ok"] for v in report["splits"].values()))
        finally:
            tmp.cleanup()

    def test_tampered_file_is_rejected(self):
        cfg = tiny_config()
        tmp, out, _ = self._frozen(cfg)
        try:
            path = out / "test.npz"
            with open(path, "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(RuntimeError):
                verify_frozen_dataset(out)
            frozen_cfg = copy.deepcopy(cfg)
            frozen_cfg["data"]["frozen_dir"] = str(out)
            with self.assertRaises(RuntimeError):
                build_dataset(frozen_cfg, "test")
        finally:
            tmp.cleanup()

    def test_missing_manifest_is_a_clear_error(self):
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["data"]["frozen_dir"] = tmp
            with self.assertRaises(FileNotFoundError):
                build_dataset(cfg, "test")

    def test_manifest_is_valid_json_with_required_keys(self):
        cfg = tiny_config()
        tmp, out, _ = self._frozen(cfg)
        try:
            payload = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            for key in ("schema_version", "dataset_version", "data_spec", "splits"):
                self.assertIn(key, payload)
            for split in ("train", "test"):
                for key in ("file", "sha256", "samples", "split_seed", "label_names"):
                    self.assertIn(key, payload["splits"][split])
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
