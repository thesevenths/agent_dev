"""S09-S12 regression suite: train/validation/adapt/test isolation.

Two layers:

* Pure unit tests -- command construction, split-aware result parsing, seed
  decoupling, stale-manifest rejection, TTA split selection. Fast, no training.
* One REAL integration test -- submit train -> evaluate(validation) ->
  evaluate(test) -> tta through the JSON CLI, then verify the artifacts prove
  which split was actually used. Set ``SCINET_SKIP_E2E=1`` to skip that part.

The critical regression: on a shared run_id a later ``--split test`` job
overwrites ``horizon_metrics.npy``. A validation job must never inherit those
numbers -- hence split-tagged metric files + ``evaluation_meta.json``.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml

from data.dataset import (
    SPLITS,
    build_dataset,
    dataset_spec,
    dataset_version,
)
from data.frozen import build_frozen_dataset
from research_api import settings
from research_api.eval_service import build_evaluate_command
from research_api.result_parser import build_result, parse_horizon_metrics
from research_api.runner import build_command

CONFIG = "configs/pendulum_smoke.yaml"


def _tiny_config() -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "data": {
            "generator": "pendulum",
            "sequence_length": 80,
            "block_size": 20,
            "t_max": 8.0,
            "train_samples": 8,
            "test_samples": 4,
        },
        "jepa": {"rollout_steps": [1, 2, 3]},
        "train": {"batch_size": 4, "epochs": 1, "lr": 0.001, "weight_decay": 0.0},
        "model": {"latent_dim": 4},
        "tta": {"n_samples": 4, "steps": 1, "lr": 0.0005,
                "adapt_encoder": False},
    }


# --------------------------------------------------------------------------- #
# S10: split flag plumbing
# --------------------------------------------------------------------------- #
class TestSplitPlumbing(unittest.TestCase):
    def test_cli_split_constant_matches_engine(self):
        """The lightweight CLI layer must not drift from data.dataset.SPLITS."""
        self.assertEqual(tuple(settings.DATA_SPLITS), tuple(SPLITS))

    def test_evaluate_command_is_unchanged_when_no_split_requested(self):
        cmd = build_evaluate_command("c.yaml", "r1", python_exe="PY")
        self.assertEqual(cmd, ["PY", "evaluate.py", "--config", "c.yaml",
                               "--run-id", "r1"])

    def test_evaluate_command_forwards_explicit_split(self):
        cmd = build_evaluate_command("c.yaml", "r1", python_exe="PY",
                                     split="validation")
        self.assertEqual(cmd[-2:], ["--split", "validation"])

    def test_split_is_only_forwarded_to_evaluate_stage(self):
        """A stray split must not silently change what other stages train on."""
        for stage in ("train", "analyze", "symbolic", "tta"):
            with self.subTest(stage=stage):
                without = build_command(stage, "c.yaml", "r1")
                with_split = build_command(stage, "c.yaml", "r1",
                                           split="test")
                self.assertEqual(without, with_split)
        self.assertIn("--split", build_command("evaluate", "c.yaml", "r1",
                                               split="test"))

    def test_cli_rejects_unknown_split_before_spawning_a_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "research_api.cli", "submit",
                 "--experiment-id", "e", "--stage", "evaluate",
                 "--config", CONFIG, "--split", "holdout_bogus",
                 "--db", str(Path(tmp) / "jobs.sqlite")],
                cwd=str(_ROOT), capture_output=True, text=True, timeout=60,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr)


# --------------------------------------------------------------------------- #
# S10: result parsing must be split-aware
# --------------------------------------------------------------------------- #
class TestSplitAwareResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation_result_never_inherits_overwritten_canonical_metrics(self):
        """The S10 regression: later --split test must not corrupt validation."""
        np.save(self.out / "horizon_metrics_validation.npy",
                np.array([[1, 0.11, 0.91]]))
        # Simulate a later test job overwriting the canonical file.
        np.save(self.out / "horizon_metrics.npy",
                np.array([[1, 0.99, 0.10]]))
        (self.out / "evaluation_meta.json").write_text(
            json.dumps({"split": "validation"}), encoding="utf-8")
        job = {"stage": "evaluate", "experiment_id": "e", "job_id": "j",
               "spec": {}}

        result = build_result(job=job, status="succeeded", exit_code=0,
                              out_dir=self.out, split="validation")
        self.assertEqual(result["evaluation_split"], "validation")
        self.assertEqual(result["requested_split"], "validation")
        self.assertAlmostEqual(result["metrics"]["latent_mse@1"], 0.11)
        # ...while the canonical file really does hold the test numbers now.
        self.assertAlmostEqual(parse_horizon_metrics(self.out)["latent_mse@1"],
                               0.99)

    def test_tagged_meta_wins_over_a_later_overwritten_canonical_meta(self):
        """Regression: a shared evaluation_meta.json is overwritten by test."""
        np.save(self.out / "horizon_metrics_validation.npy",
                np.array([[1, 0.11, 0.91]]))
        np.save(self.out / "horizon_metrics_test.npy",
                np.array([[1, 0.99, 0.10]]))
        (self.out / "evaluation_meta.json").write_text(
            json.dumps({"split": "test"}), encoding="utf-8")
        (self.out / "evaluation_meta_validation.json").write_text(
            json.dumps({"split": "validation"}), encoding="utf-8")
        job = {"stage": "evaluate", "experiment_id": "e", "job_id": "j",
               "spec": {}}
        result = build_result(job=job, status="succeeded", exit_code=0,
                              out_dir=self.out, split="validation")
        self.assertEqual(result["evaluation_split"], "validation")
        self.assertAlmostEqual(result["metrics"]["latent_mse@1"], 0.11)

    def test_missing_split_tagged_evidence_yields_empty_not_wrong_metrics(self):
        np.save(self.out / "horizon_metrics.npy", np.array([[1, 0.5, 0.5]]))
        self.assertEqual(parse_horizon_metrics(self.out, split="validation"), {})

    def test_train_stage_does_not_report_evaluation_metrics(self):
        np.save(self.out / "horizon_metrics.npy", np.array([[1, 0.5, 0.5]]))
        job = {"stage": "train", "experiment_id": "e", "job_id": "j",
               "spec": {}}
        result = build_result(job=job, status="succeeded", exit_code=0,
                              out_dir=self.out)
        self.assertEqual(result["metrics"], {})
        self.assertIsNone(result["evaluation_split"])

    def test_no_meta_falls_back_to_requested_split(self):
        np.save(self.out / "horizon_metrics_test.npy",
                np.array([[1, 0.22, 0.88]]))
        job = {"stage": "evaluate", "experiment_id": "e", "job_id": "j",
               "spec": {}}
        result = build_result(job=job, status="succeeded", exit_code=0,
                              out_dir=self.out, split="test")
        self.assertEqual(result["evaluation_split"], "test")
        self.assertAlmostEqual(result["metrics"]["latent_mse@1"], 0.22)


# --------------------------------------------------------------------------- #
# S12: seed separation + frozen dataset wiring
# --------------------------------------------------------------------------- #
class TestSeedAndFrozenIsolation(unittest.TestCase):
    def test_model_seed_does_not_change_dataset_version(self):
        cfg = _tiny_config()
        cfg["data"]["dataset_seed"] = 20261109
        baseline = dataset_version(cfg)
        changed = copy.deepcopy(cfg)
        changed["seed"] = 999
        self.assertEqual(dataset_version(changed), baseline)
        self.assertEqual(dataset_spec(cfg)["seed"], 20261109)

    def test_dataset_seed_does_change_dataset_version(self):
        cfg = _tiny_config()
        cfg["data"]["dataset_seed"] = 20261109
        changed = copy.deepcopy(cfg)
        changed["data"]["dataset_seed"] = 20261110
        self.assertNotEqual(dataset_version(cfg), dataset_version(changed))

    def test_legacy_config_without_dataset_seed_keeps_historical_version(self):
        cfg = _tiny_config()
        explicit = copy.deepcopy(cfg)
        explicit["data"]["dataset_seed"] = cfg["seed"]
        self.assertEqual(dataset_version(cfg), dataset_version(explicit))

    def test_stale_manifest_is_rejected_instead_of_silently_reused(self):
        cfg = _tiny_config()
        cfg["data"]["dataset_seed"] = 111
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "frozen"
            build_frozen_dataset(cfg, out, splits=SPLITS)
            stale = copy.deepcopy(cfg)
            stale["data"]["dataset_seed"] = 222
            stale["data"]["frozen_dir"] = str(out)
            with self.assertRaises(RuntimeError) as ctx:
                build_dataset(stale, "test")
            self.assertIn("mismatch", str(ctx.exception).lower())

    def test_smoke_config_really_loads_all_four_frozen_splits(self):
        cfg = yaml.safe_load((_ROOT / CONFIG).read_text(encoding="utf-8"))
        self.assertTrue(cfg["data"].get("frozen_dir"),
                        "smoke config must point at the frozen dataset")
        manifest = json.loads((_ROOT / cfg["data"]["frozen_dir"] /
                               "manifest.json").read_text(encoding="utf-8"))
        for split in SPLITS:
            with self.subTest(split=split):
                ds = build_dataset(cfg, split)
                self.assertEqual(ds.split, split)
                self.assertEqual(ds.dataset_version,
                                 manifest["dataset_version"])
                self.assertTrue(ds.frozen_path)
                self.assertTrue(Path(ds.frozen_path).is_file())

    def test_four_splits_are_pairwise_distinct(self):
        cfg = yaml.safe_load((_ROOT / CONFIG).read_text(encoding="utf-8"))
        blocks = {s: build_dataset(cfg, s).blocks[:2] for s in SPLITS}
        for a in SPLITS:
            for b in SPLITS:
                if a >= b:
                    continue
                with self.subTest(pair=f"{a}/{b}"):
                    self.assertFalse(
                        np.array_equal(blocks[a], blocks[b]),
                        f"{a} and {b} must not share samples",
                    )


# --------------------------------------------------------------------------- #
# S11: TTA split selection (no training needed)
# --------------------------------------------------------------------------- #
class _FakeDataset:
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"blocks": torch.zeros(4, 1)}


class TestTtaSplitSelection(unittest.TestCase):
    def test_adaptation_uses_adapt_split_and_never_mutates_config(self):
        import tta

        cfg = _tiny_config()
        original = copy.deepcopy(cfg)
        with patch.object(tta, "build_dataset",
                          return_value=_FakeDataset()) as mocked:
            blocks = tta.build_tta_batch(cfg, torch.device("cpu"), 0.0)
        self.assertEqual(blocks.shape[0], 4)
        mocked.assert_called_once_with(cfg, split="adapt")
        self.assertEqual(cfg, original,
                         "TTA must not change cfg['data'] (dataset identity)")

    def test_heldout_uses_validation_split(self):
        import tta

        cfg = _tiny_config()
        with patch.object(tta, "build_dataset",
                          return_value=_FakeDataset()) as mocked:
            tta.build_tta_batch(cfg, torch.device("cpu"), 0.0,
                                split=tta.HELDOUT_SPLIT)
        mocked.assert_called_once_with(cfg, split="validation")

    def test_test_split_is_never_the_adaptation_default(self):
        import tta

        self.assertEqual(tta.ADAPT_SPLIT, "adapt")
        self.assertEqual(tta.HELDOUT_SPLIT, "validation")
        self.assertNotIn("test", (tta.ADAPT_SPLIT, tta.HELDOUT_SPLIT))


# --------------------------------------------------------------------------- #
# REAL end-to-end: train -> validation -> test -> tta
# --------------------------------------------------------------------------- #
@unittest.skipIf(os.environ.get("SCINET_SKIP_E2E") == "1",
                 "SCINET_SKIP_E2E=1")
class TestRealSplitIsolationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = str(Path(cls.tmp.name) / "jobs.sqlite")
        cls.tag = uuid.uuid4().hex[:8]
        cls.run_id = f"iso_{cls.tag}"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _cli(self, *args, timeout=240) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "research_api.cli", *args, "--db", self.db],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=timeout,
        )
        self.assertEqual(proc.returncode, 0,
                         f"CLI failed: {args}\nstdout={proc.stdout}\n"
                         f"stderr={proc.stderr}")
        return json.loads(proc.stdout)

    def _submit_wait(self, job_id: str, stage: str, *, split=None) -> dict:
        args = ["submit", "--experiment-id", "split_iso", "--stage", stage,
                "--config", CONFIG, "--run-id", self.run_id,
                "--job-id", job_id]
        if stage == "train":
            args += ["--epochs", "1"]
        if split:
            args += ["--split", split]
        self._cli(*args)
        done = self._cli("wait", "--job-id", job_id, "--timeout", "180",
                         "--poll", "0.5")
        self.assertEqual(done["status"], "succeeded",
                         f"{stage}/{split} failed: {done.get('error')}")
        return self._cli("result", "--job-id", job_id)

    def test_train_validation_test_tta_are_isolated(self):
        # ---- 1) TRAIN must load train + validation, never test -------------
        train = self._submit_wait(f"tr_{self.tag}", "train")
        self.assertEqual(train["status"], "succeeded")
        log = Path(train["log_path"]).read_text(encoding="utf-8",
                                                errors="replace")
        self.assertIn("train split=train", log)
        self.assertIn("validation split=validation", log)
        self.assertNotIn("test split=test", log,
                         "train.py leaked the frozen test split")
        self.assertIn("frozen=", log)
        self.assertNotIn("frozen=None", log,
                         "smoke config must really load frozen .npz files")
        out_dir = Path(train["result_path"]).parent

        # ---- 2) VALIDATION evaluation --------------------------------------
        val = self._submit_wait(f"va_{self.tag}", "evaluate",
                                split="validation")
        self.assertEqual(val["evaluation_split"], "validation")
        self.assertEqual(val["requested_split"], "validation")
        val_metrics = dict(val["metrics"])
        tagged = np.load(out_dir / "horizon_metrics_validation.npy")
        self.assertAlmostEqual(val_metrics["latent_mse@1"], float(tagged[0][1]),
                               places=6)
        self.assertTrue(val["dataset_version"])

        # ---- 3) TEST evaluation on the SAME run_id -------------------------
        test = self._submit_wait(f"te_{self.tag}", "evaluate", split="test")
        self.assertEqual(test["evaluation_split"], "test")
        self.assertEqual(test["requested_split"], "test")
        self.assertTrue((out_dir / "horizon_metrics_test.npy").is_file())

        # REGRESSION: test overwrote the canonical file, but a re-collected
        # validation result must still report validation numbers.
        from research_api.job_store import JobStore
        from research_api.result_parser import build_result as rebuild
        job_record = JobStore(self.db).get(f"va_{self.tag}")
        rebuilt = rebuild(job=job_record, status="succeeded", exit_code=0,
                          out_dir=out_dir, split="validation")
        self.assertEqual(rebuilt["metrics"], val_metrics)
        self.assertEqual(rebuilt["evaluation_split"], "validation")

        # ---- 4) TTA must use adapt + validation, never test ----------------
        tta_result = self._submit_wait(f"tt_{self.tag}", "tta")
        self.assertEqual(tta_result["status"], "succeeded")
        evidence = json.loads((out_dir / "tta_metrics.json").read_text(
            encoding="utf-8"))
        self.assertEqual(evidence["adaptation_split"], "adapt")
        self.assertEqual(evidence["heldout_split"], "validation")
        self.assertFalse(evidence["test_split_used"])
        self.assertIn("generalization", evidence)
        self.assertIn("transductive", evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
