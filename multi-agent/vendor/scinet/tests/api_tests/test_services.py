"""Pure-function tests: command builders and machine-readable result parsing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from research_api.eval_service import build_evaluate_command
from research_api.result_parser import (
    build_result,
    collect_artifacts,
    describe_checkpoint,
    parse_horizon_metrics,
    read_result,
    write_result,
)
from research_api.runner import SUPPORTED_STAGES, build_command, make_run_id
from research_api.train_service import build_train_command


class TestCommandBuilders(unittest.TestCase):
    def test_train_always_passes_run_id(self):
        cmd = build_train_command("configs/pendulum.yaml", "run_abc",
                                  epochs=3, python_exe="PY")
        self.assertEqual(cmd[:5], ["PY", "train.py", "--config",
                                   "configs/pendulum.yaml", "--run-id"])
        self.assertEqual(cmd[5], "run_abc")
        self.assertIn("--epochs", cmd)
        self.assertIn("3", cmd)
        self.assertNotIn("--new", cmd)

    def test_evaluate_uses_same_run_id_and_no_new(self):
        cmd = build_evaluate_command("configs/pendulum.yaml", "run_abc",
                                     python_exe="PY")
        self.assertEqual(cmd, ["PY", "evaluate.py", "--config",
                               "configs/pendulum.yaml", "--run-id", "run_abc"])
        self.assertNotIn("--new", cmd)

    def test_stage_registry(self):
        for stage in SUPPORTED_STAGES:
            with self.subTest(stage=stage):
                cmd = build_command(stage, "c.yaml", "r1")
                self.assertIn("--run-id", cmd)
                self.assertIn("r1", cmd)
        with self.assertRaises(ValueError):
            build_command("unknown_stage", "c.yaml", "r1")

    def test_run_ids_are_unique_per_job(self):
        self.assertNotEqual(make_run_id("aaaaaaaa"), make_run_id("bbbbbbbb"))
        self.assertTrue(make_run_id("aaaaaaaa").endswith("_aaaaaaaa"))


class TestResultParser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_horizon_metrics(self):
        np.save(self.out / "horizon_metrics.npy",
                np.array([[1, 0.01, 0.9], [3, 0.03, 0.8], [5, 0.06, 0.7]]))
        m = parse_horizon_metrics(self.out)
        self.assertEqual(set(m), {
            "latent_mse@1", "cosine@1", "latent_mse@3", "cosine@3",
            "latent_mse@5", "cosine@5",
        })
        self.assertAlmostEqual(m["latent_mse@3"], 0.03)
        self.assertAlmostEqual(m["cosine@5"], 0.7)

    def test_missing_metrics_returns_empty_dict(self):
        self.assertEqual(parse_horizon_metrics(self.out), {})

    def test_checkpoint_descriptor(self):
        (self.out / "checkpoint.pt").write_bytes(b"fake weights")
        info = describe_checkpoint(self.out)
        self.assertEqual(info["bytes"], len(b"fake weights"))
        self.assertEqual(len(info["sha256"]), 64)
        self.assertIsNone(describe_checkpoint(self.out / "nope"))

    def test_artifacts_exclude_all_result_files(self):
        for name in ("checkpoint.pt", "train.log", "result.json",
                     "result_job1.json", "result_job2.json"):
            (self.out / name).write_text("x")
        self.assertEqual(collect_artifacts(self.out),
                         ["checkpoint.pt", "train.log"])

    def test_build_result_has_full_shape_on_failure(self):
        job = {
            "experiment_id": "exp1", "job_id": "job1", "stage": "train",
            "run_id": "run1", "spec": {"model_version": "v1"},
            "log_path": None, "result_path": None,
        }
        result = build_result(job=job, status="failed", exit_code=1,
                              out_dir=self.out, error="boom",
                              error_type="process_failed")
        for key in ("schema_version", "experiment_id", "job_id", "stage",
                    "status", "engine", "run_id", "model_version",
                    "dataset_version", "metrics", "artifacts", "error",
                    "error_type", "exit_code"):
            self.assertIn(key, result)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["model_version"], "v1")
        self.assertEqual(result["metrics"], {})

    def test_write_read_result_is_atomic_and_json(self):
        path = self.out / "result_abc.json"
        write_result(path, {"a": 1, "metrics": {"mse": 0.5}})
        self.assertEqual(read_result(path)["metrics"]["mse"], 0.5)
        self.assertFalse((self.out / "result_abc.json.tmp").exists())
        self.assertIsNone(read_result(self.out / "missing.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
