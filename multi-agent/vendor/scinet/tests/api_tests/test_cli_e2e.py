"""Real end-to-end acceptance: Agent-facing CLI -> worker -> train.py -> evaluate.py.

This is an INTEGRATION test: it launches genuine SciNet++ subprocesses using
``configs/pendulum_smoke.yaml`` (CPU, ~20-40s). Set ``SCINET_SKIP_E2E=1`` to
skip it in a fast unit-test run; the P0 acceptance criteria require it enabled.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@unittest.skipIf(os.environ.get("SCINET_SKIP_E2E") == "1",
                 "SCINET_SKIP_E2E=1")
class TestCliEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = str(Path(cls.tmp.name) / "jobs.sqlite")
        cls.tag = uuid.uuid4().hex[:8]
        cls.run_id = f"e2e_{cls.tag}"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _cli(self, *args, timeout=200) -> dict:
        proc = subprocess.run(
            [sys.executable, "-m", "research_api.cli", *args, "--db", self.db],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=timeout,
        )
        self.assertEqual(proc.returncode, 0,
                         f"CLI failed: {args}\nstdout={proc.stdout}\nstderr={proc.stderr}")
        return json.loads(proc.stdout)

    def test_train_then_evaluate_full_loop(self):
        train_job = f"tr_{self.tag}"
        eval_job = f"ev_{self.tag}"

        # 1) submit returns immediately with a job_id (async contract)
        submitted = self._cli(
            "submit", "--experiment-id", "e2e", "--stage", "train",
            "--config", "configs/pendulum_smoke.yaml", "--epochs", "1",
            "--run-id", self.run_id, "--job-id", train_job,
            "--spec-json", json.dumps({"model_version": "baseline-v0",
                                       "dataset_version": "smoke-v1"}),
        )
        self.assertEqual(submitted["status"], "queued")
        self.assertEqual(submitted["job_id"], train_job)
        self.assertEqual(submitted["run_id"], self.run_id)

        # 2) wait for terminal state, then read structured result
        done = self._cli("wait", "--job-id", train_job, "--timeout", "180",
                         "--poll", "0.5", timeout=220)
        self.assertEqual(done["status"], "succeeded", done.get("error"))
        train_result = self._cli("result", "--job-id", train_job)
        self.assertEqual(train_result["status"], "succeeded")
        self.assertIsNotNone(train_result["checkpoint"])
        self.assertIn("checkpoint.pt", train_result["artifacts"])
        self.assertEqual(train_result["model_version"], "baseline-v0")

        # 3) evaluate MUST reuse the training run_id (that's where ckpt lives)
        eval_submit = self._cli(
            "submit", "--experiment-id", "e2e", "--stage", "evaluate",
            "--config", "configs/pendulum_smoke.yaml",
            "--run-id", self.run_id, "--job-id", eval_job,
        )
        self.assertEqual(eval_submit["run_id"], self.run_id)
        eval_done = self._cli("wait", "--job-id", eval_job, "--timeout", "180",
                              "--poll", "0.5", timeout=220)
        self.assertEqual(eval_done["status"], "succeeded", eval_done.get("error"))
        eval_result = self._cli("result", "--job-id", eval_job)

        # 4) metrics are numeric and structured, not scraped from console text
        for k in (1, 2, 3):
            self.assertIn(f"latent_mse@{k}", eval_result["metrics"])
            self.assertIn(f"cosine@{k}", eval_result["metrics"])
            self.assertIsInstance(eval_result["metrics"][f"latent_mse@{k}"], float)
        self.assertEqual(eval_result["dataset_version"],
                         train_result["dataset_version"])

        # 5) REGRESSION: results are per-job; evaluate must NOT overwrite train's
        self.assertNotEqual(train_result["result_path"], eval_result["result_path"])
        self.assertTrue(Path(train_result["result_path"]).is_file())
        self.assertTrue(Path(eval_result["result_path"]).is_file())
        self.assertEqual(json.loads(Path(train_result["result_path"]).read_text(
            encoding="utf-8"))["stage"], "train")

        # 6) status is queryable from a fresh CLI invocation (restart-safe)
        status = self._cli("status", "--job-id", train_job)
        self.assertEqual(status["status"], "succeeded")

    def test_duplicate_job_id_is_refused_by_cli(self):
        # Pre-seed the store directly so no training process is spawned.
        from research_api.job_store import JobStore
        JobStore(self.db).create_job(job_id=f"dup_{self.tag}",
                                     experiment_id="e2e", stage="train")
        proc = subprocess.run(
            [sys.executable, "-m", "research_api.cli", "submit",
             "--experiment-id", "e2e", "--stage", "train",
             "--config", "configs/pendulum_smoke.yaml",
             "--job-id", f"dup_{self.tag}", "--db", self.db],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(proc.returncode, 0)
        payload = json.loads(proc.stdout)
        self.assertIn("error", payload)
        self.assertIn("already exists", payload["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
