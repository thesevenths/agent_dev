"""SciNetAdapter tests: command construction, JSON contract, failure fallback."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.adapters import SciNetAdapter
from research_runtime.adapters.scinet_adapter import AdapterError
from research_runtime.schemas import ExperimentSpec


class TestSciNetAdapter(unittest.TestCase):
    def setUp(self):
        # strict=False keeps the test independent of the machine's real paths.
        self.adapter = SciNetAdapter(strict=False)

    @staticmethod
    def _spec(**kw):
        defaults = dict(experiment_id="exp1", objective_id="obj1",
                        config_path="configs/pendulum_smoke.yaml")
        defaults.update(kw)
        return ExperimentSpec(**defaults)

    def test_submit_builds_correct_cli_args(self):
        spec = self._spec(budget={"epochs": 2, "job_id": "fixed_job",
                                  "timeout_s": 900})
        payload = {"job_id": "fixed_job", "status": "queued",
                   "run_id": "run_1"}
        with patch.object(self.adapter, "_cli", return_value=payload) as cli:
            job_id = self.adapter.submit(spec)
        self.assertEqual(job_id, "fixed_job")
        args = cli.call_args.args
        self.assertEqual(args[0], "submit")
        self.assertIn("--experiment-id", args)
        self.assertIn("exp1", args)
        self.assertIn("--stage", args)
        self.assertIn("train", args)
        self.assertIn("--epochs", args)
        self.assertIn("2", args)
        self.assertIn("--job-id", args)
        self.assertIn("fixed_job", args)
        spec_json = args[args.index("--spec-json") + 1]
        self.assertIn('"timeout_s": 900', spec_json)

    def test_evaluate_requires_run_id_propagated(self):
        spec = self._spec(stage="evaluate", run_id="run_42")
        with patch.object(self.adapter, "_cli",
                          return_value={"job_id": "j2"}) as cli:
            self.adapter.submit(spec)
        args = cli.call_args.args
        self.assertIn("--run-id", args)
        self.assertEqual(args[args.index("--run-id") + 1], "run_42")

    def test_unsupported_stage_is_rejected_locally(self):
        with self.assertRaises(AdapterError):
            self.adapter.submit(self._spec(stage="teleport"))

    def test_collect_result_maps_engine_payload(self):
        raw = {
            "experiment_id": "exp1", "job_id": "j1", "status": "succeeded",
            "engine": "scinet++", "stage": "evaluate", "run_id": "run1",
            "metrics": {"latent_mse@1": 0.0123, "cosine@1": 0.93},
            "artifacts": ["horizon_metrics.npy"],
        }
        with patch.object(self.adapter, "_cli", return_value=raw):
            result = self.adapter.collect_result("j1", self._spec())
        self.assertTrue(result.succeeded)
        self.assertAlmostEqual(result.metrics["latent_mse@1"], 0.0123)
        self.assertEqual(result.engine, "scinet++")
        self.assertEqual(result.raw["job_id"], "j1")

    def test_cli_failure_becomes_incomplete_result_not_exception(self):
        with patch.object(self.adapter, "_cli",
                          side_effect=AdapterError("engine unreachable")):
            result = self.adapter.collect_result("j1", self._spec())
        self.assertEqual(result.status, "incomplete")
        self.assertIn("engine unreachable", result.error)
        self.assertFalse(result.succeeded)

    def test_status_and_wait_use_cli(self):
        with patch.object(self.adapter, "_cli",
                          return_value={"status": "succeeded"}) as cli:
            self.assertEqual(self.adapter.get_status("j1"), "succeeded")
            self.assertEqual(self.adapter.wait("j1", timeout=120), "succeeded")
        self.assertEqual(cli.call_args.args[0], "wait")

    def test_capabilities_are_engine_agnostic_metadata(self):
        caps = self.adapter.capabilities()
        self.assertEqual(caps["transport"], "subprocess/json-cli")
        self.assertIn("train", caps["stages"])


class TestModelAdapterContract(unittest.TestCase):
    def test_default_wait_uses_status_polling(self):
        from .fakes import FakeAdapter
        adapter = FakeAdapter()
        spec = ExperimentSpec(experiment_id="e", objective_id="o",
                              config_path="c.yaml")
        job = adapter.submit(spec)
        self.assertEqual(adapter.wait(job, timeout=1), "succeeded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
