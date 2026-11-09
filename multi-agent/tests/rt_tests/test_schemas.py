"""Contract tests for the four shared data structures + ModelVersion."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydantic import ValidationError

from research_runtime.schemas import (
    ExperimentResult,
    ExperimentSpec,
    FailureReport,
    ModelVersion,
    ResearchObjective,
)


class TestSchemas(unittest.TestCase):
    def test_all_structures_roundtrip_json(self):
        objective = ResearchObjective(
            objective_id="obj1", domain="robotics",
            description="push cup to target",
            success_criteria={"success_rate": {"min": 0.5}},
        )
        spec = ExperimentSpec(
            experiment_id="exp1", objective_id="obj1",
            config_path="configs/pendulum.yaml", seed=7,
        )
        result = ExperimentResult(
            experiment_id="exp1", job_id="job1", status="succeeded",
            metrics={"latent_mse@1": 0.02},
        )
        failure = FailureReport(
            experiment_id="exp1", job_id="job1", category="execution_failure",
            recommended_action="retry",
        )
        version = ModelVersion(
            version_id="v1", experiment_id="exp1", job_id="job1",
            checkpoint_sha256="abc", dataset_version="d1",
        )
        cases = (
            (objective, "objective_id", "obj1"),
            (spec, "experiment_id", "exp1"),
            (result, "experiment_id", "exp1"),
            (failure, "experiment_id", "exp1"),
            (version, "version_id", "v1"),
        )
        for model, key, expected in cases:
            with self.subTest(type=type(model).__name__):
                restored = json.loads(model.model_dump_json())
                self.assertEqual(restored[key], expected)
                self.assertEqual(type(model).model_validate(restored), model)

    def test_missing_required_fields_raise(self):
        with self.assertRaises(ValidationError):
            ResearchObjective(domain="robotics")            # missing objective_id
        with self.assertRaises(ValidationError):
            ExperimentSpec(experiment_id="exp1")            # missing objective_id/config_path
        with self.assertRaises(ValidationError):
            ExperimentResult(job_id="job1", status="succeeded")  # missing experiment_id

    def test_status_rejects_unknown_value(self):
        with self.assertRaises(ValidationError):
            ExperimentResult(experiment_id="e", job_id="j", status="almost_done")

    def test_metrics_must_be_finite_numbers(self):
        result = ExperimentResult(
            experiment_id="e", job_id="j", status="succeeded",
            metrics={"mse": 0.1, "nan": float("nan"), "inf": math.inf,
                     "text": "oops", "flag": True, "int_ok": 3},
        )
        self.assertEqual(result.metrics, {"mse": 0.1, "int_ok": 3.0})
        self.assertNotIn("nan", result.metrics)
        self.assertNotIn("text", result.metrics)
        self.assertNotIn("flag", result.metrics)

    def test_failed_run_yields_complete_result(self):
        raw = {
            "experiment_id": "exp1", "job_id": "job1", "stage": "train",
            "status": "failed", "engine": "scinet++", "exit_code": 1,
            "error_type": "process_failed", "error": "CUDA OOM",
            "metrics": {}, "artifacts": ["train.log"],
            "run_id": "run1", "result_path": "/tmp/result.json",
        }
        result = ExperimentResult.from_cli_result(raw)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "process_failed")
        self.assertEqual(result.experiment_id, "exp1")
        self.assertEqual(result.metrics, {})
        # The CONTRACT object is complete even though the engine payload was
        # minimal. (``raw`` is the untouched original, never back-filled.)
        dumped = result.model_dump()
        for key in ("status", "experiment_id", "job_id", "stage", "engine",
                    "metrics", "checkpoint", "artifacts", "error", "error_type",
                    "run_id", "result_path", "dataset_version",
                    "model_version", "started_at", "finished_at",
                    "data_quality_notes", "raw"):
            self.assertIn(key, dumped)
        self.assertEqual(result.artifacts, ["train.log"])
        self.assertEqual(result.run_id, "run1")
        self.assertFalse(result.succeeded)

    def test_from_cli_result_handles_bad_payload(self):
        result = ExperimentResult.from_cli_result({})
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.job_id, "unknown")
        # Unknown engine status is downgraded, never crashes the caller.
        weird = ExperimentResult.from_cli_result(
            {"job_id": "j", "experiment_id": "e", "status": "banana"}
        )
        self.assertEqual(weird.status, "incomplete")
        self.assertTrue(weird.data_quality_notes)

    def test_metric_quality_notes_are_recorded(self):
        result = ExperimentResult.from_cli_result({
            "experiment_id": "e", "job_id": "j", "status": "succeeded",
            "metrics": {"mse": 0.1, "nan": float("nan")},
        })
        self.assertEqual(result.metrics, {"mse": 0.1})
        self.assertTrue(any("nan" in n for n in result.data_quality_notes))

    def test_three_identifiers_are_distinct(self):
        spec = ExperimentSpec(
            experiment_id="exp1", objective_id="obj1",
            config_path="c.yaml", model_version="v0",
        )
        child = spec.with_run_id("run_abc", stage="evaluate", parent_job_id="job_train")
        self.assertEqual(child.run_id, "run_abc")
        self.assertEqual(child.parent_run_id, "run_abc")
        self.assertEqual(child.parent_job_id, "job_train")
        self.assertEqual(child.experiment_id, "exp1")
        self.assertEqual(child.model_version, "v0")
        self.assertEqual(child.stage, "evaluate")
        # original is untouched (no shared mutable state)
        self.assertEqual(spec.stage, "train")
        self.assertIsNone(spec.run_id)

    def test_engine_spec_excludes_runtime_only_fields(self):
        spec = ExperimentSpec(experiment_id="e", objective_id="o",
                              config_path="c.yaml", run_id="r1")
        payload = spec.to_engine_spec()
        self.assertNotIn("run_id", payload)
        self.assertNotIn("parent_job_id", payload)
        self.assertEqual(payload["experiment_id"], "e")
        self.assertEqual(payload["budget"], {})

    def test_incomplete_factory(self):
        result = ExperimentResult.incomplete(
            experiment_id="e", job_id="j", error="engine gone",
        )
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.error_type, "result_unavailable")
        self.assertIsNotNone(result.finished_at)


if __name__ == "__main__":
    unittest.main(verbosity=2)
