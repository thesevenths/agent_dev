"""Two-layer failure diagnosis driven by structured fields only."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.failure_diagnoser import (
    FailureDiagnoser,
    _degrades_over_horizon,
)
from research_runtime.schemas import ExperimentResult


def result(**kw) -> ExperimentResult:
    base = dict(experiment_id="e", job_id="j", status="succeeded",
                stage="evaluate", engine="scinet++")
    base.update(kw)
    return ExperimentResult(**base)


class TestFailureDiagnoser(unittest.TestCase):
    def setUp(self):
        self.diag = FailureDiagnoser()

    def test_healthy_result_has_no_failure(self):
        r = result(metrics={"latent_mse@1": 0.02, "latent_mse@3": 0.021,
                            "cosine@1": 0.9, "cosine@3": 0.89})
        self.assertIsNone(self.diag.diagnose(r))

    def test_runtime_vs_model_layers_are_separate(self):
        r = result(status="failed", error_type="timeout")
        reports = self.diag.diagnose_all(r)
        self.assertTrue(all(x.layer == "runtime" for x in reports))
        self.assertEqual(reports[0].category, "execution_budget")
        self.assertEqual(reports[0].severity, "high")
        self.assertEqual(self.diag.diagnose(r, layer="model"), None)

    def test_missing_artifact_and_worker_loss(self):
        for error_type, category in (
            ("missing_artifact", "missing_artifact"),
            ("worker_interrupted", "execution_infrastructure"),
            ("process_failed", "execution_failure"),
        ):
            with self.subTest(error_type=error_type):
                r = result(status="failed", error_type=error_type)
                self.assertEqual(self.diag.diagnose(r).category, category)

    def test_no_metrics_on_evaluation_is_model_layer(self):
        r = result(metrics={})
        report = self.diag.diagnose(r, layer="model")
        self.assertIsNotNone(report)
        self.assertEqual(report.category, "evidence_gap")

    def test_metric_quality_notes_trigger_rule(self):
        r = result(metrics={"mse": 0.1},
                   data_quality_notes=["dropped non-finite metric: nan"])
        self.assertEqual(self.diag.diagnose(r, layer="model").category,
                         "data_quality")

    def test_horizon_degradation_is_generic(self):
        # lower-is-better metric rising with horizon
        self.assertTrue(_degrades_over_horizon(result(metrics={
            "prediction_mse@1": 0.01, "prediction_mse@2": 0.02,
            "prediction_mse@4": 0.04,
        })))
        # higher-is-better metric falling with horizon
        self.assertTrue(_degrades_over_horizon(result(metrics={
            "accuracy_score@1": 0.9, "accuracy_score@4": 0.3,
        })))
        # stable metrics -> no degradation
        self.assertFalse(_degrades_over_horizon(result(metrics={
            "prediction_mse@1": 0.01, "prediction_mse@4": 0.012,
        })))
        # unrelated flat metric names are ignored, not guessed at
        self.assertFalse(_degrades_over_horizon(result(metrics={
            "throughput@1": 10, "throughput@4": 100,
        })))

    def test_degradation_report_has_actionable_next_step(self):
        r = result(metrics={"loss@1": 0.01, "loss@4": 0.09})
        reports = self.diag.diagnose_all(r, layer="model")
        self.assertTrue(any(x.category == "dynamics_error_accumulation"
                            for x in reports))
        target = next(x for x in reports
                      if x.category == "dynamics_error_accumulation")
        self.assertIn("frozen test", target.recommended_action.lower())

    def test_diagnose_runtime_factory(self):
        report = FailureDiagnoser.diagnose_runtime(
            experiment_id="e", job_id="j", category="planning_failure",
            error="supervisor loop stalled",
        )
        self.assertEqual(report.layer, "runtime")
        self.assertEqual(report.severity, "high")

    def test_custom_rules_are_injectable(self):
        from research_runtime.failure_diagnoser import DiagnosticRule
        custom = DiagnosticRule(
            name="always", category="custom", layer="model", severity="low",
            hypothesis="test", recommended_action="act",
            match=lambda r: True,
        )
        diag = FailureDiagnoser(rules=[custom])
        self.assertEqual(len(diag.diagnose_all(result(metrics={"a": 1}))), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
