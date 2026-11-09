"""P0 orchestration: train -> evaluate on same run -> lineage + evidence."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import research_runtime.orchestrator as orch_mod
from research_runtime.evidence_store import EvidenceStore
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.model_registry import ModelRegistry
from research_runtime.orchestrator import ResearchOrchestrator, check_success_criteria
from research_runtime.schemas import ResearchObjective

from .fakes import FakeAdapter


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evidence = EvidenceStore(self.root / "evidence.jsonl")
        self.registry = ModelRegistry(self.root / "models.sqlite")
        self._patcher = patch.object(orch_mod, "ensure_data_dir",
                                     return_value=self.root)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def _orch(self, adapter=None):
        adapter = adapter or FakeAdapter()
        manager = ExperimentManager(adapter, evidence_store=self.evidence)
        orch = ResearchOrchestrator(manager, registry=self.registry,
                                    evidence_store=self.evidence)
        return adapter, orch

    def _objective(self, **criteria):
        return ResearchObjective(objective_id="obj1", domain="science",
                                 description="baseline world model",
                                 success_criteria=criteria)

    def test_full_baseline_loop(self):
        adapter, orch = self._orch()
        report = orch.run_baseline(self._objective(),
                                   config_path="configs/pendulum_smoke.yaml")
        self.assertTrue(report["success"])
        self.assertEqual(report["train"]["run_id"], "run1")
        self.assertEqual(report["evaluate"]["run_id"], "run1")
        self.assertEqual(len(adapter.submits), 2)
        # evaluate MUST be wired to the training run, not a fresh one
        self.assertEqual(adapter.submits[1][1].run_id, "run1")
        self.assertEqual(adapter.submits[1][1].stage, "evaluate")
        self.assertEqual(adapter.submits[1][1].parent_job_id, "job1")
        self.assertIn("latent_mse@1", report["evaluate"]["metrics"])

        mv = self.registry.get(report["model_version"]["version_id"])
        self.assertIsNotNone(mv)
        self.assertEqual(mv.checkpoint_sha256, "deadbeef")
        self.assertEqual(mv.dataset_version, "dataset-v1")
        self.assertEqual(mv.metrics["cosine@1"], 0.91)
        self.assertEqual(mv.parent_version_id, None)

        self.assertTrue(Path(report["report_path"]).is_file())
        self.assertEqual(json.loads(Path(report["report_path"]).read_text(
            encoding="utf-8"))["experiment_id"], report["experiment_id"])
        self.assertTrue(self.evidence.query(event_type="objective_complete"))

    def test_success_criteria_gate(self):
        _, orch = self._orch()
        report = orch.run_baseline(
            self._objective(**{"latent_mse@1": 0.001}),
            config_path="c.yaml",
        )
        self.assertFalse(report["success"])
        self.assertEqual(report["success_criteria"]["passed"], False)
        self.assertTrue(report["success_criteria"]["violations"])
        # a rejected candidate still leaves full evidence + lineage
        self.assertIsNotNone(report["model_version"])
        self.assertTrue(Path(report["report_path"]).is_file())

    def test_train_failure_short_circuits_with_failure_report(self):
        _, orch = self._orch(FakeAdapter(fail_stage="train"))
        report = orch.run_baseline(self._objective(), config_path="c.yaml")
        self.assertFalse(report["success"])
        self.assertNotIn("evaluate", report)
        self.assertIsNone(report["model_version"])
        self.assertTrue(report["failure_reports"])
        self.assertEqual(report["failure_reports"][0]["layer"], "runtime")

    def test_evaluate_failure_short_circuits(self):
        adapter, orch = self._orch(FakeAdapter(fail_stage="evaluate"))
        report = orch.run_baseline(self._objective(), config_path="c.yaml")
        self.assertEqual(len(adapter.submits), 2)
        self.assertFalse(report["success"])
        self.assertIsNone(report["model_version"])
        self.assertTrue(report["failure_reports"])

    def test_run_stage_and_summarize(self):
        _, orch = self._orch()
        from research_runtime.schemas import ExperimentSpec
        spec = ExperimentSpec(experiment_id="e", objective_id="o",
                              config_path="c.yaml", stage="evaluate")
        single = orch.run_stage(spec)
        self.assertEqual(single["result"]["status"], "succeeded")
        self.assertIsInstance(orch_mod.ResearchOrchestrator.summarize(
            {"experiment_id": "e", "success": True,
             "evaluate": {"metrics": {"a": 0.5}}}), str)

    def test_promotion_is_explicit(self):
        _, orch = self._orch()
        report = orch.run_baseline(self._objective(), config_path="c.yaml",
                                   promote=True)
        mv = self.registry.get(report["model_version"]["version_id"])
        self.assertTrue(mv.promoted)
        self.assertEqual(mv.promotion_reason, "P0 baseline accepted")


class TestSuccessCriteria(unittest.TestCase):
    def test_upper_lower_and_missing(self):
        metrics = {"mse": 0.05, "acc": 0.7}
        ok, why = check_success_criteria({"mse": 0.1, "acc": {"min": 0.8}},
                                         metrics)
        self.assertFalse(ok)
        self.assertEqual(len(why), 1)
        ok2, _ = check_success_criteria({"mse": {"max": 0.1, "min": 0.01},
                                         "acc": {"min": 0.6}}, metrics)
        self.assertTrue(ok2)
        ok3, why3 = check_success_criteria({"missing": 1.0}, metrics)
        self.assertFalse(ok3)
        self.assertIn("missing", why3[0])

    def test_empty_criteria_always_pass(self):
        self.assertEqual(check_success_criteria({}, {"a": 1.0}), (True, []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
