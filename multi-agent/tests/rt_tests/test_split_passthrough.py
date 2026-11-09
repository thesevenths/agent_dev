"""S10 regression: ExperimentSpec.split must reach the real engine command.

Before this fix the field was serialised into --spec-json (stored in the
JobStore) but never forwarded to evaluate.py, so the Agent could believe it was
evaluating on validation while the engine silently used test.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import research_runtime.orchestrator as orch_mod
from research_runtime.adapters import SciNetAdapter
from research_runtime.evidence_store import EvidenceStore
from research_runtime.model_registry import ModelRegistry
from research_runtime.orchestrator import ResearchOrchestrator
from research_runtime.schemas import (
    ExperimentResult,
    ExperimentSpec,
    ResearchObjective,
)


class TestAdapterSplitForwarding(unittest.TestCase):
    def setUp(self):
        self.adapter = SciNetAdapter(strict=False)

    @staticmethod
    def _spec(**kw):
        defaults = dict(experiment_id="exp1", objective_id="obj1",
                        config_path="configs/pendulum_smoke.yaml")
        defaults.update(kw)
        return ExperimentSpec(**defaults)

    def test_evaluate_submits_split_to_engine_cli(self):
        spec = self._spec(stage="evaluate", run_id="run_42",
                          split="validation")
        with patch.object(self.adapter, "_cli",
                          return_value={"job_id": "j2"}) as cli:
            self.adapter.submit(spec)
        args = cli.call_args.args
        self.assertIn("--split", args)
        self.assertEqual(args[args.index("--split") + 1], "validation")
        self.assertEqual(args[args.index("--run-id") + 1], "run_42")

    def test_non_evaluate_stage_never_gets_a_split_flag(self):
        for stage in ("train", "analyze", "symbolic", "tta"):
            with self.subTest(stage=stage):
                with patch.object(self.adapter, "_cli",
                                  return_value={"job_id": "j"}) as cli:
                    self.adapter.submit(self._spec(stage=stage))
                self.assertNotIn("--split", cli.call_args.args)

    def test_engine_reported_split_is_preserved_in_result(self):
        raw = {
            "experiment_id": "exp1", "job_id": "j1", "status": "succeeded",
            "stage": "evaluate", "evaluation_split": "validation",
            "metrics": {"latent_mse@3": 0.02},
        }
        with patch.object(self.adapter, "_cli", return_value=raw):
            result = self.adapter.collect_result("j1", self._spec())
        self.assertEqual(result.evaluation_split, "validation")
        self.assertEqual(result.metrics["latent_mse@3"], 0.02)

    def test_with_run_id_can_switch_split_and_stage(self):
        spec = self._spec(stage="train", split="train")
        changed = spec.with_run_id("r1", stage="evaluate", split="test")
        self.assertEqual(changed.split, "test")
        self.assertEqual(changed.stage, "evaluate")
        self.assertEqual(spec.split, "train")   # original untouched


class TestOrchestratorEvaluationSplit(unittest.TestCase):
    """The evaluate spec must never inherit the train spec's split='train'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._patcher = patch.object(orch_mod, "ensure_data_dir",
                                     return_value=self.root)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def _run(self, **kwargs):
        manager = MagicMock()
        manager.adapter = SimpleNamespace(name="fake", project_dir=None)
        manager.evidence = EvidenceStore(self.root / "evidence.jsonl")
        manager.submit.side_effect = [{"job_id": "j1"}, {"job_id": "j2"}]
        manager.wait.return_value = "succeeded"
        manager.collect_result.side_effect = [
            ExperimentResult(experiment_id="e", job_id="j1", status="succeeded",
                             run_id="run_1", dataset_version="dv1",
                             checkpoint={"path": "ckpt.pt", "sha256": "abc"}),
            ExperimentResult(experiment_id="e", job_id="j2", status="succeeded",
                             run_id="run_1", dataset_version="dv1",
                             evaluation_split=kwargs.get("evaluation_split", "test"),
                             metrics={"latent_mse@1": 0.01}),
        ]
        orch = ResearchOrchestrator(
            manager,
            registry=ModelRegistry(self.root / "models.sqlite"),
            evidence_store=manager.evidence,
        )
        objective = ResearchObjective(objective_id="o", domain="science",
                                      success_criteria={})
        report = orch.run_baseline(objective, config_path="c.yaml", epochs=1,
                                   **kwargs)
        specs = [call.args[0] for call in manager.submit.call_args_list]
        return report, specs

    def test_default_evaluation_uses_frozen_test_split(self):
        report, (train_spec, eval_spec) = self._run()
        self.assertEqual(train_spec.split, "train")
        self.assertEqual(eval_spec.stage, "evaluate")
        self.assertEqual(eval_spec.split, "test")
        self.assertEqual(report["evaluation_split"], "test")
        self.assertEqual(report["evaluate"]["evaluation_split"], "test")

    def test_candidate_selection_can_request_validation(self):
        report, (_, eval_spec) = self._run(evaluation_split="validation")
        self.assertEqual(eval_spec.split, "validation")
        self.assertEqual(report["evaluation_split"], "validation")
        self.assertEqual(report["evaluate"]["evaluation_split"], "validation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
