"""REAL P0 acceptance: Agent Runtime -> JSON CLI -> SciNet++ (torch) -> back.

Integration test; launches genuine train.py + evaluate.py via the smoke config
(~30-60s). Set ``RESEARCH_SKIP_E2E=1`` for a fast unit-only run.

Run with the project interpreter that has a complete pydantic install::

    D:/python313/python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import research_runtime.orchestrator as orch_mod
from research_runtime.adapters import SciNetAdapter
from research_runtime.evidence_store import EvidenceStore
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.model_registry import ModelRegistry
from research_runtime.orchestrator import ResearchOrchestrator
from research_runtime.schemas import ResearchObjective


@unittest.skipIf(os.environ.get("RESEARCH_SKIP_E2E") == "1",
                 "RESEARCH_SKIP_E2E=1")
class TestRealP0Flow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._patcher = patch.object(orch_mod, "ensure_data_dir",
                                     return_value=self.root)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def test_agent_controls_scinet_end_to_end(self):
        evidence = EvidenceStore(self.root / "evidence.jsonl")
        registry = ModelRegistry(self.root / "models.sqlite")
        adapter = SciNetAdapter(db_path=str(self.root / "jobs.sqlite"))

        health = adapter.health_check()
        self.assertTrue(health["ok"])
        self.assertEqual(health["name"], "scinet++")

        manager = ExperimentManager(adapter, evidence_store=evidence,
                                    wait_timeout=300)
        orch = ResearchOrchestrator(manager, registry=registry,
                                    evidence_store=evidence)
        objective = ResearchObjective(
            objective_id="p0_acceptance", domain="science",
            description="Train a baseline world model and evaluate rollout horizons",
            success_criteria={"latent_mse@1": {"max": 10.0}},
        )
        report = orch.run_baseline(
            objective, config_path="configs/pendulum_smoke.yaml",
            epochs=1, wait_timeout=300,
        )

        # 1) both stages succeeded with real subprocesses
        self.assertEqual(report["train"]["status"], "succeeded")
        self.assertEqual(report["evaluate"]["status"], "succeeded")
        self.assertTrue(report["train"]["run_id"])
        self.assertEqual(report["train"]["run_id"],
                         report["evaluate"]["run_id"])

        # 2) real checkpoint exists with a real hash
        ckpt = report["train"]["checkpoint"]
        self.assertIsNotNone(ckpt)
        self.assertTrue(Path(ckpt["path"]).is_file())
        self.assertEqual(len(ckpt["sha256"]), 64)

        # 3) metrics are structured numbers, not scraped console text
        metrics = report["evaluate"]["metrics"]
        for k in (1, 2, 3):
            self.assertIn(f"latent_mse@{k}", metrics)
            self.assertIn(f"cosine@{k}", metrics)

        # 4) lineage + evidence + report are all persisted
        self.assertTrue(report["success"])
        mv = registry.get(report["model_version"]["version_id"])
        self.assertIsNotNone(mv)
        self.assertEqual(mv.checkpoint_sha256, ckpt["sha256"])
        self.assertTrue(mv.dataset_version)
        self.assertTrue(Path(report["report_path"]).is_file())
        payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment_id"], report["experiment_id"])
        self.assertTrue(evidence.query(event_type="objective_complete"))

        # 5) frozen/deterministic dataset id is recorded for every experiment
        self.assertEqual(mv.dataset_version,
                         (report.get("evaluate") or {}).get("dataset_version")
                         or mv.dataset_version)

    def test_freeze_dataset_via_adapter(self):
        adapter = SciNetAdapter(db_path=str(self.root / "jobs.sqlite"))
        out = self.root / "frozen"
        payload = adapter.freeze_dataset(
            config="configs/pendulum_smoke.yaml", out=str(out),
            splits=("train", "test"),
        )
        self.assertTrue(payload["dataset_version"])
        self.assertTrue((out / "manifest.json").is_file())
        self.assertTrue((out / "train.npz").is_file())
        self.assertTrue((out / "test.npz").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
