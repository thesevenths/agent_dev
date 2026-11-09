"""A08 regression: execution success and research success are NOT the same.

The bug being pinned down here: ``report["success"]`` used to be
``passed and bool(metrics)``. With an empty or broken ``success_criteria`` that
is vacuously True, so a run that merely *executed* was reported as a research
success and the self-improvement loop had no reason to keep going.
"""
from __future__ import annotations

import sqlite3
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
from research_runtime.orchestrator import (
    ResearchOrchestrator,
    criteria_are_decidable,
    evaluate_objective,
)
from research_runtime.schemas import ModelVersion, ResearchObjective

from .fakes import FakeAdapter

METRICS = {"latent_mse@1": 0.0264, "cosine@1": 0.5437}


class TestEvaluateObjective(unittest.TestCase):
    def test_achieved_requires_a_declarable_target_and_a_good_pipeline(self):
        verdict = evaluate_objective({"latent_mse@1": 0.05}, METRICS,
                                     pipeline_success=True)
        self.assertEqual(verdict["status"], "achieved")
        self.assertTrue(verdict["objective_success"])
        self.assertTrue(verdict["pipeline_success"])
        self.assertTrue(verdict["decidable"])

    def test_pipeline_ok_but_target_missed_is_not_achieved(self):
        verdict = evaluate_objective({"latent_mse@1": 0.001}, METRICS,
                                     pipeline_success=True)
        self.assertEqual(verdict["status"], "not_achieved")
        self.assertFalse(verdict["objective_success"])
        self.assertTrue(verdict["pipeline_success"],
                        "missing the target must not look like a broken run")
        self.assertTrue(verdict["violations"])

    def test_no_or_broken_target_is_undefined_not_success(self):
        broken = ({}, {"latent_mse@1": True}, {"latent_mse@1": {}},
                  {"latent_mse@1": {"max": "loose"}},
                  {"latent_mse@1": None})
        for criteria in broken:
            with self.subTest(criteria=criteria):
                verdict = evaluate_objective(criteria, METRICS,
                                             pipeline_success=True)
                self.assertEqual(verdict["status"], "undefined")
                self.assertFalse(verdict["objective_success"])
                self.assertFalse(verdict["decidable"])

    def test_broken_pipeline_is_never_a_success_even_with_a_loose_target(self):
        verdict = evaluate_objective({}, {}, pipeline_success=False)
        self.assertEqual(verdict["status"], "pipeline_failed")
        self.assertFalse(verdict["objective_success"])
        self.assertFalse(verdict["pipeline_success"])

    def test_missing_metric_is_not_achieved_but_the_target_is_decidable(self):
        verdict = evaluate_objective({"does_not_exist": 1.0}, METRICS,
                                     pipeline_success=True)
        self.assertEqual(verdict["status"], "not_achieved")
        self.assertTrue(verdict["decidable"])
        self.assertIn("missing from result", verdict["violations"][0])

    def test_criteria_decidability(self):
        self.assertFalse(criteria_are_decidable({}))
        self.assertFalse(criteria_are_decidable({"a": None}))
        self.assertTrue(criteria_are_decidable({"a": 0.1}))
        self.assertTrue(criteria_are_decidable({"a": {"min": 0.5}}))
        self.assertTrue(criteria_are_decidable({"a": {"min": 0.0, "max": 1.0}}))


class TestReportScoreboards(unittest.TestCase):
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

    def _run(self, criteria, *, fail_stage=None):
        adapter = FakeAdapter(fail_stage=fail_stage, metrics=METRICS)
        manager = ExperimentManager(adapter, evidence_store=self.evidence)
        orch = ResearchOrchestrator(manager, registry=self.registry,
                                    evidence_store=self.evidence)
        objective = ResearchObjective(objective_id="o", domain="science",
                                      success_criteria=criteria)
        return orch.run_baseline(objective, config_path="c.yaml", epochs=1)

    def test_executed_without_a_target_is_not_a_research_success(self):
        report = self._run({})
        self.assertTrue(report["pipeline_success"])
        self.assertFalse(report["objective_success"])
        self.assertFalse(report["success"])
        self.assertEqual(report["objective_status"], "undefined")
        self.assertFalse(report["success_criteria"]["decidable"])
        # the auditable report on disk must carry the same verdict
        from json import loads
        on_disk = loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["objective_status"], "undefined")
        self.assertEqual(on_disk["success"], False)

    def test_real_target_gates_the_verdict(self):
        achieved = self._run({"latent_mse@1": 0.05,
                              "cosine@1": {"min": 0.5}})
        self.assertTrue(achieved["pipeline_success"])
        self.assertTrue(achieved["objective_success"])
        self.assertTrue(achieved["success"])
        self.assertEqual(achieved["objective_status"], "achieved")

        missed = self._run({"latent_mse@1": 0.0001})
        self.assertTrue(missed["pipeline_success"])
        self.assertFalse(missed["objective_success"])
        self.assertEqual(missed["objective_status"], "not_achieved")

    def test_failed_pipeline_short_circuits_both_scoreboards(self):
        report = self._run({"latent_mse@1": 100.0}, fail_stage="evaluate")
        self.assertFalse(report["pipeline_success"])
        self.assertFalse(report["objective_success"])
        self.assertEqual(report["objective_status"], "pipeline_failed")
        self.assertFalse(self.evidence.query(event_type="objective_complete"))

    def test_evidence_ledger_records_both_scoreboards(self):
        self._run({"latent_mse@1": 0.05})
        payload = self.evidence.query(event_type="objective_complete")[-1]["payload"]
        self.assertTrue(payload["pipeline_success"])
        self.assertTrue(payload["objective_success"])
        self.assertEqual(payload["objective_status"], "achieved")
        self.assertEqual(payload["success"], payload["objective_success"])

    def test_report_records_both_repository_heads_and_dirty_flag(self):
        with patch.object(ResearchOrchestrator, "_repo_version",
                          side_effect=lambda p: {"head": "abc1234",
                                                 "dirty": False,
                                                 "root": str(p)}):
            report = self._run({"latent_mse@1": 0.05})
        versions = report["code_versions"]
        self.assertEqual(set(versions), {"engine", "runtime"})
        self.assertEqual(versions["engine"]["head"], "abc1234")
        self.assertEqual(versions["runtime"]["head"], "abc1234")
        mv = self.registry.get(report["model_version"]["version_id"])
        self.assertEqual(mv.code_version, "abc1234")
        self.assertEqual(mv.runtime_code_version, "abc1234")
        self.assertFalse(mv.code_dirty)

    def test_repo_version_is_scoped_to_the_project_directory(self):
        info = ResearchOrchestrator._repo_version(_ROOT)
        self.assertIsInstance(info, dict)
        self.assertTrue(info["head"])
        self.assertIn("dirty", info)
        self.assertEqual(info["root"], str(_ROOT))
        self.assertIsNone(ResearchOrchestrator._repo_version(None))


class TestLegacyRegistryMigration(unittest.TestCase):
    """A registry created before A08 has no runtime_code_version / code_dirty."""

    _OLD_SCHEMA = """
    CREATE TABLE model_versions (
        version_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
        job_id TEXT NOT NULL, engine TEXT, checkpoint_path TEXT,
        checkpoint_sha256 TEXT, config_path TEXT, config_sha256 TEXT,
        code_version TEXT, dataset_version TEXT, parent_version_id TEXT,
        metrics TEXT, promoted INTEGER NOT NULL DEFAULT 0,
        promotion_reason TEXT, created_at TEXT NOT NULL
    );
    """

    def test_additive_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "legacy.sqlite"
            conn = sqlite3.connect(str(db))
            conn.executescript(self._OLD_SCHEMA)
            conn.close()

            registry = ModelRegistry(db)
            registry.register(ModelVersion(
                version_id="mv1", experiment_id="e", job_id="j",
                runtime_code_version="deadbee", code_dirty=True,
            ))
            ModelRegistry(db)  # second open must not add duplicate columns
            mv = registry.get("mv1")
            self.assertEqual(mv.runtime_code_version, "deadbee")
            self.assertTrue(mv.code_dirty)
            check = sqlite3.connect(str(db))
            try:
                cols = {row[1] for row in check.execute(
                    "PRAGMA table_info(model_versions)"
                )}
            finally:
                check.close()
            self.assertIn("code_dirty", cols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
