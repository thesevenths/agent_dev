"""ExperimentManager lifecycle + evidence ledger."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.evidence_store import EvidenceStore
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.schemas import ExperimentSpec

from .fakes import FakeAdapter


class TestExperimentManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.evidence = EvidenceStore(Path(self.tmp.name) / "evidence.jsonl")
        self.adapter = FakeAdapter()
        self.manager = ExperimentManager(self.adapter,
                                         evidence_store=self.evidence)

    def tearDown(self):
        self.tmp.cleanup()

    def _spec(self, stage="train"):
        return ExperimentSpec(experiment_id="exp1", objective_id="obj1",
                              config_path="c.yaml", stage=stage)

    def test_submit_returns_ids_and_records_spec(self):
        spec = self._spec()
        record = self.manager.submit(spec)
        self.assertEqual(record["status"], "queued")
        self.assertEqual(record["job_id"], "job1")
        self.assertEqual(record["experiment_id"], "exp1")
        self.assertIs(self.manager.get_spec("job1"), spec)

    def test_full_lifecycle(self):
        spec = self._spec()
        record = self.manager.submit(spec)
        job = record["job_id"]
        self.assertEqual(self.manager.get_status(job), "succeeded")
        self.assertEqual(self.manager.wait(job), "succeeded")
        result = self.manager.collect_result(job)
        self.assertTrue(result.succeeded)
        self.assertIsNotNone(result.checkpoint)

    def test_run_convenience(self):
        job, result = self.manager.run(self._spec(stage="evaluate"))
        self.assertEqual(job, "job1")
        self.assertIn("latent_mse@1", result.metrics)

    def test_evidence_is_append_only_and_queryable(self):
        self.manager.run(self._spec())
        self.manager.run(self._spec(stage="evaluate"))
        events = self.evidence.query(experiment_id="exp1")
        types = [e["event_type"] for e in events]
        self.assertIn("submitted", types)
        self.assertIn("result", types)
        self.assertEqual(len(self.evidence.query(event_type="submitted")), 2)
        self.assertEqual(len(self.evidence.query(event_type="result")), 2)

    def test_evidence_never_contains_raw_payload(self):
        self.manager.run(self._spec(stage="evaluate"))
        for event in self.evidence.query(event_type="result"):
            self.assertNotIn("raw", event["payload"]["result"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
