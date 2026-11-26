"""JobStore acceptance tests: restart persistence, exactly-once, stale recovery."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_api.job_store import DuplicateJobError, JobStore


class TestJobStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "jobs.sqlite"
        self.store = JobStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def _create(self, job_id="job1"):
        return self.store.create_job(
            job_id=job_id, experiment_id="exp1", stage="train",
            spec={"seed": 42}, command=["python", "train.py"],
            cwd=str(_ROOT), run_id="run1", log_path="x.log",
            result_path="result.json",
        )

    def test_create_and_get(self):
        rec = self._create()
        self.assertEqual(rec["status"], "queued")
        self.assertEqual(rec["spec"], {"seed": 42})
        self.assertEqual(rec["command"], ["python", "train.py"])
        self.assertIsNotNone(rec["created_at"])

    def test_duplicate_job_id_is_rejected(self):
        self._create()
        with self.assertRaises(DuplicateJobError):
            self._create()
        self.assertEqual(len(self.store.list_jobs()), 1)

    def test_claim_running_is_exactly_once(self):
        self._create()
        self.assertTrue(self.store.claim_running("job1", 111))
        self.assertFalse(self.store.claim_running("job1", 222))
        self.assertEqual(self.store.get("job1")["pid"], 111)

    def test_state_survives_store_restart(self):
        self._create()
        self.store.claim_running("job1", 111)
        self.store.update("job1", status="succeeded", exit_code=0)
        reopened = JobStore(self.db)  # simulate Agent process restart
        self.assertEqual(reopened.get("job1")["status"], "succeeded")
        self.assertEqual(reopened.get("job1")["exit_code"], 0)

    def test_invalid_status_and_fields_are_rejected(self):
        self._create()
        with self.assertRaises(ValueError):
            self.store.update("job1", status="banana")
        with self.assertRaises(ValueError):
            self.store.update("job1", job_id="hacked")

    def test_heartbeat_and_stale_reconcile(self):
        self._create()
        self.store.claim_running("job1", 111)
        self.assertFalse(self.store.stale_running_jobs(90.0))
        # Force an old heartbeat to simulate a crashed worker.
        with self.store._session() as conn:
            conn.execute("UPDATE jobs SET heartbeat_at='2000-01-01T00:00:00' "
                         "WHERE job_id='job1'")
        recovered = self.store.reconcile_stale(90.0)
        self.assertEqual(recovered, ["job1"])
        rec = self.store.get("job1")
        self.assertEqual(rec["status"], "failed")
        self.assertEqual(rec["error_type"], "worker_interrupted")

    def test_list_filters(self):
        self._create("job1")
        self.store.create_job(job_id="job2", experiment_id="exp2", stage="evaluate")
        self.assertEqual(len(self.store.list_jobs(experiment_id="exp1")), 1)
        self.assertEqual(len(self.store.list_jobs(status="queued")), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
