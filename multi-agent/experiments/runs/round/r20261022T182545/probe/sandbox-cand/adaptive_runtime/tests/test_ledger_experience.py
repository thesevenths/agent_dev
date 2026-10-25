import tempfile
import unittest
from pathlib import Path

from adaptive_runtime.experience import Experience, Store, confidence
from adaptive_runtime.ledger import Ledger, fingerprint


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.tmp.name) / "h.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_and_duplicate(self):
        hyp = self.ledger.create("AGENT_PARSE_TRIES", "3", "5", "EXECUTION_FAILURE", "claim")
        self.assertEqual(hyp.fingerprint, fingerprint("AGENT_PARSE_TRIES", "3", "5", "EXECUTION_FAILURE"))
        with self.assertRaises(ValueError):
            self.ledger.create("AGENT_PARSE_TRIES", "3", "5", "EXECUTION_FAILURE", "claim again")

    def test_status_flow_reopens_after_reject(self):
        fp = self.ledger.create("A", "1", "2", "T", "c").fingerprint
        self.ledger.update(fp, "rejected", {"n": 4})
        dup, prev = self.ledger.is_duplicate(fp)
        self.assertTrue(dup)
        self.assertEqual(prev["status"], "rejected")

    def test_inconclusive_streak_promotes(self):
        fp = self.ledger.create("A", "1", "2", "T", "c").fingerprint
        for _ in range(3):
            self.ledger.update(fp, "inconclusive")
        self.assertEqual(self.ledger.inconclusive_streak(fp), 3)
        row = self.ledger.update(fp, "rejected", notes="样本不足")
        self.assertIn("inconclusive", row["notes"])

    def test_rejected_context(self):
        fp = self.ledger.create("A", "1", "2", "T", "c").fingerprint
        self.ledger.update(fp, "rejected", {"n": 4, "dQ": 0.2, "reason": "within_noise"})
        ctx = self.ledger.rejected_context()
        self.assertIn(fp, ctx)

    def test_unknown_status(self):
        fp = self.ledger.create("A", "1", "2", "T", "c").fingerprint
        with self.assertRaises(ValueError):
            self.ledger.update(fp, "bogus")


class TestConfidence(unittest.TestCase):
    def test_full_sample_no_regression_sig(self):
        self.assertAlmostEqual(confidence(8, 8, 0.0, 0.01), 0.997, places=3)

    def test_small_sample_discount(self):
        self.assertLess(confidence(2, 8, 0.0, 0.01), confidence(8, 8, 0.0, 0.01))
        self.assertLess(confidence(2, 8, 0.0, 0.01), 0.75)

    def test_regression_discount(self):
        self.assertLess(confidence(8, 8, 0.5, 0.01), 0.85)

    def test_bounds(self):
        self.assertGreaterEqual(confidence(0, 0, 1.0, 1.0), 0.0)
        self.assertLessEqual(confidence(100, 8, 0.0, 0.0), 1.0)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "e.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _mk(self, problem, patch_hash, conf_tasks=("REP-001", "CAL-001"), **kw):
        return self.store.create(problem=problem, claim="c", component="L1/x",
                                 edit={"patch_hash": patch_hash},
                                 n_pairs=8, n_target=8, p_value=0.01,
                                 regression_rate=0.0, dQ=5.0, dC=0.05,
                                 positive=list(conf_tasks), negative=[], **kw)

    def test_idempotent_add(self):
        self._mk("COMPLETION_FAILURE|code|3", "p1")
        self._mk("COMPLETION_FAILURE|code|3", "p1")
        self.assertEqual(len(self.store.rows()), 1)

    def test_retrieve_exact_first(self):
        self._mk("COMPLETION_FAILURE|code|3", "p1")
        self._mk("COMPLETION_FAILURE|crawler|1", "p2")
        self._mk("PLANNING_FAILURE|supervisor|0", "p3")
        got = self.store.retrieve("COMPLETION_FAILURE|code|3", top=2)
        self.assertEqual(got[0].experience_id, self.store.rows()[0]["experience_id"])
        self.assertEqual(len(got), 2)

    def test_retrieve_skips_other_taxonomy(self):
        self._mk("PLANNING_FAILURE|supervisor|0", "p3")
        self.assertEqual(self.store.retrieve("COMPLETION_FAILURE|x|1"), [])

    def test_refute_excludes(self):
        e = self._mk("COMPLETION_FAILURE|code|3", "p1")
        self.assertTrue(self.store.refute(e.experience_id, "复现失败"))
        self.assertEqual(self.store.retrieve("COMPLETION_FAILURE|code|3"), [])

    def test_promotion_needs_two_families(self):
        self._mk("A|x|1", "p1", conf_tasks=("REP-001",))
        self.assertEqual(self.store.promotion_candidates(), [])
        self._mk("A|x|2", "p2", conf_tasks=("REP-002", "CAL-003"))
        self.assertEqual(len(self.store.promotion_candidates()), 1)

    def test_lesson_marks_unverified(self):
        e = Experience(experience_id="exp_x", problem="T|a|1", claim="c",
                       component="L1/x", confidence=0.2)
        self.assertIn("待验证", e.lesson())
        e.confidence = 0.9
        self.assertIn("已验证", e.lesson())


if __name__ == "__main__":
    unittest.main()
