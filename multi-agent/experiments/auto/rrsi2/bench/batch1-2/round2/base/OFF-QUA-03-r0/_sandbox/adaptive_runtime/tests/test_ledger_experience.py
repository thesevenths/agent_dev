import tempfile
import unittest
from pathlib import Path

from adaptive_runtime.experience import Experience, Store, confidence
from adaptive_runtime.gate import Decision, Reason, ledger_status
from adaptive_runtime.ledger import Ledger, fingerprint


class TestLedgerStatusClassification(unittest.TestCase):
    """「这轮没拿到可用证据」不能写成 rejected。

    rejected 在台账里等价于「永久不再重提」（``is_duplicate`` 会跳过它）。旧写法把
    一次超时、一次 Stage 1 止损、一次尺子抖动都写成 rejected —— 一轮 78 个 run 里
    只要挂 1 个（实测失败率 ≈ 3% ⇒ 整轮干净概率 ≈ 9%），这个假设就被永久烧掉了；
    搜索空间一共十几个旋钮，几轮之后闭环只会每轮 SKIP。
    """

    def test_incomplete_evidence_is_retryable(self):
        st = ledger_status(Decision("REJECT", Reason.INCOMPLETE_EVIDENCE.value))
        self.assertEqual(st, "inconclusive")

    def test_stop_loss_is_retryable(self):
        """止损只看了 4 个任务就收摊，那是对成本的交代，不是对假设的结论。"""
        st = ledger_status(Decision("REJECT", Reason.REGRESSION.value),
                           stopped="stop_loss")
        self.assertEqual(st, "inconclusive")

    def test_within_noise_and_judge_and_cost_are_retryable(self):
        for reason in (Reason.WITHIN_NOISE.value, Reason.JUDGE_UNSTABLE.value,
                       Reason.COST_CAP.value, Reason.COST_RULE.value):
            self.assertEqual(ledger_status(Decision("REJECT", reason)), "inconclusive",
                             reason)

    def test_refutations_burn_the_hypothesis(self):
        for reason in (Reason.REGRESSION.value, Reason.CONTRACT_FLOOR.value,
                       Reason.INERT_MECHANISM.value, Reason.BENCHMARK_LEAKAGE.value,
                       Reason.FROZEN_CORE_VIOLATION.value, Reason.EDIT_BUDGET.value):
            self.assertEqual(ledger_status(Decision("REJECT", reason)), "rejected", reason)

    def test_accept_and_missing_decision(self):
        self.assertEqual(ledger_status(Decision("ACCEPT", "VALIDATED")), "accepted")
        self.assertEqual(ledger_status(None), "inconclusive")

    def test_timeout_does_not_make_the_hypothesis_duplicate(self):
        """端到端：按新状态写台账后，proposer 的查重必须还能放行这个假设。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        L = Ledger(Path(tmp.name) / "h.jsonl")
        fp = L.create("AGENT_MAX_ITERATIONS", "15", "25", "EXECUTION_FAILURE", "c").fingerprint
        dec = Decision("REJECT", Reason.INCOMPLETE_EVIDENCE.value)
        L.update(fp, ledger_status(dec), {"reason": dec.reason, "timeout": 1})
        dup, _ = L.is_duplicate(fp)
        self.assertFalse(dup, "超时把整轮打成证据不全，却永久否决了一个还没被检验的改动")

    def test_real_regression_does_make_it_duplicate(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        L = Ledger(Path(tmp.name) / "h.jsonl")
        fp = L.create("AGENT_MAX_ITERATIONS", "15", "25", "EXECUTION_FAILURE", "c").fingerprint
        L.update(fp, ledger_status(Decision("REJECT", Reason.REGRESSION.value)))
        self.assertTrue(L.is_duplicate(fp)[0], "真证伪必须别再重提，否则闭环反复撞同一堵墙")


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
