import tempfile
import unittest
from pathlib import Path

from adaptive_runtime.budget import Budget, BudgetExceeded, degraded_n, estimate_round_tokens
from adaptive_runtime.comparator import compare
from adaptive_runtime.proposer import (SEARCH_SPACE, falsification_candidate, leakage_hits,
                                       propose, zero_change_candidate)
from adaptive_runtime.ledger import Ledger
from adaptive_runtime.settings import CFG, Settings


def scores(arm, qs, tokens=1000, A=0.8):
    out = []
    for task, vals in qs.items():
        for rep, q in enumerate(vals):
            out.append({"task_id": task, "rep": rep, "arm": arm, "Q": q,
                        "A": A, "tokens": tokens})
    return out


class TestCompare(unittest.TestCase):
    def test_pairs_and_stats(self):
        base = scores("base", {"T1": [50, 50], "T2": [60, 60]})
        cand = scores("cand", {"T1": [55, 56], "T2": [64, 66]})
        cmp = compare(base, cand)
        self.assertEqual(len(cmp.pairs), 4)
        self.assertAlmostEqual(cmp.stats["mean"], (5 + 6 + 4 + 6) / 4, places=3)
        self.assertEqual(cmp.missing_runs, 0)
        self.assertEqual(set(cmp.per_task), {"T1", "T2"})

    def test_missing_pair_counted(self):
        base = scores("base", {"T1": [50, 50]})
        cand = scores("cand", {"T1": [55]})
        cmp = compare(base, cand)
        self.assertEqual(len(cmp.pairs), 1)
        self.assertEqual(cmp.missing_runs, 1)

    def test_cost_delta(self):
        base = scores("base", {"T1": [50]}, tokens=1000)
        cand = scores("cand", {"T1": [50]}, tokens=1500)
        cmp = compare(base, cand)
        ev = cmp.evidence()
        self.assertAlmostEqual(ev.dC, 0.5)

    def test_regression_rate(self):
        base = scores("base", {f"T{i}": [60, 60] for i in range(5)})
        cand = scores("cand", {**{f"T{i}": [70, 70] for i in range(4)}, "T4": [40, 40]})
        cmp = compare(base, cand)
        self.assertAlmostEqual(cmp.stats["regression_rate"], 0.2)
        self.assertEqual(cmp.evidence().n_tasks, 5)

    def test_crash_requires_consistency_across_reps(self):
        """均值跌破阈值但只有一些重复崩 → 不计入回归率（噪声伪装不了一致性）。

        rrsi2 round1 就是把 OFF-QUA-03（重复间跌幅 -47.5/-39.9/+21.4，均值 -22）当成
        「任务崩了」，配合另一个噪声任务一起顶穿了 r_max=0.15。
        """
        base = scores("base", {**{f"T{i}": [60, 60, 60] for i in range(4)}, "T4": [80, 80, 80]})
        cand = scores("cand", {**{f"T{i}": [61, 61, 61] for i in range(4)},
                               "T4": [10, 80, 82]})
        cmp = compare(base, cand)
        self.assertLess(cmp.per_task["T4"]["mean_delta"], -CFG.regression_min_points)
        self.assertEqual(cmp.stats["crashed_tasks"], [])
        self.assertEqual(cmp.stats["regression_rate"], 0.0)
        self.assertEqual(cmp.evidence().crashed_tasks, [])

    def test_stats_and_evidence_share_regression_definition(self):
        """报告里的回归率必须就是闸门用的那个数（单一真源）。"""
        base = scores("base", {f"T{i}": [60, 60] for i in range(4)})
        cand = scores("cand", {"T0": [40, 40], "T1": [60, 60],
                               "T2": [60, 60], "T3": [60, 60]})
        cmp = compare(base, cand)
        self.assertAlmostEqual(cmp.stats["regression_rate"], 0.25)
        self.assertAlmostEqual(cmp.evidence().regression_rate, cmp.stats["regression_rate"])
        self.assertEqual(cmp.evidence().crashed_tasks, ["T0"])

    def test_A_comes_from_A_field_not_Q(self):
        base = scores("base", {"T1": [50]}, A=0.6)
        cand = scores("cand", {"T1": [50]}, A=0.9)
        ev = compare(base, cand).evidence()
        self.assertAlmostEqual(ev.A_base, 0.6)
        self.assertAlmostEqual(ev.A_cand, 0.9)


class TestBudget(unittest.TestCase):
    def test_spend_and_cap(self):
        b = Budget(cap_round=100, cap_cand=60)
        b.spend(40, "c1")
        b.spend(30)          # 轮内 70
        self.assertEqual(b.remaining_round, 30)
        with self.assertRaises(BudgetExceeded):
            b.spend(40, "c1")

    def test_negative(self):
        with self.assertRaises(ValueError):
            Budget().spend(-1)

    def test_estimate(self):
        self.assertEqual(estimate_round_tokens(1000, 8, overhead=0), 16000)

    def test_from_settings_unbounded_when_cost_not_governed(self):
        """token 不计费时，预算不得拦腰截断 run 序列（缺 run 会让整轮证据作废）。"""
        b = Budget.from_settings(Settings(cost_governed=False))
        self.assertGreaterEqual(b.cap_round, 10 ** 11)
        b.spend(10 ** 10, "c1")               # 不该抛 BudgetExceeded
        self.assertTrue(b.can_spend(10 ** 10, "c1"))

    def test_from_settings_keeps_caps_when_governed(self):
        cfg = Settings(cost_governed=True, cost_cap_round=5_000_000,
                       cost_cap_cand=4_000_000)
        b = Budget.from_settings(cfg)
        self.assertEqual((b.cap_round, b.cap_cand), (5_000_000, 4_000_000))
        with self.assertRaises(BudgetExceeded):
            b.spend(4_000_001, "c1")

    def test_degraded_n(self):
        cfg = Settings(cost_cap_round=1_000_000)
        self.assertGreaterEqual(degraded_n(200_000, cfg), 0)
        self.assertLessEqual(degraded_n(200_000, cfg), cfg.n_default)


class TestProposer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.tmp.name) / "h.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_proposes_only_level1_at_p0(self):
        cands = propose("COMPLETION_FAILURE", self.ledger, Settings(max_edit_level=1), limit=5)
        self.assertGreaterEqual(len(cands), 1)
        self.assertTrue(all(c.level == 1 for c in cands))

    def test_inert_targets_not_in_search_space(self):
        for bad in ("AGENT_CRITIC_MIN_LEN", "AGENT_CRITIC_RETRIES", "AGENT_STEP_FAILURES_MAX"):
            self.assertNotIn(bad, SEARCH_SPACE)

    def test_duplicate_hypothesis_skipped(self):
        first = propose("COMPLETION_FAILURE", self.ledger, limit=1)
        self.assertEqual(len(first), 1)
        c = first[0]
        self.ledger.create(c.target, c.old, c.new, c.taxonomy, "rejected already")
        self.ledger.update(c.hyp_fingerprint, "rejected")
        again = propose("COMPLETION_FAILURE", self.ledger, limit=1)
        self.assertNotEqual(again[0].hyp_fingerprint if again else None, c.hyp_fingerprint)

    def test_decisive_wash_is_not_re_proposed(self):
        """测完了的平局不重提（否则闭环每轮重跑同一个旋钮、永不前进）。"""
        c = propose("COMPLETION_FAILURE", self.ledger, limit=1)[0]
        self.ledger.create(c.target, c.old, c.new, c.taxonomy, "wash")
        self.ledger.update(c.hyp_fingerprint, "inconclusive",
                           {"reason": "within_noise", "n": 13, "dQ": -3.19})
        again = propose("COMPLETION_FAILURE", self.ledger, limit=3)
        self.assertNotIn(c.hyp_fingerprint, [x.hyp_fingerprint for x in again])

    def test_measurement_failure_is_re_proposable(self):
        """「没测成」与「测完是平局」必须区分：前者值得再试一次。"""
        c = propose("COMPLETION_FAILURE", self.ledger, limit=1)[0]
        self.ledger.create(c.target, c.old, c.new, c.taxonomy, "aborted")
        self.ledger.update(c.hyp_fingerprint, "inconclusive",
                           {"reason": "incomplete_evidence", "stopped": "stop_loss"})
        again = propose("COMPLETION_FAILURE", self.ledger, limit=1)
        self.assertEqual(again[0].hyp_fingerprint, c.hyp_fingerprint)

    def test_env_override_shape(self):
        c = propose("COMPLETION_FAILURE", self.ledger, limit=1)[0]
        self.assertEqual(c.env_overrides(), {c.target: c.new})

    def test_leakage_scan(self):
        self.assertTrue(leakage_hits("if task_id == 'FRS-012': pass"))
        self.assertFalse(leakage_hits("AGENT_MAX_ITERATIONS: 15 -> 25"))

    def test_controls(self):
        self.assertEqual(zero_change_candidate().env_overrides(), {})
        f = falsification_candidate()
        self.assertEqual((f.old, f.new), ("60", "12"))


if __name__ == "__main__":
    unittest.main()
