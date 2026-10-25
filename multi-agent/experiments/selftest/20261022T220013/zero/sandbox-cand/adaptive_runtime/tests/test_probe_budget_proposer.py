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
