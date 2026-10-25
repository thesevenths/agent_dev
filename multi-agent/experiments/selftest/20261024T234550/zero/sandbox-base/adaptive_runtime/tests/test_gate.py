import unittest

from adaptive_runtime.gate import (Accept, Evidence, INERT_TARGETS, Reason, explain,
                                   gate_a, gate_b,
                                   ledger_status as gate_ledger_status)
from adaptive_runtime.proposer import Candidate
from adaptive_runtime.settings import Settings
from adaptive_runtime.stats import paired_stats


def cand(**kw):
    base = dict(candidate_id="c1", target="AGENT_MAX_ITERATIONS", level=1, old="15",
                new="25", hypothesis="h", edits=1, est_cost=1000,
                patch={"kind": "env"})
    base.update(kw)
    return Candidate(**base)


def ev(**kw):
    base = dict(deltas=[6, 7, 5, 6, 7, 5, 6, 6], A_base=0.8, A_cand=0.8, dC=0.05,
                regression_rate=0.0, n_tasks=8)
    base.update(kw)
    return Evidence(**base)


class TestGateA(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()

    def test_pass(self):
        d = gate_a(cand(), [], [], self.cfg)
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, Accept.GATE_A_PASS.value)

    def test_frozen(self):
        d = gate_a(cand(), ["symbol_changed:agents.py:_critic_gate"], [], self.cfg)
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, Reason.FROZEN_CORE_VIOLATION.value)

    def test_leakage(self):
        d = gate_a(cand(), [], ["FRS-012"], self.cfg)
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, Reason.BENCHMARK_LEAKAGE.value)

    def test_inert_mechanism(self):
        for target in ("AGENT_CRITIC_MIN_LEN", "AGENT_TRUNCATED_SALVAGE"):
            self.assertIn(target, INERT_TARGETS)
            d = gate_a(cand(target=target), [], [], self.cfg)
            self.assertEqual(d.reason, Reason.INERT_MECHANISM.value)

    def test_cost_cap(self):
        d = gate_a(cand(est_cost=10 ** 9), [], [], self.cfg)
        self.assertEqual(d.reason, Reason.COST_CAP.value)

    def test_edit_budget_blocks_level2_at_p0(self):
        d = gate_a(cand(level=2), [], [], self.cfg)
        self.assertEqual(d.reason, Reason.EDIT_BUDGET.value)

    def test_gate_trace_present(self):
        d = gate_a(cand(), [], [], self.cfg)
        checks = {r["check"] for r in d.gate_trace}
        self.assertTrue({"frozen_core", "leakage", "not_inert", "cost_cap",
                         "edit_budget"} <= checks)


class TestGateB(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()

    def test_incomplete_evidence_first(self):
        d = gate_b(ev(missing_runs=1), self.cfg)
        self.assertEqual(d.reason, Reason.INCOMPLETE_EVIDENCE.value)

    def test_contract_floor_relative_to_base(self):
        # base 只有 0.60 时，0.80 是改进，必须放行到主目标
        d = gate_b(ev(A_base=0.60, A_cand=0.80), self.cfg)
        self.assertNotEqual(d.reason, Reason.CONTRACT_FLOOR.value)
        # base 0.60 → 0.50 是退步，必须拦
        d2 = gate_b(ev(A_base=0.60, A_cand=0.50), self.cfg)
        self.assertEqual(d2.reason, Reason.CONTRACT_FLOOR.value)

    def test_accept_validated(self):
        d = gate_b(ev(), self.cfg)
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, Accept.VALIDATED.value)

    def test_within_noise(self):
        d = gate_b(ev(deltas=[0.5, -0.5, 1, -1, 0.2, -0.2]), self.cfg)
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, Reason.WITHIN_NOISE.value)

    def test_regression_direct(self):
        d = gate_b(ev(deltas=[-8] * 8), self.cfg)
        self.assertEqual(d.reason, Reason.REGRESSION.value)

    def test_regression_rate(self):
        # 平均涨，但 37.5% 任务崩了 → regression_rate=0.375 > 0.15
        d = gate_b(ev(deltas=[20, 20, 20, 20, 20, -10, -10, -10], regression_rate=0.375), self.cfg)
        self.assertEqual(d.reason, Reason.REGRESSION.value)

    def test_regression_rate_ignored_for_few_tasks(self):
        # 只有 2 个任务时「回归率」没有意义，不应据此否决
        d = gate_b(ev(deltas=[20, 20], regression_rate=0.5, n_tasks=2), self.cfg)
        self.assertTrue(d.accepted)

    def test_negative_wash_is_not_regression(self):
        """均值为负但不显著 + 高回归率 ⇒ 必须是 within_noise，不是 regression。

        这是 rrsi2 round1 的真实形态（dQ=-3.19、p=0.16、2/13 个任务「均值跌 5 分」）。
        regression 属于**证伪**（gate.ledger_status → rejected），拿它记一次噪声测量
        就是永久删掉一个旋钮；闭环会在几轮内把搜索空间烧完。
        """
        d = gate_b(ev(deltas=[-8, -2, -1, 0, 2, 1, -1, 0], regression_rate=0.3), self.cfg)
        pr = paired_stats([-8, -2, -1, 0, 2, 1, -1, 0], alpha=self.cfg.alpha)
        self.assertGreater(pr.delta_crit, abs(pr.mean))     # 前提：确实不显著
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, Reason.WITHIN_NOISE.value)
        self.assertEqual(gate_ledger_status(d), "inconclusive")

    def test_positive_wash_with_stable_crash_is_rejected(self):
        """均值为正（哪怕不显著）但稳定崩掉足够多任务 → 仍然必须否决。"""
        d = gate_b(ev(deltas=[5, -4, 5, -4, 5, -4, 5, -4], regression_rate=0.4), self.cfg)
        pr = paired_stats([5, -4, 5, -4, 5, -4, 5, -4], alpha=self.cfg.alpha)
        self.assertGreater(pr.mean, 0.0)
        self.assertLessEqual(pr.mean, pr.delta_crit)        # 不是靠配对门过的
        self.assertEqual(d.reason, Reason.REGRESSION.value)
        self.assertIn("regression_rate", [r["check"] for r in d.gate_trace])

    def test_cost_rule(self):
        # +5 分却 +50% 成本：allowance = 0.05 + 0.015*5 = 0.125 < 0.5
        d = gate_b(ev(deltas=[5] * 8, dC=0.5), self.cfg)
        self.assertEqual(d.reason, Reason.COST_RULE.value)

    def test_cheaper_equal_accepted(self):
        d = gate_b(ev(deltas=[0, 0, 0], dC=-0.10), self.cfg)
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, Accept.CHEAPER_EQUAL.value)

    def test_judge_unstable(self):
        d = gate_b(ev(judge_unstable_ratio=0.6), self.cfg)
        self.assertEqual(d.reason, Reason.JUDGE_UNSTABLE.value)

    def test_dB_not_in_gate(self):
        # Evidence 里根本没有 dB 字段：行为指标无法参与判定
        self.assertNotIn("dB", Evidence.__dataclass_fields__)

    def test_explain(self):
        d = gate_b(ev(deltas=[0, 0.1, -0.1]), self.cfg)
        self.assertIn("REJECT(within_noise)", explain(d))


class TestCostNotGoverned(unittest.TestCase):
    """``ADAPTIVE_COST_GOVERNED=0``（自托管 LLM、token 不计费）下的三道成本裁决。

    它们是一个前提的三个面：token 是稀缺资源。前提不成立时，既不因贵而拒，
    也不因省而收 —— 后者会把「没变好」记成「改进」。
    """

    def setUp(self):
        self.cfg = Settings(cost_governed=False)

    def test_cost_cap_skipped_but_traced(self):
        d = gate_a(cand(est_cost=10 ** 9), [], [], self.cfg)
        self.assertTrue(d.accepted, d.to_dict())
        row = next(r for r in d.gate_trace if r["check"] == "cost_cap")
        self.assertTrue(row["pass"])
        self.assertFalse(row["governed"])
        self.assertEqual(row["est_cost"], 10 ** 9, "trace 里要能看出本来是会被拦的")

    def test_expensive_but_better_is_accepted(self):
        d = gate_b(ev(deltas=[5] * 8, dC=0.5), self.cfg)
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, Accept.VALIDATED.value)

    def test_within_noise_not_flipped_by_savings(self):
        d = gate_b(ev(deltas=[0, 0, 0], dC=-0.10), self.cfg)
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason, Reason.WITHIN_NOISE.value)

    def test_default_still_governs(self):
        """零回归：默认（计费口径）下三道照旧生效。"""
        self.assertTrue(Settings().cost_governed)
        self.assertEqual(gate_b(ev(deltas=[5] * 8, dC=0.5), Settings()).reason,
                         Reason.COST_RULE.value)


if __name__ == "__main__":
    unittest.main()
