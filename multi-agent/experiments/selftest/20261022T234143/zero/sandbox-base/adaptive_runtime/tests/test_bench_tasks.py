"""任务集分层与样本量估算的测试。

存在理由
--------
「用哪批任务验证」决定了结论能走多远。若 evolve 集与 held-out 集混用，
测得的提升无法与记忆效应区分——这是 RRSI 论文所针对的失效模式本身。
本文件锁住分层的不变量：完备、互斥、稳定。
"""
import unittest

from adaptive_runtime.bench_tasks import (
    EXTERNAL_BENCHMARKS, CostPlan, TaskSet, external_status, normalize,
    plan_cost, required_n, split_three_way,
)


def _mk(n: int, prefix: str = "T") -> list[dict]:
    return [{"task_id": f"{prefix}-{i:03d}", "query": f"分析任务 {i} 的市场规模与增长"} for i in range(1, n + 1)]


class TestNormalize(unittest.TestCase):
    def test_filters_short_query(self):
        out = normalize([{"task_id": "a", "query": "太短"}])
        self.assertEqual(out, [])

    def test_dedups_task_id(self):
        out = normalize([
            {"task_id": "X", "query": "长度足够的查询内容一"},
            {"task_id": "X", "query": "长度足够的查询内容二"},
        ])
        self.assertEqual(len(out), 1)

    def test_accepts_prompt_alias(self):
        out = normalize([{"id": "Y", "prompt": "长度足够的查询内容三"}])
        self.assertEqual(out[0]["task_id"], "Y")
        self.assertEqual(out[0]["query"], "长度足够的查询内容三")

    def test_skips_non_dict(self):
        self.assertEqual(normalize(["not a dict"]), [])


class TestSplitThreeWay(unittest.TestCase):
    def test_partition_is_complete_and_disjoint(self):
        ts = split_three_way(_mk(20), evolve_ratio=0.5, id_ratio=0.2)
        ids = ts.all_ids
        self.assertEqual(len(ids), len(set(ids)), "存在跨层重复任务")
        self.assertEqual(len(ids), 20)

    def test_rejects_invalid_ratios(self):
        with self.assertRaises(ValueError):
            split_three_way(_mk(5), evolve_ratio=0.9, id_ratio=0.2)

    def test_order_independent(self):
        a = split_three_way(_mk(20), evolve_ratio=0.5, id_ratio=0.2)
        b = split_three_way(list(reversed(_mk(20))), evolve_ratio=0.5, id_ratio=0.2)
        self.assertEqual(a.summary(), b.summary())

    def test_evolve_is_nonempty(self):
        # evolve 集为空则闭环无从优化，必须在划分层就保证非空
        ts = split_three_way(_mk(4), evolve_ratio=0.5, id_ratio=0.2)
        self.assertGreater(len(ts.evolve), 0)


class TestRequiredN(unittest.TestCase):
    def test_matches_known_values(self):
        # α=0.05, power=0.80 → (1.96+0.8416)² ≈ 7.85
        n = required_n(24.5, 10)
        self.assertEqual(n, 48)          # 7.8489 × 600.25 / 100 ≈ 47.1 → 48

    def test_larger_effect_needs_fewer_samples(self):
        self.assertGreater(required_n(24.5, 10), required_n(24.5, 30))

    def test_rejects_nonpositive_effect(self):
        with self.assertRaises(ValueError):
            required_n(24.5, 0)

    def test_unsupported_quantile_raises(self):
        # 查表法：未收录的 p 必须显式报错，而不是静默给出错误样本量
        with self.assertRaises(ValueError):
            required_n(24.5, 10, alpha=0.03)


class TestPlanCost(unittest.TestCase):
    def test_run_count(self):
        # 每轮：train × reps × 2 臂；holdout 另计 base 单臂
        p = plan_cost(train_tasks=3, reps_per_task=2, rounds=3, holdout_tasks=4)
        self.assertEqual(p.evolve_runs, 3 * 2 * 2 * 3)
        self.assertEqual(p.holdout_runs, 4 * 3)
        self.assertEqual(p.total_runs, 36 + 12)

    def test_total_tokens_and_hours(self):
        p = plan_cost(train_tasks=1, reps_per_task=1, rounds=1, holdout_tasks=0,
                      tokens_per_run=100, seconds_per_run=3600)
        self.assertEqual(p.total_runs, 2)
        self.assertEqual(p.total_tokens, 200)
        self.assertAlmostEqual(p.wall_hours, 2.0)


class TestExternalBenchmarks(unittest.TestCase):
    def test_none_available_yet(self):
        # 任何「已在外部基准验证」的声称都必须先有 loader；此时应全部为 False
        self.assertFalse(any(s.available for s in EXTERNAL_BENCHMARKS))

    def test_status_reports_all(self):
        st = external_status()
        self.assertEqual(len(st), len(EXTERNAL_BENCHMARKS))
        self.assertTrue(all("available" in s for s in st))


if __name__ == "__main__":
    unittest.main()
