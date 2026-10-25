import math
import unittest

from adaptive_runtime import stats


class TestTCrit(unittest.TestCase):
    def test_table_matches_known_values(self):
        self.assertAlmostEqual(stats.t_crit(4), 2.776, places=3)
        self.assertAlmostEqual(stats.t_crit(9), 2.262, places=3)
        self.assertAlmostEqual(stats.t_crit(24), 2.064, places=3)

    def test_asymptotic_close_to_normal(self):
        self.assertLess(abs(stats.t_crit(1000) - 1.96), 0.01)

    def test_monotonic_decreasing(self):
        vals = [stats.t_crit(df) for df in range(1, 40)]
        for a, b in zip(vals, vals[1:]):
            self.assertGreaterEqual(a + 1e-9, b)

    def test_invalid_df(self):
        with self.assertRaises(ValueError):
            stats.t_crit(0)


class TestPValue(unittest.TestCase):
    def test_known_quantile(self):
        # t=2.776, df=4 对应双侧 0.05
        self.assertAlmostEqual(stats.t_pvalue(2.776, 4), 0.05, places=2)

    def test_symmetry_and_bounds(self):
        self.assertAlmostEqual(stats.t_pvalue(-2.0, 10), stats.t_pvalue(2.0, 10), places=9)
        self.assertAlmostEqual(stats.t_pvalue(0.0, 10), 1.0, places=6)
        self.assertLess(stats.t_pvalue(10, 10), 0.001)


class TestPaired(unittest.TestCase):
    def test_basic(self):
        r = stats.paired_stats([4, 5, 3, 6])
        self.assertEqual(r.n, 4)
        self.assertAlmostEqual(r.mean, 4.5)
        self.assertAlmostEqual(r.sd, math.sqrt(1.6666667), places=5)
        # n=4 → df=3 → t=3.182
        self.assertAlmostEqual(r.delta_crit, stats.t_crit(3) * r.se, places=9)
        self.assertTrue(r.positive)

    def test_zero_variance_positive(self):
        r = stats.paired_stats([2, 2, 2, 2])
        self.assertEqual(r.sd, 0.0)
        self.assertEqual(r.delta_crit, 0.0)
        self.assertTrue(r.positive)

    def test_zero_variance_zero(self):
        r = stats.paired_stats([0, 0, 0])
        self.assertFalse(r.positive)
        self.assertAlmostEqual(r.p_value, 1.0)

    def test_negative(self):
        r = stats.paired_stats([-3, -3, -3])
        self.assertFalse(r.positive)

    def test_unstable_not_significant(self):
        # 均值 +2、波动 ±10：n=4 时不应判为显著
        r = stats.paired_stats([12, -8, 12, -8])
        self.assertGreater(r.delta_crit, 2.0)
        self.assertFalse(r.positive)

    def test_empty(self):
        r = stats.paired_stats([])
        self.assertEqual(r.n, 0)
        self.assertFalse(r.positive)


class TestPower(unittest.TestCase):
    def test_required_n_matches_doc(self):
        # 方案 §8.3：sd=7, Δ=4 → 25 对；sd=12, Δ=4 → 71 对
        self.assertEqual(stats.required_n(7, 4), 25)
        self.assertEqual(stats.required_n(12, 4), 71)

    def test_required_n_small_sd(self):
        self.assertEqual(stats.required_n(2, 4), 2)
        self.assertEqual(stats.required_n(4, 4), 8)

    def test_degenerate(self):
        self.assertEqual(stats.required_n(0, 4), 1)
        self.assertEqual(stats.required_n(5, 0), 1)

    def test_mde_grows_with_sd(self):
        self.assertLess(stats.mde(2, 8), stats.mde(7, 8))

    def test_norm_ppf(self):
        self.assertAlmostEqual(stats._norm_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(stats._norm_ppf(0.8), 0.841621, places=5)


if __name__ == "__main__":
    unittest.main()
