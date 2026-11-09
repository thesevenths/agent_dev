"""P1-05B: paired-seed statistics, dependency-free and numerically verified."""
from __future__ import annotations

import unittest
from pathlib import Path

from research_runtime.comparison import (
    ComparisonInputError,
    PairedStatisticsEngine,
    holm_adjust,
    normalize_deltas,
    student_t_critical,
    student_t_two_sided_p,
)

try:
    from scipy import stats as scipy_stats
except ImportError:                                   # pragma: no cover
    scipy_stats = None


class TestDeltaNormalization(unittest.TestCase):
    def test_minimize_and_maximize_are_unified(self):
        low = normalize_deltas([0.10, 0.20], [0.08, 0.25], "minimize")
        for got, want in zip(low, (0.02, -0.05)):
            self.assertAlmostEqual(got, want)
        high = normalize_deltas([0.10, 0.20], [0.12, 0.15], "maximize")
        for got, want in zip(high, (0.02, -0.05)):
            self.assertAlmostEqual(got, want)
        self.assertEqual(normalize_deltas([0.1, 0.2], [0.1, 0.2],
                                         "minimize"), [0.0, 0.0])

    def test_invalid_inputs_are_rejected(self):
        bad = [
            ([0.1], [0.2], "unknown"),
            ([0.1], [0.2, 0.3], "minimize"),
            ([0.1], [float("nan")], "minimize"),
            ([float("inf")], [0.2], "minimize"),
            ([True], [0.2], "minimize"),
            (["x"], [0.2], "minimize"),
        ]
        for baseline, candidate, direction in bad:
            with self.subTest(b=(baseline, candidate, direction)):
                with self.assertRaises(ComparisonInputError):
                    normalize_deltas(baseline, candidate, direction)

    def test_direction_is_never_inferred_from_a_metric_name(self):
        with self.assertRaises(ComparisonInputError):
            normalize_deltas([0.1], [0.2], "latent_mse@1")


class TestPairedStatistics(unittest.TestCase):
    def setUp(self):
        self.engine = PairedStatisticsEngine()

    def test_no_observations(self):
        s = self.engine.summarize([])
        self.assertEqual(s.n, 0)
        self.assertIsNone(s.confidence_interval)
        self.assertIsNone(s.std_delta)

    def test_single_seed_has_no_interval_or_p_value(self):
        s = self.engine.summarize([0.01])
        self.assertEqual(s.n, 1)
        self.assertAlmostEqual(s.mean_delta, 0.01)
        self.assertIsNone(s.confidence_interval)
        self.assertIsNone(s.p_value_two_sided)
        self.assertIsNone(s.std_delta)

    def test_zero_variance_is_a_point_interval(self):
        s = self.engine.summarize([0.01] * 5)
        self.assertEqual(s.confidence_interval, (0.01, 0.01))
        self.assertEqual(s.std_delta, 0.0)
        self.assertEqual(s.p_value_two_sided, 0.0)
        null = self.engine.summarize([0.0] * 5)
        self.assertEqual(null.confidence_interval, (0.0, 0.0))
        self.assertEqual(null.p_value_two_sided, 1.0)

    def test_clear_improvement_and_regression(self):
        up = self.engine.summarize([0.001, 0.0012, 0.0009, 0.0011, 0.0013])
        self.assertGreater(up.confidence_interval[0], 0.0)
        self.assertLess(up.p_value_two_sided, 0.05)
        down = self.engine.summarize([-x for x in
                                     (0.001, 0.0012, 0.0009, 0.0011, 0.0013)])
        self.assertLess(down.confidence_interval[1], 0.0)

    def test_noisy_deltas_include_zero(self):
        s = self.engine.summarize([0.001, -0.001, 0.002, -0.002, 0.0])
        lo, hi = s.confidence_interval
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_nonfinite_delta_is_rejected(self):
        for value in (float("nan"), float("inf"), True, "0.1"):
            with self.subTest(value=value), self.assertRaises(ComparisonInputError):
                self.engine.summarize([0.01, value, 0.02])

    def test_wider_alpha_widens_the_interval(self):
        deltas = [0.001, 0.0012, 0.0009, 0.0011, 0.0013]
        narrow = self.engine.summarize(deltas, alpha=0.05).confidence_interval
        wide = self.engine.summarize(deltas, alpha=0.01).confidence_interval
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])

    def test_invalid_alpha_is_rejected(self):
        with self.assertRaises(ComparisonInputError):
            self.engine.summarize([0.01, 0.02], alpha=0.0)

    def test_no_large_numeric_dependency_is_introduced(self):
        source = (Path(__file__).resolve().parents[2]
                  / "research_runtime" / "comparison" / "statistics.py")
        text = source.read_text(encoding="utf-8").lower()
        self.assertNotIn("import scipy", text)
        self.assertNotIn("import numpy", text)
        self.assertNotIn("from scipy", text)


@unittest.skipIf(scipy_stats is None, "SciPy not available (optional oracle)")
class TestAgainstSciPyOracle(unittest.TestCase):
    """The pure-stdlib implementation must match an independent oracle."""

    def test_t_critical_values(self):
        for df in (1, 2, 4, 9, 29):
            with self.subTest(df=df):
                self.assertAlmostEqual(student_t_critical(df, 0.05),
                                       scipy_stats.t.ppf(0.975, df), places=6)

    def test_two_sided_p_values(self):
        for t, df in ((0.0, 9), (2.0, 4), (3.5, 10), (8.0, 2)):
            with self.subTest(t=t, df=df):
                self.assertAlmostEqual(student_t_two_sided_p(t, df),
                                       2 * scipy_stats.t.sf(abs(t), df),
                                       places=6)


class TestHolmCorrection(unittest.TestCase):
    def test_step_down_monotonicity(self):
        result = holm_adjust([0.001, 0.04, 0.2], familywise_alpha=0.05)
        self.assertEqual(result.family_size, 3)
        self.assertEqual([e.index for e in result.entries], [0, 1, 2])
        self.assertLessEqual(result.entries[0].adjusted_p,
                              result.entries[1].adjusted_p)
        self.assertLessEqual(result.entries[1].adjusted_p,
                             result.entries[2].adjusted_p)
        self.assertTrue(result.entries[0].significant)
        self.assertFalse(result.entries[1].significant)
        self.assertFalse(result.entries[2].significant)
        self.assertAlmostEqual(result.entries[0].adjusted_alpha, 0.05 / 3)
        self.assertAlmostEqual(result.entries[2].adjusted_alpha, 0.05)

    def test_single_hypothesis_is_unchanged(self):
        result = holm_adjust([0.01])
        self.assertTrue(result.entries[0].significant)
        self.assertAlmostEqual(result.entries[0].adjusted_alpha, 0.05)
        self.assertAlmostEqual(result.entries[0].adjusted_p, 0.01)

    def test_adjusted_p_is_clipped_and_invalid_p_rejected(self):
        result = holm_adjust([0.9, 0.9])
        self.assertLessEqual(max(e.adjusted_p for e in result.entries), 1.0)
        self.assertFalse(any(e.significant for e in result.entries))
        with self.assertRaises(ComparisonInputError):
            holm_adjust([float("nan")])
        with self.assertRaises(ComparisonInputError):
            holm_adjust([0.01], familywise_alpha=1.0)

    def test_multiple_candidates_are_stricter_than_one(self):
        p = 0.02
        single = holm_adjust([p])
        family = holm_adjust([p, 0.02])
        self.assertTrue(single.entries[0].significant)
        # Two hypotheses at p=0.02: adjusted p = 0.04 < 0.05, still significant
        # but the decision CI uses alpha=0.025, so it is wider than 95%.
        self.assertLess(family.entries[0].adjusted_alpha, 0.05)
        self.assertGreater(1 - family.entries[0].adjusted_alpha, 0.95)


if __name__ == "__main__":
    unittest.main()
