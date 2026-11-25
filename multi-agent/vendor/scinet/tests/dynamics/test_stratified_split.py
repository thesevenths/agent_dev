"""
tests.dynamics.test_stratified_split
====================================
P1-B1.1 / P7: genuine stratified split allocator.

Positive + negative coverage. Confirms the allocator (a) produces real
stratification when strata are coverable, and (b) REFUSES to fabricate balance
when they are not -- reporting INCONCLUSIVE/BLOCKED instead.
"""

from __future__ import annotations

import copy
import unittest

from data.stratified_split import (
    StratifiedSplitter,
    StratificationConfig,
    bin_index,
    build_episode_metadata,
)


def _ep(eid, profile, m, L, b, th0, om0, regime):
    return {
        "episode_id": eid,
        "profile": profile,
        "physical_parameters": {"m": m, "L": L, "b": b, "g": 9.81},
        "initial_state": [th0, om0],
        "regime": regime,
    }


class TestBinIndex(unittest.TestCase):
    def test_basic_binning(self):
        self.assertEqual(bin_index(0.9, [0.8, 1.0, 1.2]), 0)
        self.assertEqual(bin_index(1.1, [0.8, 1.0, 1.2]), 1)
        self.assertEqual(bin_index(2.0, [-3.14, -1.0, 1.0, 3.14]), 2)
        self.assertEqual(bin_index(-2.0, [-3.14, -1.0, 1.0, 3.14]), 0)

    def test_out_of_range_clamped(self):
        self.assertEqual(bin_index(99.0, [0.8, 1.0, 1.2]), 1)
        self.assertEqual(bin_index(-99.0, [0.8, 1.0, 1.2]), 0)

    def test_short_edges(self):
        self.assertEqual(bin_index(5.0, []), 0)
        self.assertEqual(bin_index(5.0, [1.0]), 0)


class TestBalancedStratification(unittest.TestCase):
    """When every stratum has enough episodes, allocation must PASS and the
    global split ratios must be approximately satisfied."""

    def _build(self, per_stratum=10):
        eps = []
        n = 0
        # 2 profiles x 2 mass-bins x 2 length-bins x 3 theta0-bins x 2 om0-bins
        # x 2 regimes = 96 strata, each with `per_stratum` episodes.
        profiles = ["random", "step"]
        regimes = ["bounded_oscillation", "continuous_rotation"]
        m_vals = [0.9, 1.1]
        L_vals = [0.9, 1.1]
        b_vals = [0.07, 0.12]
        th_vals = [-2.0, 0.0, 2.0]
        om_vals = [-0.5, 0.5]
        for p in profiles:
            for m in m_vals:
                for L in L_vals:
                    for b in b_vals:
                        for th in th_vals:
                            for om in om_vals:
                                for r in regimes:
                                    for _ in range(per_stratum):
                                        eps.append(_ep(f"e{n}", p, m, L, b, th, om, r))
                                        n += 1
        return eps

    def test_pass_and_ratio_within_tolerance(self):
        eps = self._build(per_stratum=10)
        splitter = StratifiedSplitter()
        res = splitter.allocate(eps, master_seed=123)
        self.assertEqual(res.n_episodes, len(eps))
        # All strata must be coverable -> PASS.
        self.assertEqual(res.status, "PASS")
        self.assertEqual(len(res.blocked_strata), 0)
        # Episode-level: exactly one split each.
        self.assertEqual(set(res.assignments.values()),
                         {"train", "validation", "test"})
        from collections import Counter
        c = Counter(res.assignments.values())
        total = len(eps)
        # requested 0.7/0.15/0.15 -> within 5% absolute.
        self.assertLessEqual(abs(c["train"] / total - 0.7), 0.05)
        self.assertLessEqual(abs(c["validation"] / total - 0.15), 0.05)
        self.assertLessEqual(abs(c["test"] / total - 0.15), 0.05)

    def test_deterministic(self):
        eps = self._build(per_stratum=6)
        s1 = StratifiedSplitter().allocate(eps, master_seed=7)
        s2 = StratifiedSplitter().allocate(copy.deepcopy(eps), master_seed=7)
        self.assertEqual(s1.assignments, s2.assignments)


class TestImpossibleAllocation(unittest.TestCase):
    """When strata cannot be split, the allocator must NOT fabricate coverage."""

    def test_single_episode_per_stratum_is_blocked(self):
        # 5 distinct strata, each with exactly 1 episode.
        eps = [
            _ep("a", "random", 0.9, 0.9, 0.07, -2.0, -0.5, "bounded_oscillation"),
            _ep("b", "step", 1.1, 0.9, 0.07, -2.0, -0.5, "bounded_oscillation"),
            _ep("c", "random", 0.9, 1.1, 0.12, 0.0, 0.5, "continuous_rotation"),
            _ep("d", "step", 1.1, 1.1, 0.12, 2.0, 0.5, "continuous_rotation"),
            _ep("e", "random", 0.9, 0.9, 0.07, 0.0, -0.5, "bounded_oscillation"),
        ]
        res = StratifiedSplitter().allocate(eps, master_seed=1)
        self.assertEqual(res.status, "BLOCKED")
        # No fabrication: nothing can go to validation/test for a 1-episode stratum.
        self.assertEqual(set(res.assignments.values()), {"train"})
        self.assertEqual(len(res.blocked_strata), 5)

    def test_two_episode_stratum_stays_blocked(self):
        eps = [
            _ep("a", "random", 0.9, 0.9, 0.07, -2.0, -0.5, "bounded_oscillation"),
            _ep("b", "random", 0.9, 0.9, 0.07, -2.0, -0.5, "bounded_oscillation"),
        ]
        res = StratifiedSplitter().allocate(eps, master_seed=1)
        self.assertEqual(res.status, "BLOCKED")
        self.assertTrue(all(v == "train" for v in res.assignments.values()))

    def test_partial_stratum_is_inconclusive_not_fabricated(self):
        # One stratum with 5 episodes (coverable) + one with 2 (blocked).
        eps = []
        for i in range(5):
            eps.append(_ep(f"cov{i}", "random", 0.9, 0.9, 0.07, -2.0, -0.5,
                            "bounded_oscillation"))
        eps.append(_ep("blk0", "step", 1.1, 1.1, 0.12, 2.0, 0.5,
                       "continuous_rotation"))
        eps.append(_ep("blk1", "step", 1.1, 1.1, 0.12, 2.0, 0.5,
                       "continuous_rotation"))
        res = StratifiedSplitter().allocate(eps, master_seed=1)
        # The blocked 2-episode stratum forces BLOCKED overall.
        self.assertEqual(res.status, "BLOCKED")
        # The coverable stratum still got real splits; the blocked ones did not.
        self.assertTrue(all(res.assignments[e] == "train"
                           for e in ("blk0", "blk1")))


class TestConfigAndMetadata(unittest.TestCase):
    def test_from_dict_and_default_equivalence(self):
        d = {
            "requested_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "param_bins": {"mass": [0.8, 1.0, 1.2]},
            "init_bins": {"theta0": [-3.14, 0.0, 3.14]},
        }
        cfg = StratificationConfig.from_dict(d)
        self.assertEqual(cfg.requested_ratios["train"], 0.7)
        self.assertIn("mass", cfg.param_bins)

    def test_build_episode_metadata_from_manifest_shape(self):
        fake_man = {"episodes": [
            {"episode_id": "E00_random",
             "action_profile": {"name": "random"},
             "physical_parameters": {"m": 1.0, "L": 1.0, "b": 0.1},
             "initial_state": [0.0, 0.0], "regime": "bounded_oscillation"},
        ]}
        md = build_episode_metadata(fake_man)
        self.assertEqual(md[0]["profile"], "random")
        self.assertEqual(md[0]["regime"], "bounded_oscillation")


if __name__ == "__main__":
    unittest.main()
