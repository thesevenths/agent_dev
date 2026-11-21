"""ARR 1.1 / M3.0 -- statistical-gate correction regression tests.

ENGINE-FREE. Covers the exact defects found in the M2.7 review:

* P0: n=5 paired seeds used the NORMAL 1.96 critical value instead of the
  Student-t 2.776 (df=4) -> false-positive ACCEPTs.
* P0: one round tests several candidates and picks the best with NO
  multiple-comparison control -> family-wise false positives.
* P0: ``mean <= 0`` REJECTED with no negative statistical evidence.
* P0: missing ``std`` / ``baseline_mean`` silently SKIPPED the CI / practical
  floor checks instead of failing closed.
* P0: holdout kept only a MEAN, with no per-seed raw results and no direction
  confirmation; partially-failed holdout seeds were silently swallowed.

No FakeRunner and no fabricated metric in the production path. Deterministic
injected runners appear only where a unit test must stand in for a heavy engine,
exactly as the pre-existing ARR test suite does.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import yaml

from research_runtime.evolution.candidate import (
    CandidateStatus, DomainModelCandidate,
)
from research_runtime.evolution.domain_evolver import evaluate_domain_candidate
from research_runtime.evolution.evolution_controller import (
    DEFAULT_ALPHA, HOLDOUT_DIRECTION_MIN_FRACTION,
    MIN_SEEDS_FOR_FORMAL_ACCEPT,
    _decide_domain, domain_family_significance, run_dual_layer,
)
from research_runtime.evolution.statistics import (
    holm_step_down, paired_confidence_interval, paired_p_value,
    paired_t_statistic, student_t_crit_two_sided, student_t_two_sided_p,
)


def _full_rec(*, n_valid=5, mean=0.001, std=0.0002, baseline_mean=0.012,
              ho=0.0009, ho_n=5, n_invalid=0,
              ho_improvements=None, ho_n_seeds=None, ho_n_invalid=0):
    """A COMPLETE M3.0 evaluation record (positive improvement == better)."""
    if ho_improvements is None and ho is not None and ho_n:
        ho_improvements = [ho] * ho_n
    return {
        "n_valid": n_valid, "n_invalid": n_invalid,
        "mean_improvement": mean, "std_improvement": std,
        "baseline_mean": baseline_mean,
        "holdout_improvement": ho, "holdout_n_valid": ho_n,
        "holdout_improvements": ho_improvements,
        "holdout_n_seeds": ho_n if ho_n_seeds is None else ho_n_seeds,
        "holdout_n_invalid": ho_n_invalid,
    }


def _drop(rec, *keys):
    out = dict(rec)
    for k in keys:
        out.pop(k, None)
    return out


_FIVE = (50, 51, 52, 53, 54)
_HOLDOUT = (60, 61, 62, 63, 64)


class TestStudentTCriticalValues(unittest.TestCase):
    """The 1.96 normal value is NOT the 95% critical value at n=5."""

    def test_critical_values_match_standard_table(self):
        table = {
            1: 12.706204736432095, 2: 4.302652729749464,
            4: 2.7764451051977987, 9: 2.2621571627409915,
            29: 2.045229642132703, 119: 1.980099957011825,
        }
        for df, expect in table.items():
            with self.subTest(df=df):
                self.assertAlmostEqual(
                    student_t_crit_two_sided(0.05, df), expect, places=6)

    def test_df4_is_not_1_96(self):
        # THE root cause of the M2.7 false-positive ACCEPTs.
        t4 = student_t_crit_two_sided(0.05, 4)
        self.assertGreater(t4, 2.77)
        self.assertGreater(t4 - 1.96, 0.8)

    def test_large_df_converges_to_normal(self):
        self.assertAlmostEqual(
            student_t_crit_two_sided(0.05, 10 ** 7), 1.9599639845, places=5)

    def test_tail_probability_inverts_critical_value(self):
        for df in (2, 4, 9, 29):
            for alpha in (0.05, 0.01):
                with self.subTest(df=df, alpha=alpha):
                    c = student_t_crit_two_sided(alpha, df)
                    self.assertAlmostEqual(
                        student_t_two_sided_p(c, df), alpha, places=7)

    def test_zero_std_is_deterministic_not_missing(self):
        self.assertEqual(paired_t_statistic(0.001, 0.0, 5), math.inf)
        self.assertEqual(paired_p_value(0.001, 0.0, 5), 0.0)
        lo, hi = paired_confidence_interval(0.001, 0.0, 5)
        self.assertAlmostEqual(lo, 0.001)
        self.assertAlmostEqual(hi, 0.001)


class TestHolmStepDown(unittest.TestCase):
    def test_single_candidate_is_uncorrected(self):
        out = holm_step_down([0.01])
        self.assertTrue(out[0]["significant"])
        self.assertAlmostEqual(out[0]["effective_alpha"], 0.05)

    def test_thresholds_strictly_tighten_with_family_size(self):
        out = holm_step_down([0.001, 0.001, 0.001])
        self.assertEqual([o["rank"] for o in out], [0, 1, 2])
        self.assertAlmostEqual(out[0]["effective_alpha"], 0.05 / 3)
        self.assertAlmostEqual(out[1]["effective_alpha"], 0.05 / 2)
        self.assertAlmostEqual(out[2]["effective_alpha"], 0.05)
        self.assertTrue(all(o["significant"] for o in out))

    def test_step_down_stops_at_first_failure(self):
        out = holm_step_down([0.01, 0.9, 0.02], alpha=0.05)
        # ranks: 0.01(rank0, thr .0167) passes; 0.02(rank1, thr .025) passes;
        # 0.9(rank2, thr .05) fails.
        self.assertEqual([o["significant"] for o in out],
                         [True, False, True])

    def test_first_failure_blocks_every_less_significant_candidate(self):
        out = holm_step_down([0.04, 0.03, 0.02], alpha=0.05)
        self.assertEqual([o["significant"] for o in out],
                         [False, False, False])

    def test_nonfinite_p_is_least_significant_never_a_pass(self):
        out = holm_step_down([float("nan"), None, 0.001], alpha=0.05)
        self.assertFalse(out[0]["significant"])
        self.assertFalse(out[1]["significant"])
        self.assertTrue(out[2]["significant"])

    def test_family_alpha_can_only_be_tightened(self):
        recs = [_full_rec(mean=0.010, std=0.007) for _ in range(2)]
        normal = domain_family_significance(recs)
        widened = domain_family_significance(recs, alpha=0.5)
        self.assertEqual([x["effective_alpha"] for x in normal],
                         [x["effective_alpha"] for x in widened])
        stricter = domain_family_significance(recs, alpha=0.001)
        self.assertLessEqual(
            max(x["effective_alpha"] for x in stricter),
            max(x["effective_alpha"] for x in normal))


class TestBidirectionalStatisticalDecision(unittest.TestCase):
    """The 8 verdict cases required by the M3.0 task book."""

    def test_significant_positive_is_accepted(self):
        self.assertEqual(_decide_domain(_full_rec()),
                         CandidateStatus.ACCEPTED)

    def test_positive_but_not_significant_is_inconclusive(self):
        # M2.7 REAL SciNet++ data: normal 1.96 said +0.004439, Student-t says
        # -0.000728 -> the CI straddles 0, so this must NOT be an ACCEPT.
        rec = _full_rec(mean=0.016844, std=0.014152,
                        baseline_mean=0.023939, ho=0.029626)
        self.assertEqual(_decide_domain(rec),
                         CandidateStatus.INCONCLUSIVE)

    def test_significant_negative_is_rejected(self):
        rec = _full_rec(mean=-0.001, ho=-0.0005)
        self.assertEqual(_decide_domain(rec), CandidateStatus.REJECTED)

    def test_negative_but_not_significant_is_inconclusive(self):
        # M2.7 would have REJECTED purely on ``mean <= 0``. M3.0 needs a CI
        # wholly below 0.
        rec = _full_rec(mean=-0.001, std=0.01, ho=-0.0005)
        self.assertEqual(_decide_domain(rec),
                         CandidateStatus.INCONCLUSIVE)

    def test_missing_std_fails_closed(self):
        for bad in (None, float("nan"), float("inf"), -1.0, "x"):
            with self.subTest(bad=bad):
                rec = _full_rec(std=bad)
                self.assertEqual(_decide_domain(rec),
                                 CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_drop(_full_rec(), "std_improvement")),
            CandidateStatus.INCONCLUSIVE)

    def test_missing_baseline_mean_fails_closed(self):
        for bad in (None, float("nan"), float("inf"), 0.0, "x"):
            with self.subTest(bad=bad):
                self.assertEqual(
                    _decide_domain(_full_rec(baseline_mean=bad)),
                    CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_drop(_full_rec(), "baseline_mean")),
            CandidateStatus.INCONCLUSIVE)

    def test_missing_sample_counts_fail_closed(self):
        for key in ("n_valid", "n_invalid"):
            with self.subTest(key=key):
                self.assertEqual(
                    _decide_domain(_drop(_full_rec(), key)),
                    CandidateStatus.INCONCLUSIVE)
        self.assertEqual(_decide_domain(_full_rec(n_invalid=1)),
                         CandidateStatus.INCONCLUSIVE)
        self.assertEqual(_decide_domain(_full_rec(n_valid=2)),
                         CandidateStatus.INCONCLUSIVE)

    def test_zero_std_positive_is_deterministic_accept(self):
        # NOT "skipped because missing": every paired run improved identically.
        rec = _full_rec(mean=0.001, std=0.0)
        self.assertEqual(_decide_domain(rec), CandidateStatus.ACCEPTED)

    def test_practical_floor_still_applies(self):
        rec = _full_rec(mean=0.00005, std=0.00002, ho=0.0001)
        self.assertEqual(_decide_domain(rec),
                         CandidateStatus.INCONCLUSIVE)

    def test_hard_floors_cannot_be_lowered(self):
        self.assertEqual(
            _decide_domain(_full_rec(n_valid=2), min_seeds_for_accept=2),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_full_rec(mean=0.00005, std=0.00002),
                           practical_improvement_floor_rel=0.0),
            CandidateStatus.INCONCLUSIVE)
        # Holm alpha cannot be widened beyond the frozen protocol constant.
        self.assertEqual(
            _decide_domain(_full_rec(mean=0.016844, std=0.014152,
                                     baseline_mean=0.023939),
                           holm_alpha=0.5),
            CandidateStatus.INCONCLUSIVE)

    def test_provisional_dev_mode_never_formally_accepts(self):
        self.assertEqual(
            _decide_domain(_full_rec(), require_holdout_confirmation=False),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_full_rec(mean=-0.001, ho=-0.0005),
                           require_holdout_confirmation=False),
            CandidateStatus.REJECTED)


class TestHoldoutPerSeedConfirmation(unittest.TestCase):
    def test_missing_per_seed_results_fails_closed(self):
        self.assertEqual(
            _decide_domain(_drop(_full_rec(), "holdout_improvements")),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_drop(_full_rec(), "holdout_n_seeds")),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_drop(_full_rec(), "holdout_n_invalid")),
            CandidateStatus.INCONCLUSIVE)

    def test_partially_failed_holdout_fails_closed(self):
        # M2.7 silently reduced n_valid; M3.0 requires ALL requested seeds.
        self.assertEqual(
            _decide_domain(_full_rec(ho_n=5, ho_n_seeds=6)),
            CandidateStatus.INCONCLUSIVE)
        self.assertEqual(
            _decide_domain(_full_rec(ho_n_invalid=1)),
            CandidateStatus.INCONCLUSIVE)

    def test_contradictory_per_seed_direction_is_inconclusive(self):
        # Mean is positive but 4/5 holdout seeds are negative -> not confirmed.
        rec = _full_rec(ho=0.0001,
                        ho_improvements=[0.01, -0.001, -0.001, -0.001, -0.001])
        self.assertEqual(_decide_domain(rec),
                         CandidateStatus.INCONCLUSIVE)
        self.assertGreaterEqual(HOLDOUT_DIRECTION_MIN_FRACTION, 0.6)

    def test_holdout_too_small_is_inconclusive(self):
        self.assertEqual(
            _decide_domain(_full_rec(ho_n=2, ho_n_seeds=2)),
            CandidateStatus.INCONCLUSIVE)
        self.assertGreaterEqual(MIN_SEEDS_FOR_FORMAL_ACCEPT, 5)

    def test_contradiction_on_negative_side_is_inconclusive(self):
        rec = _full_rec(mean=-0.001, std=0.0002, ho=-0.0001,
                        ho_improvements=[-0.01, 0.001, 0.001, 0.001, 0.001])
        self.assertEqual(_decide_domain(rec),
                         CandidateStatus.INCONCLUSIVE)


class TestMultipleComparisonEndToEnd(unittest.TestCase):
    def _harness_stub(self, **kw):
        return {"schema_version": "stub",
                "dominant_failure": "DECISION_UNCERTAINTY",
                "records": [], "accepted": [], "n_accepted": 0}

    def test_three_marginal_candidates_cannot_all_win(self):
        # p=0.033: EACH candidate is individually significant at alpha=0.05.
        # With 3 candidates the Holm rank-0 threshold is 0.0167 -> the family
        # produces NO significant winner. This is the "lucky best of three"
        # false-positive the review flagged.
        rec = _full_rec(mean=0.010, std=0.007)
        self.assertEqual(_decide_domain(rec), CandidateStatus.ACCEPTED)
        holm = domain_family_significance([rec, rec, rec])
        self.assertEqual([h["n_compared"] for h in holm], [3, 3, 3])
        self.assertTrue(all(not h["significant"] for h in holm))
        for h in holm:
            self.assertEqual(_decide_domain(rec, holm_alpha=h["effective_alpha"]),
                             CandidateStatus.INCONCLUSIVE)

    def test_run_dual_layer_wires_holm_into_real_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "base.yaml"
            cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n",
                           encoding="utf-8")
            payload = run_dual_layer(
                base_harness=None, tasks=[], heldout_task=None,
                run_root=tmp, repo_root=tmp, rounds=1, max_minutes=1.0,
                domain="embodied", domain_base_config=str(cfg),
                project_dir=tmp, dataset_version="test-v1",
                domain_seeds=_FIVE, domain_holdout_seeds=_HOLDOUT,
                domain_require_frozen_manifest=False,
                domain_runner=lambda art: (
                    0.7 if "baseline" not in art["directory"] else 0.9),
                harness_evolve=self._harness_stub)
            dom = payload["domain_layer"]
            self.assertEqual(dom["statistics_protocol"]["name"],
                             "M3.0-student-t-holm")
            self.assertTrue(dom["statistics_protocol"]["bidirectional"])
            self.assertGreaterEqual(dom["statistics_protocol"]["n_compared"], 1)
            for r in dom["records"]:
                st = r["statistics"]
                self.assertEqual(st["protocol"], "M3.0-student-t-holm")
                self.assertEqual(st["method"], "holm-step-down")
                self.assertEqual(st["n_compared"], len(dom["records"]))
                self.assertAlmostEqual(st["family_alpha"], DEFAULT_ALPHA)
                # Deterministic runner -> p=0 -> Holm passes every rank.
                self.assertTrue(st["holm_significant"])
                self.assertEqual(r["decision"], "accepted")

    def test_significant_negative_family_is_still_rejected(self):
        rec = _full_rec(mean=-0.001, ho=-0.0005)
        holm = domain_family_significance([rec, rec, rec])
        for h in holm:
            self.assertEqual(
                _decide_domain(rec, holm_alpha=h["effective_alpha"]),
                CandidateStatus.REJECTED)


class TestEvaluatePersistsHoldoutPerSeed(unittest.TestCase):
    def _candidate(self):
        return DomainModelCandidate(
            candidate_id="m30", domain="embodied", target="train.lr",
            old=0.01, new=0.005, taxonomy="DECISION_UNCERTAINTY")

    def _eval(self, tmp, runner):
        cfg = Path(tmp) / "base.yaml"
        cfg.write_text("seed: 0\ntrain:\n  lr: 0.01\ndata: {}\n",
                       encoding="utf-8")
        return evaluate_domain_candidate(
            domain="embodied", candidate=self._candidate(),
            base_config=str(cfg), project_dir=tmp, campaign_id="m30",
            seeds=_FIVE, holdout_seeds=_HOLDOUT,
            dataset_version="test-v1", runner=runner,
            require_frozen_manifest=False)

    def test_per_seed_raw_results_are_persisted_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self._eval(
                tmp, lambda art: (0.7 if "baseline" not in art["directory"]
                                  else 0.9))
            self.assertEqual(rec["holdout_n_seeds"], 5)
            self.assertEqual(rec["holdout_n_valid"], 5)
            self.assertEqual(rec["holdout_n_invalid"], 0)
            self.assertEqual(len(rec["holdout_improvements"]), 5)
            self.assertEqual(len(rec["holdout_base_Q"]), 5)
            self.assertEqual(len(rec["holdout_cand_Q"]), 5)
            self.assertEqual(rec["holdout_n_positive"], 5)
            self.assertAlmostEqual(rec["holdout_direction_fraction"], 1.0)
            # Holdout numbers NEVER leak into the paired selection statistic.
            self.assertEqual(len(rec["improvements"]), 5)
            self.assertEqual(rec["n_valid"], 5)
            self.assertEqual(_decide_domain(rec), CandidateStatus.ACCEPTED)

    def test_partial_holdout_failure_is_recorded_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def runner(art):
                cc = yaml.safe_load(Path(art["config_path"]).read_text(
                    encoding="utf-8"))
                if int(cc.get("seed", 0)) == 60:
                    return float("nan")
                return 0.7 if "baseline" not in art["directory"] else 0.9
            rec = self._eval(tmp, runner)
            self.assertEqual(rec["holdout_n_seeds"], 5)
            self.assertEqual(rec["holdout_n_valid"], 4)
            self.assertEqual(rec["holdout_n_invalid"], 1)
            self.assertEqual(_decide_domain(rec),
                             CandidateStatus.INCONCLUSIVE)


class TestM2_7HistoricalReevaluation(unittest.TestCase):
    """Recompute the M2.7 SciNet++ round-1 ACCEPTs under the M3.0 protocol.

    Historical artifacts are NOT modified. This test pins the numeric conclusion
    so the acceptance report's 'provisional' marking cannot silently drift.
    """

    HISTORY = [
        # lr, mean, std, holdout mean (real M2.7 evidence, n_valid=5)
        (0.003, 0.011228, 0.008678, 0.018635),
        (0.005, 0.016844, 0.014152, 0.029626),
        (0.010, 0.019355, 0.019861, 0.033622),
    ]

    def test_student_t_ci_lower_bounds_match_review(self):
        for lr, mean, std, _ho in self.HISTORY:
            lo, _hi = paired_confidence_interval(mean, std, 5)
            norm_lo = mean - 1.96 * std / math.sqrt(5)
            with self.subTest(lr=lr):
                # The normal approximation is systematically optimistic.
                self.assertGreater(norm_lo, lo)
        # Exact signs reported in the review.
        self.assertGreater(
            paired_confidence_interval(0.011228, 0.008678, 5)[0], 0.0)
        self.assertLess(
            paired_confidence_interval(0.016844, 0.014152, 5)[0], 0.0)
        self.assertLess(
            paired_confidence_interval(0.019355, 0.019861, 5)[0], 0.0)

    def test_holm_removes_formal_accept_qualification_of_lr_0_01(self):
        recs = [_full_rec(mean=m, std=s, baseline_mean=0.023939, ho=h)
                for _lr, m, s, h in self.HISTORY]
        holm = domain_family_significance(recs)
        verdicts = [_decide_domain(r, holm_alpha=x["effective_alpha"]).value
                    for r, x in zip(recs, holm)]
        self.assertEqual(verdicts,
                         ["inconclusive", "inconclusive", "inconclusive"])
        # The observed metric improvement is REAL, but the formal statistical
        # ACCEPT qualification of the chosen lr=0.01 is not established.
        self.assertFalse(holm[2]["significant"])


if __name__ == "__main__":
    unittest.main()
