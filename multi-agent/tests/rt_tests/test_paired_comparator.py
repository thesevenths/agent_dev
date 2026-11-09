"""P1-05B/C: pairing integrity, decisions, Holm correction and report flow."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from research_runtime.campaign.schemas import CampaignRecord
from research_runtime.comparison import (
    ComparisonIntegrityError,
    ComparisonPolicy,
    content_sha256,
)
from research_runtime.schemas import ExperimentPlan
from tests.rt_tests.comparison_fakes import (
    BASE_HASH,
    ComparisonTestBase,
    make_plan,
    make_record,
    make_trial,
    paired_trials,
)

SEEDS = (42, 43, 44, 45, 46)
IMPROVED = [0.09] * 5
BASELINE = [0.10] * 5
REGRESSED = [0.11] * 5


class ComparatorTestBase(ComparisonTestBase):
    def build(self, *, seeds=SEEDS, baseline=BASELINE, candidate=IMPROVED,
              plan_kwargs=None, trial_kwargs=None, record_kwargs=None,
              comparator_kwargs=None):
        plan = make_plan(seeds, **(plan_kwargs or {}))
        trials = paired_trials(seeds, baseline, candidate,
                               **(trial_kwargs or {}))
        record = make_record(trials, plan, **(record_kwargs or {}))
        store = self.persist(record)
        comparator = self.comparator(store=store, **(comparator_kwargs or {}))
        return plan, record, store, comparator

    def one(self, **kwargs):
        plan, record, store, comparator = self.build(**kwargs)
        reports = comparator.compare(record.campaign_id)
        self.assertEqual(len(reports), 1,
                         f"expected one report, got {len(reports)}")
        return reports[0], store, comparator, record


class TestDirectionAndDecisions(ComparatorTestBase):
    def test_minimize_improvement_is_positive_and_accepted(self):
        report, *_ = self.one()
        self.assertEqual(report.metric_direction, "minimize")
        self.assertEqual(report.delta_convention, "positive_is_better")
        self.assertTrue(all(d > 0 for d in report.paired_deltas))
        self.assertEqual(report.decision, "accepted_for_confirmation")
        self.assertEqual(report.decision_rule, "R7_practical_effect_confirmed")
        for value in report.confidence_interval_95:
            self.assertAlmostEqual(value, 0.01)

    def test_maximize_improvement_is_also_positive(self):
        report, *_ = self.one(
            baseline=[0.50] * 5, candidate=[0.60] * 5,
            plan_kwargs={"metric_direction": "maximize",
                         "primary_metric": "r2@1"},
            trial_kwargs={"primary_metric": "r2@1"})
        self.assertTrue(all(d > 0 for d in report.paired_deltas))
        self.assertEqual(report.decision, "accepted_for_confirmation")
        self.assertEqual(report.metric_direction, "maximize")

    def test_equal_values_are_inconclusive(self):
        report, *_ = self.one(candidate=BASELINE)
        self.assertEqual(report.paired_deltas, [0.0] * 5)
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.decision_rule,
                         "R6_confidence_interval_includes_zero")

    def test_clear_regression_is_rejected(self):
        report, *_ = self.one(candidate=REGRESSED)
        self.assertTrue(all(d < 0 for d in report.paired_deltas))
        self.assertEqual(report.decision, "rejected")
        self.assertEqual(report.decision_rule,
                         "R5_confidence_interval_regression")

    def test_noisy_deltas_are_inconclusive(self):
        report, *_ = self.one(candidate=[0.102, 0.098, 0.104, 0.096, 0.100])
        self.assertEqual(report.decision, "inconclusive")
        lo, hi = report.confidence_interval_95
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)
        self.assertGreater(report.std_delta, 0.0)

    def test_too_few_seeds_never_decides(self):
        report, *_ = self.one(seeds=(42, 43, 44),
                              baseline=[0.10] * 3, candidate=[0.09] * 3)
        self.assertEqual(report.n_pairs, 3)
        self.assertIsNotNone(report.confidence_interval_95)
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.decision_rule, "R2_insufficient_paired_seeds")

    def test_single_seed_has_no_confidence_interval(self):
        report, *_ = self.one(seeds=(42,), baseline=[0.10],
                              candidate=[0.09])
        self.assertIsNone(report.confidence_interval_95)
        self.assertIsNone(report.std_delta)
        self.assertEqual(report.decision, "inconclusive")

    def test_practical_effect_threshold_is_respected(self):
        report, *_ = self.one(
            comparator_kwargs={"policy": ComparisonPolicy(
                practical_effect_threshold=0.05)})
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.decision_rule, "R8_no_decisive_rule")
        self.assertAlmostEqual(report.practical_effect_threshold, 0.05)


class TestPairingIntegrity(ComparatorTestBase):
    def _refuse(self, *, trial_mutator=None, trial_kwargs=None,
                record_kwargs=None, comparator_kwargs=None):
        plan = make_plan(SEEDS)
        trials = paired_trials(SEEDS, BASELINE, IMPROVED,
                              **(trial_kwargs or {}))
        if trial_mutator:
            trials = [trial_mutator(t) for t in trials]
        record = make_record(trials, plan, **(record_kwargs or {}))
        comparator = self.comparator(**(comparator_kwargs or {}))
        with self.assertRaises(ComparisonIntegrityError):
            comparator.compare_candidate(record, "cand_0", plan=plan)

    def test_dataset_version_mismatch_is_refused(self):
        self._refuse(trial_mutator=lambda t: (
            t.model_copy(update={"dataset_version": "ds-v2"})
            if t.arm_type == "candidate" else t))

    def test_split_mismatch_is_refused(self):
        self._refuse(trial_mutator=lambda t: (
            t.model_copy(update={"evaluation_split": "test"})
            if t.arm_type == "candidate" else t))

    def test_reserved_final_split_is_refused(self):
        # The whole campaign uses a reserved split -> every trial is refused.
        self._refuse(trial_kwargs={"evaluation_split": "test"},
                     record_kwargs={"evaluation_split": "test"})

    def test_evaluation_run_id_must_match_train_run_id(self):
        def mutate(t):
            if t.arm_type != "candidate":
                return t
            spec = dict(t.evaluation_spec or {})
            spec["run_id"] = "a_different_checkpoint"
            return t.model_copy(update={"evaluation_spec": spec})
        self._refuse(trial_mutator=mutate)

    def test_missing_evaluation_spec_is_refused(self):
        self._refuse(trial_mutator=lambda t: (
            t.model_copy(update={"evaluation_spec": None})
            if t.arm_type == "candidate" else t))

    def test_config_hash_tampering_is_refused(self):
        plan = make_plan(SEEDS)
        record = make_record(paired_trials(SEEDS, BASELINE, IMPROVED), plan)
        comparator = self.comparator(
            config_hash_verifier=lambda trial: "TAMPERED_HASH")
        with self.assertRaises(ComparisonIntegrityError):
            comparator.compare_candidate(record, "cand_0", plan=plan)
        # A verifier that agrees with the record is accepted.
        ok = self.comparator(
            config_hash_verifier=lambda trial: trial.config_sha256)
        report = ok.compare_candidate(record, "cand_0", plan=plan)
        self.assertEqual(report.n_pairs, 5)

    def test_baseline_config_hash_must_match_between_arms(self):
        def mutate(t):
            if t.arm_type == "candidate":
                return t.model_copy(update={"baseline_config_sha256": "OTHER"})
            return t
        self._refuse(trial_mutator=mutate)

    def test_duplicate_trials_for_one_seed_are_refused(self):
        plan = make_plan((42,))
        trials = paired_trials((42,), [0.10], [0.09])
        trials.append(make_trial(seed=42, arm_type="baseline",
                                 arm_id="baseline", value=0.10))
        record = make_record(trials, plan)
        with self.assertRaises(ComparisonIntegrityError):
            self.comparator().compare_candidate(record, "cand_0", plan=plan)

    def test_primary_metric_mismatch_is_refused(self):
        self._refuse(trial_mutator=lambda t: (
            t.model_copy(update={"primary_metric": "other_metric"})
            if t.arm_type == "candidate" else t))

    def test_nonfinite_primary_metric_is_refused(self):
        def mutate(t):
            if t.arm_type == "candidate":
                metrics = dict(t.validation_metrics)
                metrics["mse@1"] = float("nan")
                return t.model_copy(update={"validation_metrics": metrics})
            return t
        self._refuse(trial_mutator=mutate)

    def test_missing_pairs_are_recorded_never_dropped(self):
        plan = make_plan(SEEDS)
        trials = paired_trials(SEEDS[:4], BASELINE[:4], IMPROVED[:4])
        record = make_record(trials, plan)
        store = self.persist(record)
        report = self.comparator(store=store).compare(record.campaign_id)[0]
        self.assertEqual(report.n_planned, 5)
        self.assertEqual(report.n_pairs, 4)
        self.assertEqual(len(report.missing_pairs), 1)
        self.assertEqual(report.missing_pairs[0]["seed"], 46)
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.decision_rule, "R1_pairs_incomplete")

    def test_failed_baseline_and_failed_candidate_are_recorded(self):
        plan = make_plan((42, 43))
        trials = [
            make_trial(seed=42, arm_type="baseline", arm_id="baseline",
                       value=0.10, status="failed", error_message="oom"),
            make_trial(seed=42, arm_type="candidate", arm_id="cand_0",
                       value=0.09),
            make_trial(seed=43, arm_type="baseline", arm_id="baseline",
                       value=0.10),
            make_trial(seed=43, arm_type="candidate", arm_id="cand_0",
                       value=0.09, status="failed", error_message="crash"),
        ]
        record = make_record(trials, plan)
        store = self.persist(record)
        report = self.comparator(store=store).compare(record.campaign_id)[0]
        self.assertEqual(report.n_pairs, 0)
        self.assertEqual(len(report.missing_pairs), 2)
        reasons = " ".join(m["reason"] for m in report.missing_pairs)
        self.assertIn("oom", reasons)
        self.assertIn("crash", reasons)
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.baseline_values, [])

    def test_integrity_failure_does_not_destroy_other_candidates(self):
        plan = make_plan(SEEDS, n_candidates=2)
        trials = paired_trials(SEEDS, BASELINE, IMPROVED)
        for seed in SEEDS:
            trials.append(make_trial(seed=seed, arm_type="candidate",
                                     arm_id="cand_1", value=0.09,
                                     dataset_version="WRONG"))
        record = make_record(trials, plan)
        store = self.persist(record)
        comparator = self.comparator(store=store)
        reports = comparator.compare(record.campaign_id)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].candidate_id, "cand_0")
        events = [e["event_type"] for e in store.list_events("cmp_001")]
        self.assertIn("comparison_failed", events)


class TestGuardrailDecisions(ComparatorTestBase):
    def test_guardrail_violation_rejects(self):
        baseline_guard = [{"aux": 0.82} for _ in SEEDS]
        candidate_guard = [{"aux": 0.70} for _ in SEEDS]
        report, *_ = self.one(
            plan_kwargs={"guardrails": {"aux": {"min": 0.80}}},
            trial_kwargs={"baseline_guard": baseline_guard,
                          "candidate_guard": candidate_guard})
        self.assertEqual(report.decision, "rejected")
        self.assertEqual(report.decision_rule, "R3_guardrail_hard_violation")
        self.assertTrue(report.guardrail_violations)
        self.assertEqual(report.guardrail_results[0]["status"], "violated")

    def test_missing_guardrail_metric_is_inconclusive(self):
        report, *_ = self.one(
            plan_kwargs={"guardrails": {"ghost": {"min": 0.80}}})
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.decision_rule,
                         "R4_guardrail_insufficient_evidence")
        self.assertEqual(report.guardrail_results[0]["status"], "insufficient")

    def test_guardrail_pass_allows_acceptance(self):
        baseline_guard = [{"aux": 0.82} for _ in SEEDS]
        candidate_guard = [{"aux": 0.83} for _ in SEEDS]
        report, *_ = self.one(
            plan_kwargs={"guardrails": {"aux": {"min": 0.80}}},
            trial_kwargs={"baseline_guard": baseline_guard,
                          "candidate_guard": candidate_guard})
        self.assertEqual(report.decision, "accepted_for_confirmation")
        self.assertEqual(report.guardrail_results[0]["status"], "passed")

    def test_relative_regression_without_direction_is_inconclusive(self):
        report, *_ = self.one(
            plan_kwargs={"guardrails": {
                "aux": {"max_relative_regression_pct": 5.0}}})
        self.assertEqual(report.decision, "inconclusive")
        self.assertEqual(report.guardrail_results[0]["status"],
                         "invalid_rule")


class TestHolmAndReports(ComparatorTestBase):
    def test_two_candidates_use_holm_and_produce_two_reports(self):
        # Non-zero variance is essential here: with a point interval a wider
        # corrected interval would be indistinguishable from the 95% one.
        variable = [0.091, 0.089, 0.090, 0.088, 0.092]
        other = [0.095, 0.093, 0.094, 0.092, 0.096]
        plan = make_plan(SEEDS, n_candidates=2)
        trials = paired_trials(SEEDS, BASELINE, variable)
        for seed, value in zip(SEEDS, other):
            trials.append(make_trial(seed=seed, arm_type="candidate",
                                     arm_id="cand_1", value=value))
        record = make_record(trials, plan)
        store = self.persist(record)
        reports = self.comparator(store=store).compare(record.campaign_id)
        self.assertEqual(len(reports), 2)
        for report in reports:
            mc = report.multiple_comparison
            self.assertEqual(mc["method"], "holm")
            self.assertEqual(mc["family_size"], 2)
            self.assertIn("adjusted_p", mc)
            self.assertIn("rank", mc)
            self.assertLessEqual(mc["adjusted_alpha"], 0.05)
            a, b = report.confidence_interval_95
            c, d = report.decision_confidence_interval
            if mc["rank"] == 0:
                # Holm reduces alpha for the leading hypothesis => wider CI.
                self.assertLess(mc["adjusted_alpha"], 0.05)
                self.assertGreater(d - c, b - a)
            else:
                # The last step-down rank keeps alpha=0.05 but may still be
                # blocked by the monotone adjusted p (Holm's stop rule).
                self.assertAlmostEqual(mc["adjusted_alpha"], 0.05)
                self.assertGreaterEqual(d - c, b - a - 1e-12)
        self.assertEqual(len(self.report_store().list_reports("cmp_001")), 2)

    def test_report_fields_are_complete_and_hash_consistent(self):
        report, store, comparator, record = self.one()
        for field in ("campaign_id", "metric_direction", "delta_convention",
                      "n_planned", "n_pairs", "std_delta", "interval_method",
                      "confidence_level", "decision_confidence_interval",
                      "practical_effect_threshold", "decision_rule",
                      "multiple_comparison", "guardrail_results",
                      "missing_pairs", "paired_observations", "evidence_refs",
                      "comparison_algorithm_version", "baseline_config_sha256",
                      "candidate_config_sha256", "dataset_version",
                      "evaluation_split", "report_sha256", "created_at"):
            self.assertIsNotNone(getattr(report, field), field)
        self.assertEqual(report.n_pairs, 5)
        self.assertEqual(len(report.paired_observations), 5)
        self.assertEqual(len(report.evidence_refs), 10)
        self.assertEqual(report.report_sha256, content_sha256(report))
        loaded = self.report_store().load("cmp_001", report.comparison_id)
        self.assertEqual(loaded.report_sha256, content_sha256(loaded))
        self.assertEqual(loaded.model_dump(), report.model_dump())
        events = [e["event_type"] for e in store.list_events("cmp_001")]
        self.assertIn("comparison_completed", events)
        self.assertEqual(comparator.event_errors, [])

    def test_repeated_compare_does_not_overwrite_history(self):
        report, store, _, record = self.one()
        path = Path(self.report_store().latest("cmp_001", "cand_0")["file"])
        full = self.campaign_root / "cmp_001" / "comparisons" / path
        before = full.read_bytes()
        again = self.comparator(store=store).compare("cmp_001")[0]
        self.assertEqual(again.comparison_id, report.comparison_id)
        self.assertEqual(again.model_dump(), report.model_dump())
        self.assertEqual(full.read_bytes(), before)
        self.assertEqual(len(self.report_store().list_reports("cmp_001")), 1)

    def test_new_algorithm_version_creates_a_new_report(self):
        first, store, _, record = self.one()
        second = self.comparator(
            store=store, algorithm_version="2.0.0").compare("cmp_001")[0]
        self.assertNotEqual(first.comparison_id, second.comparison_id)
        self.assertEqual(len(self.report_store().list_reports("cmp_001")), 2)
        old_path = (self.campaign_root / "cmp_001" / "comparisons"
                    / self.report_store().list_reports("cmp_001")[0]["file"])
        self.assertTrue(old_path.is_file())

    def test_compare_candidate_matches_single_candidate_compare(self):
        report, store, comparator, record = self.one()
        single = comparator.compare_candidate(record, "cand_0")
        self.assertEqual(single.comparison_id, report.comparison_id)
        self.assertEqual(single.decision, report.decision)

    def test_unknown_campaign_is_refused(self):
        with self.assertRaises(ComparisonIntegrityError):
            self.comparator().compare("does_not_exist")

    def test_event_log_failure_does_not_erase_the_report(self):
        class BrokenEvents:
            def append_event(self, *args, **kwargs):
                raise RuntimeError("simulated DB outage")
        report, *_ = self.one(comparator_kwargs={"event_store": BrokenEvents()})
        self.assertEqual(report.decision, "accepted_for_confirmation")
        self.assertTrue(self.report_store().list_reports("cmp_001"))


if __name__ == "__main__":
    unittest.main()
