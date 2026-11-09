"""P1-05A: comparison protocol, backward compatibility, immutable storage."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from research_runtime.comparison import (
    COMPARISON_ALGORITHM_VERSION,
    ComparisonPolicy,
    ComparisonReportStore,
    PairedComparisonReport,
    content_sha256,
)
from research_runtime.schemas import ComparisonReport, ExperimentPlan
from tests.rt_tests.comparison_fakes import ComparisonTestBase, make_plan


def make_report(**overrides):
    base = dict(
        comparison_id="cmp_" + "a" * 24, plan_id="plan_001",
        candidate_id="cand_0", primary_metric="mse@1",
        baseline_values=[0.10, 0.11, 0.10, 0.12, 0.11],
        candidate_values=[0.09, 0.10, 0.09, 0.11, 0.10],
        paired_deltas=[0.01, 0.01, 0.01, 0.01, 0.01],
        mean_delta=0.01, median_delta=0.01,
        confidence_interval_95=(0.01, 0.01),
        decision="accepted_for_confirmation", campaign_id="cmp_001",
    )
    base.update(overrides)
    return PairedComparisonReport(**base)


def final_report(**overrides):
    """A report whose digest is computed from ITS OWN content (no clock race)."""
    report = make_report(**overrides)
    return report.model_copy(update={"report_sha256": content_sha256(report)})


class TestComparisonProtocol(unittest.TestCase):
    def test_p1_01_construction_remains_valid(self):
        """Every new field has a default: P1-01 callers are unaffected."""
        old = ComparisonReport(
            comparison_id="c1", plan_id="p1", candidate_id="A",
            primary_metric="m", baseline_values=[0.1],
            candidate_values=[0.09], paired_deltas=[0.01],
            mean_delta=0.01, median_delta=0.01,
            decision="accepted_for_confirmation")
        self.assertEqual(old.decision, "accepted_for_confirmation")
        self.assertEqual(old.delta_convention, "positive_is_better")
        self.assertEqual(old.n_pairs, 0)
        self.assertIsNone(old.report_sha256)

    def test_base_class_still_rejects_empty_arrays(self):
        with self.assertRaises(ValidationError):
            ComparisonReport(
                comparison_id="c", plan_id="p", candidate_id="A",
                primary_metric="m", baseline_values=[],
                candidate_values=[], paired_deltas=[], mean_delta=0.0,
                median_delta=0.0, decision="inconclusive")

    def test_zero_pair_report_is_allowed_but_never_accepted(self):
        report = make_report(
            baseline_values=[], candidate_values=[], paired_deltas=[],
            decision="inconclusive")
        self.assertEqual(report.n_pairs, 0)
        with self.assertRaises(ValidationError):
            make_report(baseline_values=[], candidate_values=[],
                        paired_deltas=[], decision="rejected")
        with self.assertRaises(ValidationError):
            make_report(baseline_values=[], candidate_values=[],
                        paired_deltas=[],
                        decision="accepted_for_confirmation")

    def test_unequal_arrays_are_rejected(self):
        with self.assertRaises(ValidationError):
            make_report(paired_deltas=[0.01, 0.02])

    def test_comparison_policy_validation(self):
        self.assertEqual(ComparisonPolicy().min_paired_seeds, 5)
        for kwargs in ({"min_paired_seeds": 1}, {"familywise_alpha": 0.0},
                       {"practical_effect_threshold": -1.0},
                       {"practical_effect_mode": "magic"},
                       {"multiple_comparison_method": "bonferonni"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ComparisonPolicy(**kwargs)

    def test_relative_effect_threshold_resolution(self):
        policy = ComparisonPolicy(practical_effect_threshold=0.05,
                                  practical_effect_mode="relative")
        value, mode = policy.effective_threshold([0.10, 0.20])
        self.assertAlmostEqual(value, 0.0075)
        self.assertEqual(mode, "relative_to_baseline_mean")
        value, mode = policy.effective_threshold([0.0, 0.0])
        self.assertEqual(value, 0.0)
        self.assertEqual(mode, "relative_undefined_nonpositive_baseline")

    def test_guardrail_rules_can_declare_direction(self):
        """The P1-05A widening: a string direction is no longer rejected."""
        plan = make_plan(guardrails={
            "quality_score": {"min": 0.8,
                              "max_relative_regression_pct": 5.0,
                              "direction": "maximize"}})
        self.assertEqual(plan.guardrails["quality_score"]["direction"],
                         "maximize")
        self.assertIsInstance(plan, ExperimentPlan)
        # Plain float rules from P1-01 still validate.
        self.assertEqual(make_plan(guardrails={"quality_score": {"min": 0.8}})
                         .guardrails["quality_score"]["min"], 0.8)

    def test_production_modules_contain_no_engine_specific_keys(self):
        import re
        root = Path(__file__).resolve().parents[2] / "research_runtime" / "comparison"
        pattern = re.compile(
            r"predictor_hidden|latent_dim|ema_momentum|pendulum|latent_mse|"
            r"cosine@|jepa|torch|mujoco|scinet", re.I)
        offenders = []
        for path in sorted(root.glob("*.py")):
            if path.name == "scinet.py":       # the one domain wiring module
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8")
                                           .splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{line_no}: {line.strip()}")
        self.assertEqual(offenders, [])

    def test_algorithm_version_is_fixed(self):
        self.assertEqual(COMPARISON_ALGORITHM_VERSION, "1.0.0")


class TestImmutableReportStore(ComparisonTestBase):
    def test_save_load_and_hashes(self):
        report = final_report()
        store = self.report_store()
        artifact = store.save(report, campaign_id="cmp_001")
        self.assertTrue(artifact.created_new)
        payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        self.assertEqual(payload["comparison_id"], report.comparison_id)
        self.assertEqual(payload["report_sha256"], artifact.content_sha256)
        loaded = store.load("cmp_001", report.comparison_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.decision, report.decision)
        self.assertEqual(loaded.report_sha256, report.report_sha256)
        self.assertEqual(store.latest("cmp_001", "cand_0")["comparison_id"],
                         report.comparison_id)

    def test_repeated_save_never_overwrites(self):
        report = final_report()
        store = self.report_store()
        first = store.save(report, campaign_id="cmp_001")
        before = Path(first.path).read_bytes()
        second = store.save(report, campaign_id="cmp_001")
        self.assertFalse(second.created_new)
        self.assertEqual(Path(first.path).read_bytes(), before)
        self.assertEqual(len(store.list_reports("cmp_001")), 1)

    def test_new_algorithm_version_creates_a_new_file(self):
        old = final_report()
        new = final_report(comparison_id="cmp_" + "b" * 24,
                           comparison_algorithm_version="2.0.0")
        store = self.report_store()
        first = store.save(old, campaign_id="cmp_001")
        before = Path(first.path).read_bytes()
        second = store.save(new, campaign_id="cmp_001")
        self.assertTrue(second.created_new)
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(Path(first.path).read_bytes(), before)
        self.assertEqual(len(store.list_reports("cmp_001")), 2)
        self.assertEqual(store.latest("cmp_001", "cand_0")["comparison_id"],
                         new.comparison_id)

    def test_tampered_file_is_detected(self):
        report = final_report()
        store = self.report_store()
        artifact = store.save(report, campaign_id="cmp_001")
        Path(artifact.path).write_text(
            Path(artifact.path).read_text(encoding="utf-8") + " ",
            encoding="utf-8")
        from research_runtime.comparison import ComparisonStoreError
        with self.assertRaises(ComparisonStoreError):
            store.load("cmp_001", report.comparison_id)

    def test_unsafe_identifiers_are_rejected(self):
        report = final_report()
        store = self.report_store()
        from research_runtime.comparison import ComparisonStoreError
        with self.assertRaises(ComparisonStoreError):
            store.save(report, campaign_id="../../escape")
        with self.assertRaises(ComparisonStoreError):
            store.save(make_report(candidate_id="../../evil"),
                       campaign_id="cmp_001")


if __name__ == "__main__":
    unittest.main()
