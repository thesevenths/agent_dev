"""P1-05B: guardrail rule semantics -- missing evidence is NEVER a pass."""
from __future__ import annotations

import unittest

from research_runtime.comparison import GuardrailEvaluator, PairedObservation


def make_pair(seed: int = 42, *, baseline_aux: float = 0.82,
              candidate_aux: float = 0.83) -> PairedObservation:
    return PairedObservation(
        seed=seed, baseline_value=0.10, candidate_value=0.09, delta=0.01,
        baseline_trial_id=f"baseline_seed_{seed}",
        candidate_trial_id=f"cand_0_seed_{seed}",
        baseline_guardrail_metrics={"aux": baseline_aux},
        candidate_guardrail_metrics={"aux": candidate_aux},
    )


def pairs(*specs) -> list[PairedObservation]:
    return [make_pair(seed=40 + index,
                      baseline_aux=b, candidate_aux=c)
            for index, (b, c) in enumerate(specs)]


class TestGuardrailEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = GuardrailEvaluator()

    def status(self, rules, observations):
        result = self.evaluator.evaluate(rules, observations)
        self.assertEqual(len(result.results), len(rules))
        return result.results[0].status if result.results else "passed"

    def test_absolute_min_and_max(self):
        self.assertEqual(self.status({"aux": {"min": 0.80}},
                                     pairs((0.82, 0.83))), "passed")
        self.assertEqual(self.status({"aux": {"min": 0.90}},
                                     pairs((0.82, 0.83))), "violated")
        self.assertEqual(self.status({"aux": {"max": 0.90}},
                                     pairs((0.82, 0.83))), "passed")
        self.assertEqual(self.status({"aux": {"max": 0.80}},
                                     pairs((0.82, 0.83))), "violated")

    def test_relative_regression_requires_an_explicit_direction(self):
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0}},
                        pairs((0.82, 0.70))), "invalid_rule")
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0,
                                 "direction": "sideways"}},
                        pairs((0.82, 0.70))), "invalid_rule")

    def test_relative_regression_maximize(self):
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0,
                                 "direction": "maximize"}},
                        pairs((0.82, 0.80))), "passed")
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0,
                                 "direction": "maximize"}},
                        pairs((0.82, 0.70))), "violated")

    def test_relative_regression_minimize(self):
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0,
                                 "direction": "minimize"}},
                        pairs((0.10, 0.104))), "passed")
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0,
                                 "direction": "minimize"}},
                        pairs((0.10, 0.12))), "violated")

    def test_missing_metric_is_insufficient_never_passed(self):
        pair = make_pair()
        pair.candidate_guardrail_metrics = {}
        assessment = self.evaluator.evaluate({"ghost": {"min": 0.1}}, [pair])
        self.assertEqual(assessment.results[0].status, "insufficient")
        self.assertTrue(assessment.insufficient)
        self.assertFalse(assessment.has_hard_violation)

    def test_nonfinite_metric_is_insufficient(self):
        for value in (float("nan"), float("inf"), None, "bad", True):
            with self.subTest(value=value):
                # Bypass pydantic's dict[str, float] coercion: the evaluator
                # must defend against raw engine payloads that are not floats.
                pair = make_pair()
                pair.candidate_guardrail_metrics = {"aux": value}
                assessment = self.evaluator.evaluate(
                    {"aux": {"min": 0.1}}, [pair])
                self.assertEqual(assessment.results[0].status, "insufficient")

    def test_invalid_rules(self):
        invalid = [
            {"aux": 0.8},
            {"aux": {}},
            {"aux": {"min": "high"}},
            {"aux": {"max": float("nan")}},
            {"aux": {"min": 0.9, "max": 0.1}},
            {"aux": {"min": 0.1, "direction": "magic"}},
            {"aux": {"min": 0.1, "unknown_key": 1}},
            {"aux": {"max_relative_regression_pct": -1.0,
                     "direction": "minimize"}},
        ]
        for rules in invalid:
            with self.subTest(rules=rules):
                self.assertEqual(self.status(rules, pairs((0.82, 0.83))),
                                 "invalid_rule")

    def test_zero_baseline_makes_relative_rule_insufficient(self):
        self.assertEqual(
            self.status({"aux": {"max_relative_regression_pct": 5.0,
                                 "direction": "minimize"}},
                        pairs((0.0, 0.01))), "insufficient")

    def test_no_rules_pass_vacuously(self):
        assessment = self.evaluator.evaluate({}, pairs((0.82, 0.83)))
        self.assertEqual(assessment.results, [])
        self.assertTrue(assessment.all_passed)
        self.assertFalse(assessment.has_hard_violation)

    def test_no_valid_pairs_is_insufficient(self):
        assessment = self.evaluator.evaluate({"aux": {"min": 0.1}}, [])
        self.assertEqual(assessment.results[0].status, "insufficient")

    def test_one_bad_seed_is_not_hidden_by_averaging(self):
        mixed = pairs((0.82, 0.83), (0.82, 0.70), (0.82, 0.85))
        assessment = self.evaluator.evaluate({"aux": {"min": 0.80}}, mixed)
        result = assessment.results[0]
        self.assertEqual(result.status, "violated")
        self.assertEqual(len(result.per_seed), 3)
        self.assertEqual(result.per_seed[1]["status"], "violated")
        self.assertTrue(assessment.has_hard_violation)
        self.assertEqual(len(assessment.hard_violations), 1)

    def test_hard_violations_separate_from_insufficient(self):
        good = make_pair(seed=42, baseline_aux=0.82, candidate_aux=0.83)
        missing = make_pair(seed=43)
        missing.candidate_guardrail_metrics = {}
        assessment = self.evaluator.evaluate(
            {"aux": {"min": 0.80}, "ghost": {"min": 0.1}}, [good, missing])
        self.assertTrue(assessment.has_hard_violation is False)
        self.assertTrue(assessment.insufficient)
        # An insufficient rule is not a hard violation, but IS a non-passing one.
        self.assertEqual(assessment.hard_violations, [])
        self.assertEqual(len(assessment.violations), 2)


if __name__ == "__main__":
    unittest.main()
