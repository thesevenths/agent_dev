"""P1-01: the research decision protocol.

A Hypothesis is NOT a ``FailureReport.hypothesis`` string: it must carry
evidence, a concrete intervention, predicted effects and a falsification rule
before any experiment is allowed to run. Candidate selection must never touch
the reserved final-evaluation split.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import ClassVar

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydantic import ValidationError

from research_runtime import (
    CandidateExperiment,
    ComparisonReport,
    ExperimentPlan,
    FailureReport,
    Hypothesis,
    MetricPrediction,
)


def hypothesis(**kw) -> Hypothesis:
    defaults = dict(
        hypothesis_id="H001",
        objective_id="obj1",
        source_experiment_id="exp_baseline_001",
        title="Predictor Capacity Limitation",
        statement="Increasing predictor capacity may improve long-horizon prediction.",
        mechanism="The current predictor may not capture nonlinear latent dynamics.",
        evidence_refs=["experiment:exp_baseline_001:metric:latent_mse@3"],
        intervention={"model.predictor_hidden": [64, 64]},
        predicted_effects=[MetricPrediction(metric="latent_mse@3",
                                           direction="decrease")],
        falsification_criteria=["Validation latent_mse@3 does not improve across paired seeds."],
        confounders=["Training budget"],
        confidence=0.65,
    )
    defaults.update(kw)
    return Hypothesis(**defaults)


def candidate(**kw) -> CandidateExperiment:
    defaults = dict(
        candidate_id="candidate_A",
        hypothesis_id="H001",
        config_overrides={"model.predictor_hidden": [64, 64]},
        seeds=[42, 43, 44],
        dataset_version="pendulum-v1",
    )
    defaults.update(kw)
    return CandidateExperiment(**defaults)


def plan(**kw) -> ExperimentPlan:
    defaults = dict(
        plan_id="campaign_001",
        objective_id="obj1",
        baseline_experiment_id="exp_baseline_001",
        dataset_version="pendulum-v1",
        candidates=[candidate()],
        primary_metric="latent_mse@3",
        metric_direction="minimize",
        paired_seeds=[42, 43, 44],
    )
    defaults.update(kw)
    return ExperimentPlan(**defaults)


class TestHypothesisSchema(unittest.TestCase):
    def test_valid_hypothesis(self):
        h = hypothesis()
        self.assertEqual(h.status, "proposed")
        self.assertEqual(h.predicted_effects[0].direction, "decrease")

    def test_roundtrip_json(self):
        h = hypothesis()
        restored = Hypothesis.model_validate(json.loads(h.model_dump_json()))
        self.assertEqual(restored, h)

    def test_missing_evidence_refs_is_rejected(self):
        with self.assertRaises(ValidationError):
            hypothesis(evidence_refs=[])

    def test_missing_falsification_criteria_is_rejected(self):
        with self.assertRaises(ValidationError):
            hypothesis(falsification_criteria=[])

    def test_blank_evidence_or_criterion_is_rejected(self):
        with self.assertRaises(ValidationError):
            hypothesis(evidence_refs=["  "])
        with self.assertRaises(ValidationError):
            hypothesis(falsification_criteria=[""])

    def test_confidence_above_one_is_rejected(self):
        with self.assertRaises(ValidationError):
            hypothesis(confidence=1.2)

    def test_missing_intervention_or_prediction_is_rejected(self):
        with self.assertRaises(ValidationError):
            hypothesis(intervention={})
        with self.assertRaises(ValidationError):
            hypothesis(predicted_effects=[])

    def test_intervention_keys_must_be_dotted_config_paths(self):
        with self.assertRaises(ValidationError):
            hypothesis(intervention={"../../etc/passwd": 1})

    def test_hypothesis_is_not_a_failure_report_string(self):
        """P1 keeps the two concepts separate on purpose."""
        fr = FailureReport(experiment_id="e", job_id="j",
                           category="x", recommended_action="act",
                           hypothesis="a free-form guess")
        self.assertIsInstance(fr.hypothesis, str)
        self.assertIsInstance(hypothesis(), Hypothesis)


class TestCandidateExperimentSchema(unittest.TestCase):
    def test_default_split_is_validation(self):
        self.assertEqual(candidate().evaluation_split, "validation")
        self.assertEqual(candidate().train_split, "train")

    def test_final_confirmation_split_is_rejected(self):
        with self.assertRaises(ValidationError):
            candidate(evaluation_split="test")

    def test_other_held_out_names_are_allowed(self):
        """Generic across domains: 'dev' is a legitimate selection split."""
        self.assertEqual(candidate(evaluation_split="dev").evaluation_split, "dev")

    def test_train_and_evaluation_split_must_differ(self):
        with self.assertRaises(ValidationError):
            candidate(train_split="validation")

    def test_duplicate_seeds_are_rejected(self):
        with self.assertRaises(ValidationError):
            candidate(seeds=[42, 42])

    def test_empty_overrides_are_rejected(self):
        with self.assertRaises(ValidationError):
            candidate(config_overrides={})

    def test_reserved_split_set_is_overridable(self):
        """A domain whose final split is 'holdout_final' can say so."""
        class DomainCandidate(CandidateExperiment):
            reserved_final_splits: ClassVar[frozenset] = frozenset(
                {"holdout_final", "test"})

        with self.assertRaises(ValidationError):
            DomainCandidate(**(candidate().model_dump()
                               | {"evaluation_split": "holdout_final"}))
        self.assertEqual(
            DomainCandidate(**(candidate().model_dump()
                               | {"evaluation_split": "validation"})).evaluation_split,
            "validation",
        )


class TestExperimentPlanSchema(unittest.TestCase):
    def test_valid_plan_json_serialization(self):
        p = plan()
        restored = ExperimentPlan.model_validate(json.loads(p.model_dump_json()))
        self.assertEqual(restored, p)
        self.assertEqual(restored.candidates[0].seeds, [42, 43, 44])

    def test_candidate_with_different_dataset_version_is_rejected(self):
        with self.assertRaises(ValidationError):
            plan(candidates=[candidate(dataset_version="other-v2")])

    def test_candidate_with_unpaired_seeds_is_rejected(self):
        with self.assertRaises(ValidationError):
            plan(candidates=[candidate(seeds=[42, 43])])

    def test_duplicate_candidate_ids_are_rejected(self):
        with self.assertRaises(ValidationError):
            plan(candidates=[candidate(), candidate(candidate_id="candidate_A")])

    def test_at_least_one_candidate_is_required(self):
        with self.assertRaises(ValidationError):
            plan(candidates=[])

    def test_concurrency_is_a_deployment_policy_not_a_protocol_limit(self):
        """No hard-coded GPU-size ceiling in a data contract."""
        self.assertEqual(plan().max_parallel, 1)
        self.assertEqual(plan(max_parallel=8).max_parallel, 8)
        with self.assertRaises(ValidationError):
            plan(max_parallel=0)


class TestComparisonReportSchema(unittest.TestCase):
    def test_paired_arrays_must_match(self):
        base = dict(comparison_id="c1", plan_id="p1", candidate_id="A",
                    primary_metric="latent_mse@3",
                    baseline_values=[0.10, 0.11], candidate_values=[0.08, 0.09],
                    paired_deltas=[-0.02, -0.02], mean_delta=-0.02,
                    median_delta=-0.02, decision="accepted_for_confirmation")
        self.assertEqual(ComparisonReport(**base).decision,
                         "accepted_for_confirmation")
        with self.assertRaises(ValidationError):
            ComparisonReport(**(base | {"paired_deltas": [-0.02]}))
        with self.assertRaises(ValidationError):
            ComparisonReport(**(base | {"baseline_values": []}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
