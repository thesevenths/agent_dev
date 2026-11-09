"""P1-02: the reviewer is deterministic and an LLM advisor cannot overrule it."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.hypothesis_critic import HypothesisCritic
from research_runtime.schemas import Hypothesis, MetricPrediction


def hypothesis(**kw) -> Hypothesis:
    defaults = dict(
        hypothesis_id="H001", objective_id="obj1",
        source_experiment_id="exp_baseline_001",
        title="Predictor Capacity Limitation",
        statement="Increasing capacity may improve long-horizon prediction.",
        mechanism="The current model may under-fit nonlinear dynamics.",
        evidence_refs=["experiment:exp_baseline_001:metric:latent_mse@3"],
        intervention={"model.predictor_hidden": [64, 64]},
        predicted_effects=[MetricPrediction(metric="latent_mse@3",
                                           direction="decrease")],
        falsification_criteria=["Validation latent_mse@3 does not improve across paired seeds."],
        confidence=0.65,
    )
    defaults.update(kw)
    return Hypothesis(**defaults)


def broken(**kw) -> Hypothesis:
    """Bypass schema validation to simulate an untrusted upstream object."""
    return hypothesis().model_copy(update=kw)


class TestHypothesisCritic(unittest.TestCase):
    def test_valid_hypothesis_is_approved(self):
        ok, errors = HypothesisCritic().review(hypothesis())
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_missing_evidence_is_rejected(self):
        ok, errors = HypothesisCritic().review(broken(evidence_refs=[]))
        self.assertFalse(ok)
        self.assertIn("missing evidence", errors)

    def test_missing_falsification_criteria_is_rejected(self):
        ok, errors = HypothesisCritic().review(
            broken(falsification_criteria=[]))
        self.assertFalse(ok)
        self.assertIn("missing falsification criteria", errors)

    def test_missing_intervention_is_rejected(self):
        ok, errors = HypothesisCritic().review(broken(intervention={}))
        self.assertFalse(ok)
        self.assertIn("missing intervention", errors)

    def test_empty_metric_is_rejected(self):
        h = broken(predicted_effects=[
            MetricPrediction.model_construct(metric="", direction="decrease")])
        ok, errors = HypothesisCritic().review(h)
        self.assertFalse(ok)
        self.assertIn("empty metric in predicted_effects", errors)

    def test_high_confidence_with_thin_evidence_is_rejected(self):
        h = hypothesis(confidence=0.95)
        ok, errors = HypothesisCritic().review(h)
        self.assertFalse(ok)
        self.assertIn("confidence too high for limited evidence", errors)

    def test_high_confidence_with_enough_evidence_is_accepted(self):
        h = hypothesis(
            confidence=0.95,
            evidence_refs=[
                "experiment:exp_baseline_001:metric:latent_mse@3",
                "experiment:exp_baseline_001:log:train",
                "failure:fr_001",
            ],
        )
        ok, _ = HypothesisCritic().review(h)
        self.assertTrue(ok)

    def test_evidence_must_reference_the_source_experiment(self):
        h = hypothesis(evidence_refs=["experiment:some_other_run:metric:mse"])
        ok, errors = HypothesisCritic().review(h)
        self.assertFalse(ok)
        self.assertTrue(any("source_experiment_id" in error for error in errors))

    def test_metric_policy_is_injected_not_built_in(self):
        # Default: generic runtime does not dictate a domain's metric names.
        ok, _ = HypothesisCritic().review(hypothesis())
        self.assertTrue(ok)
        # With an injected policy, out-of-family metrics are refused.
        critic = HypothesisCritic(allowed_metric_patterns=(r"^latent_mse",))
        ok, errors = critic.review(broken(predicted_effects=[
            MetricPrediction(metric="throughput@3", direction="increase")]))
        self.assertFalse(ok)
        self.assertTrue(any("outside the injected" in error for error in errors))

    def test_advisor_is_advice_only(self):
        critic = HypothesisCritic(
            advisor=lambda h: ["consider a cheaper experiment",
                               "alternative explanation: optimizer drift"])
        ok, _ = critic.review(hypothesis())
        self.assertTrue(ok)
        self.assertEqual(len(critic.advisories), 2)

        rejecting = HypothesisCritic(advisor=lambda h: "this looks fine")
        ok, _ = rejecting.review(broken(evidence_refs=[]))
        self.assertFalse(ok, "an LLM note must not overrule a deterministic gate")
        self.assertEqual(len(rejecting.advisories), 1)

    def test_advisor_exception_never_breaks_the_gate(self):
        critic = HypothesisCritic(
            advisor=lambda h: (_ for _ in ()).throw(RuntimeError("offline")))
        ok, _ = critic.review(hypothesis())
        self.assertTrue(ok)
        self.assertTrue(critic.advisories)

    def test_review_all_marks_approved_status(self):
        critic = HypothesisCritic()
        good = hypothesis()
        bad = broken(hypothesis_id="H002", evidence_refs=[])
        approved = critic.review_all([good, bad])
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0].status, "approved")
        self.assertEqual(critic.rejections[0]["hypothesis_id"], "H002")


if __name__ == "__main__":
    unittest.main(verbosity=2)
