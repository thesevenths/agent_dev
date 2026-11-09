"""P1-02: generation is evidence-gated and LLM output is untrusted."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.failure_diagnoser import FailureDiagnoser
from research_runtime.hypothesis_manager import (
    HORIZON_DIAGNOSIS,
    HypothesisManager,
    LLMProposer,
    RuleBasedProposer,
    build_experiment_plan,
    degraded_families,
    improvement_direction,
)
from research_runtime.schemas import (
    ExperimentResult,
    FailureReport,
    Hypothesis,
    MetricPrediction,
)

# Domain policy is INJECTED by the caller. The production module deliberately
# contains no engine-specific config key.
POLICY = {HORIZON_DIAGNOSIS: {"model.predictor_hidden": [64, 64]}}
ALLOWED_KEYS = {"model.predictor_hidden", "train.lr", "model.latent_dim"}


def result(metrics=None, *, experiment_id="exp_baseline_001") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id, job_id="j1", status="succeeded",
        stage="evaluate",
        metrics=metrics if metrics is not None else {
            "latent_mse@1": 0.01, "latent_mse@2": 0.02, "latent_mse@4": 0.04,
        },
    )


def horizon_report() -> FailureReport:
    return FailureReport(
        experiment_id="exp_baseline_001", job_id="j1",
        category=HORIZON_DIAGNOSIS,
        evidence={"rule": "horizon_degradation"},
        recommended_action="compare on validation",
    )


def llm_payload(**overrides) -> str:
    item = dict(
        hypothesis_id="H_LLM", objective_id="obj1",
        source_experiment_id="exp_FORGED",
        title="LLM proposal", statement="A testable statement",
        mechanism="A plausible mechanism",
        evidence_refs=["experiment:exp_baseline_001:metric:latent_mse@4"],
        intervention={"model.predictor_hidden": [64, 64]},
        predicted_effects=[dict(metric="latent_mse@4", direction="decrease")],
        falsification_criteria=["Validation latent_mse@4 does not improve across seeds."],
        confidence=0.6,
    )
    item.update(overrides)
    return json.dumps([item])


class TestGenericDegradation(unittest.TestCase):
    def test_direction_is_inferred_from_generic_metric_names(self):
        self.assertEqual(improvement_direction("prediction_mse"), "decrease")
        self.assertEqual(improvement_direction("accuracy_score"), "increase")
        self.assertIsNone(improvement_direction("throughput"))

    def test_any_horizon_family_is_detected(self):
        families = degraded_families({
            "backtest_error@1": 0.01, "backtest_error@2": 0.02,
            "backtest_error@4": 0.04,
            "accuracy_score@1": 0.9, "accuracy_score@4": 0.3,
        })
        self.assertEqual({family for family, _, _ in families},
                         {"backtest_error", "accuracy_score"})

    def test_flat_and_unknown_semantics_are_ignored(self):
        self.assertEqual(degraded_families({"mse@1": 0.01, "mse@4": 0.012}), [])
        self.assertEqual(degraded_families({
            "throughput@1": 1.0, "throughput@4": 100.0,
        }), [])


class TestRuleBasedProposer(unittest.TestCase):
    def test_degradation_produces_evidence_grounded_hypothesis(self):
        manager = HypothesisManager(proposer=RuleBasedProposer(POLICY))
        proposals = manager.propose(result(), [horizon_report()], "obj1")
        self.assertEqual(len(proposals), 1)
        h = proposals[0]
        self.assertEqual(h.objective_id, "obj1")
        self.assertEqual(h.source_experiment_id, "exp_baseline_001")
        self.assertIn("exp_baseline_001", h.evidence_refs[0])
        self.assertEqual(h.intervention, POLICY[HORIZON_DIAGNOSIS])
        self.assertEqual(h.predicted_effects[0].metric, "latent_mse@4")
        self.assertEqual(h.predicted_effects[0].direction, "decrease")
        self.assertIn("validation", h.falsification_criteria[0].lower())
        self.assertLessEqual(h.confidence, 0.9)
        self.assertEqual(h.status, "proposed")
        # a reviewer must still approve it before it can become a candidate
        approved = manager.review(proposals)
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0].status, "approved")

    def test_no_relevant_metrics_means_no_speculation(self):
        proposer = RuleBasedProposer(POLICY)
        out = proposer.propose(result({"latent_mse@1": 0.01}),
                               [horizon_report()], "obj1")
        self.assertEqual(out, [])
        self.assertTrue(proposer.skipped_reasons)

    def test_no_diagnosis_means_no_hypothesis(self):
        proposer = RuleBasedProposer(POLICY)
        self.assertEqual(proposer.propose(result(), [], "obj1"), [])

    def test_no_injected_policy_means_no_invented_root_cause(self):
        proposer = RuleBasedProposer()
        self.assertEqual(proposer.propose(result(), [horizon_report()], "obj1"), [])
        self.assertIn("refusing to invent", " ".join(proposer.skipped_reasons))

    def test_generated_hypothesis_is_not_specific_to_one_metric_family(self):
        proposer = RuleBasedProposer(POLICY)
        out = proposer.propose(
            result({"backtest_error@1": 0.01, "backtest_error@4": 0.05}),
            [horizon_report()], "obj1",
        )
        self.assertEqual(len(out), 1)
        self.assertIn("backtest_error", out[0].title)


class TestLLMProposer(unittest.TestCase):
    def _manager(self, raw, **kw):
        proposer = LLMProposer(lambda prompt: raw,
                               allowed_intervention_keys=ALLOWED_KEYS, **kw)
        return HypothesisManager(proposer=proposer), proposer

    def test_valid_json_becomes_hypothesis(self):
        manager, _ = self._manager(llm_payload())
        proposals = manager.propose(result(), [horizon_report()], "obj1")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].source_experiment_id,
                         "exp_baseline_001")

    def test_fenced_json_is_accepted(self):
        manager, _ = self._manager(f"```json\n{llm_payload()}\n```")
        self.assertEqual(len(manager.propose(result(), [], "obj1")), 1)

    def test_invalid_json_rejects_whole_batch_and_runs_no_experiment(self):
        manager, proposer = self._manager("{not json at all")
        self.assertEqual(manager.propose(result(), [], "obj1"), [])
        self.assertTrue(proposer.rejections)

    def test_unknown_intervention_key_rejects_whole_batch(self):
        manager, proposer = self._manager(llm_payload(
            intervention={"optimizer.magic": True}))
        self.assertEqual(manager.propose(result(), [], "obj1"), [])
        self.assertIn("unknown intervention key",
                      " ".join(proposer.rejections))

    def test_schema_violation_rejects_whole_batch(self):
        manager, proposer = self._manager(llm_payload(confidence=1.7))
        self.assertEqual(manager.propose(result(), [], "obj1"), [])
        self.assertTrue(proposer.rejections)

    def test_generator_failure_never_escapes_into_the_loop(self):
        proposer = LLMProposer(
            lambda prompt: (_ for _ in ()).throw(RuntimeError("model offline")),
            allowed_intervention_keys=ALLOWED_KEYS,
        )
        manager = HypothesisManager(proposer=proposer)
        self.assertEqual(manager.propose(result(), [], "obj1"), [])
        self.assertTrue(proposer.rejections)

    def test_prompt_carries_real_evidence(self):
        captured = {}

        def generate(prompt):
            captured["prompt"] = json.loads(prompt)
            return llm_payload()

        manager = HypothesisManager(
            proposer=LLMProposer(generate, allowed_intervention_keys=ALLOWED_KEYS))
        manager.propose(result(), [horizon_report()], "obj1")
        self.assertEqual(captured["prompt"]["source_experiment_id"],
                         "exp_baseline_001")
        self.assertIn("latent_mse@4", captured["prompt"]["metrics"])


class TestManagerAndPlanner(unittest.TestCase):
    def test_failure_diagnoser_report_feeds_the_manager(self):
        diagnoser = FailureDiagnoser()
        reports = diagnoser.diagnose_all(
            result({"loss@1": 0.01, "loss@4": 0.09}), layer="model")
        manager = HypothesisManager(proposer=RuleBasedProposer(POLICY))
        proposals = manager.propose(
            result({"loss@1": 0.01, "loss@4": 0.09}), reports, "obj1")
        self.assertEqual(len(proposals), 1)
        self.assertIn("loss", proposals[0].title)

    def test_review_rejects_bad_hypothesis_before_planning(self):
        manager = HypothesisManager(proposer=RuleBasedProposer(POLICY))
        proposals = manager.propose(result(), [horizon_report()], "obj1")
        broken = proposals[0].model_copy(update={"evidence_refs": []})
        self.assertEqual(manager.review([broken]), [])
        self.assertTrue(manager.rejections)

    def test_approved_hypotheses_become_a_paired_plan(self):
        manager = HypothesisManager(proposer=RuleBasedProposer(POLICY))
        proposals = manager.propose(result(), [horizon_report()], "obj1")
        approved = manager.review(proposals)
        plan = build_experiment_plan(
            plan_id="campaign_001", objective_id="obj1",
            baseline_experiment_id="exp_baseline_001",
            dataset_version="pendulum-a0e03386be89",
            hypotheses=approved, primary_metric="latent_mse@4",
            paired_seeds=[42, 43],
        )
        self.assertEqual(plan.candidates[0].seeds, [42, 43])
        self.assertEqual(plan.candidates[0].config_overrides,
                         POLICY[HORIZON_DIAGNOSIS])
        self.assertEqual(plan.candidates[0].evaluation_split, "validation")

    def test_unreviewed_hypothesis_cannot_become_a_candidate(self):
        h = Hypothesis(
            hypothesis_id="H1", objective_id="o",
            source_experiment_id="exp_baseline_001",
            title="t", statement="s", mechanism="m",
            evidence_refs=["experiment:exp_baseline_001:metric:loss@4"],
            intervention={"train.lr": 0.0005},
            predicted_effects=[MetricPrediction(metric="loss@4",
                                               direction="decrease")],
            falsification_criteria=["Validation loss@4 does not improve."],
            confidence=0.5,
        )
        with self.assertRaises(ValueError):
            build_experiment_plan(
                plan_id="p", objective_id="o",
                baseline_experiment_id="exp_baseline_001",
                dataset_version="v1", hypotheses=[h], primary_metric="loss@4",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
