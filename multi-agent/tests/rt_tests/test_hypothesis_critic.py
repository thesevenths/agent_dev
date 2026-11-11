"""P1-02: the reviewer is deterministic and an LLM advisor cannot overrule it.
ARR 0.1 / R1: evidence references are grounded in REAL results, with explicit
MATCH / MISMATCH / UNKNOWN / UNRESOLVED verdicts.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.hypothesis_critic import (
    EvidenceVerdict,
    HypothesisCritic,
    verify_evidence_ref,
)
from research_runtime.schemas import Hypothesis, MetricPrediction


def _real_result(**overrides):
    """A duck-typed stand-in for a persisted ExperimentResult."""
    data = dict(metrics={"latent_mse@3": 0.1, "cosine@1": 0.9},
                evaluation_split="validation", dataset_version="ds_frozen_v1",
                run_id="run_42", artifacts=[], result_path=None)
    data.update(overrides)
    return types.SimpleNamespace(**data)


class TestEvidenceIntegrity(unittest.TestCase):
    """ARR 0.1 / R1: evidence references must be grounded in REAL results."""

    def _critic(self, store: dict[str, object] | None = None,
                strict: bool = True):
        db = store if store is not None else {"exp_baseline_001": _real_result()}
        def resolver(experiment_id: str):
            return db.get(experiment_id)
        return HypothesisCritic(evidence_resolver=resolver, strict=strict)

    def test_real_metric_is_accepted(self):
        ok, errors = self._critic().review(hypothesis())
        self.assertTrue(ok, errors)

    def test_fabricated_metric_is_rejected(self):
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:metric:fake_metric@9"])
        ok, errors = self._critic().review(h)
        self.assertFalse(ok)
        self.assertTrue(any("fake_metric@9" in e and "not present" in e
                            for e in errors))

    def test_unknown_experiment_is_rejected(self):
        h = hypothesis(evidence_refs=[
            "experiment:exp_missing:metric:latent_mse@3"])
        ok, errors = self._critic().review(h)
        self.assertFalse(ok)
        self.assertTrue(any("unknown experiment_id" in e
                            or EvidenceVerdict.UNRESOLVED in e for e in errors))

    def test_existence_only_ref_accepted(self):
        h = hypothesis(evidence_refs=["experiment:exp_baseline_001"])
        ok, errors = self._critic().review(h)
        self.assertTrue(ok, errors)

    def test_split_mismatch_is_rejected(self):
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:metric:latent_mse@3:split:test"])
        ok, errors = self._critic().review(h)
        self.assertFalse(ok)
        self.assertTrue(any("split" in e and "test" in e for e in errors))

    def test_dataset_mismatch_is_rejected(self):
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:dataset:ds_wrong"])
        ok, errors = self._critic().review(h)
        self.assertFalse(ok)
        self.assertTrue(any("dataset_version" in e for e in errors))

    def test_run_mismatch_is_rejected(self):
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:run:run_999"])
        ok, errors = self._critic().review(h)
        self.assertFalse(ok)
        self.assertTrue(any("run_id" in e for e in errors))

    def test_no_resolver_keeps_backward_compatible(self):
        # Without a resolver the runtime does NOT invent a metric check, so a
        # non-existent metric cannot be rejected on those grounds alone.
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:metric:fake_metric@9"])
        ok, _ = HypothesisCritic().review(h)
        self.assertTrue(ok)


class TestEvidenceVerdicts(unittest.TestCase):
    """ARR 0.1 / R1 hardening: UNKNOWN != PASS, artifact identity, malformed."""

    def _resolver(self, store):
        def r(eid):
            return store.get(eid)
        return r

    def test_unknown_field_is_unverifiable_and_rejected_in_strict(self):
        # Real result never recorded a split, yet the ref claims one -> UNKNOWN.
        res = _real_result(evaluation_split=None)
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:metric:latent_mse@3:split:validation"])
        ok, errors = HypothesisCritic(
            evidence_resolver=self._resolver({"exp_baseline_001": res})).review(h)
        self.assertFalse(ok)
        self.assertTrue(any(EvidenceVerdict.UNKNOWN in e for e in errors))

    def test_unknown_field_passes_in_non_strict_legacy_mode(self):
        res = _real_result(evaluation_split=None)
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:metric:latent_mse@3:split:validation"])
        ok, _ = HypothesisCritic(
            evidence_resolver=self._resolver({"exp_baseline_001": res}),
            strict=False).review(h)
        self.assertTrue(ok, "non-strict legacy path must not reject UNKNOWN")

    def test_verify_evidence_ref_returns_explicit_verdicts(self):
        res = _real_result()
        v = verify_evidence_ref(
            "experiment:exp_baseline_001:metric:latent_mse@3",
            self._resolver({"exp_baseline_001": res}))
        self.assertEqual(v.verdict, EvidenceVerdict.MATCH)
        v2 = verify_evidence_ref(
            "experiment:exp_missing:metric:x", self._resolver({}))
        self.assertEqual(v2.verdict, EvidenceVerdict.UNRESOLVED)

    def test_artifact_missing_on_disk_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            res = _real_result(artifacts=["ckpt.pt"],
                               result_path=os.path.join(d, "report.json"))
            h = hypothesis(evidence_refs=[
                "experiment:exp_baseline_001:artifact:ckpt.pt"])
            ok, errors = HypothesisCritic(
                evidence_resolver=self._resolver({"exp_baseline_001": res}),
            ).review(h)
            self.assertFalse(ok)
            self.assertTrue(any("artifact" in e and "missing" in e
                                for e in errors))

    def test_artifact_sha256_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            import hashlib
            p = Path(d) / "ckpt.pt"
            p.write_bytes(b"original weights")
            good = hashlib.sha256(b"original weights").hexdigest()
            bad = "0" * 64
            res = _real_result(artifacts=["ckpt.pt"],
                               result_path=os.path.join(d, "report.json"))
            # cited hash differs from actual -> MISMATCH
            h = hypothesis(evidence_refs=[
                f"experiment:exp_baseline_001:artifact:ckpt.pt@{bad}"])
            ok, errors = HypothesisCritic(
                evidence_resolver=self._resolver({"exp_baseline_001": res}),
            ).review(h)
            self.assertFalse(ok)
            self.assertTrue(any("SHA-256 mismatch" in e for e in errors))
            # correct hash -> MATCH
            h2 = hypothesis(evidence_refs=[
                f"experiment:exp_baseline_001:artifact:ckpt.pt@{good}"])
            ok2, _ = HypothesisCritic(
                evidence_resolver=self._resolver({"exp_baseline_001": res}),
            ).review(h2)
            self.assertTrue(ok2)

    def test_malformed_reference_is_rejected(self):
        h = hypothesis(evidence_refs=["this reference has no key value pairs"])
        ok, errors = HypothesisCritic(
            evidence_resolver=self._resolver({})).review(h)
        self.assertFalse(ok)
        self.assertTrue(any("malformed" in e for e in errors))

    def test_resolver_failure_is_treated_as_unresolved(self):
        def boom(eid):
            raise RuntimeError("evidence store offline")
        h = hypothesis(evidence_refs=[
            "experiment:exp_baseline_001:metric:latent_mse@3"])
        ok, errors = HypothesisCritic(evidence_resolver=boom).review(h)
        self.assertFalse(ok)
        self.assertTrue(any(EvidenceVerdict.UNRESOLVED in e for e in errors))


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
