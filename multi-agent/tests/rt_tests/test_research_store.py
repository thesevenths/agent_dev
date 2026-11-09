"""P1-06 A1: trusted hypothesis / plan / approval / session stores."""
from __future__ import annotations

import tempfile
import threading
import unittest
import uuid
from pathlib import Path

from research_runtime.research_store import (
    ResearchStores,
    ResearchStoreConflict,
    ResearchStoreError,
    hypothesis_sha256,
    plan_sha256,
)
from research_runtime.research_store.schemas import CampaignBinding, ResearchReport
from research_runtime.schemas import (
    CandidateExperiment,
    ExperimentPlan,
    Hypothesis,
    MetricPrediction,
    ResearchObjective,
)


def _hypothesis(hid: str = "H1", lr: float = 0.001) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hid, objective_id="obj1", source_experiment_id="exp1",
        title="learning rate", statement="a smaller lr improves the metric",
        mechanism="optimisation step size",
        evidence_refs=["experiment:exp1:metric:latent_mse@1"],
        intervention={"train.lr": lr},
        predicted_effects=[MetricPrediction(metric="latent_mse@1",
                                             direction="decrease")],
        falsification_criteria=["no improvement across paired seeds"],
        confidence=0.5,
    )


def _plan(plan_id: str = "plan1", lr: float = 0.001,
          seeds: tuple[int, ...] = (42, 43)) -> ExperimentPlan:
    return ExperimentPlan(
        plan_id=plan_id, objective_id="obj1", baseline_experiment_id="exp1",
        dataset_version="ds1", primary_metric="latent_mse@1",
        paired_seeds=list(seeds),
        candidates=[CandidateExperiment(
            candidate_id="cand_H1", hypothesis_id="H1",
            config_overrides={"train.lr": lr}, seeds=list(seeds),
            dataset_version="ds1",
        )],
    )


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stores = ResearchStores(Path(self.tmp.name) / "research.sqlite")

    def _approved(self, plan_sha: str, *, by: str = "alice") -> str:
        request = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=plan_sha, requested_by="research_agent")
        granted = self.stores.approvals.grant(
            request.approval_id, approved_by=by,
            current_resource_sha256=plan_sha)
        return granted.approval_id


class HypothesisStoreTests(StoreFixture):
    def test_save_is_idempotent_and_volatile_fields_do_not_change_hash(self):
        h = _hypothesis()
        first = self.stores.hypotheses.save(h)
        self.assertEqual(first.version, 1)
        again = self.stores.hypotheses.save(h)
        self.assertEqual(again.version, 1)
        self.assertEqual(again.content_sha256, first.content_sha256)
        reviewed = h.model_copy(update={"status": "approved"})
        self.assertEqual(self.stores.hypotheses.save(reviewed).version, 1)

    def test_content_change_creates_new_immutable_version(self):
        first = self.stores.hypotheses.save(_hypothesis(lr=0.001))
        second = self.stores.hypotheses.save(_hypothesis(lr=0.002))
        self.assertEqual(second.version, 2)
        self.assertNotEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(len(self.stores.hypotheses.versions("H1")), 2)
        self.assertEqual(self.stores.hypotheses.get("H1").content_sha256,
                         second.content_sha256)
        self.assertEqual(hypothesis_sha256(_hypothesis()), first.content_sha256)

    def test_critic_review_is_bound_to_the_exact_hypothesis_hash(self):
        first = self.stores.hypotheses.save(_hypothesis())
        review = self.stores.hypotheses.save_review("H1", passed=True)
        self.assertTrue(review.passed)
        self.assertEqual(review.hypothesis_sha256, first.content_sha256)
        self.assertIsNotNone(
            self.stores.hypotheses.passed_review("H1", first.content_sha256))
        # Re-reviewing identical content never silently overwrites the verdict.
        self.assertEqual(
            self.stores.hypotheses.save_review("H1", passed=False,
                                              errors=["forged"]).review_id,
            review.review_id)
        # Editing the hypothesis INHERITS NOTHING from the old review.
        second = self.stores.hypotheses.save(_hypothesis(lr=0.003))
        self.assertIsNone(
            self.stores.hypotheses.passed_review("H1", second.content_sha256))

    def test_failed_review_is_persisted_as_failed(self):
        self.stores.hypotheses.save(_hypothesis())
        review = self.stores.hypotheses.save_review(
            "H1", passed=False, errors=["missing evidence"])
        self.assertFalse(review.passed)
        self.assertEqual(review.errors, ["missing evidence"])
        self.assertIsNone(self.stores.hypotheses.passed_review(
            "H1", self.stores.hypotheses.get("H1").content_sha256))

    def test_review_of_unknown_hypothesis_is_rejected(self):
        with self.assertRaises(ResearchStoreError):
            self.stores.hypotheses.save_review("missing", passed=True)


class ExperimentPlanStoreTests(StoreFixture):
    def test_plan_is_content_addressed(self):
        plan = _plan()
        first = self.stores.plans.save(plan)
        self.assertEqual(first.plan_sha256, plan_sha256(plan))
        self.assertEqual(self.stores.plans.save(plan).version, 1)

    def test_edit_creates_new_version_and_old_hash_stays_readable(self):
        first = self.stores.plans.save(_plan())
        second = self.stores.plans.save(_plan(lr=0.005))
        self.assertEqual(second.version, 2)
        self.assertNotEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(self.stores.plans.current_sha256("plan1"),
                         second.plan_sha256)
        old = self.stores.plans.get_version("plan1", first.plan_sha256)
        self.assertEqual(old.plan["candidates"][0]["config_overrides"]["train.lr"],
                         0.001)
        self.assertEqual(len(self.stores.plans.versions("plan1")), 2)


class ApprovalStoreTests(StoreFixture):
    def test_request_is_pending_and_grant_is_human(self):
        sha = self.stores.plans.save(_plan()).plan_sha256
        request = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=sha, requested_by="agent")
        self.assertEqual(request.status, "pending")
        self.assertEqual(request.approved_by, "")
        self.assertIsNone(request.approved_at)
        granted = self.stores.approvals.grant(
            request.approval_id, approved_by="alice",
            current_resource_sha256=sha)
        self.assertEqual(granted.status, "approved")
        self.assertEqual(granted.source, "human")
        self.assertEqual(granted.approved_by, "alice")
        self.assertTrue(granted.approved_at)

    def test_grant_requires_a_real_operator(self):
        sha = self.stores.plans.save(_plan()).plan_sha256
        request = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=sha)
        with self.assertRaises(ResearchStoreError):
            self.stores.approvals.grant(request.approval_id, approved_by="  ")

    def test_approval_decision_is_immutable(self):
        sha = self.stores.plans.save(_plan()).plan_sha256
        request = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=sha)
        self.stores.approvals.grant(request.approval_id, approved_by="alice")
        with self.assertRaises(ResearchStoreConflict):
            self.stores.approvals.grant(request.approval_id, approved_by="bob")
        self.assertTrue(any(e["event_type"] == "approval_granted"
                            for e in self.stores.approvals.events(request.approval_id)))

    def test_grant_refuses_a_plan_that_changed_while_waiting(self):
        old = self.stores.plans.save(_plan()).plan_sha256
        request = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=old)
        self.stores.plans.save(_plan(lr=0.009))
        with self.assertRaises(ResearchStoreConflict):
            self.stores.approvals.grant(
                request.approval_id, approved_by="alice",
                current_resource_sha256=self.stores.plans.current_sha256("plan1"))

    def test_reject_and_revoke_are_distinct_statuses(self):
        sha = self.stores.plans.save(_plan()).plan_sha256
        r1 = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=sha)
        self.assertEqual(self.stores.approvals.reject(
            r1.approval_id, decided_by="alice").status, "rejected")
        r2 = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=sha)
        self.stores.approvals.grant(r2.approval_id, approved_by="alice")
        self.assertEqual(self.stores.approvals.revoke(
            r2.approval_id, decided_by="alice").status, "revoked")

    def test_system_policy_source_is_recorded_distinctly(self):
        sha = self.stores.plans.save(_plan()).plan_sha256
        request = self.stores.approvals.request(
            resource_type="experiment_plan", resource_id="plan1",
            resource_sha256=sha)
        granted = self.stores.approvals.grant(
            request.approval_id, approved_by="policy:auto",
            source="system_policy")
        self.assertEqual(granted.source, "system_policy")
        self.assertNotEqual(granted.source, "human")

    def test_unknown_approval_returns_none(self):
        self.assertIsNone(self.stores.approvals.get("approval_forged"))


class SessionStoreTests(StoreFixture):
    def test_session_links_and_append_only_events(self):
        session = self.stores.sessions.create_session(
            ResearchObjective(objective_id="obj1", description="improve rollout"))
        session.link("hypothesis_ids", ["H1", "H1"])
        session.link("plan_ids", ["plan1"])
        self.stores.sessions.save(session, event_type="hypotheses_proposed",
                                  payload={"count": 1})
        loaded = self.stores.sessions.get_session(session.session_id)
        self.assertEqual(loaded.hypothesis_ids, ["H1"])
        self.assertEqual(loaded.plan_ids, ["plan1"])
        events = self.stores.sessions.events(session.session_id)
        self.assertEqual([e["event_type"] for e in events],
                         ["session_created", "hypotheses_proposed"])
        self.assertEqual(self.stores.sessions.session_for_objective(
            "obj1").session_id, session.session_id)

    def test_campaign_binding_is_idempotent(self):
        binding = CampaignBinding(
            campaign_id="campaign_a", plan_id="plan1",
            approval_id="approval_a", plan_sha256="sha1",
            objective_id="obj1", session_id="session_a")
        first, created = self.stores.sessions.bind_campaign(binding)
        self.assertTrue(created)
        again, created2 = self.stores.sessions.bind_campaign(
            binding.model_copy(update={"campaign_id": "campaign_b"}))
        self.assertFalse(created2)
        self.assertEqual(again.campaign_id, first.campaign_id)
        self.assertEqual(len(self.stores.sessions.campaigns_for_objective("obj1")), 1)
        different, created3 = self.stores.sessions.bind_campaign(
            binding.model_copy(update={"campaign_id": "campaign_c",
                                       "plan_sha256": "sha2"}))
        self.assertTrue(created3)
        self.assertEqual(different.campaign_id, "campaign_c")
        self.assertIsNotNone(self.stores.sessions.campaign_for(
            "plan1", "approval_a", "sha1"))

    def test_concurrent_bindings_create_only_one_campaign(self):
        results: list[tuple[str, bool]] = []
        barrier = threading.Barrier(4)

        def worker(index: int) -> None:
            barrier.wait()
            binding = CampaignBinding(
                campaign_id=f"campaign_{index}", plan_id="plan1",
                approval_id="approval_a", plan_sha256="sha1",
                objective_id="obj1")
            stored, created = self.stores.sessions.bind_campaign(binding)
            results.append((stored.campaign_id, created))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(1 for _, created in results if created), 1)
        self.assertEqual(len({cid for cid, _ in results}), 1)
        self.assertEqual(len(self.stores.sessions.campaigns_for_objective("obj1")), 1)

    def test_reports_are_immutable_and_experience_is_append_only(self):
        report = ResearchReport(report_id="report_1", session_id="s1",
                                objective_id="obj1", outcome="rejected")
        saved = self.stores.sessions.save_report(report)
        self.assertTrue(saved.report_sha256)
        self.assertEqual(self.stores.sessions.save_report(report).report_sha256,
                         saved.report_sha256)
        with self.assertRaises(ResearchStoreError):
            self.stores.sessions.save_report(
                report.model_copy(update={"outcome": "accepted_for_confirmation"}))
        self.assertEqual(self.stores.sessions.get_report("report_1").outcome,
                         "rejected")
        ref1 = self.stores.sessions.record_experience(
            session_id="s1", objective_id="obj1", kind="negative_result",
            payload={"decision": "rejected"})
        ref2 = self.stores.sessions.record_experience(
            session_id="s1", objective_id="obj1", kind="negative_result",
            payload={"decision": "rejected"})
        self.assertNotEqual(ref1, ref2)
        self.assertEqual(len(self.stores.sessions.experience("s1")), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
