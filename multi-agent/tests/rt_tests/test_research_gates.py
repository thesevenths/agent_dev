"""P1-06 A6: every safety boundary in the task brief, plus the full workflow.

The service under test is the REAL production ``ResearchService``; only the
engine / stores paths / comparator are injected fakes. No gate logic is stubbed.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from research_runtime.research_service import make_campaign_id
from research_runtime.schemas import ExperimentPlan, Hypothesis
from tests.rt_tests.research_fakes import (
    DATASET_VERSION,
    HYPOTHESIS_ID,
    OBJECTIVE_ID,
    PLAN_ID,
    SOURCE_EXPERIMENT_ID,
    Harness,
    operator_hypothesis,
)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.close)

    def _assert_rejected(self, payload: dict, code: str) -> None:
        self.assertFalse(payload.get("ok"), payload)
        self.assertEqual(payload.get("error_code"), code, payload)
        self.assertTrue(payload.get("error"))

    # ------------------------------------------------------- approval gates
    def test_unapproved_plan_is_rejected(self):
        ctx = self.h.chain(grant=False)
        self.assertEqual(ctx["request"]["status"], "pending")
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "approval_not_approved")
        self.assertEqual(self.h.runner.run_calls, [])

    def test_forged_approval_id_is_rejected(self):
        ctx = self.h.chain()
        self._assert_rejected(self.h.run("approval_forged_by_llm"),
                              "approval_not_found")
        self.assertEqual(self.h.runner.run_calls, [])

    def test_plan_modified_after_approval_invalidates_the_hash(self):
        ctx = self.h.chain()
        stored = self.h.stores.plans.get(PLAN_ID)
        old_sha = stored.plan_sha256
        edited = ExperimentPlan.model_validate(stored.plan).model_copy(
            update={"guardrails": {"mse@1": {"max": 999.0}}})
        new = self.h.stores.plans.save(edited)
        self.assertEqual(new.version, 2)
        self.assertNotEqual(new.plan_sha256, old_sha)
        self.assertEqual(self.h.stores.plans.current_sha256(PLAN_ID),
                         new.plan_sha256)
        # The old immutable version is still auditable.
        self.assertEqual(self.h.stores.plans.get_version(
            PLAN_ID, old_sha).version, 1)
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "plan_hash_mismatch")
        self.assertEqual(self.h.runner.run_calls, [])

    def test_system_policy_approval_is_not_trusted_for_execution(self):
        ctx = self.h.chain(source="system_policy")
        self.assertEqual(self.h.stores.approvals.get(ctx["approval_id"]).source,
                         "system_policy")
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "approval_source_rejected")

    def test_approval_scope_must_include_run(self):
        ctx = self.h.chain()
        limited = self.h.stores.approvals.request(
            resource_type="experiment_plan", resource_id=PLAN_ID,
            resource_sha256=ctx["plan"]["plan_sha256"],
            scope=["resume_campaign"])
        self.h.stores.approvals.grant(limited.approval_id, approved_by="alice")
        self._assert_rejected(self.h.run(limited.approval_id),
                              "approval_scope_missing")

    def test_revoked_approval_cannot_be_replayed(self):
        ctx = self.h.chain()
        self.h.stores.approvals.revoke(ctx["approval_id"], decided_by="alice")
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "approval_not_approved")

    # ----------------------------------------------------------- Critic gates
    def test_hypothesis_that_fails_critic_cannot_become_a_plan(self):
        ctx = self.h.chain(hypothesis=operator_hypothesis(
            evidence_refs=["experiment:unrelated:metric:mse@1"]))
        review = ctx["reviewed"]["reviews"][0]
        self.assertFalse(review["passed"])
        self.assertTrue(any("evidence" in e for e in review["errors"]))
        self.assertEqual(ctx["plan"]["error_code"], "critic_review_missing")
        self.assertIsNone(self.h.stores.hypotheses.passed_review(
            HYPOTHESIS_ID, review["hypothesis_sha256"]))

    def test_hypothesis_edited_after_review_loses_its_verdict(self):
        ctx = self.h.chain(grant=False)
        stored = self.h.stores.hypotheses.get(HYPOTHESIS_ID)
        edited = Hypothesis.model_validate(stored.hypothesis).model_copy(
            update={"intervention": {"train.lr": 0.009}})
        self.assertEqual(self.h.stores.hypotheses.save(edited).version, 2)
        self.h.stores.approvals.grant(
            ctx["approval_id"], approved_by="alice",
            current_resource_sha256=ctx["plan"]["plan_sha256"])
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "critic_review_missing")

    def test_candidate_overrides_must_match_the_reviewed_intervention(self):
        ctx = self.h.chain()
        stored = self.h.stores.plans.get(PLAN_ID)
        payload = dict(stored.plan)
        payload["candidates"] = [
            {**payload["candidates"][0],
             "config_overrides": {"train.lr": 0.5}}]
        self.h.stores.plans.save(payload)
        # The plan edit alone already invalidates the approval hash; the
        # integrity gate must fire BEFORE any weaker check.
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "plan_hash_mismatch")

    # --------------------------------------------------------- config gates
    def test_config_outside_whitelist_is_rejected(self):
        ctx = self.h.chain(hypothesis=operator_hypothesis(
            intervention={"data.frozen_dir": "datasets/hacked"}))
        self.assertTrue(ctx["plan"]["ok"], ctx["plan"])
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "config_key_not_allowed")
        self.assertIn("protected root", self.h.run(
            ctx["approval_id"])["error"])

    def test_unknown_config_key_is_rejected(self):
        ctx = self.h.chain(hypothesis=operator_hypothesis(
            intervention={"model.unknown_field": 1}))
        payload = self.h.run(ctx["approval_id"])
        self._assert_rejected(payload, "config_key_not_allowed")
        self.assertIn("whitelist", payload["error"])

    # --------------------------------------------------------- data / budget
    def test_undeclared_dataset_version_is_rejected(self):
        ctx = self.h.chain()
        self.h.service.deps.dataset_verifier = (
            lambda base, version: {"verified": True})
        self.h.replace_domain(allowed_dataset_versions=("some_other_dataset",))
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "dataset_version_not_allowed")

    def test_missing_or_tampered_frozen_dataset_is_rejected(self):
        ctx = self.h.chain()
        self.h.service.deps.dataset_verifier = (
            lambda base, version: {"verified": False, "reason": "hash mismatch"})
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "dataset_not_verified")

    def test_dataset_verification_exception_is_fail_closed(self):
        ctx = self.h.chain()

        def broken(base, version):
            raise FileNotFoundError("manifest.json missing")
        self.h.service.deps.dataset_verifier = broken
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "dataset_not_verified")

    def test_budget_over_limit_is_rejected(self):
        ctx = self.h.chain()
        self.h.replace_domain(max_jobs=2)
        payload = self.h.run(ctx["approval_id"])
        self._assert_rejected(payload, "budget_exceeded")
        self.assertIn("needs 8 jobs", payload["error"])
        self.assertEqual(self.h.runner.run_calls, [])

    def test_frozen_final_split_is_never_used_for_selection(self):
        ctx = self.h.chain()
        self.h.replace_domain(evaluation_split="test")
        self._assert_rejected(self.h.run(ctx["approval_id"]),
                              "final_split_forbidden")

    # ------------------------------------------------------ idempotency/loop
    def test_duplicate_submission_never_relaunches_jobs(self):
        ctx = self.h.chain()
        first = self.h.run(ctx["approval_id"])
        second = self.h.run(ctx["approval_id"])
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(second["already_submitted"])
        self.assertEqual(second["campaign_id"], first["campaign_id"])
        self.assertEqual(len(self.h.runner.run_calls), 1)
        binding = self.h.stores.sessions.campaign_for(
            PLAN_ID, ctx["approval_id"], ctx["plan"]["plan_sha256"])
        self.assertEqual(binding.campaign_id, make_campaign_id(
            PLAN_ID, ctx["approval_id"], ctx["plan"]["plan_sha256"]))

    def test_round_limit_caps_campaigns_per_objective(self):
        ctx1 = self.h.chain(plan_id="plan_round1")
        first = self.h.run(ctx1["approval_id"], plan_id="plan_round1")
        ctx2 = self.h.chain(plan_id="plan_round2")
        second = self.h.run(ctx2["approval_id"], plan_id="plan_round2")
        self.assertTrue(first["ok"] and second["ok"])
        self.assertEqual(len(self.h.stores.sessions.campaigns_for_objective(
            OBJECTIVE_ID)), 2)
        ctx3 = self.h.chain(plan_id="plan_round3")
        self._assert_rejected(self.h.run(ctx3["approval_id"],
                                        plan_id="plan_round3"),
                              "round_limit_reached")
        self.assertEqual(len(self.h.runner.run_calls), 2)

    # ------------------------------------------------------------- resume
    def test_running_campaign_cannot_be_resumed(self):
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        record = self.h.campaign_store.load_campaign(run["campaign_id"])
        record.status = "running"
        self.h.campaign_store.put(record)
        payload = self.h.service.resume_research_campaign(
            campaign_id=run["campaign_id"])
        self._assert_rejected(payload, "campaign_not_resumable")
        self.assertEqual(self.h.runner.resume_calls, [])

    def test_terminal_campaign_cannot_be_resumed(self):
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        payload = self.h.service.resume_research_campaign(
            campaign_id=run["campaign_id"])
        self._assert_rejected(payload, "campaign_terminal")

    def test_unbound_campaign_cannot_be_resumed(self):
        from research_runtime.campaign.schemas import CampaignRecord
        self.h.campaign_store.put(CampaignRecord(
            campaign_id="campaign_orphan", plan_id=PLAN_ID,
            objective_id=OBJECTIVE_ID, status="paused",
            dataset_version=DATASET_VERSION))
        self._assert_rejected(self.h.service.resume_research_campaign(
            campaign_id="campaign_orphan"), "campaign_unbound")

    def test_paused_campaign_resumes_only_with_the_bound_approval(self):
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        record = self.h.campaign_store.load_campaign(run["campaign_id"])
        record.status = "paused"
        self.h.campaign_store.put(record)
        self._assert_rejected(self.h.service.resume_research_campaign(
            campaign_id=run["campaign_id"], approval_id="approval_forged"),
            "approval_mismatch")
        resumed = self.h.service.resume_research_campaign(
            campaign_id=run["campaign_id"])
        self.assertTrue(resumed["ok"], resumed)
        self.assertEqual(self.h.runner.resume_calls, [run["campaign_id"]])

    # --------------------------------------------------------- comparison
    def test_missing_comparison_report_is_an_explicit_error(self):
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        self._assert_rejected(
            self.h.service.get_comparison_report(
                campaign_id=run["campaign_id"]),
            "comparison_report_not_found")
        self._assert_rejected(
            self.h.service.get_comparison_report(
                campaign_id=run["campaign_id"], comparison_id="cmp_forged"),
            "comparison_report_not_found")

    def test_comparator_returning_no_report_never_invents_a_result(self):
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        self.h.comparator.reports = []
        self._assert_rejected(self.h.service.compare_research_campaign(
            campaign_id=run["campaign_id"]), "comparison_unavailable")

    def test_mixed_candidate_decisions_are_conservatively_inconclusive(self):
        from tests.rt_tests.research_fakes import FakeComparisonReport
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        self.h.comparator.reports = [
            FakeComparisonReport(comparison_id="cmp_a",
                                 decision="accepted_for_confirmation"),
            FakeComparisonReport(comparison_id="cmp_b",
                                 decision="inconclusive"),
        ]
        compared = self.h.service.compare_research_campaign(
            campaign_id=run["campaign_id"])
        self.assertTrue(compared["ok"], compared)
        self.assertEqual(compared["research_outcome"], "inconclusive")

    def test_incomplete_campaign_cannot_be_compared(self):
        ctx = self.h.chain()
        run = self.h.run(ctx["approval_id"])
        record = self.h.campaign_store.load_campaign(run["campaign_id"])
        record.status = "running"
        self.h.campaign_store.put(record)
        self._assert_rejected(self.h.service.compare_research_campaign(
            campaign_id=run["campaign_id"]), "campaign_not_terminal")

    # ------------------------------------------------------- full workflow
    def test_full_workflow_objective_to_research_report(self):
        ctx = self.h.chain()
        proposed = ctx["proposed"]
        self.assertEqual(proposed["source_experiment_id"],
                         SOURCE_EXPERIMENT_ID)
        self.assertTrue(proposed["hypotheses"][0]["content_sha256"])
        self.assertTrue(ctx["reviewed"]["approved_ids"] == [HYPOTHESIS_ID])

        run = self.h.run(ctx["approval_id"])
        self.assertTrue(run["ok"], run)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(len(self.h.runner.run_calls), 1)
        spec = self.h.runner.run_calls[0]["run_spec"]
        self.assertEqual(spec.evaluation_split, "validation")
        self.assertEqual(spec.dataset_version, DATASET_VERSION)
        self.assertEqual(spec.approved_hypothesis_ids, [HYPOTHESIS_ID])
        self.assertEqual(spec.max_parallel, 1)

        status = self.h.service.get_campaign_status(
            campaign_id=run["campaign_id"])
        self.assertTrue(status["ok"])
        self.assertEqual(status["plan_id"], PLAN_ID)
        self.assertEqual(status["approval_id"], ctx["approval_id"])

        compared = self.h.service.compare_research_campaign(
            campaign_id=run["campaign_id"])
        self.assertTrue(compared["ok"], compared)
        self.assertEqual(compared["research_outcome"], "rejected")

        report = self.h.service.get_comparison_report(
            campaign_id=run["campaign_id"], comparison_id="cmp_1")
        self.assertTrue(report["ok"])
        self.assertEqual(report["report"]["decision"], "rejected")

        final = self.h.service.get_research_report(
            session_id=ctx["session"]["session_id"])
        self.assertTrue(final["ok"], final)
        payload = final["report"]
        self.assertEqual(payload["outcome"], "rejected")
        self.assertEqual(payload["promotion"], "not_requested")
        self.assertFalse(payload["frozen_test_used"])
        self.assertEqual(payload["campaign_count"], 1)
        self.assertEqual(payload["comparison_count"], 1)
        self.assertTrue(payload["reviews"][0]["passed"])
        self.assertEqual(payload["approvals"][0]["source"], "human")
        self.assertEqual(payload["plans"][0]["plan_sha256"],
                         ctx["plan"]["plan_sha256"])
        self.assertEqual(payload["decision_rules"],
                         ["R3_guardrail_hard_violation"])
        self.assertTrue(payload["experience_refs"])
        self.assertTrue(payload["report_sha256"])
        self.assertTrue(Path(final["report_path"]).is_file())
        on_disk = json.loads(Path(final["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["report_sha256"], payload["report_sha256"])
        # Negative evidence is retained as structured experience.
        experience = self.h.stores.sessions.experience(
            ctx["session"]["session_id"])
        self.assertEqual(experience[0]["kind"], "research_outcome")
        self.assertEqual(experience[0]["payload"]["outcome"], "rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
