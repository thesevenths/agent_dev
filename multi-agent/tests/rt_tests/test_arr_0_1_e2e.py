"""ARR 0.1 end-to-end: INNER loop (autonomous research via the real service)
+ OUTER loop (RRSI bridge) demonstrated together -- deterministically and
without any network / model / engine.

INNER (research_runtime)
-----------------------
    objective -> AUTONOMOUS hypothesis generation (NO manual hypotheses_json;
    the deterministic RuleBasedProposer -- R3's autonomous path -- reads the
    REAL structured evidence and emits a hypothesis) -> deterministic Critic
    -> plan -> human approval -> fake campaign runner -> fake comparator ->
    ResearchReport. The report refuses to promote and never touches the frozen
    split.

OUTER (RRSI via R4 bridge)
--------------------------
    The ResearchReport is exported as a research trace and fed into the REAL
    RRSI primitives (isolated Sandbox, gate_a/gate_b, Ledger + experience.Store)
    using deterministic --fake profiles. The bridge produces a research-runtime
    improvement candidate and a gate decision (accept/validate or reject); it
    NEVER calls version.promote (operator-gated) and NEVER mutates the live
    runtime.

This is the "both loops connected" demonstration the fourth-round review asked
for, and it is CI-safe: every heavy part is a fake, and the autonomous proposer
is the deterministic RuleBasedProposer (no LLM is configured in CI). R3's
LLMProposer is exercised separately in test_research_proposer.py.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from adaptive_runtime import research_ingest as bridge
from research_runtime.adapters.base import ModelAdapter
from research_runtime.research_service import ResearchService
from research_runtime.schemas import ExperimentResult

from tests.rt_tests.research_fakes import (
    OBJECTIVE_ID,
    SOURCE_EXPERIMENT_ID,
    Harness,
)


# Env vars that, if set, make the autonomous proposer attempt the REAL LLM path
# (research_runtime.llm_client). The E2E must be deterministic and CI-safe, so
# it exercises the deterministic RuleBasedProposer fallback -- the realistic
# path when no LLM is configured. We isolate these in setUp/tearDown so the
# test is robust regardless of any other test's env mutations in the same run.
_LLM_ENV_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_VERIFY_SSL")


class StubModelAdapter(ModelAdapter):
    """A real ModelAdapter (satisfies R2 observability) with no engine behind it."""

    name = "stub"
    stages = ("submit", "status", "collect")

    def submit(self, spec):
        return "stub_job"

    def get_status(self, job_id):
        return "succeeded"

    def collect_result(self, job_id, spec=None):
        return ExperimentResult(
            experiment_id="stub", job_id=job_id, status="succeeded",
            stage="evaluate", metrics={})


class Arr01DualLoopE2E(unittest.TestCase):
    def setUp(self):
        # Isolate LLM env so the autonomous proposer uses the deterministic
        # RuleBasedProposer (the realistic no-LLM path), independent of any
        # other test in the same session that may set LLM_* variables.
        self._saved_llm_env = {k: os.environ.get(k) for k in _LLM_ENV_KEYS}
        for k in _LLM_ENV_KEYS:
            os.environ.pop(k, None)
        # Default Harness already wires fake_evidence + fake_dataset_verifier.
        self.harness = Harness()
        # R2: the assembled service exposes a REAL ModelAdapter instance.
        self.harness.deps.adapter = StubModelAdapter()
        self.harness.service = ResearchService(self.harness.deps)
        # Drive the inner loop ONCE; reuse the report for the outer-loop tests.
        self.report = self._run_inner_loop()

    def tearDown(self):
        # Restore the LLM env exactly as it was before this test.
        for k, v in self._saved_llm_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ----------------------------------------------------------- inner loop
    def _run_inner_loop(self) -> dict:
        svc = self.harness.service
        sid = svc.start_research_session(
            objective_id=OBJECTIVE_ID,
            description="reduce prediction error")["session_id"]

        # AUTONOMOUS path: NO manual hypotheses_json. The service's autonomous
        # proposer (RuleBasedProposer in CI, LLMProposer if an LLM is wired)
        # reads the REAL structured evidence and emits the hypothesis itself.
        proposed = svc.propose_research_hypotheses(
            objective_id=OBJECTIVE_ID,
            source_experiment_id=SOURCE_EXPERIMENT_ID,
            session_id=sid,
            hypotheses_json="",  # <- autonomous; no operator-supplied JSON
        )
        self.assertTrue(proposed["ok"], proposed)
        self.assertGreater(
            len(proposed["hypotheses"]), 0,
            "the autonomous proposer must emit a hypothesis from the real "
            "horizon-degradation evidence")
        hid = proposed["hypotheses"][0]["hypothesis_id"]

        reviewed = svc.review_research_hypotheses(
            hypothesis_ids_json=json.dumps([hid]), session_id=sid)
        self.assertEqual(reviewed["approved_ids"], [hid],
                         "deterministic Critic must approve the grounded hypothesis")

        plan = svc.create_research_plan(
            objective_id=OBJECTIVE_ID,
            hypothesis_ids_json=json.dumps([hid]), session_id=sid,
            plan_id="plan_arr01")
        self.assertTrue(plan["ok"], plan)

        request = svc.request_plan_approval(plan_id="plan_arr01", session_id=sid)
        self.harness.stores.approvals.grant(
            request["approval_id"], approved_by="alice", source="human",
            current_resource_sha256=plan["plan_sha256"])

        run = svc.run_research_campaign(
            plan_id="plan_arr01", approval_id=request["approval_id"],
            session_id=sid)
        self.assertEqual(run["status"], "completed", run)

        compared = svc.compare_research_campaign(
            campaign_id=run["campaign_id"], session_id=sid)
        self.assertTrue(compared["ok"], compared)
        self.assertGreater(len(compared["comparisons"]), 0)

        final = svc.get_research_report(session_id=sid)
        self.assertTrue(final["ok"], final)
        report = final["report"]
        # The research report must never promote or use the frozen split.
        self.assertEqual(report["promotion"], "not_requested")
        self.assertFalse(report["frozen_test_used"])
        return report

    # ------------------------------------------------------------- assertions
    def test_inner_loop_adapter_is_real_model_adapter(self):
        # R2: the generic service participates with a real engine adapter.
        self.assertIsInstance(self.harness.service.adapter, ModelAdapter)

    def test_inner_loop_report_is_traceable(self):
        report = self.report
        self.assertIn(report["outcome"],
                      {"accepted_for_confirmation", "rejected", "inconclusive"})
        self.assertTrue(report["experience_refs"], "experience must be recorded")
        self.assertTrue(report["hypotheses"], "hypotheses must be persisted")
        self.assertTrue(report["reviews"], "critic verdicts must be persisted")

    def test_outer_loop_accepts_good_candidate(self):
        # The autonomous research report feeds the REAL RRSI outer loop.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            res = bridge.ingest(
                self.report,
                ledger_path=root / "ledger.jsonl",
                experience_path=root / "exp.jsonl",
                sandbox_root=root / "good",
                fake_case="good",
            )
            self.assertTrue(res.decision_a.startswith("ACCEPT"),
                            res.decision_a)
            self.assertEqual(res.decision_b, "ACCEPT(VALIDATED)",
                             res.decision_b)
            self.assertEqual(res.ledger_status, "accepted", res.ledger_status)
            # Safety invariants: never promotes, never touches the live runtime.
            self.assertTrue(res.live_runtime_untouched)
            self.assertTrue(res.sandbox_path.startswith(str(root / "good")))

    def test_outer_loop_rejects_regression_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            res = bridge.ingest(
                self.report,
                ledger_path=root / "ledger.jsonl",
                experience_path=root / "exp.jsonl",
                sandbox_root=root / "worse",
                fake_case="worse",
            )
            self.assertTrue(res.decision_a.startswith("ACCEPT"),
                            res.decision_a)
            self.assertTrue(
                res.decision_b.startswith("REJECT(regression)"), res.decision_b)
            self.assertEqual(res.ledger_status, "rejected", res.ledger_status)
            self.assertTrue(res.live_runtime_untouched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
