"""A6: JSON tool contract, feature-flag zero regression, HITL and console gate."""
from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import research_runtime.tools as tmod
from research_runtime.research_store import approval_cli
from tests.rt_tests.research_fakes import (
    HYPOTHESIS_ID, OBJECTIVE_ID, PLAN_ID, SOURCE_EXPERIMENT_ID,
    Harness, operator_hypothesis,
)


def _call(tool_obj, kwargs: dict) -> str:
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(kwargs)
    func = getattr(tool_obj, "func", tool_obj)
    return func(**kwargs)


def _payload(tool_obj, kwargs: dict) -> dict:
    raw = _call(tool_obj, kwargs)
    assert isinstance(raw, str), type(raw)
    return json.loads(raw)


TOOL_NAMES = [getattr(x, "name", getattr(x, "__name__", str(x)))
              for x in tmod.ALL_RESEARCH_TOOLS]


class ToolSurfaceTests(unittest.TestCase):
    def test_no_tool_can_grant_a_human_approval(self):
        self.assertFalse(any("approve" in n or "grant" in n for n in TOOL_NAMES),
                         TOOL_NAMES)
        self.assertIn("request_plan_approval", TOOL_NAMES)
        self.assertEqual(len(tmod.RESEARCH_TOOLS), 4)
        self.assertEqual(len(tmod.CAMPAIGN_RESEARCH_TOOLS), 11)
        self.assertEqual(len(tmod.ALL_RESEARCH_TOOLS), 15)

    def test_run_tool_only_accepts_identifiers(self):
        args = set(getattr(tmod.run_research_campaign, "args", {}) or {})
        self.assertEqual(args, {"plan_id", "approval_id", "session_id"})
        self.assertNotIn("config_json", args)
        self.assertNotIn("approved_hypothesis_ids", args)
        self.assertNotIn("config_overrides", args)

    def test_flag_off_keeps_all_six_role_tool_lists_unchanged(self):
        import agents
        os.environ["AGENT_RESEARCH_TOOLS"] = "0"
        captured: dict[str, list[str]] = {}
        original = agents.create_resilient_agent

        def fake(llm, tools, system_prompt, agent_name="Agent", middleware=None):
            captured[agent_name] = [
                getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
            return object()

        agents.create_resilient_agent = fake
        try:
            agents.create_agents()
        finally:
            agents.create_resilient_agent = original
        self.assertEqual(sorted(captured), [
            "ChatAgent", "CodeAgent", "ContextEngineer",
            "CrawlerAgent", "DBAgent", "RAGAgent"])
        self.assertEqual(captured["CodeAgent"], [
            "python_repl", "create_file", "read_file", "grep_files",
            "str_replace", "shell_exec", "get_current_time"])
        for role, names in captured.items():
            self.assertEqual(set(names) & set(TOOL_NAMES), set(), role)

    def test_flag_on_registers_research_tools_only_on_code_agent(self):
        import agents
        os.environ["AGENT_RESEARCH_TOOLS"] = "1"
        captured: dict[str, list[str]] = {}
        original = agents.create_resilient_agent

        def fake(llm, tools, system_prompt, agent_name="Agent", middleware=None):
            captured[agent_name] = [
                getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
            return object()

        agents.create_resilient_agent = fake
        try:
            agents.create_agents()
        finally:
            agents.create_resilient_agent = original
            os.environ["AGENT_RESEARCH_TOOLS"] = "0"
        self.assertEqual(set(captured["CodeAgent"]) & set(TOOL_NAMES),
                         set(TOOL_NAMES))
        for role, names in captured.items():
            if role != "CodeAgent":
                self.assertEqual(set(names) & set(TOOL_NAMES), set(), role)

    def test_high_risk_tools_are_registered_as_danger_for_hitl(self):
        import hitl
        self.assertEqual(hitl.risk_of("run_research_campaign"), "danger")
        self.assertEqual(hitl.risk_of("resume_research_campaign"), "danger")
        self.assertEqual(hitl.risk_of("request_plan_approval"), "warn")
        self.assertEqual(hitl.risk_of("get_campaign_status"), "safe")
        self.assertIn("run_research_campaign", hitl.DANGER_TOOLS)
        gate = hitl.build_interrupt_on(TOOL_NAMES)
        self.assertIn("run_research_campaign", gate)
        self.assertIn("resume_research_campaign", gate)
        self.assertNotIn("compare_research_campaign", gate)


class ToolWorkflowTests(unittest.TestCase):
    """The ACTUAL @tool objects, backed by the real service + injected fakes."""

    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.close)
        self.patcher = patch.object(tmod, "_service",
                                   return_value=self.h.service)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_complete_chain_through_tools(self):
        start = _payload(tmod.start_research_session, {
            "objective_id": OBJECTIVE_ID, "description": "reduce rollout error"})
        self.assertTrue(start["ok"])
        sid = start["session_id"]
        proposed = _payload(tmod.propose_research_hypotheses, {
            "objective_id": OBJECTIVE_ID,
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "session_id": sid,
            "hypotheses_json": json.dumps([operator_hypothesis()])})
        self.assertTrue(proposed["ok"], proposed)
        reviewed = _payload(tmod.review_research_hypotheses, {
            "hypothesis_ids_json": json.dumps([HYPOTHESIS_ID]),
            "session_id": sid})
        self.assertEqual(reviewed["approved_ids"], [HYPOTHESIS_ID])
        plan = _payload(tmod.create_research_plan, {
            "objective_id": OBJECTIVE_ID,
            "hypothesis_ids_json": json.dumps([HYPOTHESIS_ID]),
            "session_id": sid, "plan_id": PLAN_ID})
        self.assertTrue(plan["ok"], plan)
        request = _payload(tmod.request_plan_approval, {
            "plan_id": PLAN_ID, "session_id": sid})
        self.assertEqual(request["status"], "pending")
        self.assertIn("approval_cli", request["human_action"])

        # Before a human grants: the danger tool must refuse, as JSON.
        denied = _payload(tmod.run_research_campaign, {
            "plan_id": PLAN_ID, "approval_id": request["approval_id"]})
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["error_code"], "approval_not_approved")

        # Human console (out-of-band), then execute.
        with contextlib.redirect_stdout(io.StringIO()):
            rc = approval_cli.main([
                "--db", str(self.h.root / "research.sqlite"),
                "grant", request["approval_id"],
                "--approved-by", "alice", "--yes"])
        self.assertEqual(rc, 0)
        run = _payload(tmod.run_research_campaign, {
            "plan_id": PLAN_ID, "approval_id": request["approval_id"],
            "session_id": sid})
        self.assertTrue(run["ok"], run)
        self.assertEqual(run["status"], "completed")

        status = _payload(tmod.get_campaign_status, {
            "campaign_id": run["campaign_id"]})
        self.assertEqual(status["status"], "completed")
        compared = _payload(tmod.compare_research_campaign, {
            "campaign_id": run["campaign_id"], "session_id": sid})
        self.assertTrue(compared["ok"], compared)
        reports = _payload(tmod.get_comparison_report, {
            "campaign_id": run["campaign_id"], "comparison_id": "cmp_1"})
        self.assertTrue(reports["ok"])
        final = _payload(tmod.get_research_report, {"session_id": sid})
        self.assertTrue(final["ok"])
        self.assertEqual(final["report"]["promotion"], "not_requested")
        self.assertFalse(final["report"]["frozen_test_used"])

    def test_all_tools_return_structured_errors_never_tracebacks(self):
        broken = type("Broken", (), {
            "run_research_campaign": lambda self, **kw: (_ for _ in ()).throw(
                RuntimeError("engine down")),
        })()
        with patch.object(tmod, "_service", return_value=broken):
            payload = _payload(tmod.run_research_campaign, {
                "plan_id": PLAN_ID, "approval_id": "approval_x"})
        self.assertFalse(payload["ok"])
        self.assertIn("engine down", payload["error"])
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertNotIn("Traceback", json.dumps(payload))

    def test_invalid_json_is_a_structured_error(self):
        payload = _payload(tmod.review_research_hypotheses, {
            "hypothesis_ids_json": "not-json"})
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "invalid_json")

    def test_missing_report_returns_explicit_tool_error(self):
        payload = _payload(tmod.get_comparison_report, {
            "campaign_id": "campaign_never_ran"})
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_code"], "comparison_report_not_found")


class GenericnessTests(unittest.TestCase):
    """Cross-project rule: generic research layers carry NO domain vocabulary.

    Engine names belong ONLY to ``domains/``, ``campaign/scinet.py`` and
    ``comparison/scinet.py`` (domain wiring) and to ``research_environment.py``
    (the composition root). A robotics or finance domain must be able to reuse
    every other module unchanged.
    """

    def test_generic_layers_contain_no_engine_specific_terms(self):
        root = Path(__file__).resolve().parents[2] / "research_runtime"
        files = sorted((root / "research_store").glob("*.py")) + [
            root / "research_policy.py",
            root / "research_service.py",
        ]
        forbidden = ("pendulum", "pusher", "mujoco", "latent_mse", "cosine@1",
                     "predictor_hidden", "ema_momentum", "pred_weight",
                     "scinet++", "jepa", "pytorch", "qpos", "qvel")
        self.assertGreaterEqual(len(files), 8)
        for path in files:
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text,
                                 f"{path.name} hard-codes {token!r}")

    def test_domain_wiring_is_isolated(self):
        root = Path(__file__).resolve().parents[2] / "research_runtime"
        self.assertTrue((root / "domains" / "scinet.py").is_file())
        self.assertTrue((root / "campaign" / "scinet.py").is_file())
        self.assertTrue((root / "comparison" / "scinet.py").is_file())


class ApprovalConsoleTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.close)
        self.ctx = self.h.chain(grant=False)
        self.db = str(self.h.root / "research.sqlite")

    def test_console_grants_with_a_real_operator(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = approval_cli.main(["--db", self.db, "grant",
                                     self.ctx["approval_id"],
                                     "--approved-by", "alice", "--yes"])
        self.assertEqual(rc, 0)
        record = self.h.stores.approvals.get(self.ctx["approval_id"])
        self.assertEqual(record.status, "approved")
        self.assertEqual(record.source, "human")
        self.assertTrue(self.h.run(self.ctx["approval_id"])["ok"])

    def test_console_refuses_to_grant_a_plan_edited_while_waiting(self):
        from research_runtime.schemas import ExperimentPlan
        stored = self.h.stores.plans.get(PLAN_ID)
        self.h.stores.plans.save(ExperimentPlan.model_validate(
            stored.plan).model_copy(update={"guardrails": {"mse@1": {"max": 9.0}}}))
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = approval_cli.main(["--db", self.db, "grant",
                                     self.ctx["approval_id"],
                                     "--approved-by", "alice", "--yes"])
        self.assertEqual(rc, 1)
        self.assertEqual(self.h.stores.approvals.get(
            self.ctx["approval_id"]).status, "pending")
        self.assertIn("changed", err.getvalue())

    def test_console_cannot_grant_twice(self):
        with contextlib.redirect_stdout(io.StringIO()):
            approval_cli.main(["--db", self.db, "grant",
                               self.ctx["approval_id"],
                               "--approved-by", "alice", "--yes"])
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = approval_cli.main(["--db", self.db, "grant",
                                     self.ctx["approval_id"],
                                     "--approved-by", "bob", "--yes"])
        self.assertEqual(rc, 1)
        self.assertEqual(self.h.stores.approvals.get(
            self.ctx["approval_id"]).approved_by, "alice")


if __name__ == "__main__":
    unittest.main(verbosity=2)
