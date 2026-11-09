"""Agent integration: zero-regression proof + tool JSON contract."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.schemas import ExperimentResult, ExperimentSpec


class _FakeManager:
    def __init__(self):
        self.submitted: list[ExperimentSpec] = []

    def submit(self, spec):
        self.submitted.append(spec)
        return {"experiment_id": spec.experiment_id, "job_id": "job1",
                "status": "queued", "engine": spec.engine,
                "stage": spec.stage, "run_id": None}

    def get_status(self, job_id):
        return "succeeded"

    def collect_result(self, job_id):
        return ExperimentResult(
            experiment_id="exp1", job_id=job_id, status="succeeded",
            stage="evaluate", run_id="run1",
            metrics={"latent_mse@1": 0.02},
        )


class _FakeOrchestrator:
    def run_baseline(self, objective, **kw):
        return {
            "experiment_id": "exp1", "success": True,
            "pipeline_success": True, "objective_success": True,
            "objective_status": "achieved",
            "success_criteria": {"violations": []},
            "evaluate": {"metrics": {"latent_mse@1": 0.02}},
            "model_version": {"version_id": "mv1"},
            "failure_reports": [], "report_path": "/tmp/report.json",
        }


def _call(tool_obj, kwargs: dict) -> str:
    """StructuredTool.invoke(dict) works across langchain-core versions."""
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(kwargs)
    func = getattr(tool_obj, "func", tool_obj)
    return func(**kwargs)


class TestZeroRegression(unittest.TestCase):
    """With the flag OFF, every role's tool list is byte-for-byte unchanged."""

    def _capture(self, enabled: bool) -> dict:
        os.environ["AGENT_RESEARCH_TOOLS"] = "1" if enabled else "0"
        import agents
        captured: dict[str, list[str]] = {}
        original = agents.create_resilient_agent

        def fake(llm, tools, system_prompt, agent_name="Agent", middleware=None):
            captured[agent_name] = [
                getattr(t, "name", getattr(t, "__name__", str(t)))
                for t in tools
            ]
            return object()

        agents.create_resilient_agent = fake
        try:
            agents.create_agents()
        finally:
            agents.create_resilient_agent = original
        return captured

    def test_flag_off_preserves_all_six_roles(self):
        off = self._capture(False)
        self.assertEqual(sorted(off), [
            "ChatAgent", "CodeAgent", "ContextEngineer",
            "CrawlerAgent", "DBAgent", "RAGAgent",
        ])
        self.assertEqual(off["ChatAgent"], [
            "read_file", "grep_files", "create_file", "str_replace",
            "send_qq_email", "get_current_time",
        ])
        self.assertEqual(off["CodeAgent"], [
            "python_repl", "create_file", "read_file", "grep_files",
            "str_replace", "shell_exec", "get_current_time",
        ])
        self.assertEqual(len(off["DBAgent"]), 7)
        self.assertEqual(len(off["CrawlerAgent"]), 5)
        self.assertEqual(len(off["RAGAgent"]), 4)
        self.assertEqual(len(off["ContextEngineer"]), 5)

    def test_flag_on_adds_tools_only_to_code_agent(self):
        from research_runtime.tools import ALL_RESEARCH_TOOLS
        off = self._capture(False)
        on = self._capture(True)
        research = {
            getattr(item, "name", getattr(item, "__name__", str(item)))
            for item in ALL_RESEARCH_TOOLS
        }
        # P0 (4 tools) + P1-06 campaign loop (11 tools) = 15, still ONLY
        # on CodeAgent. Flag OFF is asserted byte-for-byte above.
        for role in off:
            with self.subTest(role=role):
                delta = set(on[role]) - set(off[role])
                if role == "CodeAgent":
                    self.assertEqual(delta, research)
                    self.assertEqual(len(on[role]),
                                     len(off[role]) + len(ALL_RESEARCH_TOOLS))
                else:
                    self.assertEqual(delta, set())
                    self.assertEqual(on[role], off[role])


class TestToolContract(unittest.TestCase):
    def setUp(self):
        import research_runtime.tools as tmod
        self.tmod = tmod
        self.manager = _FakeManager()

    def test_train_tool_returns_job_id_json(self):
        with patch.object(self.tmod, "_manager", return_value=self.manager):
            payload = json.loads(_call(self.tmod.train_world_model, {
                "experiment_id": "exp1", "epochs": 2,
            }))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["job_id"], "job1")
        spec = self.manager.submitted[0]
        self.assertEqual(spec.stage, "train")
        self.assertEqual(spec.budget["epochs"], 2)

    def test_evaluate_tool_requires_run_id(self):
        with patch.object(self.tmod, "_manager", return_value=self.manager):
            payload = json.loads(_call(self.tmod.evaluate_world_model, {
                "experiment_id": "exp1", "run_id": "",
            }))
        self.assertFalse(payload["ok"])
        self.assertIn("run_id", payload["error"])

    def test_evaluate_tool_propagates_run_id(self):
        with patch.object(self.tmod, "_manager", return_value=self.manager):
            json.loads(_call(self.tmod.evaluate_world_model, {
                "experiment_id": "exp1", "run_id": "run_abc",
            }))
        self.assertEqual(self.manager.submitted[0].run_id, "run_abc")
        self.assertEqual(self.manager.submitted[0].stage, "evaluate")

    def test_status_tool_returns_metrics_when_done(self):
        with patch.object(self.tmod, "_manager", return_value=self.manager):
            payload = json.loads(_call(self.tmod.get_experiment_status, {
                "job_id": "job1", "include_result": True,
            }))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "succeeded")
        self.assertIn("latent_mse@1", payload["metrics"])

    def test_baseline_tool_returns_compact_summary(self):
        with patch.object(self.tmod, "_orchestrator",
                          return_value=_FakeOrchestrator()):
            payload = json.loads(_call(self.tmod.run_research_baseline, {
                "description": "smoke", "epochs": 1,
            }))
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["success"])
        self.assertTrue(payload["pipeline_success"])
        self.assertTrue(payload["objective_success"])
        self.assertEqual(payload["objective_status"], "achieved")
        self.assertIn("report_path", payload)
        # Large reports must NOT be inlined into an Agent message.
        self.assertNotIn("raw", payload)
        self.assertNotIn("train", payload)

    def test_tool_never_leaks_exception(self):
        broken = type("Broken", (), {
            "submit": lambda self, spec: (_ for _ in ()).throw(
                RuntimeError("engine down"))
        })()
        with patch.object(self.tmod, "_manager", return_value=broken):
            payload = json.loads(_call(self.tmod.train_world_model, {
                "experiment_id": "exp1",
            }))
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "RuntimeError")

    def test_enabled_flag_reads_env(self):
        os.environ["AGENT_RESEARCH_TOOLS"] = "0"
        self.assertFalse(self.tmod.research_tools_enabled())
        os.environ["AGENT_RESEARCH_TOOLS"] = "1"
        self.assertTrue(self.tmod.research_tools_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
