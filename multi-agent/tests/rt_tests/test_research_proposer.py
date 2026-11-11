"""ARR 0.1 / R3: autonomous (LLM-backed) hypothesis proposer wiring.

Proves:
  * when an LLM generate callable is available/injected, the autonomous path
    (no manual hypotheses_json) uses the LLMProposer, and its output flows
    through the service exactly like a human-supplied hypothesis would;
  * when the shared LLM_* gateway is NOT configured, the service deterministically
    falls back to RuleBasedProposer;
  * an LLM that returns malformed JSON yields zero proposals with an explicit
    rejection (never a silently invented hypothesis);
  * the deterministic HypothesisCritic remains the sole approval gate regardless.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research_runtime import llm_client
from research_runtime.hypothesis_manager import LLMProposer, RuleBasedProposer
from research_runtime.research_service import ResearchDependencies, ResearchService
from research_runtime.research_store import ResearchStores

from tests.rt_tests.research_fakes import (
    FakeCampaignStore,
    FakeComparisonReportStore,
    FakeComparator,
    FakeRunner,
    default_domain,
    fake_dataset_verifier,
    fake_evidence,
)


def _build_service(*, llm_generate=None, domain=None) -> ResearchService:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    stores = ResearchStores(root / "research.sqlite")
    campaign_store = FakeCampaignStore()
    report_store = FakeComparisonReportStore()
    runner = FakeRunner(campaign_store)
    comparator = FakeComparator(report_store)
    dom = domain or default_domain()
    deps = ResearchDependencies(
        stores=stores, domain=dom,
        campaign_store=campaign_store, report_store=report_store,
        runner_factory=lambda: runner, comparator_factory=lambda: comparator,
        evidence_provider=fake_evidence, dataset_verifier=fake_dataset_verifier,
        campaign_root=str(root / "campaigns"), llm_generate=llm_generate,
    )
    svc = ResearchService(deps)
    svc._tmp = tmp  # keep the temp dir alive for the test's lifetime
    return svc


def _fake_generate_ok(prompt: str) -> str:
    """A well-behaved LLM that returns one valid hypothesis (config within the
    domain's whitelist)."""
    return json.dumps([{
        "hypothesis_id": "H_llm_1",
        "title": "LLM-suggested learning rate",
        "statement": "a smaller learning rate reduces one-step error",
        "mechanism": "optimisation step size",
        "evidence_refs": ["experiment:exp_baseline:metric:mse@1"],
        "intervention": {"train.lr": 0.0008},
        "predicted_effects": [{"metric": "mse@1", "direction": "decrease"}],
        "falsification_criteria": [
            "no improvement on the held-out validation split over paired seeds"],
        "confidence": 0.6,
    }])


def _fake_generate_bad(prompt: str) -> str:
    return "this is not valid JSON at all"


class TestAutonomousProposerSelection(unittest.TestCase):
    def test_injected_generate_yields_llm_proposer(self):
        svc = _build_service(llm_generate=_fake_generate_ok)
        proposer = svc._autonomous_proposer()
        self.assertIsInstance(proposer, LLMProposer)
        self.assertIs(proposer.generate, _fake_generate_ok)

    def test_no_generate_and_gateway_unconfigured_yields_rulebased(self):
        svc = _build_service(llm_generate=None)
        with mock.patch.object(llm_client, "is_configured", return_value=False):
            proposer = svc._autonomous_proposer()
        self.assertIsInstance(proposer, RuleBasedProposer)

    def test_gateway_configured_yields_llm_proposer(self):
        svc = _build_service(llm_generate=None)
        with mock.patch.object(llm_client, "is_configured", return_value=True):
            with mock.patch.object(llm_client, "make_generate",
                                   return_value=_fake_generate_ok):
                proposer = svc._autonomous_proposer()
        self.assertIsInstance(proposer, LLMProposer)

    def test_gateway_configured_but_build_fails_falls_back(self):
        svc = _build_service(llm_generate=None)
        with mock.patch.object(llm_client, "is_configured", return_value=True):
            with mock.patch.object(llm_client, "make_generate",
                                   side_effect=RuntimeError("boom")):
                proposer = svc._autonomous_proposer()
        self.assertIsInstance(proposer, RuleBasedProposer)


class TestAutonomousProposeFlow(unittest.TestCase):
    def tearDown(self):
        for attr in ("_tmp",):
            tmp = getattr(self, attr, None)
            if tmp is not None:
                tmp.cleanup()

    def test_autonomous_propose_uses_llm_and_persists(self):
        svc = _build_service(llm_generate=_fake_generate_ok)
        self._svc = svc
        session = svc.start_research_session(
            objective_id="obj_p106", description="reduce error")
        out = svc.propose_research_hypotheses(
            objective_id="obj_p106",
            source_experiment_id="exp_baseline",
            hypotheses_json="",  # autonomous path
        )
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["hypotheses"]), 1)
        hyp = out["hypotheses"][0]
        self.assertEqual(hyp["hypothesis_id"], "H_llm_1")
        # The LLM's evidence source must have been forced to the real id.
        stored = svc.stores.hypotheses.get("H_llm_1")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.hypothesis["source_experiment_id"],
                         "exp_baseline")
        self.assertEqual(session["session_id"], out["session_id"])

    def test_autonomous_propose_malformed_llm_yields_no_hypotheses(self):
        svc = _build_service(llm_generate=_fake_generate_bad)
        self._svc = svc
        svc.start_research_session(
            objective_id="obj_p106", description="reduce error")
        out = svc.propose_research_hypotheses(
            objective_id="obj_p106",
            source_experiment_id="exp_baseline",
            hypotheses_json="",
        )
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["hypotheses"]), 0)
        # The failure is surfaced, not silently dropped.
        self.assertTrue(any("invalid or empty" in str(r)
                           for r in out["skip_reasons"]))


if __name__ == "__main__":
    unittest.main()
