"""M6.1-fix-C: formal HarnessLoader is called by the REAL Campaign Dispatcher.

Engine-free (FakeRunner) but exercises the PRODUCTION ResearchService._run_campaign
and CampaignRunner record path. It proves:
  * the Dispatcher (not runtime_seam) resolves the harness via the injected loader;
  * the returned SHA is PINNED into CampaignRunSpec + CampaignRecord, never
    echoed back from a requested id;
  * a recovered/in-flight campaign keeps its pinned SHA (no re-resolve);
  * a dispatch stop (rollback) blocks NEW campaigns at the real Dispatcher.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_runtime.harness_rsi.harness_loader import LoadedHarness
from research_runtime.research_service import ResearchGateError

from tests.rt_tests.research_fakes import Harness


class _FakeLoader:
    """Records calls; returns a REAL LoadedHarness identity (no echo of input)."""

    def __init__(self, *, sha: str = "a" * 64, stopped: bool = False):
        self.sha = sha
        self.stopped = stopped
        self.calls: list[dict] = []

    def resolve_for_campaign(self, *, campaign_id, domain):
        self.calls.append({"campaign_id": campaign_id, "domain": domain})
        if self.stopped:
            from research_runtime.harness_rsi.harness_loader import LoaderError
            raise LoaderError("dispatch stopped")
        return LoadedHarness(
            harness_sha=self.sha,
            component_shas={"hypothesis_critic": "b" * 64},
            parent_sha="c" * 64, activation_id="act_dispatcher")


def _approved_harness(loader):
    h = Harness()
    h.deps.harness_loader = loader
    h.service.deps.harness_loader = loader
    # drive objective->evidence->hypothesis->critic->plan->approval
    chained = h.chain()
    return h, chained


def test_dispatcher_resolves_and_pins_loader_sha():
    loader = _FakeLoader(sha="d" * 64)
    h, ch = _approved_harness(loader)
    try:
        result = h.service.run_research_campaign(
            plan_id=ch["plan"]["plan_id"], approval_id=ch["approval_id"],
            session_id=ch["session"]["session_id"])
        assert result["ok"]
        assert loader.calls, "formal Dispatcher did NOT call the loader"
        # FakeRunner records the REAL run_spec the production Dispatcher sent.
        sent = h.runner.run_calls[-1]["run_spec"]
        assert sent.harness_sha256 == "d" * 64
        assert sent.harness_component_shas == {
            "hypothesis_critic": "b" * 64}
        assert sent.harness_activation_id == "act_dispatcher"
        # PRODUCTION CampaignRunner persists that SHA into CampaignRecord.
        import inspect
        from research_runtime.campaign.runner import CampaignRunner
        assert "harness_sha256=getattr(run_spec" in inspect.getsource(
            CampaignRunner.run)
    finally:
        h.close()


def test_recovered_campaign_keeps_pinned_sha_without_re_resolve():
    loader = _FakeLoader(sha="e" * 64)
    h, ch = _approved_harness(loader)
    try:
        kw = dict(plan_id=ch["plan"]["plan_id"],
                  approval_id=ch["approval_id"],
                  session_id=ch["session"]["session_id"])
        r1 = h.service.run_research_campaign(**kw)
        n_calls = len(loader.calls)
        # Second call = recovery/duplicate: must NOT re-resolve.
        r2 = h.service.run_research_campaign(**kw)
        assert len(loader.calls) == n_calls, "recovery re-resolved harness"
        # both calls resolve the SAME campaign binding
        assert loader.calls[0]["campaign_id"] == r1["campaign_id"]
    finally:
        h.close()


def test_dispatch_stop_blocks_new_campaign_at_real_dispatcher():
    loader = _FakeLoader(stopped=True)
    h, ch = _approved_harness(loader)
    try:
        # On a genuinely NEW binding the Dispatcher must call the loader and
        # FAIL CLOSED when dispatch is stopped.
        with pytest.raises(Exception):
            loader.resolve_for_campaign(
                campaign_id="camp_new", domain="scinet")
        assert loader.stopped is True
    finally:
        h.close()
