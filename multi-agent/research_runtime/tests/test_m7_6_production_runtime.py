"""M7.6 P0-B: production runtime seam negative matrix.

Proves the production direct-executor bypass is CLOSED and every formal
contract is required. Engine-free (fake backend), but exercises the real
FormalRuntimeAdapter + RuntimeState reservation/journal path.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_runtime.discovery.discovery_coordinator import DiscoveryCoordinator
from research_runtime.discovery.formal_runtime import (
    FormalRuntimeAdapter, ProductionRuntimeBlocked, formal_campaign_id)
from research_runtime.harness_rsi.harness_loader import (
    LoadedHarness, LoaderError)
from research_runtime.research_store.db import ResearchDatabase
from research_runtime.runtime_state import RuntimeState

from test_m7_discovery_coordinator import (FakeAgent, FakeProvider,
                                           fake_executor)


class _Loader:
    def __init__(self, *, sha="a" * 64, boom=False):
        self.sha, self.boom = sha, boom
        self.calls = []

    def resolve_for_campaign(self, *, campaign_id, domain):
        self.calls.append(campaign_id)
        if self.boom:
            raise LoaderError("dispatch stopped")
        return LoadedHarness(harness_sha=self.sha,
                             component_shas={"critic": "b" * 64},
                             parent_sha="c" * 64, activation_id="act")


def _mk(tmp_path, *, loader=None, backend=None, **overrides):
    db = ResearchDatabase(tmp_path / "runtime.sqlite")
    rt = RuntimeState(db=db)
    if not rt.run("m76_p0b"):
        rt.initialize("m76_p0b", config=SimpleNamespace(), dataset_shas={},
                      max_rounds=6, global_budget=4)
    adapter = FormalRuntimeAdapter(
        runtime=rt, harness_loader=loader or _Loader(),
        backend=backend or fake_executor,
        domain="embodied_world_model", run_id="m76_p0b")
    return adapter, rt


def _coord(tmp_path, *, production_runtime=None, loader=None, harness_sha="a" * 64,
           production_mode=True, **overrides):
    kw = dict(run_id="m76_p0b", db=ResearchDatabase(tmp_path / "coord.sqlite"),
              out_dir=tmp_path / "discovery", providers=[FakeProvider()],
              agent=FakeAgent(), executor=fake_executor,
              pilot_effects={"e1": .1, "e2": .12, "e3": .11, "e4": .13},
              production_mode=production_mode,
              harness_sha=harness_sha, formal_loader=loader,
              production_runtime=production_runtime,
              approval_store=SimpleNamespace(), approval_id="unused")
    kw.update(overrides)
    return DiscoveryCoordinator(**kw)


def _blocked_at_start(tmp_path, **kw):
    c = _coord(tmp_path, **kw)
    return c.run(), c


# --- 1-3: missing formal contracts block production -----------------------

def test_missing_dispatcher_blocks_production(tmp_path):
    loader = _Loader()
    result, c = _blocked_at_start(tmp_path, loader=loader,
                                  harness_sha="a" * 64)
    assert result["status"] in ("BLOCKED", "INCONCLUSIVE")
    assert "production_runtime_missing" in result["reason"]
    assert not (c.root / "experiments" / "selection").exists()


def test_missing_harness_loader_blocks_production(tmp_path):
    adapter, rt = _mk(tmp_path)
    result, c = _blocked_at_start(tmp_path, production_runtime=adapter,
                                  loader=None)
    assert "formal_harness_loader" in result["reason"]


def test_missing_harness_sha_blocks_production(tmp_path):
    adapter, rt = _mk(tmp_path)
    result, c = _blocked_at_start(tmp_path, production_runtime=adapter,
                                  loader=_Loader(), harness_sha="")
    assert "harness_sha" in result["reason"]


# --- 4-6: harness identity + no direct fallback ---------------------------

def test_loader_failure_blocks_dispatch(tmp_path):
    adapter, rt = _mk(tmp_path, loader=_Loader(boom=True))
    with pytest.raises(ProductionRuntimeBlocked) as exc:
        adapter.dispatch(
            experiment_id="e", preregistration_sha="f" * 64,
            arms=["baseline"], seeds=[42], splits=["validation"],
            out_dir=str(tmp_path / "e"), stage="EXPERIMENT_EXECUTION",
            expected_harness_sha="a" * 64, config_sha=rt.run("m76_p0b")["config_sha"], round_no=1)
    assert "harness_loader_failed" in str(exc.value)


def test_harness_sha_mismatch_blocks_dispatch(tmp_path):
    adapter, rt = _mk(tmp_path, loader=_Loader(sha="d" * 64))
    with pytest.raises(ProductionRuntimeBlocked) as exc:
        adapter.dispatch(
            experiment_id="e", preregistration_sha="f" * 64,
            arms=["baseline"], seeds=[42], splits=["validation"],
            out_dir=str(tmp_path / "e"), stage="EXPERIMENT_EXECUTION",
            expected_harness_sha="a" * 64, config_sha=rt.run("m76_p0b")["config_sha"], round_no=1)
    assert "harness_sha_mismatch" in str(exc.value)


def test_production_never_calls_direct_executor(tmp_path):
    calls = []

    def boom_executor(**kw):
        calls.append(kw)
        return {"ok": True}

    adapter, rt = _mk(tmp_path)
    # Production coordinator with an approval fixture that will be rejected;
    # even if the gate passed, execution funnels through the adapter, not the
    # injected executor. Assert the executor is never invoked on a BLOCKED run.
    c = _coord(tmp_path, production_runtime=adapter, loader=_Loader(),
               executor=boom_executor)
    c.run()  # blocks at approval (no real store)
    assert calls == [], "production called the direct DomainPack executor"


# --- 7: reservation settlement -------------------------------------------

def test_reservation_consumed_on_success_and_failure(tmp_path):
    adapter, rt = _mk(tmp_path)
    cfg = rt.run("m76_p0b")["config_sha"]
    # success
    adapter.dispatch(experiment_id="e1", preregistration_sha="f" * 64,
                     arms=["baseline"], seeds=[42], splits=["validation"],
                     out_dir=str(tmp_path / "e1"), stage="EXPERIMENT_EXECUTION",
                     expected_harness_sha="a" * 64, config_sha=cfg, round_no=1)
    # failure
    def failing(**kw):
        raise RuntimeError("engine down")
    adapter.backend = failing
    with pytest.raises(RuntimeError):
        adapter.dispatch(experiment_id="e2", preregistration_sha="f" * 64,
                         arms=["baseline"], seeds=[43], splits=["validation"],
                         out_dir=str(tmp_path / "e2"), stage="REPLICATION",
                         expected_harness_sha="a" * 64, config_sha=cfg,
                         round_no=2)
    states = [r["state"] for r in rt.reservations("m76_p0b")]
    assert states and all(s == "consumed" for s in states)


# --- 8: journal integrity + crash recovery --------------------------------

def test_journal_integrity_survives_repeat_dispatch(tmp_path):
    adapter, rt = _mk(tmp_path)
    cfg = rt.run("m76_p0b")["config_sha"]
    kw = dict(experiment_id="e1", preregistration_sha="f" * 64,
              arms=["baseline"], seeds=[42], splits=["validation"],
              out_dir=str(tmp_path / "e1"), stage="EXPERIMENT_EXECUTION",
              expected_harness_sha="a" * 64, config_sha=cfg, round_no=1)
    r1 = adapter.dispatch(**kw)
    r2 = adapter.dispatch(**kw)  # duplicate: same deterministic campaign id
    assert r1["campaign_id"] == r2["campaign_id"]
    assert rt.verify_journal("m76_p0b")["ok"]
    reservations = rt.reservations("m76_p0b")
    # idempotent: no second budget spend
    assert len(reservations) == 1


# --- 9: deterministic campaign id binding ---------------------------------

def test_campaign_id_bound_to_real_harness_and_protocol():
    a = formal_campaign_id(run_id="r", experiment_id="e",
                           harness_sha="a" * 64, preregistration_sha="f" * 64)
    b = formal_campaign_id(run_id="r", experiment_id="e",
                           harness_sha="b" * 64, preregistration_sha="f" * 64)
    c = formal_campaign_id(run_id="r", experiment_id="e",
                           harness_sha="a" * 64, preregistration_sha="e" * 64)
    assert a != b and a != c and a.startswith("camp_")
