"""M5-00.1 / P0-01 -- durable budget ledger (ENGINE-FREE)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from research_runtime.autonomous_loop_state import (
    LoopEventStore, LoopRecord, S_PREFLIGHTED,
)
from research_runtime.autonomous_research_loop import (
    AutonomousResearchLoop, LoopConfig, LoopPreflightError,
)
from research_runtime.loop_budget_ledger import (
    LoopBudgetExhausted, LoopBudgetLedger, config_sha256,
)
from research_runtime.research_store import ResearchStoreConflict
from research_runtime.research_store.db import ResearchDatabase


def _ledger(path):
    return LoopBudgetLedger(db=ResearchDatabase(path))


def test_budget_survives_restart_and_never_reclaims_spent_units(tmp_path):
    path = tmp_path / "budget.sqlite"
    sha = "sha_config_v1"
    first = _ledger(path)
    first.initialize("L1", config_sha=sha, max_campaigns=2,
                     max_wall_time_s=0, started_epoch=time.time())
    first.reserve("L1", "campaign_a", config_sha=sha)
    first.consume("L1", "campaign_a", outcome="COMPLETED", config_sha=sha)

    # Simulate a fresh process: only SQLite state is available.
    restarted = _ledger(path)
    snap = restarted.snapshot("L1")
    assert (snap["consumed"], snap["reserved"], snap["remaining"]) == (1, 0, 1)

    # Same campaign can never be charged twice.
    restarted.reserve("L1", "campaign_a", config_sha=sha)
    assert restarted.snapshot("L1")["consumed"] == 1

    restarted.reserve("L1", "campaign_b", config_sha=sha)
    assert restarted.snapshot("L1")["remaining"] == 0
    with pytest.raises(LoopBudgetExhausted):
        restarted.reserve("L1", "campaign_c", config_sha=sha)

    # Failure/interruption also spends the reserved unit.
    restarted.consume("L1", "campaign_b", outcome="interrupted", config_sha=sha)
    final = restarted.snapshot("L1")
    assert (final["consumed"], final["reserved"], final["remaining"]) == (2, 0, 0)


def test_budget_config_sha_is_bound(tmp_path):
    ledger = _ledger(tmp_path / "sha.sqlite")
    ledger.initialize("L2", config_sha="sha_v1", max_campaigns=1,
                      max_wall_time_s=0, started_epoch=time.time())
    with pytest.raises(ResearchStoreConflict):
        ledger.initialize("L2", config_sha="sha_v2", max_campaigns=1,
                          max_wall_time_s=0, started_epoch=time.time())
    with pytest.raises(ResearchStoreConflict):
        ledger.reserve("L2", "campaign_x", config_sha="sha_v2")


def test_loop_uses_persisted_wall_clock_not_process_start(tmp_path):
    db = ResearchDatabase(tmp_path / "loop.sqlite")
    svc = SimpleNamespace(stores=SimpleNamespace(db=db))
    cfg = LoopConfig(loop_id="L3", max_wall_time_s=10.0)
    loop = AutonomousResearchLoop(service=svc, config=cfg,
                                  out_dir=tmp_path / "out")
    loop.record = LoopRecord(loop_id="L3", state=S_PREFLIGHTED)
    loop.budget.initialize("L3", config_sha=loop.config_sha,
                           max_campaigns=0, max_wall_time_s=10.0,
                           started_epoch=time.time() - 100)
    with pytest.raises(LoopBudgetExhausted, match="max_wall_time_s"):
        loop._check_budget()


def test_loop_refuses_changed_config_after_creation(tmp_path):
    db = ResearchDatabase(tmp_path / "cfg.sqlite")
    svc = SimpleNamespace(stores=SimpleNamespace(db=db))
    cfg = LoopConfig(loop_id="L4", max_campaigns=2)
    LoopEventStore(db=db).save(LoopRecord(
        loop_id="L4", state=S_PREFLIGHTED, config=cfg.__dict__.copy(),
        config_sha="sha_old"))
    with pytest.raises(LoopPreflightError, match="config SHA mismatch"):
        AutonomousResearchLoop(service=svc, config=cfg,
                               out_dir=tmp_path / "out")
    assert config_sha256(cfg) == config_sha256(cfg.__dict__)
