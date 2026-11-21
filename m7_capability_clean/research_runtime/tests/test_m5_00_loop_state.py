"""M5-00 -- autonomous loop state machine + recovery (ENGINE-FREE)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from research_runtime.autonomous_research_loop import (
    AutonomousResearchLoop, LoopConfig, _ReusePriorObjectiveStore,
)
from research_runtime.autonomous_loop_state import (
    IllegalTransition, LoopEventStore, LoopRecord, S_AWAITING_APPROVAL,
    S_CREATED, S_DISCOVERING, S_EXECUTING, S_LOOP_COMPLETED, S_OBJECTIVE_READY,
    S_PREFLIGHTED, S_RECOVERING, S_ROUND_COMPLETED, S_STOPPED, STATES,
    TERMINAL, can_transition,
)
from research_runtime.autonomous_loop_recovery import (
    apply_recovery, load_round_records, reconcile,
)
from research_runtime.research_store import (
    ResearchStores, ResearchStoreConflict, from_research_objective,
)
from research_runtime.research_store.db import ResearchDatabase
from research_runtime.schemas import ResearchObjective


def _store(tmp_path):
    return LoopEventStore(db=ResearchDatabase(tmp_path / "loop.sqlite"))


# ------------------------------------------------------------- state machine
def test_state_vocabulary_and_terminal():
    assert len(STATES) == 16
    assert S_LOOP_COMPLETED in TERMINAL and S_STOPPED in TERMINAL
    assert can_transition(S_CREATED, "preflight")
    assert can_transition(S_PREFLIGHTED, "begin_round")
    assert not can_transition(S_LOOP_COMPLETED, "begin_round")


def test_event_log_is_hash_chained(tmp_path):
    s = _store(tmp_path)
    rec = LoopRecord(loop_id="L1")
    s.save(rec)
    s.append(loop_id="L1", round_no=1, event="e1", payload={"a": 1})
    s.append(loop_id="L1", round_no=1, event="e2", payload={"a": 2})
    assert s.verify_integrity("L1") is True
    assert len(s.history("L1")) == 2


def test_event_log_detects_tampering(tmp_path):
    s = _store(tmp_path)
    s.append(loop_id="L1", round_no=1, event="e1", payload={"a": 1})
    assert s.verify_integrity("L1")
    with s.db.session() as conn:
        conn.execute("UPDATE m5_loop_events SET payload_json=? WHERE seq=1",
                     ("{}",))
    assert s.verify_integrity("L1") is False


def test_illegal_transition_raises():
    from research_runtime.autonomous_loop_state import E_EXECUTE
    assert not can_transition(S_CREATED, E_EXECUTE)


# ---------------------------------------------------------------- recovery
def test_reconcile_reports_incomplete_round_without_changing_state(tmp_path):
    s = _store(tmp_path)
    rec = LoopRecord(loop_id="L2", state=S_DISCOVERING, round_no=1)
    s.save(rec)
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    (rounds / "round_01.json").write_text(
        '{"round_no":1,"objective_id":"o","plan_id":"p"}', encoding="utf-8")
    report = reconcile(store=s, record=rec, rounds_dir=rounds)
    assert report.integrity_ok is True
    assert report.rounds[0].complete is False
    assert "approval" in report.rounds[0].reasons
    assert rec.state == S_DISCOVERING          # read-only until applied
    assert report.resolved_state == S_RECOVERING


def test_reconcile_complete_round_resolves_and_applies(tmp_path):
    s = _store(tmp_path)
    rec = LoopRecord(loop_id="L3", state=S_DISCOVERING, round_no=1)
    s.save(rec)
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    (rounds / "round_01.json").write_text(json.dumps({
        "round_no": 1, "objective_id": "o", "plan_id": "p",
        "approval_id": "a", "campaign_id": "c", "report_id": "r",
        "experience_id": "e"}), encoding="utf-8")
    report = reconcile(store=s, record=rec, rounds_dir=rounds)
    assert report.rounds[0].complete is True
    assert report.resolved_state == S_ROUND_COMPLETED
    apply_recovery(store=s, record=rec, report=report)
    assert report.applied is True and rec.state == S_ROUND_COMPLETED
    assert s.load("L3").state == S_ROUND_COMPLETED
    # idempotent: applying again does not create a new transition
    n = len(s.history("L3"))
    apply_recovery(store=s, record=rec, report=report)
    assert len(s.history("L3")) == n


def test_load_round_records_ignores_corrupt_files(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    (rounds / "round_01.json").write_text("{not json", encoding="utf-8")
    out = load_round_records(rounds)
    assert out and out[0].get("corrupt") is True


# ------------------------------------------- later-round discovery persistence
def _objective_record(oid: str, constraints: dict, provenance: dict):
    return from_research_objective(
        ResearchObjective(objective_id=oid, domain="scinet",
                          description="test objective",
                          constraints=constraints),
        provenance=provenance)


def test_events_are_tagged_with_the_executing_round(tmp_path):
    svc = SimpleNamespace(stores=SimpleNamespace(
        db=ResearchDatabase(tmp_path / "loop.sqlite")))
    loop = AutonomousResearchLoop(
        service=svc, config=LoopConfig(loop_id="L4"),
        out_dir=tmp_path / "out")
    loop._emit("before_round", {"a": 1})
    loop._current_round = 2
    loop._emit("during_round", {"a": 2})
    assert [e["round_no"] for e in loop.store.history("L4")] == [0, 2]
    assert loop.store.verify_integrity("L4") is True


def test_later_round_reuses_same_candidate_without_overwriting_history(tmp_path):
    store = ResearchStores(tmp_path / "objectives.sqlite").objectives
    prov = {"candidate_id": "cand_1", "generation_source": "negative_evidence"}
    store.save(_objective_record("o1", {"v": 1}, prov))

    changed = _objective_record(
        "o1", {"v": 2, "source_objective_ids": ["prior"]}, prov)
    adapter = _ReusePriorObjectiveStore(store, {"o1"})
    assert adapter.save(changed) == {"status": "reused_prior",
                                     "objective_id": "o1"}
    assert store.get("o1").constraints == {"v": 1}   # history untouched

    # Not previously proposed by this loop -> still fail closed.
    with pytest.raises(ResearchStoreConflict):
        _ReusePriorObjectiveStore(store, set()).save(changed)

    # Same id but a DIFFERENT candidate is never silently reused.
    other = _objective_record("o1", {"v": 3},
                              {**prov, "candidate_id": "cand_other"})
    with pytest.raises(ResearchStoreConflict):
        _ReusePriorObjectiveStore(store, {"o1"}).save(other)
