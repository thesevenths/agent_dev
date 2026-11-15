"""M4.3-00 / P0-04 -- Director recovery state & audit semantics (engine-free).

Legacy defects being fixed:
  * ``director_recovery.reconcile_director_state`` treated the EXISTENCE of a
    ResearchReport as proof that Experience was written, and returned
    ``S_COMPLETED`` -- but a Report row can exist with empty experience_refs or
    with NO matching Experience Store rows.
  * ``S_COMPLETED`` is not in ``ACTIVE_STATES``, so ``recover()`` fell into its
    generic halt branch and wrote COMPLETED with an ``E_FAIL`` event (#3).
  * Step methods wrote ``context.state`` directly, bypassing the StateMachine
    (#4), so journal events could disagree with the persisted state (#7).

Fix: a dedicated legal ``RECOVERING --RECOVER_COMPLETE--> COMPLETED`` path;
COMPLETED (both via finalize and via recovery) requires THREE proofs (real
Report + non-empty experience_refs + real Experience Store rows); ALL
transitions go through ``StateMachine.require``.

All engine-free (Harness FakeRunner / FakeComparator). No SciNet++ engine used.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_runtime.director_recovery import (
    reconcile_director_state, report_has_real_experience,
)
from research_runtime.director_state_machine import (
    ACTIVE_STATES, DirectorStateMachine, E_FAIL, E_RECOVER_COMPLETE,
    IllegalTransitionError, S_COMPLETED, S_EVALUATING, S_EXECUTING,
    S_IDLE, S_RECORDING_EXPERIENCE, S_RECOVERING,
)
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.research_director import ResearchDirector

from tests.rt_tests.research_fakes import (
    Harness, SOURCE_EXPERIMENT_ID,
)


# --------------------------------------------------------------------- helpers
def _proposed_objective(h: Harness) -> str:
    """Discovery persists a Proposed Objective into the SAME store."""
    from research_runtime.discovery_context_provider import DiscoverySnapshot
    snap = DiscoverySnapshot(
        run_id="r_p004",
        schema_version="m4.1-00-discovery-snapshot-v1",
        generated_at="2026-01-01T00:00:00",
        evidence_view={"schema_version": "m4.0-evidence-view-v1",
                       "run_id": "r_p004", "n_evidence": 1,
                       "evidence": [{"evidence_id": "sha256:neg1",
                                     "verdicts": ["rejected"], "sources": []}],
                       "provenance": {"domains": {}}, "unresolved": []},
        objectives=[], links={"run_stats": {}, "by_objective": {}},
        sessions=[], sessions_by_objective={}, experience_available=False,
        failures={"session_failure_signals": [], "evidence_unresolved": [],
                  "structured_diagnoses": []},
        provenance={"read_only": True, "run_id": "r_p004"},
        unresolved_gaps=[])
    res = ObjectiveDiscovery(domain="scinet").discover(
        snap, objective_store=h.stores.objectives, domain="scinet")
    assert res.proposed_objective_ids
    return res.proposed_objective_ids[0]


def _director(h: Harness, oid: str, state_dir: str = "director") -> ResearchDirector:
    return ResearchDirector(
        service=h.service, objective_store=h.stores.objectives,
        state_dir=h.root / state_dir, objective_id=oid,
        source_experiment_id=SOURCE_EXPERIMENT_ID)


def _grant(h: Harness, d: ResearchDirector) -> None:
    h.stores.approvals.grant(
        d.context.approval_id, approved_by="alice", source="human",
        current_resource_sha256=d.context.plan_sha256)


def _completed_run(h: Harness):
    """Drive a FULL real chain: Report + Experience rows are persisted."""
    oid = _proposed_objective(h)
    d = _director(h, oid)
    d.run_to_awaiting_approval()
    _grant(h, d)
    final = d.run_to_completion()
    assert final["state"] == S_COMPLETED, final
    return d


def _delete_experience_rows(h: Harness, session_id: str) -> None:
    """Simulate the fault: Report exists but Experience Store rows are gone."""
    with h.stores.sessions.db.session() as conn:
        conn.execute("DELETE FROM research_experience WHERE session_id=?",
                     (session_id,))


# --------------------------------------------------------------------- #1/#2 three proofs
def test_report_exists_but_experience_missing_is_NOT_completed():
    """#1/#6: a Report row alone is NOT proof of Experience."""
    h = Harness()
    try:
        d = _completed_run(h)
        sid = d.context.session_id
        rep = h.service.get_research_report(session_id=sid)
        assert rep["ok"] and rep["report"]["experience_refs"], \
            "a completed run has a real report with refs"

        _delete_experience_rows(h, sid)
        verified, detail = report_has_real_experience(h.service, sid, rep["report"])
        assert verified is False, "no Experience Store rows -> NOT verified"
        assert detail["experience_refs_count"] >= 1
        assert detail["experience_store_rows"] == 0

        # Reconciliation must NOT conclude COMPLETED; it resolves back to the
        # honest in-progress state so the Experience can actually be written.
        target, info = reconcile_director_state(
            h.service, session_id=sid, plan_id=d.context.plan_id,
            approval_id=d.context.approval_id,
            campaign_id=d.context.campaign_id)
        assert target == S_RECORDING_EXPERIENCE, info
        assert info.get("experience_verified") is False
        assert "Experience" in info.get("reason", "")
    finally:
        h.close()


def test_three_proofs_present_resolve_to_completed():
    """#2: real Report + non-empty refs + real rows -> COMPLETED is legitimate."""
    h = Harness()
    try:
        d = _completed_run(h)
        rep = h.service.get_research_report(session_id=d.context.session_id)
        verified, detail = report_has_real_experience(
            h.service, d.context.session_id, rep["report"])
        assert verified is True, detail
        assert detail["experience_refs_matched"] >= 1
        target, info = reconcile_director_state(
            h.service, session_id=d.context.session_id,
            plan_id=d.context.plan_id, approval_id=d.context.approval_id,
            campaign_id=d.context.campaign_id)
        assert target == S_COMPLETED, info
        assert info.get("experience_verified") is True
    finally:
        h.close()


def test_finalize_requires_real_experience_store_rows():
    """#2: finalize() (the COMPLETED gate) must fail-closed without rows."""
    h = Harness()
    try:
        oid = _proposed_objective(h)
        d = _director(h, oid)
        d.run_to_awaiting_approval()
        _grant(h, d)
        d.execute()
        d.compare()
        d.record_experience()
        assert d.context.state == S_RECORDING_EXPERIENCE
        # Corrupt the Experience Store AFTER record_experience but before finalize.
        _delete_experience_rows(h, d.context.session_id)
        with pytest.raises(Exception) as exc:
            d.finalize()
        assert "COMPLETED" in str(exc.value) or "Experience" in str(exc.value)
        assert d.context.state == S_RECORDING_EXPERIENCE, \
            "finalize must not jump to COMPLETED without real Experience"
    finally:
        h.close()


# --------------------------------------------------------------------- #3/#5 recovery-to-COMPLETED path
def test_recovery_uses_dedicated_complete_event_never_E_FAIL():
    """#3/#5: recovery to COMPLETED uses RECOVER_COMPLETE, never E_FAIL."""
    h = Harness()
    try:
        d = _completed_run(h)
        # Build a FRESH director at an in-progress state (simulated crash where
        # the journal lagged behind the engine, which already finished).
        oid = d.context.objective_id
        stale = _director(h, oid, "director_stale")
        stale.context.session_id = d.context.session_id
        stale.context.plan_id = d.context.plan_id
        stale.context.approval_id = d.context.approval_id
        stale.context.campaign_id = d.context.campaign_id
        stale.context.state = S_EXECUTING
        stale._write_ctx(stale.context)

        resolved = stale.recover()
        assert resolved == S_COMPLETED, resolved
        history = stale.history()
        complete = [e for e in history if e["event"] == E_RECOVER_COMPLETE]
        assert len(complete) == 1, history
        assert complete[0]["to"] == S_COMPLETED
        # An E_FAIL event must NEVER carry a COMPLETED target.
        assert not [e for e in history
                    if e["event"] == E_FAIL and e["to"] == S_COMPLETED], history
    finally:
        h.close()


def test_state_machine_rejects_E_FAIL_to_completed():
    """#3: the machine itself forbids E_FAIL -> COMPLETED."""
    sm = DirectorStateMachine()
    assert sm.next(E_FAIL, S_RECOVERING) != S_COMPLETED
    assert sm.next(E_RECOVER_COMPLETE, S_RECOVERING) == S_COMPLETED
    assert sm.next(E_RECOVER_COMPLETE, S_IDLE) is None, \
        "RECOVER_COMPLETE is legal ONLY from RECOVERING"
    assert S_COMPLETED not in ACTIVE_STATES


# --------------------------------------------------------------------- #4 all transitions via StateMachine
def test_illegal_state_write_is_rejected():
    """#4: _record refuses a transition the StateMachine does not allow."""
    h = Harness()
    try:
        oid = _proposed_objective(h)
        d = _director(h, oid)
        # From IDLE, an E_FINALIZE -> COMPLETED write must be rejected by the
        # StateMachine inside _record (no direct context.state mutation).
        with pytest.raises(IllegalTransitionError):
            d._record("FINALIZE", S_COMPLETED, {})
        assert d.context.state == S_IDLE, "state unchanged after illegal write"
    finally:
        h.close()


# --------------------------------------------------------------------- #7 journal/state consistency
def test_journal_events_consistent_with_final_state():
    """#7: after recovery the last journal event's target == persisted state,
    and every (from,event,to) triple is a legal StateMachine transition."""
    h = Harness()
    try:
        d = _completed_run(h)
        stale = _director(h, d.context.objective_id, "dir_consistent")
        stale.context.session_id = d.context.session_id
        stale.context.plan_id = d.context.plan_id
        stale.context.approval_id = d.context.approval_id
        stale.context.campaign_id = d.context.campaign_id
        stale.context.state = S_EVALUATING
        stale._write_ctx(stale.context)

        resolved = stale.recover()
        history = stale.history()
        ctx = json.loads((stale.state_dir / f"{stale.director_id}.json").read_text(
            encoding="utf-8"))
        assert ctx["state"] == resolved, "persisted state == returned state"
        assert history[-1]["to"] == resolved, "last journal target == final state"

        sm = DirectorStateMachine()
        # Validate recovery-origin transitions (ENTER_RECOVERY + resolution).
        # The initial 'director initialised at IDLE' bootstrap event is a
        # no-op marker (from==to==IDLE), not a state transition; skip it.
        for e in history:
            if e["from"] == e["to"] == S_IDLE and e["event"] == "SELECT_OBJECTIVE":
                continue
            if e["event"] == "RESOLVE_RECOVERY":
                assert sm.next(e["event"], e["from"],
                               recovery_target=e["to"]) == e["to"], e
            else:
                assert sm.next(e["event"], e["from"]) == e["to"], e
    finally:
        h.close()


def test_recovery_with_report_but_no_experience_resolves_to_in_progress():
    """#6/#7 end-to-end fault: recovery with a Report but no Experience must
    NOT claim COMPLETED, and must record the failure honestly in the journal."""
    h = Harness()
    try:
        d = _completed_run(h)
        _delete_experience_rows(h, d.context.session_id)
        stale = _director(h, d.context.objective_id, "dir_fault")
        stale.context.session_id = d.context.session_id
        stale.context.plan_id = d.context.plan_id
        stale.context.approval_id = d.context.approval_id
        stale.context.campaign_id = d.context.campaign_id
        stale.context.state = S_EVALUATING
        stale._write_ctx(stale.context)

        resolved = stale.recover()
        assert resolved == S_RECORDING_EXPERIENCE, \
            "Report without Experience must NOT resolve to COMPLETED"
        assert stale.context.state == resolved
        assert stale.history()[-1]["to"] == resolved
        assert not any(e["to"] == S_COMPLETED for e in stale.history())
    finally:
        h.close()
