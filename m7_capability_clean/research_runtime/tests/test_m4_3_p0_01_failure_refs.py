"""M4.3-00 / P0-01 -- Failure Evidence Reference contract tests (engine-free).

Bug being fixed (M4-E2E-01.1 legacy):
``_resolve_ref`` pooled ``session_id`` and ``objective_id`` into ONE undirected
set and never looked at ``:event:<type>``. As a result
``session:<real_id>:event:<any_fabricated_type>`` resolved, an objective id
cited under a ``session:`` prefix resolved, and any real session made an event
ref "grounded" even when no such failure event ever happened.

Fix: a single shared ref contract (``make_failure_ref`` / ``parse_failure_ref``
in objective_candidate_generator.py) and a strict resolver in the Critic that
matches BOTH the scope-correct id field AND the exact event_type against the
REAL ``snapshot.failures.session_failure_signals`` entries.
"""
from __future__ import annotations

from research_runtime.discovery_context_provider import DiscoverySnapshot
from research_runtime.objective_candidate_generator import (
    ObjectiveCandidateGenerator, make_failure_ref, parse_failure_ref,
)
from research_runtime.objective_critic import ObjectiveCritic, _resolve_ref


def _snapshot(*, signals=None, sessions=None,
              sessions_by_objective=None) -> DiscoverySnapshot:
    """A real-shaped snapshot carrying explicit failure signals + sessions."""
    signals = signals if signals is not None else [
        {"objective_id": "obj_x", "session_id": "s_x",
         "session_index": 0, "event_type": "train_failed", "created_at": ""},
        # A signal with NO session_id -> the canonical ref is objective-scoped.
        {"objective_id": "obj_y", "session_id": "",
         "session_index": 0, "event_type": "eval_failed", "created_at": ""},
    ]
    sessions = sessions if sessions is not None else [
        {"objective_id": "obj_x", "session_id": "s_x",
         "session_available": True, "session_index": 0,
         "events": [{"event_type": "train_failed"}], "reports": [],
         "experience": [], "experience_available": False},
        {"objective_id": "obj_y", "session_id": "s_y",
         "session_available": True, "session_index": 0,
         "events": [{"event_type": "eval_failed"}], "reports": [],
         "experience": [], "experience_available": False},
    ]
    sbo = sessions_by_objective if sessions_by_objective is not None else \
        {"obj_x": [sessions[0]], "obj_y": [sessions[1]]}
    return DiscoverySnapshot(
        run_id="r_p001",
        schema_version="m4.1-00-discovery-snapshot-v1",
        generated_at="2026-01-01T00:00:00",
        evidence_view={"schema_version": "m4.0-evidence-view-v1",
                       "run_id": "r_p001", "n_evidence": 0, "evidence": [],
                       "provenance": {"domains": {}}, "unresolved": []},
        objectives=[],
        links={"run_stats": {}, "by_objective": {}},
        sessions=sessions,
        sessions_by_objective=sbo,
        experience_available=False,
        failures={"session_failure_signals": signals,
                  "evidence_unresolved": [], "structured_diagnoses": []},
        provenance={"read_only": True, "run_id": "r_p001"},
        unresolved_gaps=[],
    )


def _candidate(ref: str) -> object:
    from research_runtime.objective_candidate_generator import ObjectiveCandidate
    return ObjectiveCandidate(
        candidate_id="cand_failure", domain="scinet",
        description="investigate a recorded failure event",
        evidence_refs=[ref],
        source_objective_ids=["obj_x"], hypothesis_hint="h",
        proposed_intervention={"axis": "avoid_prior_failure"},
        success_criteria={"criterion": "avoid repeating failure",
                          "measurable": True},
        budget={"max_campaign_rounds": 1}, provenance={},
        generation_source="historical_failure")


# --------------------------------------------------------------- ref contract
def test_parse_failure_ref_is_strict_and_typed():
    assert parse_failure_ref("session:s_x:event:train_failed") == \
        ("session", "s_x", "train_failed")
    assert parse_failure_ref("objective:obj_y:event:eval_failed") == \
        ("objective", "obj_y", "eval_failed")
    assert parse_failure_ref("session:s_x") == ("session", "s_x", None)
    # Malformed / fabricated shapes must not parse at all.
    for bad in ("session:s_x:event:", "session::event:fail",
                "session:s_x:oops:fail", "session:s_x:event:fail:extra",
                "other:s_x:event:fail", "", "no_colon"):
        assert parse_failure_ref(bad) is None, bad


def test_make_failure_ref_selects_correct_scope():
    assert make_failure_ref({"session_id": "s_x", "objective_id": "obj_x",
                             "event_type": "train_failed"}) == \
        "session:s_x:event:train_failed"
    assert make_failure_ref({"session_id": "", "objective_id": "obj_y",
                             "event_type": "eval_failed"}) == \
        "objective:obj_y:event:eval_failed"


# --------------------------------------------------------------- P0-01 (1)(2)
def test_real_session_failure_ref_resolves():
    snap = _snapshot()
    assert _resolve_ref("session:s_x:event:train_failed", snap) is True


def test_real_objective_failure_ref_resolves():
    snap = _snapshot()
    assert _resolve_ref("objective:obj_y:event:eval_failed", snap) is True


# --------------------------------------------------------------- P0-01 (3)(4)
def test_unknown_id_is_rejected():
    snap = _snapshot()
    assert _resolve_ref("session:does_not_exist:event:train_failed", snap) is False
    assert _resolve_ref("objective:does_not_exist:event:eval_failed", snap) is False


def test_fabricated_event_ref_is_rejected():
    snap = _snapshot()
    # The id is real but this event_type never happened -> NOT grounded.
    assert _resolve_ref("session:s_x:event:fabricated_event", snap) is False
    assert _resolve_ref("objective:obj_y:event:fabricated_event", snap) is False


def test_scope_type_confusion_is_rejected():
    """The old bug: objective ids were pooled with session ids."""
    snap = _snapshot()
    # obj_x is an OBJECTIVE id, not a session id -> session-scoped ref must fail.
    assert _resolve_ref("session:obj_x:event:train_failed", snap) is False
    # s_x is a SESSION id, not an objective id -> objective-scoped ref must fail.
    assert _resolve_ref("objective:s_x:event:train_failed", snap) is False
    # Event type from a different signal must not cross-resolve.
    assert _resolve_ref("session:s_x:event:eval_failed", snap) is False


def test_plain_entity_ref_resolves_only_real_scope_id():
    snap = _snapshot()
    assert _resolve_ref("session:s_x", snap) is True
    assert _resolve_ref("objective:obj_x", snap) is True
    assert _resolve_ref("session:obj_x", snap) is False   # wrong kind
    assert _resolve_ref("session:s_missing", snap) is False


# --------------------------------------------------------------- P0-01 (5)
def test_failure_derived_candidate_passes_critic():
    """Generator output + Critic must agree on the SAME real failure ref."""
    snap = _snapshot()
    gen = ObjectiveCandidateGenerator(domain="scinet")
    failure_cands = [c for c in gen.generate(snap)
                     if c.generation_source == "historical_failure"]
    assert failure_cands, "generator must derive failure candidates from signals"
    critic = ObjectiveCritic()
    for c in failure_cands:
        passed, errors = critic.critique(c, snapshot=snap)
        assert passed is True, (c.evidence_refs, errors)


def test_fabricated_event_candidate_is_rejected_by_critic():
    snap = _snapshot()
    critic = ObjectiveCritic()
    passed, errors = critic.critique(
        _candidate("session:s_x:event:fabricated_event"), snapshot=snap)
    assert passed is False
    assert any("resolv" in e.lower() for e in errors)
