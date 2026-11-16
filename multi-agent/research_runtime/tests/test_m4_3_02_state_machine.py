"""M4.3-02 -- experiment state machine, controller, budget and recovery.

ENGINE-FREE: no SciNet++ process, no torch. Everything here is the
coordination layer (state vocabulary, legal transitions, append-only events,
budget guard, restart reconciliation).
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
from research_runtime.m4_3.experiment_controller import (
    BudgetExhausted, BudgetGuard, ExperimentController,
)
from research_runtime.m4_3.experiment_recovery import (
    assert_campaigns_not_duplicated, recover_experiment,
)
from research_runtime.m4_3.experiment_state import (
    ACTIVE_STATES, E_FAIL, E_RECOVER, E_STOP, STATES, TERMINAL_STATES,
    IllegalTransition, ExperimentStateMachine, S_AWAITING_APPROVAL,
    S_BLOCK_COMPLETED, S_COMPLETED, S_CREATED, S_EXECUTING, S_INTERRUPTED,
    S_PILOT_ANALYZING, S_PILOT_COMPLETED, S_RECORDING, S_REPORTING,
    S_SELECTING_OBJECTIVES, S_SNAPSHOTS_FROZEN, S_STOPPED,
)


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(snapshot_ids=("snap_1",), sample_size=0, seed=42,
                          model_version="m", prompt_version="p",
                          ranker_version="r", exclusion_rules=("none",),
                          statistical_method="paired-block",
                          stopping_rule="all blocks")


def _controller(tmp_path, **cfg) -> ExperimentController:
    return ExperimentController.create(
        experiment_id="exp_1", db_path=tmp_path / "exp.sqlite",
        config=cfg or None,
        budget=BudgetGuard(max_campaigns=cfg.pop("max_campaigns", 6),
                           max_wall_time_s=cfg.pop("max_wall_time_s", 600.0)))


# ------------------------------------------------------------------ states
def test_every_required_state_is_declared():
    required = {
        "CREATED", "PREFLIGHTING", "SNAPSHOTS_FROZEN", "SELECTING_OBJECTIVES",
        "AWAITING_REVIEW", "AWAITING_APPROVAL", "EXECUTING", "EVALUATING",
        "RECORDING", "BLOCK_COMPLETED", "PILOT_ANALYZING", "PILOT_COMPLETED",
        "CONFIRMATORY_PREPARING", "CONFIRMATORY_FROZEN",
        "CONFIRMATORY_AWAITING_APPROVAL", "CONFIRMATORY_EXECUTING",
        "CONFIRMATORY_COMPLETED", "REPORTING", "COMPLETED", "FAILED",
        "INTERRUPTED", "RECOVERING", "STOPPED",
    }
    assert required == set(STATES)
    assert TERMINAL_STATES == {"COMPLETED", "FAILED", "STOPPED"}
    assert "INTERRUPTED" not in ACTIVE_STATES      # resumable, not active
    assert "AWAITING_REVIEW" in ACTIVE_STATES


def test_illegal_transitions_are_refused():
    sm = ExperimentStateMachine()
    # cannot execute straight out of CREATED
    with pytest.raises(IllegalTransition):
        sm.require("executing", S_CREATED)
    # cannot report from CREATED
    with pytest.raises(IllegalTransition):
        sm.require("report", S_CREATED)
    # unknown event
    with pytest.raises(IllegalTransition):
        sm.require("teleport", S_CREATED)
    # terminal states accept nothing
    for event in (E_FAIL, E_RECOVER, E_STOP, "begin_block"):
        with pytest.raises(IllegalTransition):
            sm.require(event, S_COMPLETED)
    # recovery resolves ONLY to an active state
    with pytest.raises(IllegalTransition):
        sm.require("resolve_recovery", "RECOVERING",
                   recovery_target=S_COMPLETED)


def test_legal_happy_path_is_ordered():
    sm = ExperimentStateMachine()
    chain = [("preflight", S_CREATED, "PREFLIGHTING"),
             ("freeze_snapshots", "PREFLIGHTING", S_SNAPSHOTS_FROZEN),
             ("begin_block", S_SNAPSHOTS_FROZEN, S_SELECTING_OBJECTIVES),
             ("request_review", S_SELECTING_OBJECTIVES, "AWAITING_REVIEW"),
             ("review_complete", "AWAITING_REVIEW", S_AWAITING_APPROVAL),
             ("executing", S_AWAITING_APPROVAL, S_EXECUTING),
             ("evaluating", S_EXECUTING, "EVALUATING"),
             ("recording", "EVALUATING", S_RECORDING),
             ("complete_block", S_RECORDING, S_BLOCK_COMPLETED),
             ("analyze_pilot", S_BLOCK_COMPLETED, S_PILOT_ANALYZING),
             ("pilot_completed", S_PILOT_ANALYZING, S_PILOT_COMPLETED),
             ("report", S_PILOT_COMPLETED, S_REPORTING),
             ("complete", S_REPORTING, S_COMPLETED)]
    for event, frm, to in chain:
        assert sm.require(event, frm) == to


# --------------------------------------------------------------- controller
def test_controller_persists_every_transition_and_reloads(tmp_path):
    c = _controller(tmp_path)
    c.preflight(detail={"engine": "scinet"})
    c.freeze_snapshots(["snap_1"])
    c.begin_block("b1")
    c.request_review("b1")
    c.defer_review("b1", reason="no human reviewers available yet")
    for arm in ("A", "B", "C"):
        c.confirm_approval("b1", arm=arm, approval_id=f"ap_{arm}",
                           plan_id=f"plan_{arm}")
    c.begin_execution("b1")
    c.mark_evaluating("b1", campaign_id="c1", seconds=30.0)
    c.mark_recording("b1", outcomes={"A": "valid_finding"})
    c.complete_block("b1")

    reloaded = ExperimentController.load("exp_1", db_path=tmp_path / "exp.sqlite")
    assert reloaded.state == S_BLOCK_COMPLETED
    assert reloaded.blocks["b1"].review_status == "deferred"
    assert reloaded.blocks["b1"].outcomes == {"A": "valid_finding"}
    assert reloaded.budget.campaigns_used == c.budget.campaigns_used
    events = reloaded.history()
    assert len(events) >= 10
    assert [e["event"] for e in events][:2] == ["create", "preflight"]
    assert events[-1]["to_state"] == S_BLOCK_COMPLETED
    # append-only: every event keeps its own from/to pair
    assert all(e["from_state"] != e["to_state"] or e["event"] == "approval_observed"
               for e in events)


def test_duplicate_experiment_id_is_refused(tmp_path):
    _controller(tmp_path)
    with pytest.raises(ValueError):
        ExperimentController.create(experiment_id="exp_1",
                                    db_path=tmp_path / "exp.sqlite")


def test_review_completion_requires_two_distinct_reviewers(tmp_path):
    c = _controller(tmp_path)
    c.preflight()
    c.freeze_snapshots(["snap_1"])
    c.begin_block("b1")
    c.request_review("b1")
    with pytest.raises(ValueError):
        c.complete_review("b1", reviewers=["only_one"], review_ids=["r1"])
    c.complete_review("b1", reviewers=["r1", "r2"], review_ids=["rev_a", "rev_b"])
    assert c.state == S_AWAITING_APPROVAL


def test_execution_requires_all_three_real_approvals(tmp_path):
    c = _controller(tmp_path)
    c.preflight()
    c.freeze_snapshots(["snap_1"])
    c.begin_block("b1")
    c.request_review("b1")
    c.defer_review("b1", reason="pending")
    c.confirm_approval("b1", arm="A", approval_id="ap_A")
    with pytest.raises(IllegalTransition):
        c.begin_execution("b1")
    # A verifier can refuse an approval that is not in the trusted store.
    c2 = _controller(tmp_path / "sub")
    c2.preflight()
    c2.freeze_snapshots(["snap_1"])
    c2.begin_block("b1")
    c2.request_review("b1")
    c2.defer_review("b1", reason="pending")
    c2._approval_verifier = lambda aid, pid: False
    with pytest.raises(IllegalTransition):
        c2.confirm_approval("b1", arm="A", approval_id="forged")


def test_budget_guard_stops_honestly():
    guard = BudgetGuard(max_campaigns=2, max_wall_time_s=10.0)
    guard.reserve_campaign()
    guard.reserve_campaign()
    with pytest.raises(BudgetExhausted) as exc:
        guard.reserve_campaign()
    assert "budget exhausted" in str(exc.value)
    assert guard.stopped_reason
    guard2 = BudgetGuard(max_wall_time_s=5.0)
    guard2.charge_seconds(5.0)
    with pytest.raises(BudgetExhausted):
        guard2.reserve_campaign()


def test_interrupt_is_resumable_and_terminal_is_not(tmp_path):
    c = _controller(tmp_path)
    c.preflight()
    c.freeze_snapshots(["snap_1"])
    c.begin_block("b1")
    c.interrupt("machine rebooted")
    assert c.state == S_INTERRUPTED
    resumed = ExperimentController.load("exp_1", db_path=tmp_path / "exp.sqlite")
    assert resumed.recover(resolved_state=S_SELECTING_OBJECTIVES) == \
        S_SELECTING_OBJECTIVES
    resumed.stop("operator stopped")
    assert resumed.state == S_STOPPED
    with pytest.raises(IllegalTransition):
        resumed.recover(resolved_state=S_SELECTING_OBJECTIVES)


# ---------------------------------------------------------------- recovery
def test_recovery_reports_progress_and_never_creates_a_campaign(tmp_path):
    c = _controller(tmp_path)
    c.preflight()
    c.freeze_snapshots(["snap_1"])
    c.begin_block("b1")
    c.request_review("b1")
    c.defer_review("b1", reason="pending")

    # A REAL registry bound to the SAME sqlite database.
    reg = ExperimentRegistry(protocol=_protocol(), db=c.store.db)
    reg.create_block(block_id="b1", snapshot_id="snap_1", seed=42)
    for arm in ("A", "B", "C"):
        reg.register_run(block_id="b1", arm=arm)
    reg.finalize("b1_A", outcome="valid_finding", campaign_id="camp_A")
    reg.finalize("b1_B", outcome="interrupted")

    before = len(reg.store.load_runs())
    report = recover_experiment(c, db=c.store.db)
    after = len(reg.store.load_runs())
    assert before == after, "recovery must not create runs / campaigns"
    assert report.applied is True
    assert report.resolved_state == S_SELECTING_OBJECTIVES
    block = next(b for b in report.blocks if b.block_id == "b1")
    assert block.arms_finalized == ["A", "B"]
    assert block.unfinalized_arms == ["C"]
    assert block.complete is False
    assert any("finalize arm(s) C" in a for a in report.next_actions)


def test_recovery_of_a_complete_registry_moves_to_block_completed(tmp_path):
    c = _controller(tmp_path)
    c.preflight()
    c.freeze_snapshots(["snap_1"])
    c.begin_block("b1")
    c.request_review("b1")
    c.defer_review("b1", reason="pending")
    reg = ExperimentRegistry(protocol=_protocol(), db=c.store.db)
    reg.create_block(block_id="b1", snapshot_id="snap_1", seed=42)
    for i, arm in enumerate(("A", "B", "C")):
        reg.register_run(block_id="b1", arm=arm)
        reg.finalize(f"b1_{arm}", outcome="valid_finding",
                     campaign_id=f"camp_{arm}")
    report = recover_experiment(c, db=c.store.db)
    assert report.resolved_state == S_BLOCK_COMPLETED
    assert c.state == S_BLOCK_COMPLETED
    assert_campaigns_not_duplicated(c.store.db)


def test_campaign_reuse_is_detected_by_integrity_check(tmp_path):
    c = _controller(tmp_path)
    reg = ExperimentRegistry(protocol=_protocol(), db=c.store.db)
    reg.create_block(block_id="b1", snapshot_id="snap_1", seed=42)
    reg.create_block(block_id="b2", snapshot_id="snap_1", seed=43)
    reg.register_run(block_id="b1", arm="A")
    reg.register_run(block_id="b2", arm="A")
    # Bypass the registry guard on purpose: write the same campaign into SQL.
    with c.store.db.session() as conn:
        for rid in ("b1_A", "b2_A"):
            conn.execute("UPDATE m4_3_runs SET campaign_id='dup' WHERE run_id=?",
                         (rid,))
    with pytest.raises(ValueError) as exc:
        assert_campaigns_not_duplicated(c.store.db)
    assert "dup" in str(exc.value)
    report = recover_experiment(c, db=c.store.db, apply=False)
    assert report.duplicate_campaign_ids == ["dup"]
    assert report.corruption
