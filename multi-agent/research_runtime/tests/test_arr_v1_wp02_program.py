"""ARR v1.0 WP-02: research program orchestration tests.

Engine-free: round_fn is injected; verifies program state machine, evidence-
backed next-round planning, memory inheritance, budget policy and resume.
"""
from __future__ import annotations

import pytest

from research_runtime.program import (
    ProgramBudgetManager, ProgramStateError, ResearchProgramDriver,
    ResearchProgramState, new_round_record)
from research_runtime.program.history_index import ResearchHistoryIndex


def _driver(tmp_path, *, families=("f1", "f2", "f3"), max_rounds=3,
            round_fn=None, diagnosis_fn=None, policy=None):
    return ResearchProgramDriver(
        root=tmp_path / "prog", program_id="P1",
        objective={"question": "improve latent dynamics prediction"},
        domain_scope="embodied_world_model", research_families=list(families),
        resource_policy=policy or {"round": {"experiments": 10}},
        max_rounds=max_rounds, round_fn=round_fn,
        diagnosis_fn=diagnosis_fn)


def test_program_state_transitions_and_integrity(tmp_path):
    st = ResearchProgramState(
        root=tmp_path, program_id="P", objective={"question": "q"},
        domain_scope="d", resource_policy={}, approval_policy={})
    assert st.status == "PROGRAM_CREATED"
    st.transition("OBJECTIVE_VALIDATED")
    with pytest.raises(ProgramStateError):
        st.transition("PROGRAM_COMPLETED")  # illegal skip
    assert st.verify_integrity()["ok"]


def test_duplicate_round_rejected(tmp_path):
    st = ResearchProgramState(
        root=tmp_path, program_id="P", objective={}, domain_scope="d",
        resource_policy={}, approval_policy={})
    r = new_round_record(program_id="P", round_id="r1", family_id="f1")
    st.add_round(r)
    with pytest.raises(ProgramStateError):
        st.add_round(r)


def _realistic_round(**kw):
    r = kw["round"]
    return {"status": "COMPLETED", "scientific_verdict": "INCONCLUSIVE",
            "question_id": "rq_" + r["round_id"],
            "hypothesis_ids": ["H1", "H2"],
            "preregistration_sha": "a" * 64, "harness_sha": "b" * 64,
            "method_sha": "c" * 64, "dataset_sha": "d" * 64,
            "campaign_ids": ["camp_" + r["round_id"]],
            "evidence_refs": ["ev_" + r["round_id"]],
            "claim_ids": ["cl_" + r["round_id"]],
            "budget_used": {"tokens": 100},
            "trace_ref": "tr_" + r["round_id"]}


def test_three_rounds_planned_from_artifacts_not_hardcoded(tmp_path):
    calls = []
    def rf(**kw):
        calls.append(kw["round"]["round_id"])
        return _realistic_round(**kw)
    d = _driver(tmp_path, round_fn=rf, max_rounds=3)
    rep = d.run()
    assert rep["ok"] and rep["status"] == "PROGRAM_COMPLETED"
    assert calls == ["P1_r1", "P1_r2", "P1_r3"]
    # round 2 family differs from round 1, parent chain set
    r1, r2, r3 = [d.state.round(x) for x in calls]
    assert len({r1["family_id"], r2["family_id"], r3["family_id"]}) == 3
    assert r2["parent_round_id"] == "P1_r1"
    assert r3["parent_round_id"] == "P1_r2"
    # memory refs from prior rounds are injected into later rounds
    assert r2["input_memory_refs"], "round 2 did not inherit prior knowledge"


def test_next_round_uses_real_digest(tmp_path):
    seen = []
    def rf(**kw):
        seen.append(kw["planning_digest"]["digest_sha"])
        return _realistic_round(**kw)
    d = _driver(tmp_path, round_fn=rf, max_rounds=2)
    d.run()
    assert len(set(seen)) == 2, "planning did not change with prior artifacts"


def test_budget_overrun_blocks_program(tmp_path):
    def rf(**kw):
        return {**_realistic_round(**kw), "budget_used": {"tokens": 100}}
    d = _driver(tmp_path, round_fn=rf, max_rounds=3,
                policy={"round": {"experiments": 3, "tokens": 150}})
    rep = d.run()
    # first round ok (100<150), second would exceed -> BLOCKED
    assert rep["status"] == "BLOCKED"
    assert len(d.state.obj["rounds"]) == 1


def test_resume_is_idempotent(tmp_path):
    d = _driver(tmp_path, round_fn=_realistic_round, max_rounds=2)
    r1 = d.run()
    d2 = _driver(tmp_path, round_fn=lambda **kw: pytest.fail("should not re-run"),
                 max_rounds=2)
    r2 = d2.run()
    assert r2["resumed"] is True and r2["status"] == r1["status"]


def test_history_index_untested_gaps(tmp_path):
    st = ResearchProgramState(
        root=tmp_path / "hist", program_id="P", objective={}, domain_scope="d",
        resource_policy={}, approval_policy={})
    st.add_round(new_round_record(program_id="P", round_id="r1",
                                  family_id="f1", objective="a"))
    st.update_round("r1", scientific_verdict="INCONCLUSIVE")
    idx = ResearchHistoryIndex(program_state=st)
    assert idx.verdict_summary()["INCONCLUSIVE"] == 1
    assert len(idx.untested_gaps()) == 1


def test_budget_manager_refuses_silent_overrun():
    bm = ProgramBudgetManager({"round": {"tokens": 10}})
    assert bm.check("round", {"tokens": 8})["ok"]
    bm.consume({"tokens": 8})
    assert not bm.check("round", {"tokens": 5})["ok"]


def test_discovery_round_adapter_maps_real_state(tmp_path):
    from research_runtime.program import DiscoveryRoundAdapter

    class FakeCoord:
        harness_sha = "b" * 64
        def __init__(self, **kw):
            self.kw = kw
            self.state = {
                "scientific_verdict": "INCONCLUSIVE",
                "selected_question": {"question_id": "rq_x"},
                "artifacts": {
                    "hypotheses": {"obj": {"hypotheses": [{"kind": "H1"}]}},
                    "claim_graph": {"obj": {"claim_id": "cl_x"}},
                    "evidence": {"obj": {"evidence_sha": "e" * 64}},
                },
                "preregistration_sha": "a" * 64,
                "approval": {"approval_id": "ap_x", "method_sha": "m" * 64},
                "dataset_shas": {"dataset_manifest_sha": "d" * 64},
                "trajectories": [{"total_tokens": 42}],
                "question_memory_trace": {"retrieved_ids": ["mem_x"]},
            }
        def run(self):
            return {"ok": True, "status": "COMPLETED",
                    "runtime_reservations": [{"campaign_id": "camp_x"}]}

    adapter = DiscoveryRoundAdapter(coordinator_factory=FakeCoord)
    out = adapter(program_id="P", round={
        "round_id": "r1", "family_id": "f2", "input_memory_refs": ["mem_x"]},
        planning_digest={"digest_sha": "z" * 64}, root=str(tmp_path))
    assert out["status"] == "COMPLETED"
    assert out["scientific_verdict"] == "INCONCLUSIVE"
    assert out["question_id"] == "rq_x"
    assert out["hypothesis_ids"] == ["H1"]
    assert out["campaign_ids"] == ["camp_x"]
    assert out["claim_ids"] == ["cl_x"]
    assert out["evidence_refs"] == ["e" * 64]
    assert out["budget_used"]["tokens"] == 42
    assert out["memory_used"] is True
