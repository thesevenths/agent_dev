"""M4-E2E-01.1 P6 -- Real Engine Validation (engine-free, CI-runnable subset).

The heavy "real SciNet++ trains a model" path lives in
``test_m4_e2e_01_real_engine.py`` (gated by ``M4_E2E_01_RUN_REAL=1`` + a SciNet++
checkout). This file proves the TWO new P6 behaviours that do NOT require the
heavy engine and therefore run in ordinary CI:

  (1) GENUINE OUT-OF-BAND GRANT. The campaign gate is fail-closed: without a
      human approval the campaign cannot run. The approval must come from the
      SEPARATE ``approval_cli`` process (a real subprocess), exactly as a human
      operator would grant -- never from in-process code. This closes the P0
      boundary at the engine boundary, not just at the Director.

  (2) ROUND-2 DISCOVERY READS ROUND-1 EXPERIENCE. A real campaign (engine-free
      FakeRunner) writes a REAL Experience row into the session store. A second
      Discovery, driven by the REAL ``DiscoveryContextProvider`` over that SAME
      store, surfaces a candidate that references ``experience:<round1_session_id>``.
      The hardened ``_resolve_ref`` (P2) accepts the specific id and rejects a
      fabricated id -- proving the closed loop on REAL stored data.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from research_runtime.discovery_context_provider import DiscoveryContextProvider
from research_runtime.objective_candidate_generator import ObjectiveCandidate
from research_runtime.objective_critic import ObjectiveCritic, _resolve_ref
from research_runtime.objective_discovery import ObjectiveDiscovery
from tests.rt_tests.research_fakes import Harness, OBJECTIVE_ID, PLAN_ID

ROOT = Path(__file__).resolve().parents[2]


def _grant_via_cli(db_path: Path, approval_id: str, *,
                   approved_by: str = "alice") -> int:
    """Invoke the REAL human approval console as a separate process.

    This is exactly the out-of-band path a human operator uses; it is the ONLY
    way to write ``source="human"``. Returns the subprocess exit code.
    """
    # NOTE: argparse puts --db on the TOP-LEVEL parser, so it must precede the
    # subcommand (the same way a human operator runs it). This is exactly the
    # out-of-band path; only the console writes source="human".
    result = subprocess.run(
        [sys.executable, "-m", "research_runtime.research_store.approval_cli",
         "--db", str(db_path), "grant", approval_id,
         "--approved-by", approved_by, "--yes"],
        cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"approval_cli grant failed rc={result.returncode}\n"
            f"STDOUT={result.stdout}\nSTDERR={result.stderr}")
    return result.returncode


def _experience_sid(h: Harness, approval_id: str, plan_id: str = PLAN_ID,
                    *, grant: bool = True) -> str:
    """Optionally grant out-of-band, then run the real campaign + comparison.

    ``grant`` is False when the caller has ALREADY granted (an approval decision
    is immutable, so the console must not be asked to grant twice).
    """
    if grant:
        assert _grant_via_cli(h.root / "research.sqlite", approval_id) == 0
    run = h.service.run_research_campaign(plan_id=plan_id, approval_id=approval_id)
    assert run.get("ok"), run
    cmp = h.service.compare_research_campaign(campaign_id=run["campaign_id"])
    assert cmp.get("ok"), cmp
    return cmp["session_id"]


# --------------------------------------------------------------------- P6 (1)
def test_approval_cli_subprocess_grant_is_out_of_band(tmp_path):
    """P6: the campaign gate is fail-closed; only an out-of-band subprocess grant
    (source='human') can open it, and a real Experience is written."""
    h = Harness()
    out = h.chain(grant=False)
    approval_id = out["approval_id"]
    plan_id = PLAN_ID

    # (a) Fail-closed: no approval in the store -> campaign gate refuses.
    assert h.service.stores.approvals.get(approval_id).status != "approved"
    bare = h.service.run_research_campaign(plan_id=plan_id, approval_id=approval_id)
    assert bare.get("ok") is False, bare
    assert bare.get("error_code") in (
        "approval_not_approved", "approval_not_found",
        "approval_resource_mismatch", "approval_source_rejected")

    # (b) Genuine out-of-band grant via the SEPARATE human console process.
    rc = _grant_via_cli(h.root / "research.sqlite", approval_id)
    assert rc == 0, "approval_cli grant must succeed as a standalone process"
    approval = h.service.stores.approvals.get(approval_id)
    assert approval.status == "approved"
    # P0 closure: only the console writes source="human"; the engine never can.
    assert approval.source == "human"
    assert approval.resource_sha256, "grant must be SHA-bound to the plan"

    # (c) Now the campaign runs and writes a REAL Experience (not a fake).
    # The approval is ALREADY granted in (b); re-granting is illegal (immutable).
    sid = _experience_sid(h, approval_id, plan_id, grant=False)
    rep = h.service.get_research_report(session_id=sid)
    assert rep.get("ok"), rep
    assert rep["report"]["experience_refs"], "REAL Experience must be recorded"
    assert h.service.stores.sessions.experience(sid), "Experience row persisted"


# --------------------------------------------------------------------- P6 (2)
def test_round2_discovery_reads_round1_experience(tmp_path):
    """P6: round-2 Discovery, driven by the REAL DiscoveryContextProvider over the
    SAME store, reads round-1's real Experience and the hardened ref resolver
    (P2) accepts the specific id while rejecting a fabricated one."""
    h = Harness()
    out = h.chain(grant=False)
    approval_id = out["approval_id"]

    # Round 1: real campaign writes a real Experience row.
    sid = _experience_sid(h, approval_id)
    exps = h.service.stores.sessions.experience(sid)
    assert exps, "round-1 must write Experience"
    assert sid

    # Round 2: the REAL provider reads the store (no hand-built fake snapshot).
    provider = DiscoveryContextProvider(stores=h.stores)
    snapshot = provider.build_snapshot("round2", objective_ids=[OBJECTIVE_ID])
    # The snapshot carries the round-1 session with its real Experience.
    assert any(s.get("session_id") == sid and s.get("experience")
               for s in snapshot.sessions), \
        "snapshot must surface round-1 Experience"

    # The hardened ref resolver (P2) accepts the SPECIFIC id, rejects a fake.
    assert _resolve_ref(f"experience:{sid}", snapshot) is True
    assert _resolve_ref("experience:does_not_exist", snapshot) is False

    # End-to-end: round-2 Discovery produces a candidate that cites the real
    # session, and the Critic accepts it while rejecting a fabricated ref.
    result = ObjectiveDiscovery(domain="scinet").discover(
        snapshot, objective_store=h.stores.objectives, domain="scinet")
    assert result.n_candidates > 0
    exp_ranked = [c for c in result.ranked
                  if any(r.startswith("experience:") for r in c["evidence_refs"])]
    assert exp_ranked, "round-2 must surface an experience-sourced candidate"
    assert any(r == f"experience:{sid}" for c in exp_ranked
               for r in c["evidence_refs"]), \
        "candidate must cite the real round-1 session"

    critic = ObjectiveCritic()
    good = ObjectiveCandidate(
        candidate_id="c_good", domain="scinet",
        description="refine from recorded experience",
        evidence_refs=[f"experience:{sid}"],
        source_objective_ids=[OBJECTIVE_ID], hypothesis_hint="h",
        proposed_intervention={"axis": "x"},
        success_criteria={"criterion": "improve", "measurable": True},
        budget={"max_campaign_rounds": 1}, provenance={},
        generation_source="historical_experience")
    passed, _ = critic.critique(good, snapshot=snapshot)
    assert passed is True, "specific experience ref must resolve"

    bad = ObjectiveCandidate(
        candidate_id="c_bad", domain="scinet",
        description="refine from recorded experience",
        evidence_refs=["experience:does_not_exist"],
        source_objective_ids=[OBJECTIVE_ID], hypothesis_hint="h",
        proposed_intervention={"axis": "x"},
        success_criteria={"criterion": "improve", "measurable": True},
        budget={"max_campaign_rounds": 1}, provenance={},
        generation_source="historical_experience")
    passed2, errors2 = critic.critique(bad, snapshot=snapshot)
    assert passed2 is False
    assert any("resolv" in e.lower() for e in errors2)
