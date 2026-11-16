"""M4.3-01.1 shared engine-free REAL gate fixture.

Uses the real ``ResearchService`` + real SQLite ``ResearchStores`` (via
``research_fakes.Harness``). The runner/comparator are injected doubles that
only PRODUCE persisted records; the gate adapter under test reads those records
through the *real* read-only store queries. No fake/mocked Gate verdict is used:
every ``GateEvidence`` is derived from rows that actually exist in SQLite (or a
real CampaignRecord in the real campaign store).

``break_*`` helpers corrupt the REAL persisted state to create genuine missing
artefacts (fail-closed evidence), never by injecting a fake evidence object.

This fixture is explicitly ENGINE-FREE. The real SciNet++ engine E2E is a
separate, environment-gated stage (P1-03 / test_m4_3_01_1_real_engine.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from research_runtime.m4_3.research_service_gate import (
    GateRunRef, ResearchServiceGateAdapter,
)
from tests.rt_tests.research_fakes import PLAN_ID, Harness


@dataclass
class GateFixture:
    harness: Harness
    adapter: ResearchServiceGateAdapter
    ref: GateRunRef
    approval_id: str
    campaign_id: str
    session_id: str
    plan_id: str

    # ------------------------------------------------------------- mutation
    def break_approval(self) -> None:
        """Remove the REAL ApprovalRecord (no real human approval exists)."""
        with self.harness.stores.db.session() as conn:
            conn.execute("DELETE FROM approvals WHERE approval_id=?",
                         (self.approval_id,))

    def downgrade_approval_to_pending(self) -> None:
        """Real ApprovalRecord exists but is NOT approved (fail-closed)."""
        with self.harness.stores.db.session() as conn:
            conn.execute("UPDATE approvals SET status='pending' "
                         "WHERE approval_id=?", (self.approval_id,))

    def set_approval_source(self, source: str) -> None:
        """Real approved record but NOT an out-of-band human decision."""
        with self.harness.stores.db.session() as conn:
            conn.execute("UPDATE approvals SET source=? WHERE approval_id=?",
                         (source, self.approval_id))

    def break_campaign(self) -> None:
        """Remove the REAL persisted CampaignRecord."""
        self.harness.campaign_store.records.pop(self.campaign_id, None)

    def break_report(self) -> None:
        """Remove the REAL persisted ResearchReport."""
        with self.harness.stores.db.session() as conn:
            conn.execute("DELETE FROM research_reports WHERE session_id=?",
                         (self.session_id,))

    def break_experience_rows(self) -> None:
        """Remove the REAL Experience rows (report refs may still exist)."""
        with self.harness.stores.db.session() as conn:
            conn.execute("DELETE FROM research_experience WHERE session_id=?",
                         (self.session_id,))

    def break_plan(self) -> None:
        """Remove the REAL current ExperimentPlan pointer (get() joins on it)."""
        with self.harness.stores.db.session() as conn:
            conn.execute("DELETE FROM plan_current WHERE plan_id=?",
                         (self.plan_id,))
            conn.execute("DELETE FROM plan_versions WHERE plan_id=?",
                         (self.plan_id,))

    def fresh_adapter(self) -> ResearchServiceGateAdapter:
        """A NEW adapter over the SAME real service (restart-like)."""
        return ResearchServiceGateAdapter(self.harness.service)

    def close(self) -> None:
        self.harness.close()


def build_gate_fixture(*, grant: bool = True) -> GateFixture:
    """Drive ONE real engine-free chain to its full persisted gate artefacts."""
    h = Harness()
    out = h.chain(grant=grant)
    aid = out["approval_id"]
    run = h.service.run_research_campaign(plan_id=PLAN_ID, approval_id=aid)
    if not run.get("ok"):
        # A non-granted chain cannot run; still return a usable fixture whose
        # gate evidence is genuinely incomplete (fail-closed).
        sid = out["session"]["session_id"]
        return GateFixture(
            harness=h, adapter=ResearchServiceGateAdapter(h.service),
            ref=GateRunRef(session_id=sid, plan_id=PLAN_ID, approval_id=aid,
                           campaign_id=""),
            approval_id=aid, campaign_id="", session_id=sid, plan_id=PLAN_ID)
    cid = run["campaign_id"]
    cmp = h.service.compare_research_campaign(campaign_id=cid)
    sid = cmp["session_id"]
    ref = GateRunRef(session_id=sid, plan_id=PLAN_ID, approval_id=aid,
                     campaign_id=cid)
    return GateFixture(
        harness=h, adapter=ResearchServiceGateAdapter(h.service), ref=ref,
        approval_id=aid, campaign_id=cid, session_id=sid, plan_id=PLAN_ID)


@dataclass
class MultiArmFixture:
    """ONE real service, N DISTINCT real chains (one campaign each).

    M4.3-02 P0 #7 forbids one campaign's evidence being reused by a second
    run, so an arm-level test MUST give every arm its own plan / approval /
    campaign / session. All arms still share ONE service and ONE adapter --
    that is what "the same gate" means.
    """
    harness: Harness
    adapter: ResearchServiceGateAdapter
    refs: dict[str, GateRunRef]
    plan_ids: dict[str, str]
    approval_ids: dict[str, str]
    campaign_ids: dict[str, str]
    session_ids: dict[str, str]

    def close(self) -> None:
        self.harness.close()


def build_multi_arm_fixture(arms: tuple[str, ...] = ("A", "B", "C")) -> MultiArmFixture:
    """Build DISTINCT persisted chains for every arm over ONE real service.

    The domain policy caps campaigns per objective (round limit); the fixture
    raises that cap so each arm gets its OWN plan + approval + campaign. The
    arms still share one real service and one real adapter -- the gate itself
    is single.
    """
    from tests.rt_tests.research_fakes import default_domain

    h = Harness(domain=default_domain(max_campaign_rounds=len(arms) + 2))
    adapter = ResearchServiceGateAdapter(h.service)
    refs: dict[str, GateRunRef] = {}
    plan_ids: dict[str, str] = {}
    approval_ids: dict[str, str] = {}
    campaign_ids: dict[str, str] = {}
    session_ids: dict[str, str] = {}
    for arm in arms:
        pid = f"plan_arm_{arm.lower()}"
        out = h.chain(plan_id=pid)
        aid = out["approval_id"]
        run = h.service.run_research_campaign(plan_id=pid, approval_id=aid)
        assert run.get("ok"), run
        cid = run["campaign_id"]
        cmp = h.service.compare_research_campaign(campaign_id=cid)
        sid = cmp["session_id"]
        refs[arm] = GateRunRef(session_id=sid, plan_id=pid, approval_id=aid,
                               campaign_id=cid)
        plan_ids[arm], approval_ids[arm] = pid, aid
        campaign_ids[arm], session_ids[arm] = cid, sid
    return MultiArmFixture(
        harness=h, adapter=adapter, refs=refs, plan_ids=plan_ids,
        approval_ids=approval_ids, campaign_ids=campaign_ids,
        session_ids=session_ids)


@dataclass
class ChainFixture:
    """N DISTINCT real chains (plan/approval/campaign/session) on ONE service."""
    harness: Harness
    adapter: ResearchServiceGateAdapter
    refs: list[GateRunRef]

    def close(self) -> None:
        self.harness.close()


def build_chain_fixture(n_chains: int) -> ChainFixture:
    """Build ``n_chains`` independent persisted chains over ONE service.

    Needed because M4.3-02 P0 #7 forbids reusing one campaign as success
    evidence for a second run: a test that records N successful runs must own N
    real campaigns.
    """
    from tests.rt_tests.research_fakes import default_domain

    h = Harness(domain=default_domain(max_campaign_rounds=n_chains + 2))
    adapter = ResearchServiceGateAdapter(h.service)
    refs: list[GateRunRef] = []
    for i in range(n_chains):
        pid = f"plan_chain_{i}"
        out = h.chain(plan_id=pid)
        run = h.service.run_research_campaign(plan_id=pid,
                                              approval_id=out["approval_id"])
        assert run.get("ok"), run
        cid = run["campaign_id"]
        sid = h.service.compare_research_campaign(campaign_id=cid)["session_id"]
        refs.append(GateRunRef(session_id=sid, plan_id=pid,
                               approval_id=out["approval_id"],
                               campaign_id=cid))
    return ChainFixture(harness=h, adapter=adapter, refs=refs)


__all__ = ["GateFixture", "build_gate_fixture", "MultiArmFixture",
           "build_multi_arm_fixture", "complete_gate_evidence",
           "ChainFixture", "build_chain_fixture"]


def complete_gate_evidence(*, campaign_id: str = "campaign_x",
                           plan_id: str = "plan_x",
                           approval_id: str = "approval_x",
                           session_id: str = "session_x",
                           report_id: str = "report_x",
                           objective_id: str = "obj_x",
                           report_outcome: str = "rejected",
                           campaign_status: str = "completed",
                           succeeded_trials: int = 2,
                           experience_refs: tuple[str, ...] = ("exp_1",),
                           broken: str = "",
                           ):
    """A COMPLETE, internally consistent ``GateEvidence``.

    Engine-free metric / analysis tests need gate evidence that is complete
    without driving a whole ResearchService chain. This helper builds evidence
    that satisfies EVERY derived check (association, resolved experience refs,
    successful campaign) so those tests measure metrics rather than the gate.
    ``broken`` removes exactly ONE link to prove fail-closed behaviour.

    The REAL gate path is covered by ``build_gate_fixture`` /
    ``build_multi_arm_fixture`` (persisted ResearchService records). Nothing
    here is used by the production code.
    """
    from research_runtime.m4_3.research_service_gate import (
        ASSOCIATION_CHECKS, GateEvidence,
    )

    ref = GateRunRef(session_id=session_id, plan_id=plan_id,
                     approval_id=approval_id, campaign_id=campaign_id,
                     report_id=report_id)
    trials = succeeded_trials
    ev: dict = dict(
        ref=ref,
        plan_found=True, plan_sha256="sha", plan_objective_id=objective_id,
        approval_found=True, approval_status="approved", approval_source="human",
        approval_resource_id=plan_id, approval_resource_sha256="sha",
        approval_hash_matches_plan=True, approval_approved_by="alice",
        campaign_found=True, campaign_id=campaign_id,
        campaign_status=campaign_status,
        campaign_terminal=campaign_status in (
            "completed", "completed_with_failures", "failed", "cancelled"),
        campaign_successful=campaign_status in (
            "completed", "completed_with_failures") and trials > 0,
        campaign_plan_id=plan_id, campaign_objective_id=objective_id,
        campaign_trials=trials, campaign_succeeded_trials=trials,
        session_found=True, session_objective_id=objective_id,
        session_report_id=report_id, session_campaign_ids=(campaign_id,),
        report_found=True, report_id=report_id, report_session_id=session_id,
        report_objective_id=objective_id, report_outcome=report_outcome,
        report_experience_refs=tuple(experience_refs),
        report_campaign_ids=(campaign_id,),
        experience_rows=len(experience_refs),
        experience_ids=tuple(experience_refs),
        experience_refs_resolved=len(experience_refs),
        experience_refs_unresolved=(),
        association={k: True for k in ASSOCIATION_CHECKS},
        scope_conflicts=(),
        errors=(),
    )
    if broken == "campaign":
        ev.update(campaign_found=False, campaign_successful=False,
                  campaign_terminal=False)
    elif broken == "approval":
        ev.update(approval_found=False, approval_status="pending")
    elif broken == "report":
        ev.update(report_found=False)
    elif broken == "experience":
        ev.update(experience_rows=0, experience_ids=(),
                  experience_refs_resolved=0,
                  experience_refs_unresolved=tuple(experience_refs))
    elif broken == "session":
        ev.update(session_found=False)
    elif broken == "association":
        ev.update(association={k: False for k in ASSOCIATION_CHECKS})
    return GateEvidence(**ev)


