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


__all__ = ["GateFixture", "build_gate_fixture"]
