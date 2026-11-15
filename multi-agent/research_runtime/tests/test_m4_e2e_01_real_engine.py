"""M4-E2E-01C / M4.3-00 P0-05 -- REAL SciNet++ engine E2E (OPT-IN, skipped in CI).

This is the "no FakeRunner, no synthetic Snapshot" path. Every entity is REAL:

  * REAL historical Evidence      -- DiscoveryContextProvider (unified Evidence
                                     Reader) over the committed historical run.
  * REAL Objective               -- ObjectiveDiscovery (generator + Critic +
                                     Ranker) on that real Snapshot, persisted.
  * REAL engine                  -- build_default_research_service("scinet"):
                                     SciNetAdapter + CampaignRunner +
                                     comparator (models are actually trained).
  * REAL out-of-band approval    -- the separate ``approval_cli`` SUBPROCESS.
  * REAL Evaluation + Experience -- ResearchReport + Experience Store rows.
  * ROUND-2 closed loop          -- the SAME real Provider builds a second
                                     Snapshot; a candidate cites the EXACT
                                     round-1 Experience session id.

Design invariants (spec M4.3-00 / P0-05):
  #1 ONE ResearchStores / SQLite: the SAME db_path backs both the explicitly
     built ResearchStores and the real ResearchService, so Objective / Session /
     Hypothesis / Plan / Approval / Experience are all in one trusted store.
  #2-#4 NO hand-made ``sha256:neg1`` and NO synthetic Snapshot. If the real
     historical Evidence is missing, the test SKIPS -- it never falls back to a
     fabricated view and then claims "real historical E2E passed".
  #8-#9 round-2 Snapshot is built by the SAME provider over the SAME store and
     must surface the round-1 Experience.

Gating (ALL required, otherwise an explicit SKIP -- never a fabricated pass):
  * ``M4_E2E_01_RUN_REAL=1`` (operator opt-in; trains models, minutes-hours),
  * a SciNet++ checkout at ``settings.DEFAULT_SCINET_DIR``,
  * REAL historical Evidence: the provider must return ``n_evidence > 0`` for
    the historical run (probed once, read-only, at collection time).

Run manually::

    M4_E2E_01_RUN_REAL=1 python -m pytest \\
        research_runtime/tests/test_m4_e2e_01_real_engine.py -q

Engine-free CI proves the state machine / approval / persistence / recovery and
the Experience writeback MECHANISM without an engine; this file proves the SAME
Director drives the REAL engine unchanged. The two are reported SEPARATELY.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from research_runtime.discovery_context_provider import DiscoveryContextProvider
from research_runtime.objective_candidate_generator import ObjectiveCandidateGenerator
from research_runtime.objective_critic import ObjectiveCritic, _resolve_ref
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.objective_ranker import ObjectiveRanker
from research_runtime.research_director import ResearchDirector
from research_runtime.research_environment import build_default_research_service
from research_runtime.research_store import ResearchStores
from research_runtime.settings import DEFAULT_SCINET_DIR

ROOT = Path(__file__).resolve().parents[2]

# The committed REAL historical run (from which the unified Evidence Reader
# derives 100+ real evidence entities across six carriers). It is a real run id,
# not a synthetic placeholder.
HISTORICAL_RUN_ID = "arr_m34_001"
# A REAL experiment id from the historical evidence ledger, used as the source
# for hypotheses. Never a fabricated "exp_baseline".
SOURCE_EXPERIMENT_ID = "p0_demo_20261109"


def _real_evidence_available() -> tuple[bool, str]:
    """Read-only probe: can the REAL provider see REAL historical Evidence?

    Uses the SAME DiscoveryContextProvider + unified Evidence Reader as the E2E
    itself. A missing/empty view is an explicit gating failure, NOT a reason to
    substitute a synthetic Snapshot.
    """
    try:
        import tempfile
        probe_path = Path(tempfile.mkdtemp()) / "probe.sqlite"
        stores = ResearchStores(probe_path)
        view = DiscoveryContextProvider(stores=stores).build_snapshot(
            HISTORICAL_RUN_ID, objective_ids=[]).evidence_view
        n = int(view.get("n_evidence") or 0)
        if n <= 0:
            return False, f"no real evidence entities for {HISTORICAL_RUN_ID!r}"
        return True, f"{n} real evidence entities"
    except Exception as exc:  # noqa: BLE001 - gating probe must never raise
        return False, f"real evidence probe failed: {type(exc).__name__}: {exc}"


_SCIENT_PRESENT = bool(DEFAULT_SCINET_DIR) and os.path.isdir(str(DEFAULT_SCINET_DIR))
_RUN_REAL = os.environ.get("M4_E2E_01_RUN_REAL") == "1"
_EVIDENCE_OK, _EVIDENCE_REASON = _real_evidence_available()

pytestmark = pytest.mark.skipif(
    not (_SCIENT_PRESENT and _RUN_REAL and _EVIDENCE_OK),
    reason=("REAL SciNet++ E2E is opt-in and requires ALL of: "
            "M4_E2E_01_RUN_REAL=1, a SciNet++ checkout, and real historical "
            f"Evidence (SciNet++ present={_SCIENT_PRESENT}, "
            f"real evidence={_EVIDENCE_REASON}). Never falls back to a "
            "synthetic Snapshot."))


def _grant_via_cli(db_path: Path, approval_id: str) -> None:
    """Genuine OUT-OF-BAND human grant via the separate console process."""
    grant = subprocess.run(
        [sys.executable, "-m", "research_runtime.research_store.approval_cli",
         "--db", str(db_path), "grant", approval_id,
         "--approved-by", "alice", "--yes"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert grant.returncode == 0, (
        f"approval_cli grant failed rc={grant.returncode}\n"
        f"STDOUT={grant.stdout}\nSTDERR={grant.stderr}")


def test_real_engine_full_chain_writes_experience_and_round2_reads_it(tmp_path):
    # (P0-05 #1) ONE SQLite database AND the SAME ResearchStores INSTANCE for
    # EVERY trusted entity. The real service is built from db_path, then we use
    # service.stores directly for Provider / Discovery / Director, so there is
    # no second store object and no second database anywhere in the chain.
    db_path = tmp_path / "research.sqlite"
    service = build_default_research_service("scinet", db_path=db_path)
    stores = service.stores
    assert stores is service.stores

    # (P0-05 #3) REAL DiscoveryContextProvider over REAL historical Evidence.
    provider = DiscoveryContextProvider(stores=stores)
    snapshot1 = provider.build_snapshot(HISTORICAL_RUN_ID)
    assert snapshot1.evidence_view.get("n_evidence", 0) > 0, \
        "round-1 Snapshot must carry REAL historical Evidence (no synthetic view)"
    assert snapshot1.provenance.get("read_only") is True

    # (P0-05 #4) REAL ObjectiveDiscovery from that Snapshot, into the SAME store.
    discovery = ObjectiveDiscovery(
        generator=ObjectiveCandidateGenerator(domain="scinet"),
        critic=ObjectiveCritic(), ranker=ObjectiveRanker(), domain="scinet")
    res = discovery.discover(
        snapshot1, objective_store=stores.objectives, domain="scinet")
    assert res.proposed_objective_ids, "real Discovery must persist an Objective"
    oid = res.proposed_objective_ids[0]
    assert stores.objectives.get(oid) is not None, \
        "Proposed Objective must live in the SAME ResearchStores"
    # Evidence refs must point at REAL entities from the historical Evidence,
    # never a hand-made "sha256:neg1".
    assert all("sha256:neg1" not in r
               for c in res.ranked for r in c["evidence_refs"])

    # (P0-05 #5) The Director drives the REAL SciNet++ engine unchanged.
    director = ResearchDirector(
        service=service, objective_store=stores.objectives,
        state_dir=tmp_path / "director", objective_id=oid,
        source_experiment_id=SOURCE_EXPERIMENT_ID)
    director.run_to_awaiting_approval()
    assert director.context.session_id
    assert stores.sessions.get_session(director.context.session_id) is not None

    # (P0-05 #6) Genuine out-of-band approval_cli SUBPROCESS grant.
    _grant_via_cli(db_path, director.context.approval_id)
    approval = stores.approvals.get(director.context.approval_id)
    assert approval.status == "approved" and approval.source == "human"

    # (P0-05 #7) REAL Evaluation + Experience writeback.
    final = director.run_to_completion()
    assert final["state"] == "COMPLETED", final
    session_id = director.context.session_id
    rep = service.get_research_report(session_id=session_id)
    assert rep["ok"], rep
    report = rep["report"]
    assert report["experience_refs"], "REAL Experience must be recorded"
    assert report["frozen_test_used"] is False
    assert report["promotion"] == "not_requested"
    experience_rows = stores.sessions.experience(session_id)
    assert experience_rows, "REAL Experience rows must exist in the SAME store"
    assert any(r["experience_id"] in report["experience_refs"]
               for r in experience_rows), \
        "report refs must match real Experience Store rows"

    # (P0-05 #8/#9) ROUND-2: the SAME real Provider builds a second Snapshot
    # over the SAME store, and surfaces a candidate citing the EXACT round-1
    # Experience session id. No hand-built Snapshot, no synthetic Experience.
    snapshot2 = provider.build_snapshot(
        f"{HISTORICAL_RUN_ID}_round2", objective_ids=[oid])
    assert any(s.get("session_id") == session_id and s.get("experience")
               for s in snapshot2.sessions), \
        "round-2 Snapshot must surface round-1 REAL Experience"
    ref = f"experience:{session_id}"
    assert _resolve_ref(ref, snapshot2) is True
    assert _resolve_ref("experience:fabricated_session", snapshot2) is False

    res2 = discovery.discover(
        snapshot2, objective_store=stores.objectives, domain="scinet")
    exp_candidates = [c for c in res2.ranked
                      if any(r == ref for r in c["evidence_refs"])]
    assert exp_candidates, \
        "round-2 Discovery must cite the EXACT round-1 Experience session"
