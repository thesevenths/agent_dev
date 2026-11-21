"""M4.0.1 P1-04 -- Discovery Evidence Adapter contract tests.

Enforces the READ-ONLY / no-verdict / no-fabrication contract between the M4.1
Discovery facade (DiscoveryEvidenceAdapter) and the legacy 12-step fail-closed
gate evidence source (default_evidence_provider).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from M4_0_runs.backfill import backfill
from research_runtime.discovery_evidence_adapter import (
    DiscoveryEvidenceAdapter, _LEGACY_GATE_PROVIDER,
    _self_source_uses_legacy_gate_provider,
)
from research_runtime.evidence_read_model import query_run_evidence
from research_runtime.tests.m4_0_seed_helper import build_equivalent_db

WS = Path(__file__).resolve().parents[2]
if str(WS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(WS))
RUN_ID = "arr_m34_001"


@pytest.fixture()
def real_env(tmp_path: Path):
    db = tmp_path / "research.sqlite"
    meta = build_equivalent_db(db, workspace=WS)
    assert meta["source"] in ("real_db_copy", "committed_p1_06_seed")
    # backfill so objective/candidate/campaign/evidence links exist; the
    # adapter itself reads the unified EvidenceView, independent of backfill,
    # but we reuse the same seed DB for realism.
    conn = __import__("sqlite3").connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    backfill(run_root=WS / "M3_4_runs" / RUN_ID, db_path=db, workspace=WS,
             report_path=tmp_path / "bf.json")
    return tmp_path, db


def test_adapter_delegates_to_canonical_unified_reader(real_env):
    tmp, db = real_env
    adapter = DiscoveryEvidenceAdapter(workspace=WS)
    # The adapter delegates to the SAME shared entry point that ResearchService
    # and UnifiedResearchScheduler use (proved equal in
    # test_m4_0_evidence_read_model.py), so Discovery, the service and the
    # scheduler all see one identical evidence set.
    assert adapter.query_run(RUN_ID) == query_run_evidence(RUN_ID, workspace=WS)


def test_adapter_returns_read_only_view_with_raw_verdicts_only(real_env):
    tmp, db = real_env
    view = DiscoveryEvidenceAdapter(workspace=WS).query_run(RUN_ID)
    # READ-ONLY contract: the view declares itself read-only and preserves raw
    # verdicts verbatim; it must NOT carry a recomputed gate verdict.
    assert "read-only" in view.get("note", "").lower()
    assert "raw verdict" in view.get("note", "").lower()
    # No gate-style decision key at the view level.
    forbidden = {"verdict", "decision", "accept", "reject", "gate",
                 "significant", "p_value"}
    assert not (set(view) & forbidden), set(view) & forbidden
    # The raw-verdict census is a REPORT, not a gate -- it is still present.
    assert "raw_verdict_census" in view
    assert isinstance(view["raw_verdict_census"], dict)


def test_adapter_does_not_fabricate_entities(real_env):
    tmp, db = real_env
    entities = DiscoveryEvidenceAdapter(workspace=WS).discover_entities(RUN_ID)
    assert entities, "expected evidence entities on real history"
    # NO FABRICATION: every returned entity must be backed by >=1 REAL carrier
    # that was actually read; the adapter never invents an entity.
    for e in entities:
        assert e.get("carriers"), f"fabricated entity with no carrier: {e['evidence_id']}"
    # Missing Experience must stay UNRESOLVED: the adapter never synthesises an
    # Experience evidence entity. No entity may claim an 'experience' carrier.
    assert not any("experience" in (e.get("carriers") or [])
                   for e in entities)


def test_adapter_never_routes_through_legacy_gate_provider():
    # Static contract: the Discovery facade must NOT use
    # default_evidence_provider (the 12-step gate's legacy evidence source).
    assert not _self_source_uses_legacy_gate_provider(), (
        f"{_LEGACY_GATE_PROVIDER} must not appear in DiscoveryEvidenceAdapter")
    src = inspect.getsource(DiscoveryEvidenceAdapter)
    assert _LEGACY_GATE_PROVIDER not in src


def test_adapter_is_idempotent_and_deterministic(real_env):
    tmp, db = real_env
    adapter = DiscoveryEvidenceAdapter(workspace=WS)
    first = adapter.query_run(RUN_ID)
    second = adapter.query_run(RUN_ID)
    assert first == second
