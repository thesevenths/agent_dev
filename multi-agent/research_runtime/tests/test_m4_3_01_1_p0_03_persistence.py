"""M4.3-01.1 P0-03 -- durable, append-only ExperimentRegistry.

Verifies the spec's six required behaviours:
  * process restart recovery (incl. unfinished blocks)
  * duplicate Run refused
  * completed Run cannot be overwritten
  * event append order stable
  * SHA integrity verification
  * recovery after interruption creates NO duplicate campaign
plus: persisted Protocol/Snapshot SHA + all trusted IDs, real failure /
interruption / recovery events, independent export.
"""
from __future__ import annotations

import pytest

from research_runtime.m4_3 import (
    ARMS, BlockSpec, EvaluationHarness, ExperimentRegistry, FrozenProtocol,
    IndependentReviewer,
)
from research_runtime.m4_3.metrics import frozen_metric_contract
from research_runtime.m4_3.registry_store import (
    RegistryConflict, RegistryStore, snapshot_sha256,
)
from research_runtime.m4_3.research_service_gate import GateRunRef
from tests.rt_tests.m4_3_gate import build_gate_fixture


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("snap_1",), sample_size=0, seed=42,
        model_version="m1", prompt_version="p1",
        ranker_version="deterministic-v1",
        budget={"paired_seeds": [42, 43]}, exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks complete")


def _reviewer(rid: str = "r1") -> IndependentReviewer:
    def scorer(_b):
        return {d: 3 for d in frozen_metric_contract()["rubric_dimensions"]} | {
            "reason": "grounded, falsifiable, measurable"}
    return IndependentReviewer(reviewer_id=rid, scorer=scorer)


def _objective(oid: str) -> dict:
    return {"objective_id": oid, "description": "reduce prediction error",
            "evidence_refs": ["sha256:real"],
            "success_criteria": {"metric": "mse@1", "measurable": True}}


def _block(bid: str = "block_1") -> BlockSpec:
    return BlockSpec(block_id=bid, snapshot_id="snap_1", seed=42,
                     current_evidence={"n": 1})


@pytest.fixture
def gate():
    fx = build_gate_fixture()
    yield fx
    fx.close()


def _harness(gate, db_path) -> EvaluationHarness:
    return EvaluationHarness(
        protocol=_protocol(), reviewer=_reviewer(),
        gate_adapter=gate.adapter, default_gate_ref=gate.ref,
        db_path=db_path)


def _seed_one(h: EvaluationHarness, bid: str = "block_1"):
    h.add_block(_block(bid), candidate_budget=1, human_time_budget_s=1.0)
    for arm in ARMS:
        h.select_objective(bid, arm, objective=_objective(f"{bid}_{arm}"))


# ------------------------------------------------------- #1 restart recovery
def test_process_restart_recovers_blocks_and_runs(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h1 = _harness(gate, db)
    _seed_one(h1)
    # Only A is finalized -> block is UNFINISHED at "crash" time.
    h1.record_outcome("block_1", "A", outcome="valid_finding",
                      compute_units=2.0)
    assert h1.registry.unfinished_blocks() == ["block_1"]

    # Fresh process / fresh objects over the SAME db.
    store2 = RegistryStore(db_path=db)
    h2 = EvaluationHarness.recover(
        protocol=_protocol(), db_path=db,
        gate_adapter=gate.adapter, default_gate_ref=gate.ref)
    assert h2.registry.unfinished_blocks() == ["block_1"]
    assert set(h2.registry.blocks) == {"block_1"}
    assert len(h2.registry.all_runs()) == 3
    rec_a = h2.registry.runs["block_1_A"]
    assert rec_a.finalized and rec_a.outcome == "valid_finding"
    assert rec_a.protocol_sha256 == h1.registry.freeze_sha256
    assert rec_a.snapshot_sha256
    assert rec_a.campaign_id == gate.campaign_id
    assert rec_a.plan_id == gate.plan_id
    assert rec_a.approval_id == gate.approval_id
    assert rec_a.review_id.startswith("rev_")
    assert store2.verify_integrity() is True


def test_recovered_registry_can_finish_the_remaining_arms(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h1 = _harness(gate, db)
    _seed_one(h1)
    h1.record_outcome("block_1", "A", outcome="valid_finding")

    h2 = EvaluationHarness.recover(
        protocol=_protocol(), db_path=db,
        gate_adapter=gate.adapter, default_gate_ref=gate.ref)
    # B / C were registered before the crash but not finalized.
    for arm in ("B", "C"):
        r = h2.record_outcome("block_1", arm, outcome="valid_finding")
        assert r.finalized
    assert h2.registry.complete_blocks() == ["block_1"]
    assert h2.registry.unfinished_blocks() == []
    assert h2.verify_integrity() is True


# ------------------------------------------------------- #2 duplicate run
def test_duplicate_run_is_refused_in_memory_and_on_disk(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    with pytest.raises(ValueError):
        h.registry.register_run(block_id="block_1", arm="A")
    # A brand-new registry over the same DB must also refuse (real PK).
    h2 = EvaluationHarness.recover(
        protocol=_protocol(), db_path=db, gate_adapter=gate.adapter,
        default_gate_ref=gate.ref)
    with pytest.raises(ValueError):
        h2.registry.register_run(block_id="block_1", arm="A")
    store = RegistryStore(db_path=db)
    with pytest.raises(RegistryConflict):
        store.save_run(h.registry.runs["block_1_A"])


# ------------------------------------------------------- #3 no overwrite
def test_completed_run_cannot_be_overwritten(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    h.record_outcome("block_1", "A", outcome="valid_finding")
    with pytest.raises(RegistryConflict) as exc:
        h.record_outcome("block_1", "A", outcome="negative_finding")
    assert "already finalized" in str(exc.value)
    with pytest.raises(RegistryConflict):
        h.registry.update("block_1_A", outcome="failed")
    # Original outcome is intact in memory AND on disk.
    assert h.registry.runs["block_1_A"].outcome == "valid_finding"
    recovered = ExperimentRegistry.recover(protocol=_protocol(), db_path=db)
    assert recovered.runs["block_1_A"].outcome == "valid_finding"
    assert recovered.runs["block_1_A"].finalized is True


def test_finalize_is_exactly_once_even_with_direct_store(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    rec = h.registry.finalize("block_1_A", outcome="valid_finding")
    store = RegistryStore(db_path=db)
    with pytest.raises(RegistryConflict):
        store.finalize_run(rec)


# ------------------------------------------------------- #4 append order / #5 SHA
def test_events_are_append_only_and_ordered(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    h.record_outcome("block_1", "A", outcome="valid_finding")
    h.record_outcome("block_1", "B", outcome="failed")
    h.record_outcome("block_1", "C", outcome="interrupted")
    ev = h.registry.store.load_events()
    seqs = [e["seq"] for e in ev]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    types = [e["event_type"] for e in ev]
    assert types[0] == "block_created"
    assert "run_registered" in types and "review_recorded" in types
    assert "run_finalized" in types and "failed" in types \
        and "interrupted" in types
    # chain linkage
    for prev, cur in zip(ev, ev[1:]):
        assert cur["prev_sha256"] == prev["chain_sha256"]
    assert h.verify_integrity() is True


def test_sha_integrity_detects_tampering_and_truncation(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    h.record_outcome("block_1", "A", outcome="valid_finding")
    bundle = h.export()
    assert bundle["persisted"] is True
    assert bundle["export_sha256"]
    assert RegistryStore.verify_export(bundle) is True
    # Tamper the exported chain hash.
    tampered = {**bundle, "events": [dict(e) for e in bundle["events"]]}
    tampered["events"][1]["chain_sha256"] = "0" * 64
    assert RegistryStore.verify_export(tampered) is False
    # Truncate the exported chain.
    truncated = {**bundle, "events": bundle["events"][:-1]}
    assert RegistryStore.verify_export(truncated) is False
    # Tamper the live DB.
    with h.registry.store.db.session() as conn:
        conn.execute("UPDATE m4_3_events SET payload_sha256=? WHERE seq=1",
                     ("0" * 64,))
    assert h.verify_integrity() is False


def test_export_contains_all_trusted_ids_and_hashes(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    h.record_outcome("block_1", "A", outcome="valid_finding")
    exp = h.export()
    run = next(r for r in exp["runs"] if r["run_id"] == "block_1_A")
    for key in ("plan_id", "approval_id", "campaign_id", "report_id",
                "experience_id", "protocol_sha256", "snapshot_sha256",
                "objective_id", "review_id"):
        assert key in run
    assert run["plan_id"] == gate.plan_id
    assert run["campaign_id"] == gate.campaign_id
    assert run["approval_id"] == gate.approval_id
    assert run["protocol_sha256"] == h.registry.freeze_sha256
    # The snapshot SHA covers the FULL shared research opportunity, and is
    # identical on the block and on every run of that block.
    assert len(exp["blocks"][0]["snapshot_sha256"]) == 64
    assert exp["blocks"][0]["snapshot_sha256"] == run["snapshot_sha256"]
    assert snapshot_sha256({"snapshot_id": "snap_1", "seed": 42}) != \
        exp["blocks"][0]["snapshot_sha256"]   # full snapshot, not minimal id


# ------------------------------------------------------- #6 no duplicate campaign
def test_interruption_recovery_creates_no_duplicate_campaign(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h1 = _harness(gate, db)
    _seed_one(h1)
    h1.record_outcome("block_1", "A", outcome="interrupted")
    assert h1.registry.runs["block_1_A"].campaign_id == gate.campaign_id

    h2 = EvaluationHarness.recover(
        protocol=_protocol(), db_path=db, gate_adapter=gate.adapter,
        default_gate_ref=gate.ref)
    # The SAME (plan, approval, hash) triple still maps to the SAME campaign.
    ev = h2.gate_adapter.inspect(gate.ref)
    assert ev.gate_passed is True
    assert gate.ref.campaign_id == gate.campaign_id
    # Recovery cannot re-finalize A; no second campaign record is produced.
    with pytest.raises(RegistryConflict):
        h2.record_outcome("block_1", "A", outcome="valid_finding")
    campaigns = [e for e in h2.registry.store.load_events()
                 if e["event_type"] == "run_finalized"]
    assert len(campaigns) == 1
    assert h2.verify_integrity() is True
    # Recovery event is itself persisted and auditable.
    assert any(e["event_type"] == "registry_recovered"
               for e in h2.registry.store.load_events())


def test_failed_and_recovered_events_are_persisted(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    h.record_outcome("block_1", "A", outcome="failed")
    h.record_outcome("block_1", "B", outcome="interrupted")
    EvaluationHarness.recover(protocol=_protocol(), db_path=db,
                              gate_adapter=gate.adapter,
                              default_gate_ref=gate.ref)
    types = [e["event_type"] for e in h.registry.store.load_events()]
    assert "failed" in types and "interrupted" in types
    assert "registry_recovered" in types
    assert h.verify_integrity() is True


def test_review_rows_are_append_only_in_sqlite(gate, tmp_path):
    db = tmp_path / "registry.sqlite"
    h = _harness(gate, db)
    _seed_one(h)
    store = RegistryStore(db_path=db)
    rows = store.load_reviews()
    assert len(rows) == 3   # one per arm
    with pytest.raises(RegistryConflict):
        store.save_review(rows[0])
    assert len(store.load_reviews()) == 3
    assert store.verify_integrity() is True


def test_memory_only_registry_is_still_supported(gate):
    """No db -> engine-free in-memory mode (backwards compatible)."""
    h = EvaluationHarness(protocol=_protocol(), reviewer=_reviewer(),
                          gate_adapter=gate.adapter,
                          default_gate_ref=gate.ref)
    _seed_one(h)
    h.record_outcome("block_1", "A", outcome="valid_finding")
    assert h.export()["persisted"] is False
    assert h.verify_integrity() is False
    assert GateRunRef() is not None
