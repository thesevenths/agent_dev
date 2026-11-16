"""M4.3-02.1 P0-01 -- paired eligibility unified across analysis & report.

ENGINE-FREE: real ExperimentRegistry in SQLite. Verifies that a block enters
paired statistics ONLY when all three arms are finalized, successful, share one
snapshot and one protocol version, and that the SAME rule drives both the pilot
analysis and the value report.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from research_runtime.m4_3 import ExperimentRegistry, FrozenProtocol
from research_runtime.m4_3.registry_store import RegistryStore
from research_runtime.m4_3.paired_eligibility import (
    ELIGIBILITY_RULES, select_paired_blocks,
)
from research_runtime.m4_3.pilot_analysis import analyze_pilot, load_runs
from research_runtime.m4_3.value_report import generate_value_report
from research_runtime.research_store.db import ResearchDatabase


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        snapshot_ids=("s1", "s2", "s3"), sample_size=0, seed=42,
        model_version="m", prompt_version="p", ranker_version="r",
        paired_seeds=(42, 43), exclusion_rules=("none",),
        statistical_method="paired-block", stopping_rule="all blocks")


def _finalize(reg, bid, arm, *, outcome="negative_finding", finalized=True,
              snapshot_id=None, protocol_sha256=None):
    rid = f"{bid}_{arm}"
    changes = dict(campaign_id=f"c_{rid}", plan_id=f"p_{rid}",
                   approval_id=f"a_{rid}", report_id=f"r_{rid}",
                   experience_id=f"e_{rid}", gate_steps_passed=12,
                   external_approval=True, compute_units=1.0,
                   wall_time_s=10.0, paired_seeds=[42, 43],
                   gate_evidence={"gate_passed": True})
    if snapshot_id is not None:
        changes["snapshot_id"] = snapshot_id
    if protocol_sha256 is not None:
        changes["protocol_sha256"] = protocol_sha256
    reg.finalize(rid, outcome=outcome, **changes)


def _three_arms(reg, bid, *, finalize_a=True, finalize_b=True, finalize_c=True,
                outcomes=None, snapshots=None, protocols=None):
    outcomes = outcomes or {}
    snapshots = snapshots or {}
    protocols = protocols or {}
    reg.create_block(block_id=bid, snapshot_id=f"s{1}", seed=41)
    for arm in ("A", "B", "C"):
        reg.register_run(block_id=bid, arm=arm)
    if finalize_a:
        _finalize(reg, bid, "A", outcome=outcomes.get("A", "negative_finding"),
                  snapshot_id=snapshots.get("A"),
                  protocol_sha256=protocols.get("A"))
    if finalize_b:
        _finalize(reg, bid, "B", outcome=outcomes.get("B", "negative_finding"),
                  snapshot_id=snapshots.get("B"),
                  protocol_sha256=protocols.get("B"))
    if finalize_c:
        _finalize(reg, bid, "C", outcome=outcomes.get("C", "negative_finding"),
                  snapshot_id=snapshots.get("C"),
                  protocol_sha256=protocols.get("C"))


def _block_from_db(registry, bid):
    for b in RegistryStore(db=registry.store.db).load_blocks():
        if b["block_id"] == bid:
            return b
    return None


# --------------------------------------------------------------- rule 1/2/3
def test_all_three_finalized_arms_enter_paired_analysis(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    _three_arms(reg, "b1")
    blocks = RegistryStore(db=db).load_blocks()
    paired = select_paired_blocks(runs=load_runs(db), blocks=blocks)
    assert len(paired["eligible"]) == 1
    assert paired["eligible"][0]["block_id"] == "b1"
    assert paired["excluded"] == []
    assert tuple(paired["rules"]) == tuple(ELIGIBILITY_RULES)


def test_one_unfinalized_arm_excludes_the_whole_block(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    _three_arms(reg, "b2", finalize_b=False)
    blocks = RegistryStore(db=db).load_blocks()
    paired = select_paired_blocks(runs=load_runs(db), blocks=blocks)
    assert paired["eligible"] == []
    assert paired["excluded"][0]["block_id"] == "b2"
    assert any("not finalized" in r for r in paired["excluded"][0]["reasons"])


def test_missing_arm_excludes_the_block(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    _three_arms(reg, "b3", finalize_c=False)
    # unregister C by deleting its persisted row from the in-memory view: the
    # shared selector only sees runs, so remove it via direct DB is complex;
    # instead register only A and B in a fresh block.
    reg2 = ExperimentRegistry(protocol=_protocol(), db=ResearchDatabase(
        tmp_path / "r2.sqlite"))
    reg2.create_block(block_id="b4", snapshot_id="s1", seed=41)
    reg2.register_run(block_id="b4", arm="A")
    reg2.register_run(block_id="b4", arm="B")
    _finalize(reg2, "b4", "A")
    _finalize(reg2, "b4", "B")
    blocks2 = RegistryStore(db=reg2.store.db).load_blocks()
    paired = select_paired_blocks(runs=load_runs(reg2.store.db), blocks=blocks2)
    assert paired["eligible"] == []
    assert any("arm missing: C" in r for r in paired["excluded"][0]["reasons"])


def test_finalized_but_failed_outcome_is_not_a_paired_observation(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    _three_arms(reg, "b5", outcomes={"C": "invalid"})
    blocks = RegistryStore(db=db).load_blocks()
    paired = select_paired_blocks(runs=load_runs(db), blocks=blocks)
    assert paired["eligible"] == []
    assert any("successful scientific outcome" in r
               for r in paired["excluded"][0]["reasons"])


# ------------------------------------------------------------- rule 5 / 6
def _diverge(runs, block_id, arm, **changes):
    """Return runs with one arm replaced (in-memory) for rule-level tests."""
    out = []
    for r in runs:
        if r.block_id == block_id and r.arm == arm:
            r = replace(r, **changes)
        out.append(r)
    return out


def test_cross_snapshot_pairing_is_refused(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    canonical_snap = "s1"
    reg.create_block(block_id="bx", snapshot_id=canonical_snap, seed=41)
    for arm in ("A", "B", "C"):
        reg.register_run(block_id="bx", arm=arm)
    for arm in ("A", "B", "C"):
        _finalize(reg, "bx", arm)
    blocks = RegistryStore(db=db).load_blocks()
    runs = _diverge(load_runs(db), "bx", "C",
                    snapshot_id="a-different-snapshot")
    paired = select_paired_blocks(runs=runs, blocks=blocks)
    assert paired["eligible"] == []
    assert any("snapshot" in r for r in paired["excluded"][0]["reasons"])


def test_cross_protocol_version_pairing_is_refused(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    reg.create_block(block_id="bp", snapshot_id="s1", seed=41)
    for arm in ("A", "B", "C"):
        reg.register_run(block_id="bp", arm=arm)
    for arm in ("A", "B", "C"):
        _finalize(reg, "bp", arm)
    blocks = RegistryStore(db=db).load_blocks()
    runs = _diverge(load_runs(db), "bp", "C",
                    protocol_sha256="a-different-protocol-version")
    paired = select_paired_blocks(runs=runs, blocks=blocks)
    assert paired["eligible"] == []
    assert any("protocol" in r for r in paired["excluded"][0]["reasons"])


# ------------------------------------------------- analysis == report paired n
def test_analysis_and_value_report_agree_on_paired_n(tmp_path):
    db = ResearchDatabase(tmp_path / "r.sqlite")
    reg = ExperimentRegistry(protocol=_protocol(), db=db)
    for i, bid in enumerate(("b1", "b2", "b3"), start=1):
        _three_arms(reg, bid)
    # a non-finalized block that must be excluded by BOTH
    reg.create_block(block_id="b4", snapshot_id="s1", seed=44)
    for arm in ("A", "B", "C"):
        reg.register_run(block_id="b4", arm=arm)
    _finalize(reg, "b4", "A")
    _finalize(reg, "b4", "B")

    analysis = analyze_pilot(db=db, out_dir=tmp_path / "a", write=False)
    report = generate_value_report(db=db, out_dir=tmp_path / "v",
                                   stage="pilot", write=False)

    assert analysis["data_integrity"]["n_blocks_eligible"] == 3
    assert report["data_integrity"]["n_blocks_eligible_for_pairing"] == 3
    assert len(analysis["data_integrity"]["excluded_blocks"]) == 1
    assert len(report["data_integrity"]["excluded_blocks"]) == 1
    # The paired n for the primary contrast must be identical in both.
    a_n = analysis["paired_contrasts"]["C-B"]["metrics"][
        "valid_research_findings_per_compute_budget"]["n"]
    r_n = report["primary"]["contrasts"][
        "valid_research_findings_per_compute_budget"]["contrasts"]["C-B"]["n"]
    assert a_n == r_n == 3
