"""M4.0-05b -- P1-03 SHA-verification statistics contract tests.

Verifies that validate_integrity.py reports three NON-conflated SHA metrics:
  * n_unique_artifact_sha_verified        -- distinct sha256 of real files
  * n_link_sha_verified                   -- distinct verified evidence links
  * n_candidate_evidence_associations_verified -- per (candidate, evidence)
and that the historical n_evidence_sha_verified is retained (== associations)
without over-counting physically shared evidence files.

Works on the REAL arr_m34_001 history (via the committed/real seed DB) AND on a
synthetic multi-domain run that shares files across domains, so the dedup
semantics are exercised directly against validate().
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from M4_0_runs.backfill import backfill
from M4_0_runs.validate_integrity import validate
from research_runtime.research_store import ResearchStores
from research_runtime.research_store.entity_link_store import EntityLinkStore
from research_runtime.tests.m4_0_seed_helper import build_equivalent_db

WS = Path(__file__).resolve().parents[2]      # multi-agent/
if str(WS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(WS))
RUN_ROOT = WS / "M3_4_runs" / "arr_m34_001"
M34_RUN_ID = "arr_m34_001"


@pytest.fixture()
def real_env(tmp_path: Path):
    db = tmp_path / "research.sqlite"
    meta = build_equivalent_db(db, workspace=WS)
    assert meta["source"] in ("real_db_copy", "committed_p1_06_seed")
    # Start from a CLEAN relation layer while KEEPING the P1-06 historical rows.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    backfill(run_root=RUN_ROOT, db_path=db, workspace=WS,
             report_path=tmp_path / "bf.json")
    return tmp_path, db


def _recompute_metrics(db: Path, run_id: str) -> dict[str, int]:
    """Independently recompute the three P1-03 metrics from entity_links."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    # Mirror validate(): only campaign->evidence links count toward the three
    # P1-03 SHA metrics (candidate->evidence identity_confirmed_by links point
    # at the same files and are a DISTINCT relation, not re-counted here).
    rows = conn.execute(
        "SELECT src_type, src_id, dst_type, dst_id, source_path, "
        "source_sha256 FROM entity_links "
        "WHERE src_type='campaign' AND dst_type='evidence' AND run_id=?",
        (run_id,)).fetchall()
    verified_keys: set[tuple] = set()
    verified_shas: set[str] = set()
    for r in rows:
        if not r["source_sha256"] or not r["source_path"]:
            continue
        if hashlib.sha256(
                (WS / r["source_path"]).read_bytes()).hexdigest() == \
                r["source_sha256"]:
            key = (r["src_type"], r["src_id"], r["dst_type"],
                   r["dst_id"], r["source_path"])
            verified_keys.add(key)
            verified_shas.add(r["dst_id"].removeprefix("sha256:"))
    # candidate-evidence associations: per (candidate) count the verified
    # campaign->evidence links of its domain (== pre-P1-03 n_evidence_sha_verified).
    obj_cands = conn.execute(
        "SELECT objective_id, dst_id FROM entity_links "
        "WHERE src_type='objective' AND dst_type='candidate' "
        "AND relation='drives' AND run_id=?", (run_id,)).fetchall()
    n_cands_per_obj: dict[str, int] = {}
    for r in obj_cands:
        n_cands_per_obj[r["objective_id"]] = \
            n_cands_per_obj.get(r["objective_id"], 0) + 1
    assoc = 0
    for oid, n_cands in n_cands_per_obj.items():
        domain = oid.split("_")[-1]
        camp_links = conn.execute(
            "SELECT COUNT(*) AS c FROM entity_links "
            "WHERE src_type='campaign' AND dst_type='evidence' "
            "AND objective_id=? AND run_id=?", (oid, run_id)).fetchone()["c"]
        # campaign->evidence links that actually verified (sha matches)
        cl_rows = conn.execute(
            "SELECT source_path, source_sha256 FROM entity_links "
            "WHERE src_type='campaign' AND dst_type='evidence' "
            "AND objective_id=? AND run_id=?", (oid, run_id)).fetchall()
        verified_camp = 0
        for cl in cl_rows:
            if cl["source_sha256"] and cl["source_path"] and \
                    hashlib.sha256(
                        (WS / cl["source_path"]).read_bytes()).hexdigest() == \
                    cl["source_sha256"]:
                verified_camp += 1
        assoc += verified_camp * n_cands
    conn.close()
    return {
        "n_link_sha_verified": len(verified_keys),
        "n_unique_artifact_sha_verified": len(verified_shas),
        "n_candidate_evidence_associations_verified": assoc,
    }


def test_sha_stat_three_way_split_present_and_ordered(real_env):
    tmp, db = real_env
    rep = validate(db_path=db, workspace=WS, out=tmp / "integ.json")
    for k in ("n_unique_artifact_sha_verified", "n_link_sha_verified",
              "n_candidate_evidence_associations_verified",
              "n_evidence_sha_verified", "n_evidence_sha_verified_note"):
        assert k in rep, f"missing P1-03 report field {k}"
    u, lk, a = (rep["n_unique_artifact_sha_verified"],
                rep["n_link_sha_verified"],
                rep["n_candidate_evidence_associations_verified"])
    # strict non-conflated ordering: unique files <= link rows <= associations
    assert u <= lk <= a, (u, lk, a)
    # the historical aggregate equals the association count (backward compat)
    assert rep["n_evidence_sha_verified"] == a


def test_sha_stat_three_way_split_recomputable(real_env):
    tmp, db = real_env
    rep = validate(db_path=db, workspace=WS, out=tmp / "integ.json")
    recomputed = _recompute_metrics(db, M34_RUN_ID)
    for k in ("n_link_sha_verified", "n_unique_artifact_sha_verified",
              "n_candidate_evidence_associations_verified"):
        assert rep[k] == recomputed[k], (k, rep[k], recomputed[k])


def test_shared_evidence_file_across_domains_is_deduped(tmp_path: Path):
    """Synthetic 2-domain run where evidence_index.json and scheduler_state.json
    are SHARED across both domains (same file, same sha) while domain_summary.json
    and candidate_registry.jsonl are domain-specific. validate() must report the
    shared files ONCE in the unique-artifact count, but each campaign->evidence
    link row is still counted, and each (candidate, evidence) association still
    counts them per candidate -- proving the historical over-count is gone from
    the artifact metric while links/associations retain their own semantics."""
    run = "syn_sha_x"
    root = tmp_path / "M3_4_runs" / run
    root.mkdir(parents=True, exist_ok=True)
    domains = ["scinet", "embodied"]
    ev_index = {"run_id": run, "domains": {}}
    for d in domains:
        ev_index["domains"][d] = {
            "campaign_id": f"{run}_{d}", "dataset_version": "v1",
            "decisions": {"fp_a": {"decision": "inconclusive"},
                          "fp_b": {"decision": "inconclusive"}},
            "ood_provenance": {"ood_dataset": {"dataset_version": "o1"},
                               "ood_metrics": {"artifacts": []}}}
    (root / "evidence_index.json").write_text(
        json.dumps(ev_index, sort_keys=True), encoding="utf-8")
    (root / "scheduler_state.json").write_text("{}", encoding="utf-8")
    for d in domains:
        (root / d).mkdir()
        # DIFFERENT content per domain so their shas differ (not collapsed).
        (root / d / "domain_summary.json").write_text(
            json.dumps({"records": [{"decision": "inconclusive"}],
                        "domain": d}), encoding="utf-8")
        (root / d / "candidate_registry.jsonl").write_text(
            (json.dumps({"event": "register", "fingerprint": "fp_a",
                         "domain": d}) + "\n"),
            encoding="utf-8")

    db = tmp_path / "research.sqlite"
    ResearchStores(db)                      # create ALL tables idempotently
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM entity_links")
    conn.execute("DELETE FROM objectives")
    conn.commit()
    conn.close()
    backfill(run_root=root, db_path=db, workspace=tmp_path,
             report_path=tmp_path / "bf.json")
    rep = validate(db_path=db, workspace=tmp_path,
                   out=tmp_path / "integ.json", run_id=run)

    # 2 domains x 4 campaign->evidence files = 8 link rows.
    # Shared across domains: evidence_index.json (1 file / 2 links),
    # scheduler_state.json (1 file / 2 links). Domain-specific:
    # domain_summary.json (2 distinct), candidate_registry.jsonl (2 distinct)
    # -> 6 unique files.
    assert rep["n_link_sha_verified"] == 8, rep["n_link_sha_verified"]
    assert rep["n_unique_artifact_sha_verified"] == 6, \
        rep["n_unique_artifact_sha_verified"]
    # 2 candidates/domain x 4 verified/domain x 2 domains = 16 associations.
    assert rep["n_candidate_evidence_associations_verified"] == 16, \
        rep["n_candidate_evidence_associations_verified"]
    assert rep["n_evidence_sha_verified"] == 16
    # The dedup property: the SAME physical file is NOT reported as many files.
    assert rep["n_unique_artifact_sha_verified"] < rep["n_link_sha_verified"]
    assert rep["n_link_sha_verified"] < \
        rep["n_candidate_evidence_associations_verified"]
