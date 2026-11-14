"""M4.0-02 -- Entity Link Store tests (temp SQLite, never the real DB)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_runtime.research_store import ResearchStores, database_snapshot
from research_runtime.research_store.db import ResearchStoreConflict
from research_runtime.research_store.entity_link_store import (
    UNRESOLVED_SENTINEL, EntityLinkStore, sha256_file,
)


@pytest.fixture()
def env(tmp_path: Path):
    db = tmp_path / "research.sqlite"
    stores = ResearchStores(db)
    links = EntityLinkStore(stores.db, workspace=tmp_path)
    return tmp_path, db, stores, links


def _add(links, **kw):
    base = dict(run_id="run_A", objective_id="obj_A",
                src_type="objective", src_id="obj_A",
                dst_type="candidate", dst_id="cand_1",
                relation="drives", basis="explicit_id")
    base.update(kw)
    return links.add_link(**base)


def test_migration_adds_entity_links_idempotently(env):
    _, db, stores, _ = env
    a = database_snapshot(db)
    assert "entity_links" in a["schema"]
    ResearchStores(db)                      # second migration
    b = database_snapshot(db)
    assert a["structure_signature"] == b["structure_signature"]


def test_add_link_is_deduplicated_but_allows_multiple_relations(env):
    _, _, _, links = env
    assert _add(links)["status"] == "created"
    assert _add(links)["status"] == "duplicate"     # exact same link
    # SAME entity, DIFFERENT relation / dst -> allowed
    assert _add(links, relation="evaluated_by", dst_type="campaign",
                dst_id="camp_1")["status"] == "created"
    assert _add(links, dst_id="cand_2")["status"] == "created"
    assert len(links.find_by_objective("obj_A", run_id="run_A")) == 3


def test_forward_and_reverse_queries(env):
    _, _, _, links = env
    _add(links)
    _add(links, src_type="candidate", src_id="cand_1",
         dst_type="campaign", dst_id="camp_1", relation="evaluated_in")
    out = links.find_by_entity("candidate", "cand_1", run_id="run_A", direction="out")
    assert [l["dst_id"] for l in out] == ["camp_1"]
    rev = links.find_by_entity("candidate", "cand_1", run_id="run_A", direction="in")
    assert [l["src_id"] for l in rev] == ["obj_A"]
    both = links.find_by_entity("candidate", "cand_1", run_id="run_A")
    assert len(both) == 2
    # Reverse query from a campaign back to the candidate.
    assert links.find_by_entity("campaign", "camp_1", run_id="run_A",
                                direction="in")[0]["src_id"] == "cand_1"


def test_cross_run_isolation(env):
    _, _, _, links = env
    _add(links)
    _add(links, run_id="run_B", objective_id="obj_B", src_id="obj_B",
         dst_id="cand_B")
    assert len(links.find_by_entity("candidate", "cand_1", run_id="run_A")) == 1
    assert links.find_by_entity("candidate", "cand_B", run_id="run_A") == []
    assert links.find_by_objective("obj_B", run_id="run_A") == []
    g = links.list_objective_graph("obj_A", run_id="run_A")
    assert all(e["run_id"] if "run_id" in e else True for e in g["edges"])
    assert g["n_links"] == 1
    assert "cand_B" not in json.dumps(g)


def test_sha_mismatch_fails_closed_and_match_succeeds(env):
    tmp, _, _, links = env
    src = tmp / "evidence" / "e.json"
    src.parent.mkdir(exist_ok=True)
    src.write_text('{"verdict":"inconclusive"}', encoding="utf-8")
    good = sha256_file(src)
    assert _add(links, source_path="evidence/e.json", source_sha256=good,
                src_type="candidate", src_id="cand_1", dst_type="evidence",
                dst_id="ev_1", relation="produces")["status"] == "created"
    with pytest.raises(ResearchStoreConflict):
        _add(links, source_path="evidence/e.json", source_sha256="0" * 64,
             src_type="candidate", src_id="cand_2", dst_type="evidence",
             dst_id="ev_2", relation="produces")
    # validate_links also detects a file mutated AFTER insertion.
    src.write_text('{"verdict":"TAMPERED"}', encoding="utf-8")
    v = links.validate_links(run_id="run_A", workspace=tmp)
    assert v["ok"] is False
    assert v["n_sha_checked"] == 1
    assert any("SHA mismatch" in x["reason"] for x in v["violations"])


def test_unresolved_requires_sentinel_and_reason(env):
    _, _, _, links = env
    with pytest.raises(ValueError):
        _add(links, status="unresolved", dst_id="guessed_obj", notes="x")
    with pytest.raises(ValueError):
        _add(links, status="unresolved", dst_id=UNRESOLVED_SENTINEL, notes="")
    assert _add(links, status="unresolved", dst_id=UNRESOLVED_SENTINEL,
                dst_type="hypothesis", relation="hypothesis_unknown",
                notes="no explicit hypothesis_id in any historical source"
                )["status"] == "created"
    v = links.validate_links(run_id="run_A")
    assert v["ok"] is True and v["n_unresolved"] == 1
    g = links.list_objective_graph("obj_A", run_id="run_A")
    assert g["n_unresolved"] == 1
    assert "no explicit hypothesis_id" in g["unresolved"][0]


def test_validate_links_clean_and_types_checked(env):
    _, _, _, links = env
    _add(links, basis="explicit_id")
    assert links.validate_links(run_id="run_A")["ok"] is True
    with pytest.raises(ValueError):
        _add(links, src_type="NOT_A_TYPE")
    with pytest.raises(ValueError):
        _add(links, status="bogus")


def test_transaction_rollback_keeps_no_partial_link(env):
    _, _, stores, links = env
    _add(links)
    with pytest.raises(RuntimeError):
        with stores.db.session() as conn:
            conn.execute(
                """INSERT INTO entity_links
                   (run_id, objective_id, src_type, src_id, dst_type, dst_id,
                    relation, status, basis, source_path, source_sha256,
                    notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("run_A", "obj_A", "candidate", "cand_x", "evidence", "ev_x",
                 "produces", "confirmed", "explicit_id", None, None, "", "t"))
            raise RuntimeError("forced failure")
    assert links.find_by_entity("candidate", "cand_x") == []
    assert links.stats(run_id="run_A") == {"confirmed": 1}


def test_source_path_is_posix_on_windows(env):
    """Windows/Linux path compatibility: stored paths are always POSIX."""
    _, _, _, links = env
    _add(links, source_path=r"evidence\sub\e.json", source_sha256=None,
         verify_source_sha=False, src_type="candidate", src_id="c1",
         dst_type="evidence", dst_id="ev1", relation="produces")
    row = links.find_by_entity("candidate", "c1")[0]
    assert row["source_path"] == "evidence/sub/e.json"
    assert "\\" not in row["source_path"]


def test_graph_is_deterministic_order(env):
    _, _, _, links = env
    for i in (3, 1, 2):
        _add(links, dst_id=f"cand_{i}")
    g1 = links.list_objective_graph("obj_A", run_id="run_A")
    g2 = links.list_objective_graph("obj_A", run_id="run_A")
    assert g1 == g2
    assert g1["nodes"]["candidate"] == sorted(g1["nodes"]["candidate"])


def test_same_entity_id_is_isolated_by_objective(env):
    """M4.0-05: three domains reuse the SAME hyp_fingerprint. Preserving the
    original ID must not mix their chains; objective_id is the boundary."""
    _, _, _, links = env
    shared = "candidate:domain_model::train.lr|None->0.003|X"
    for domain in ("scinet", "embodied", "quant"):
        oid = f"obj_run_A_{domain}"
        _add(links, objective_id=oid, src_id=oid, dst_id=shared,
             relation="drives")
        _add(links, objective_id=oid, src_type="candidate", src_id=shared,
             dst_type="campaign", dst_id=f"camp_{domain}",
             relation="evaluated_in")
        _add(links, objective_id=oid, src_type="candidate", src_id=shared,
             dst_type="hypothesis", dst_id=UNRESOLVED_SENTINEL,
             relation="hypothesis_unknown", status="unresolved",
             basis="no_explicit_id", notes="no explicit hypothesis id")
    all_links = links.find_by_entity("candidate", shared, run_id="run_A")
    assert len(all_links) == 9           # 3 domains x 3 relations
    scoped = links.find_by_entity("candidate", shared, run_id="run_A",
                                  objective_id="obj_run_A_scinet")
    assert len(scoped) == 3
    assert {l["objective_id"] for l in scoped} == {"obj_run_A_scinet"}
    assert [l["dst_id"] for l in scoped if l["dst_type"] == "campaign"] \
        == ["camp_scinet"]
    g = links.list_objective_graph("obj_run_A_quant", run_id="run_A")
    assert g["n_links"] == 3 and g["n_unresolved"] == 1


def test_same_entity_id_is_isolated_by_objective(env):
    """M4.0-05: three domains reuse the SAME hyp_fingerprint. Preserving the
    original ID must not mix their chains; objective_id is the boundary."""
    _, _, _, links = env
    shared = "candidate:domain_model::train.lr|None->0.003|X"
    for domain in ("scinet", "embodied", "quant"):
        oid = f"obj_run_A_{domain}"
        _add(links, objective_id=oid, src_id=oid, dst_id=shared,
             relation="drives")
        _add(links, objective_id=oid, src_type="candidate", src_id=shared,
             dst_type="campaign", dst_id=f"camp_{domain}",
             relation="evaluated_in")
        _add(links, objective_id=oid, src_type="candidate", src_id=shared,
             dst_type="hypothesis", dst_id=UNRESOLVED_SENTINEL,
             relation="hypothesis_unknown", status="unresolved",
             basis="no_explicit_id", notes="no explicit hypothesis id")
    all_links = links.find_by_entity("candidate", shared, run_id="run_A")
    assert len(all_links) == 9           # 3 domains x 3 relations
    scoped = links.find_by_entity("candidate", shared, run_id="run_A",
                                  objective_id="obj_run_A_scinet")
    assert len(scoped) == 3
    assert {l["objective_id"] for l in scoped} == {"obj_run_A_scinet"}
    assert [l["dst_id"] for l in scoped if l["dst_type"] == "campaign"] \
        == ["camp_scinet"]
    g = links.list_objective_graph("obj_run_A_quant", run_id="run_A")
    assert g["n_links"] == 3 and g["n_unresolved"] == 1
