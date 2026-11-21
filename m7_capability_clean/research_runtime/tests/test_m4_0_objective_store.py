"""M4.0-01 -- objectives table migration / persistence tests.

Uses a TEMPORARY SQLite file (injected via ResearchStores(db_path=...)), never
the real research.sqlite. All paths are Path-based / as_posix where compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_runtime.research_store import (
    ResearchStores, database_snapshot, describe_schema,
    from_research_objective, schema_structure,
)
from research_runtime.research_store.db import ResearchStoreConflict
from research_runtime.research_store.objective_store import (
    ObjectiveStore, REQUIRED_COLUMNS,
)
from research_runtime.schemas import ResearchObjective


def _objective(oid: str = "obj_probe", **kw) -> ResearchObjective:
    return ResearchObjective(
        objective_id=oid, domain=kw.pop("domain", "scinet"),
        description=kw.pop("description", "probe objective"),
        success_criteria=kw.pop("success_criteria", {"metric": "latent_mse@1"}),
        constraints=kw.pop("constraints", {"min_seeds": 5}),
        created_at="2026-11-14T00:00:00",
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "research.sqlite"


def test_migration_is_idempotent_and_structure_identical(db_path):
    """Acceptance A: two consecutive migrations succeed, structure identical."""
    s1 = ResearchStores(db_path)
    before = database_snapshot(db_path)
    assert describe_schema(db_path)["required_columns_present"] is True
    assert set(REQUIRED_COLUMNS) <= set(describe_schema(db_path)["column_names"])

    s2 = ResearchStores(db_path)          # SECOND migration, fresh connection
    after = database_snapshot(db_path)

    assert before["structure_signature"] == after["structure_signature"]
    assert schema_structure(db_path) == schema_structure(db_path)
    assert "objectives" in after["schema"]
    assert "objectives" in after["statistics"]
    assert s2.objectives is not None
    assert s1.objectives.all() == []


def test_existing_tables_are_preserved_by_migration(db_path):
    """Additive migration must not drop any P1-06 table or its columns."""
    ResearchStores(db_path)
    snap = database_snapshot(db_path)
    for t in ("hypothesis_versions", "plan_versions", "approvals",
              "research_sessions", "research_experience",
              "research_session_events", "campaign_bindings"):
        assert t in snap["schema"], f"existing table {t} lost"


def test_save_is_idempotent_and_conflict_is_fail_closed(db_path):
    stores = ResearchStores(db_path)
    rec = from_research_objective(
        _objective(), status="reconstructed",
        provenance={"basis": "m4_backfill", "source_path": "a/b.json"})
    assert stores.objectives.save(rec)["status"] == "created"
    assert stores.objectives.save(rec)["status"] == "unchanged"
    assert len(stores.objectives.all()) == 1
    got = stores.objectives.get("obj_probe")
    assert got is not None and got.status == "reconstructed"
    assert got.provenance["basis"] == "m4_backfill"
    assert got.constraints == {"min_seeds": 5}

    # Same id, different content -> must REFUSE (never overwrite history).
    changed = rec.model_copy(update={"description": "ALTERED"})
    with pytest.raises(ResearchStoreConflict):
        stores.objectives.save(changed)
    assert stores.objectives.get("obj_probe").description == "probe objective"


def test_content_sha256_is_stable_over_created_at(db_path):
    stores = ResearchStores(db_path)
    a = from_research_objective(_objective())
    b = from_research_objective(_objective().model_copy(
        update={"created_at": "2027-01-01T00:00:00"}))
    assert a.content_sha256 == b.content_sha256, \
        "created_at is metadata and must not change objective identity"


def test_parent_and_domain_filters(db_path):
    stores = ResearchStores(db_path)
    stores.objectives.save(from_research_objective(
        _objective("obj_parent", domain="scinet")))
    stores.objectives.save(from_research_objective(
        _objective("obj_child", domain="embodied"),
        parent_objective_id="obj_parent"))
    child = stores.objectives.get("obj_child")
    assert child.parent_objective_id == "obj_parent"
    assert [o.objective_id for o in stores.objectives.all(domain="embodied")] \
        == ["obj_child"]
    assert [o.objective_id for o in stores.objectives.all(status="proposed")] \
        == ["obj_child", "obj_parent"]


def test_transaction_rollback_leaves_no_partial_objective(db_path):
    """A failed write inside ResearchDatabase.session() must roll back."""
    stores = ResearchStores(db_path)
    with pytest.raises(RuntimeError):
        with stores.db.session() as conn:
            conn.execute(
                """INSERT INTO objectives
                   (objective_id, domain, description, constraints_json,
                    success_criteria_json, parent_objective_id, status,
                    origin_experience_id, content_sha256, created_at,
                    provenance_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                ("obj_rollback", "scinet", "x", "{}", "{}", None,
                 "proposed", None, "deadbeef", "2026-11-14", "{}"))
            raise RuntimeError("forced failure")
    assert stores.objectives.get("obj_rollback") is None
    assert describe_schema(db_path)["row_count"] == 0


def test_schema_snapshot_is_json_serialisable(db_path):
    ResearchStores(db_path)
    snap = database_snapshot(db_path)
    json.dumps(snap)                      # must be portable for the report
    assert isinstance(snap["statistics"], dict)
    assert snap["table_count"] == len(snap["schema"])
