"""M7 -- append-only scientific discovery schema (single existing ResearchDatabase).

All tables carry PK, SHA, version, provenance, status, created_at; mutable
projections use updated_at but original events are never modified. No parallel
database is created -- this reuses the M5/M6 ResearchDatabase connection.
"""
from __future__ import annotations

from typing import Any

M7_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m7_literature_papers (
    paper_id TEXT PRIMARY KEY,
    doi TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    metadata_sha TEXT NOT NULL,
    content_sha TEXT NOT NULL,
    source TEXT NOT NULL,
    query_id TEXT NOT NULL,
    year INTEGER NOT NULL DEFAULT 0,
    citations INTEGER NOT NULL DEFAULT 0,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_search_runs (
    run_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    query_sha TEXT NOT NULL,
    n_returned INTEGER NOT NULL,
    success INTEGER NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_research_questions (
    question_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    title TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    evidence_sha TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_value_assessments (
    assessment_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    scores TEXT NOT NULL,
    info_gain REAL NOT NULL,
    priority REAL NOT NULL,
    evaluator_version TEXT NOT NULL,
    evidence_sha TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    falsification_sha TEXT NOT NULL,
    review_sha TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_experiment_protocols (
    protocol_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    prereg_sha TEXT NOT NULL,
    family_manifest_sha TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    mme REAL NOT NULL,
    power_report_sha TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_scientific_claims (
    claim_id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_claim_relations (
    relation_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    evidence_sha TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_verified_memory (
    memory_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    evidence_shas TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    source_sha TEXT NOT NULL,
    confidence REAL NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    applicability TEXT NOT NULL,
    limitations TEXT NOT NULL,
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS m7_discovery_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload_sha TEXT NOT NULL,
    harness_sha TEXT NOT NULL DEFAULT '',
    experiment_sha TEXT NOT NULL DEFAULT '',
    campaign_id TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL,
    status TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def install_m7_schema(db: Any) -> dict[str, Any]:
    """Create all M7 tables on the EXISTING ResearchDatabase (idempotent)."""
    with db.session() as conn:
        for stmt in M7_SCHEMA_SQL.split(";"):
            if stmt.strip():
                conn.execute(stmt)
    return {"ok": True, "tables": M7_SCHEMA_SQL.count("CREATE TABLE")}


__all__ = ["M7_SCHEMA_SQL", "install_m7_schema"]
