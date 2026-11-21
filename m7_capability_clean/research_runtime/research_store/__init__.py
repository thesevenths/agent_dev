"""P1-06 trusted research state (A1).

    HypothesisStore        hypothesis versions + REAL Critic verdicts
    ExperimentPlanStore    versioned, SHA-256-addressed ExperimentPlans
    ApprovalStore          human / system-policy decisions bound to plan hashes
    ResearchSessionStore   objective context, campaign idempotency, reports,
                           append-only structured experience

All four share ONE SQLite database and ONE transaction helper, so a check plus
its write is atomic across stores. Nothing here imports an engine, an LLM, a
tool or a campaign runner; this layer stores and verifies, it never executes.

The tool surface may call ``request()`` (creates ``pending`` only). ``grant()``
is reserved for the out-of-band console in ``approval_cli.py``.
"""
from __future__ import annotations

from pathlib import Path

from ..settings import DATA_DIR
from .approval_store import ApprovalStore, SCHEMA_SQL as APPROVAL_SCHEMA
from .db import (
    ResearchDatabase,
    ResearchStoreConflict,
    ResearchStoreError,
    canonical_bytes,
    content_sha256,
)
from .experiment_plan_store import (
    ExperimentPlanStore,
    SCHEMA_SQL as PLAN_SCHEMA,
    canonical_plan,
    plan_sha256,
)
from .hypothesis_store import (
    HypothesisStore,
    SCHEMA_SQL as HYPOTHESIS_SCHEMA,
    canonical_hypothesis,
    hypothesis_sha256,
)
from .objective_store import (
    ObjectiveStore, SCHEMA_SQL as OBJECTIVE_SCHEMA, database_snapshot,
    describe_schema, from_research_objective, schema_structure,
)
from .entity_link_store import (
    EntityLinkStore, SCHEMA_SQL as ENTITY_LINK_SCHEMA, ensure_link_schema,
    sha256_file,
)
from .research_session_store import ResearchSessionStore, SCHEMA_SQL as SESSION_SCHEMA
from .schemas import (
    APPROVAL_SCOPES,
    ApprovalRecord,
    ApprovalSource,
    CampaignBinding,
    CriticReviewRecord,
    ResearchOutcome,
    ResearchReport,
    ResearchSession,
    StoredExperimentPlan,
    StoredHypothesis,
    StoredObjective,
    safe_id,
)

# M4.0-01/02: the two additive tables join ALL_SCHEMA_SQL so a plain
# ``ResearchStores()`` performs a single atomic, idempotent migration.
ALL_SCHEMA_SQL = "\n".join((
    HYPOTHESIS_SCHEMA, PLAN_SCHEMA, APPROVAL_SCHEMA, SESSION_SCHEMA,
    OBJECTIVE_SCHEMA, ENTITY_LINK_SCHEMA,
))


def default_db_path() -> Path:
    """Runtime state (git-ignored), never a reproducibility artefact."""
    return DATA_DIR / "research.sqlite"


class ResearchStores:
    """Single construction point; inject a temp path for tests."""

    def __init__(self, db_path: str | Path | None = None):
        self.db = ResearchDatabase(db_path or default_db_path(),
                                   schema_sql=ALL_SCHEMA_SQL)
        self.hypotheses = HypothesisStore(self.db)
        self.plans = ExperimentPlanStore(self.db)
        self.approvals = ApprovalStore(self.db)
        self.sessions = ResearchSessionStore(self.db)
        self.objectives = ObjectiveStore(self.db)
        ensure_link_schema(self.db)       # idempotent, fail-closed
        self.links = EntityLinkStore(self.db, workspace=Path.cwd())


__all__ = [
    "ResearchStores", "ResearchDatabase", "default_db_path",
    "HypothesisStore", "ExperimentPlanStore", "ApprovalStore",
    "ResearchSessionStore", "ObjectiveStore", "EntityLinkStore",
    "ensure_link_schema", "sha256_file",
    "canonical_hypothesis", "hypothesis_sha256",
    "canonical_plan", "plan_sha256",
    "canonical_bytes", "content_sha256", "safe_id",
    "StoredHypothesis", "CriticReviewRecord", "StoredExperimentPlan",
    "StoredObjective", "from_research_objective", "describe_schema",
    "schema_structure", "database_snapshot",
    "ApprovalRecord", "ApprovalSource", "CampaignBinding",
    "ResearchSession", "ResearchReport", "ResearchOutcome",
    "APPROVAL_SCOPES", "ALL_SCHEMA_SQL",
    "ResearchStoreError", "ResearchStoreConflict",
]
