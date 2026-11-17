"""M6.1 -- formal Harness Loader + campaign version binding.

M6 adopted a candidate into a LOCAL activation record but the Driver kept
passing the candidate object straight to the runner. That is not proof of
Runtime Adoption. This module provides the missing formal seam:

  ``HarnessLoader.resolve_for_campaign(campaign_id, domain)``

It reads the promoted version from the REAL EvolutionRegistry, verifies the
candidate artifact on disk, its parent, the structured promotion evidence, the
SHA-bound human approval and the activation record, then returns an immutable
``LoadedHarness``. Any failure is FAIL_CLOSED.

``CampaignVersionBinding`` persists a binding per campaign in SQLite. Once a
campaign is bound to a harness SHA, it can NEVER change -- a rollback affects
only NEW campaigns. This directly implements the user's hard requirement:
in-flight campaigns must finish, be isolated, or be explicitly cancelled, but
must not silently switch version identity mid-run.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..research_store.db import ResearchDatabase
from ..rrsi.evolution_registry import EvolutionRegistry
from .adoption import AdoptionManager
from .harness import load_frozen_bundle
from .identity import sha256_json, verify_frozen

LOADER_VERSION = "m6_1_harness_loader_v1"


class LoaderError(RuntimeError):
    """Any verification failure -> fail closed, no campaign may dispatch."""


@dataclass(frozen=True)
class LoadedHarness:
    harness_sha: str
    component_shas: dict[str, str]
    parent_sha: str
    activation_id: str
    loader_version: str = LOADER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CampaignVersionBinding:
    """Append-only campaign -> immutable harness version binding (SQLite)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS m61_campaign_binding (
        campaign_id TEXT PRIMARY KEY,
        domain TEXT NOT NULL,
        harness_sha TEXT NOT NULL,
        parent_sha TEXT NOT NULL DEFAULT '',
        component_shas TEXT NOT NULL,
        activation_id TEXT NOT NULL,
        state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        journal_sha TEXT NOT NULL
    );
    """

    def __init__(self, db: ResearchDatabase):
        self.db = db
        with self.db.session() as conn:
            for stmt in self.SCHEMA.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    def bind(self, *, campaign_id: str, domain: str,
             loaded: LoadedHarness) -> dict[str, Any]:
        if not campaign_id:
            raise LoaderError("cannot bind a campaign without an id")
        blob = {"campaign_id": campaign_id, "domain": domain,
                "loaded": loaded.to_dict()}
        journal_sha = sha256_json(blob)
        with self.db.session() as conn:
            old = conn.execute(
                "SELECT * FROM m61_campaign_binding WHERE campaign_id=?",
                (campaign_id,)).fetchone()
            if old is not None:
                if old["harness_sha"] != loaded.harness_sha:
                    raise LoaderError(
                        f"campaign {campaign_id} is ALREADY bound to "
                        f"{old['harness_sha']}; refusing mid-campaign switch")
                return {"ok": True, "already_bound": True,
                        "harness_sha": old["harness_sha"]}
            conn.execute(
                """INSERT INTO m61_campaign_binding
                   (campaign_id,domain,harness_sha,parent_sha,component_shas,
                    activation_id,state,created_at,journal_sha)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (campaign_id, domain, loaded.harness_sha, loaded.parent_sha,
                 json.dumps(loaded.component_shas, sort_keys=True),
                 loaded.activation_id, "BOUND",
                 time.strftime("%Y-%m-%dT%H:%M:%S"), journal_sha))
        return {"ok": True, "already_bound": False,
                "harness_sha": loaded.harness_sha}

    def resolve(self, campaign_id: str) -> LoadedHarness | None:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT * FROM m61_campaign_binding WHERE campaign_id=?",
                (campaign_id,)).fetchone()
        if row is None:
            return None
        return LoadedHarness(
            harness_sha=row["harness_sha"],
            component_shas=json.loads(row["component_shas"]),
            parent_sha=row["parent_sha"],
            activation_id=row["activation_id"])

    def bound_campaigns(self, harness_sha: str | None = None
                        ) -> list[dict[str, Any]]:
        with self.db.session() as conn:
            if harness_sha:
                rows = conn.execute(
                    "SELECT * FROM m61_campaign_binding WHERE harness_sha=?",
                    (harness_sha,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM m61_campaign_binding").fetchall()
        return [dict(r) for r in rows]

    def state(self, campaign_id: str) -> str:
        with self.db.session() as conn:
            row = conn.execute(
                "SELECT state FROM m61_campaign_binding WHERE campaign_id=?",
                (campaign_id,)).fetchone()
        return str(row["state"]) if row else ""

    def mark(self, campaign_id: str, state: str) -> dict[str, Any]:
        allowed = {"BOUND", "RUNNING", "COMPLETED", "CANCELLED", "FAILED"}
        if state not in allowed:
            raise LoaderError(f"unknown campaign binding state {state}")
        with self.db.session() as conn:
            conn.execute(
                "UPDATE m61_campaign_binding SET state=? WHERE campaign_id=?",
                (state, campaign_id))
        return {"ok": True, "campaign_id": campaign_id, "state": state}


class HarnessLoader:
    """Resolve the exact approved harness version for a new campaign."""

    def __init__(self, *, registry: EvolutionRegistry,
                 adoption: AdoptionManager,
                 bindings: CampaignVersionBinding,
                 artifact_index: dict[str, str] | None = None):
        self.registry = registry
        self.adoption = adoption
        self.bindings = bindings
        # harness_sha -> on-disk artifact path (populated by the run driver
        # from the REAL frozen artifacts; no in-memory candidate object allowed)
        self.artifact_index = dict(artifact_index or {})

    def verify_approved(self, harness_sha: str) -> dict[str, Any]:
        """Verify promotion + approval + activation from durable records."""
        records = [r for r in self.registry.all("research_harness")
                   if r.get("candidate_sha") == harness_sha]
        if not records:
            raise LoaderError(f"no registry record for {harness_sha}")
        promoted = [r for r in records if r.get("status") == "PROMOTED"]
        if not promoted:
            raise LoaderError("candidate was never PROMOTED")
        rec = promoted[-1]
        # EvolutionRegistry stores the caller's evidence payload under the
        # "evidence" key; read the approval from there (with a legacy fallback).
        evidence_payload = rec.get("evidence") or {}
        approval = (evidence_payload.get("human_approval")
                    or rec.get("human_approval") or {})
        if approval.get("approved") is not True:
            raise LoaderError("no explicit human approval")
        if str(approval.get("reviewed_candidate_sha", "")) != harness_sha:
            raise LoaderError("approval not bound to this candidate SHA")
        act = self.adoption._read()
        if not act or act.get("loaded_sha") != harness_sha:
            raise LoaderError("activation record does not load this SHA")
        if act.get("dispatch_stopped"):
            raise LoaderError("dispatch is stopped; no new campaign")
        if not self.adoption.verify_loaded()["ok"]:
            raise LoaderError("loaded artifact verification failed")
        path = self.artifact_index.get(harness_sha, act.get(
            "loaded_artifact_path", ""))
        check = load_frozen_bundle(
            path, expected_candidate_sha=harness_sha,
            expected_file_sha=act.get("loaded_file_sha256", ""))
        if not check["ok"]:
            raise LoaderError("candidate artifact failed disk re-read: "
                              + "|".join(check["reasons"]))
        bundle = check["bundle"]
        components = {k: sha256_json(v) for k, v in
                      (bundle.get("components") or {}).items()}
        return {"record": rec, "activation": act,
                "bundle": bundle, "components": components, "path": path}

    def resolve_for_campaign(self, *, campaign_id: str,
                             domain: str) -> LoadedHarness:
        """Bind a NEW campaign to the exact approved immutable version."""
        current = self.adoption.active_sha()
        if not current:
            raise LoaderError("no harness adopted")
        verified = self.verify_approved(current)
        loaded = LoadedHarness(
            harness_sha=current,
            component_shas=verified["components"],
            parent_sha=str(verified["record"].get("parent_sha", "")),
            activation_id=sha256_json({
                "experiment_id": verified["record"].get("experiment_id", ""),
                "candidate_sha": current,
                "activation_file": verified["activation"].get(
                    "loaded_file_sha256", "")}))
        self.bindings.bind(campaign_id=campaign_id, domain=domain,
                           loaded=loaded)
        self.adoption.bind_campaign(campaign_id)
        return loaded


__all__ = [
    "LOADER_VERSION", "LoaderError", "LoadedHarness",
    "CampaignVersionBinding", "HarnessLoader",
]
