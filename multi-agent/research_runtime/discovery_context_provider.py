"""M4.1-00 -- Discovery Context Provider (READ-ONLY aggregation facade).

Purpose
------
``DiscoveryEvidenceAdapter`` (M4.0.1 P1-04) is a correct, thin, read-only
facade over the unified Evidence Reader, but it is NOT a complete Research
Context. M4.1 Discovery needs to read and cross-link:

  * historical Evidence          (DiscoveryEvidenceAdapter / UnifiedEvidenceReader)
  * historical Objectives         (ObjectiveStore)
  * Objective/Candidate/Evidence relations (EntityLinkStore)
  * Research Sessions             (ResearchSessionStore)
  * Research Experience           (research_experience table)
  * Session events / reports      (ResearchSessionStore)
  * Failure signals              (FailureDiagnoser, applied to real results)

This module provides a SINGLE, read-only ``DiscoveryContextProvider`` that
combines those sources into one deterministic ``DiscoverySnapshot``. It does
NOT invent any entity and it does NOT recompute any verdict.

Two wiring constraints from the M4.0.1 review are enforced here:

  (1) SINGLE READ PER RUN. The unified reader re-reads the (up to 38 MB)
      ``evidence.jsonl`` on every ``query_run()`` call. The adapter's
      ``discover_entities()`` / ``carriers()`` / ``raw_verdict_census()`` /
      ``unresolved()`` each call ``query_run()`` again. Discovery MUST therefore
      call ``query_run()`` exactly ONCE, cache the view, and slice it -- never
      re-invoke the per-access accessors in a hot loop.

  (2) NO FABRICATED EXPERIENCE. M3.4 historical runs have NO native
      ``research_experience`` rows (the table is an A1 addition; history was
      backfilled for objectives + links only). The provider must mark a missing
      Experience as an explicit UNRESOLVED gap and never synthesise one. New
      runs accumulate real Experience over time; old data stays unresolved.

Design rules
-----------
  * READ-ONLY: uses only store READ methods (get / all / find_by_* /
    session_for_objective / events / experience / reports_for_session /
    list_objective_graph / stats). No writes, no verdict recomputation, no gate.
  * STABLE CONTRACT: M4.1 Discovery depends only on ``build_snapshot()`` and the
    ``DiscoverySnapshot`` shape; it never reaches into the individual stores.
  * DETERMINISTIC: the same (db, workspace, run_id) inputs yield the same
    ``DiscoverySnapshot.data_dict()`` (only ``generated_at`` is a build stamp).
  * TRACEABLE: every aggregated entity carries its source id + a SHA/source
    reference (evidence_id = sha256:..., objective content_sha256, link
    source_sha256, session created_at) so an Objective can always be traced to
    concrete Evidence / Experience / Failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .discovery_evidence_adapter import DiscoveryEvidenceAdapter
from .failure_diagnoser import FailureDiagnoser
from .research_store import ResearchStores


DISCOVERY_SNAPSHOT_SCHEMA_VERSION = "m4.1-00-discovery-snapshot-v1"

# Session event types that signal a research failure worth surfacing to
# Discovery. Conservative: only explicit failure events, never guessed.
_FAILURE_EVENT_MARKERS = ("fail",)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    """Serialise a pydantic model or dict uniformly."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass
    return dict(obj)


def _domain_names_from_view(view: dict[str, Any]) -> list[str]:
    domains = (view.get("provenance") or {}).get("domains") or {}
    return sorted(domains.keys())


@dataclass
class DiscoverySnapshot:
    """A deterministic, read-only aggregation of one run's research context.

    Every field is derived from real stored data. ``unresolved_gaps`` makes
    explicit what is missing (sessions / experience / evidence collisions) so
    Discovery never has to guess.
    """

    run_id: str
    schema_version: str
    generated_at: str
    evidence_view: dict[str, Any]
    objectives: list[dict[str, Any]]
    links: dict[str, Any]
    sessions: list[dict[str, Any]]
    experience_available: bool
    failures: dict[str, Any]
    provenance: dict[str, Any]
    unresolved_gaps: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "evidence_view": self.evidence_view,
            "objectives": self.objectives,
            "links": self.links,
            "sessions": self.sessions,
            "experience_available": self.experience_available,
            "failures": self.failures,
            "provenance": self.provenance,
            "unresolved_gaps": self.unresolved_gaps,
        }

    def data_dict(self) -> dict[str, Any]:
        """Deterministic comparison form: excludes the build stamp."""
        d = self.to_dict()
        d.pop("generated_at", None)
        return d


class DiscoveryContextProvider:
    """Read-only combiner of Evidence + Objectives + Links + Sessions/Experience.

    Construction
    ------------
    Inject an already-built ``ResearchStores`` (which bundles ObjectiveStore,
    EntityLinkStore and ResearchSessionStore) and an optional
    ``DiscoveryEvidenceAdapter``. Wiring ``FailureDiagnoser`` is provided for
    later sub-steps (M4.1-01/02) that diagnose real ``ExperimentResult``s.
    """

    def __init__(self, *, stores: ResearchStores,
                 evidence_adapter: DiscoveryEvidenceAdapter | None = None,
                 workspace: str | Path | None = None,
                 failure_diagnoser: FailureDiagnoser | None = None):
        self.stores = stores
        self.workspace = Path(workspace) if workspace is not None else None
        # (1) SINGLE READ PER RUN: the provider owns exactly one query_run()
        # call per build_snapshot; it never uses the adapter's per-access
        # accessors (discover_entities/carriers/raw_verdict_census/unresolved)
        # in a loop.
        self.evidence_adapter = (
            evidence_adapter
            if evidence_adapter is not None
            else DiscoveryEvidenceAdapter(workspace=self.workspace))
        self.failure_diagnoser = (
            failure_diagnoser if failure_diagnoser is not None
            else FailureDiagnoser())

    # -- single read ----------------------------------------------------
    def build_snapshot(self, run_id: str,
                       objective_ids: list[str] | None = None) -> DiscoverySnapshot:
        """Build one deterministic snapshot for ``run_id``.

        Calls the unified Evidence Reader exactly ONCE, then cross-links the
        objectives / links / sessions / experience from the injected stores.
        """
        view = self.evidence_adapter.query_run(run_id)  # <-- the ONLY read

        domains = _domain_names_from_view(view)
        if objective_ids is None:
            objective_ids = [f"obj_{run_id}_{d}" for d in domains]

        objectives = self._load_objectives(objective_ids)
        sessions = self._load_sessions(objective_ids)
        links = self._load_links(run_id, objective_ids)
        failures = self._load_failures(view, sessions)
        provenance = self._build_provenance(
            run_id, view, objective_ids, objectives, sessions, links)
        unresolved_gaps = self._build_unresolved_gaps(
            view, objective_ids, sessions, failures)

        any_experience = any(s.get("experience_available") for s in sessions)

        return DiscoverySnapshot(
            run_id=run_id,
            schema_version=DISCOVERY_SNAPSHOT_SCHEMA_VERSION,
            generated_at=_now(),
            evidence_view=view,
            objectives=objectives,
            links=links,
            sessions=sessions,
            experience_available=any_experience,
            failures=failures,
            provenance=provenance,
            unresolved_gaps=unresolved_gaps,
        )

    # -- sub-reads (all READ-ONLY, all post-snapshot) -------------------
    def _load_objectives(self, objective_ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for oid in objective_ids:
            rec = self.stores.objectives.get(oid)
            if rec is None:
                continue
            out.append(_obj_to_dict(rec))
        return out

    def _load_sessions(self, objective_ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for oid in objective_ids:
            session = self.stores.sessions.session_for_objective(oid)
            if session is None:
                out.append({
                    "objective_id": oid,
                    "session_available": False,
                    "session_id": None,
                    "session_status": None,
                    "events": [],
                    "reports": [],
                    "experience": [],
                    "experience_available": False,
                })
                continue
            sid = session.session_id
            exp = self.stores.sessions.experience(sid)
            out.append({
                "objective_id": oid,
                "session_available": True,
                "session_id": sid,
                "session_status": getattr(session, "status", None),
                "events": self.stores.sessions.events(sid),
                "reports": [_obj_to_dict(r) for r
                            in self.stores.sessions.reports_for_session(sid)],
                # (2) NO FABRICATION: experience is echoed verbatim; if empty,
                # experience_available is False and the gap is recorded upstream.
                "experience": exp,
                "experience_available": len(exp) > 0,
            })
        return out

    def _load_links(self, run_id: str,
                    objective_ids: list[str]) -> dict[str, Any]:
        by_objective: dict[str, Any] = {}
        for oid in objective_ids:
            by_objective[oid] = self.stores.links.list_objective_graph(oid, run_id)
        return {
            "run_stats": self.stores.links.stats(run_id=run_id),
            "by_objective": by_objective,
        }

    def _load_failures(self, view: dict[str, Any],
                       sessions: list[dict[str, Any]]) -> dict[str, Any]:
        # Honest, stored-only failure signals. We do NOT manufacture
        # FailureReports here; the FailureDiagnoser is wired for later sub-steps
        # that receive real ExperimentResult objects.
        evidence_unresolved = view.get("unresolved", [])
        session_failure_events: list[dict[str, Any]] = []
        for s in sessions:
            for ev in s.get("events", []):
                etype = str(ev.get("event_type", "")).lower()
                if any(m in etype for m in _FAILURE_EVENT_MARKERS):
                    session_failure_events.append({
                        "objective_id": s["objective_id"],
                        "session_id": s.get("session_id"),
                        "event_type": ev.get("event_type"),
                        "created_at": ev.get("created_at"),
                    })
        return {
            "evidence_unresolved": evidence_unresolved,
            "session_failure_events": session_failure_events,
        }

    def _build_provenance(self, run_id: str, view: dict[str, Any],
                          objective_ids: list[str],
                          objectives: list[dict[str, Any]],
                          sessions: list[dict[str, Any]],
                          links: dict[str, Any]) -> dict[str, Any]:
        n_experience = sum(len(s.get("experience", [])) for s in sessions)
        n_links = sum(
            g.get("n_links", 0) for g in links.get("by_objective", {}).values())
        return {
            "read_only": True,
            "schema_version": DISCOVERY_SNAPSHOT_SCHEMA_VERSION,
            "run_id": run_id,
            "objective_ids": objective_ids,
            "sources": [
                "DiscoveryEvidenceAdapter.query_run",
                "ObjectiveStore.get",
                "EntityLinkStore.list_objective_graph",
                "EntityLinkStore.stats",
                "ResearchSessionStore.session_for_objective",
                "ResearchSessionStore.events",
                "ResearchSessionStore.reports_for_session",
                "ResearchSessionStore.experience",
            ],
            "counts": {
                "n_evidence_entities": view.get("n_evidence", 0),
                "n_carriers_read": view.get("n_carriers_read", 0),
                "n_objectives": len(objectives),
                "n_objective_ids_resolved": len(objectives),
                "n_objective_ids_missing": len(objective_ids) - len(objectives),
                "n_session_records": len(sessions),
                "n_sessions_available": sum(
                    1 for s in sessions if s.get("session_available")),
                "n_experience_records": n_experience,
                "n_links": n_links,
                "n_evidence_unresolved": len(
                    view.get("unresolved", [])),
            },
        }

    def _build_unresolved_gaps(self, view: dict[str, Any],
                               objective_ids: list[str],
                               sessions: list[dict[str, Any]],
                               failures: dict[str, Any]) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        for s in sessions:
            if not s.get("session_available"):
                gaps.append({
                    "kind": "missing_session",
                    "objective_id": s["objective_id"],
                    "reason": ("no research_session recorded for this historical "
                               "objective; sessions are created by the A1 runtime, "
                               "not by M4.0 backfill"),
                })
            elif not s.get("experience_available"):
                gaps.append({
                    "kind": "missing_experience",
                    "objective_id": s["objective_id"],
                    "session_id": s.get("session_id"),
                    "reason": ("objective has a session but no recorded "
                               "research_experience; M3.4 history has no native "
                               "experience -- must NOT be fabricated"),
                })
        for u in view.get("unresolved", []):
            gaps.append({
                "kind": "evidence_unresolved",
                "detail": u,
                "reason": "evidence carrier could not be resolved to a real file",
            })
        return gaps

    # -- forward wiring for later sub-steps (not used during snapshot build) -
    def diagnose_result(self, result: Any) -> list[dict[str, Any]]:
        """Wrap FailureDiagnoser for M4.1-01/02 candidate diagnosis.

        Receives a real ``ExperimentResult`` and returns structured
        FailureReports. Kept separate from snapshot building so historical
        (result-less) runs are never forced to fabricate a diagnosis.
        """
        reports = self.failure_diagnoser.diagnose_all(result)
        return [_obj_to_dict(r) for r in reports]
