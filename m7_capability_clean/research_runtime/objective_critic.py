"""M4-E2E-01A -- Objective Candidate Critic (rules-only, deterministic).

Checks the four required dimensions (spec M4-E2E-01A.4) plus a data-gap guard:

  * evidence        -- at least one evidence_ref resolves to REAL data in the
                      snapshot (an evidence entity, a provenance domain, a session
                      or a recorded Experience). Unresolvable refs are rejected;
                      a candidate whose ONLY refs are unresolved is rejected.
  * verifiability   -- success_criteria carries a non-empty, measurable criterion.
  * intervention    -- proposed_intervention is a non-empty, directional hint.
  * duplicate       -- does NOT duplicate an existing Objective (same domain +
                      normalised description).
  * data gap        -- if every evidence_ref is unresolved, reject (cannot ground).

The Critic NEVER produces an ACCEPT/REJECT/INCONCLUSIVE of the research itself;
it only gates whether a *candidate* is well-formed and evidence-grounded enough to
become a Proposed Objective. An LLM may not influence these checks.
"""
from __future__ import annotations

import re
from typing import Any

from .discovery_context_provider import DiscoverySnapshot
from .objective_candidate_generator import (
    FAILURE_ID_FIELD, ID_OOD_METRIC_REF_PREFIX, ObjectiveCandidate,
    parse_failure_ref, resolve_id_ood_metric_ref, _reserved_final_splits,
)

# Refs we accept as resolvable, keyed by their colon-delimited prefix.
_RESOLVABLE_PREFIXES = ("sha256:", "domain:", "session:", "objective:",
                        "experience:", ID_OOD_METRIC_REF_PREFIX)


def _resolve_entity_ref(ref: str, snapshot: DiscoverySnapshot) -> bool:
    """M4.3-00 / P0-01: resolve ``session:`` / ``objective:`` refs strictly.

    An EVENT ref (``<scope>:<id>:event:<type>``) is real ONLY when
    ``snapshot.failures.session_failure_signals`` contains a signal whose id
    field -- selected by the ref's scope (``session_id`` for ``session:``,
    ``objective_id`` for ``objective:``) -- equals ``<id>`` AND whose
    ``event_type`` equals ``<type>``. This rejects:
      * fabricated events (real id, non-existent event type),
      * unknown ids,
      * scope/type confusion (an objective id cited under a ``session:`` ref),
      * malformed refs (handled by ``parse_failure_ref`` -> None).

    A PLAIN entity ref (``<scope>:<id>``, no event) resolves against the real
    session/objective ids in the snapshot, with the same scope discipline.
    Unlike the old logic, session ids and objective ids are NEVER pooled into
    one undirected id set.
    """
    parsed = parse_failure_ref(ref)
    if parsed is None:
        return False
    scope, rid, etype = parsed
    id_field = FAILURE_ID_FIELD[scope]

    if etype is not None:
        signals = (snapshot.failures or {}).get("session_failure_signals", []) or []
        for sig in signals:
            if (str(sig.get(id_field) or "") == rid
                    and str(sig.get("event_type") or "") == etype):
                return True
        return False

    # Plain entity ref: match a real session_id / objective_id (never the other
    # kind), across both the flat and nested session views.
    for s in (snapshot.sessions or []):
        if str(s.get(id_field) or "") == rid:
            return True
    for lst in (snapshot.sessions_by_objective or {}).values():
        for s in lst:
            if str(s.get(id_field) or "") == rid:
                return True
    return False


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _resolve_ref(ref: str, snapshot: DiscoverySnapshot) -> bool:
    """True iff ``ref`` points at real data in the snapshot (no fabrication)."""
    if not ref or ":" not in ref:
        return False
    prefix = ref.split(":", 1)[0] + ":"
    if prefix not in _RESOLVABLE_PREFIXES:
        return False
    if prefix == "sha256:":
        ids = {e.get("evidence_id") for e in
               (snapshot.evidence_view or {}).get("evidence", [])}
        return ref in ids
    if prefix == ID_OOD_METRIC_REF_PREFIX:
        # M4.3-00 / P0-02: a gap-metric ref resolves ONLY to a REAL, comparable
        # ID/OOD record (four identity checks, non-frozen split, real evidence).
        return resolve_id_ood_metric_ref(ref, snapshot)
    if prefix == "domain:":
        # domain:<name>[:ood_dataset_version:<v>] -> provenance.domains must
        # contain <name> AND, when a version is specified, it must MATCH exactly.
        parts = ref.split(":")
        if len(parts) < 2:
            return False
        name = parts[1]
        domains = (snapshot.evidence_view or {}).get(
            "provenance", {}).get("domains", {})
        if name not in domains:
            return False
        if len(parts) == 2:
            return True
        # domain:<name>:ood_dataset_version:<v> -> verify the EXACT version so a
        # fabricated or stale ood_dataset_version cannot masquerade as real.
        if len(parts) >= 4 and parts[2] == "ood_dataset_version":
            return str(domains[name].get("ood_dataset_version")) == parts[3]
        return False
    if prefix in ("session:", "objective:"):
        return _resolve_entity_ref(ref, snapshot)
    if prefix == "experience:":
        # A SPECIFIC Experience id (session_id or objective_id) must resolve to a
        # real, non-empty Experience record. "Some experience exists somewhere"
        # is NOT sufficient -- a fabricated id must not verify.
        rid = ref[len(prefix):]
        if not rid:
            return False
        for s in (snapshot.sessions or []):
            sid = s.get("session_id") or ""
            oid = s.get("objective_id") or ""
            exp = s.get("experience") or []
            if (rid == sid or rid == oid) and exp:
                return True
        return False
    return False


class ObjectiveCritic:
    """Deterministic gate for objective candidates (no LLM, no fabrication)."""

    def critique(self, candidate: ObjectiveCandidate, *,
                 snapshot: DiscoverySnapshot) -> tuple[bool, list[str]]:
        errors: list[str] = []
        cands = candidate.evidence_refs or []
        if not cands:
            errors.append("no evidence references")
        else:
            resolved = [r for r in cands if _resolve_ref(r, snapshot)]
            if not resolved:
                errors.append(
                    "no evidence_ref resolves to real snapshot data "
                    f"(checked {len(cands)} ref(s))")
            # Data-gap guard: if EVERY ref is unresolved, reject hard.
            if len(resolved) == 0 and any(
                not _resolve_ref(r, snapshot) for r in cands):
                errors.append(
                    "all evidence_refs are unresolved; candidate cannot be grounded")

        # M4.3-00 / P0-02: evidence-semantics guards.
        # (a) A "close the gap" claim (generation_source=id_ood_gap) MUST cite a
        #     REAL comparable ID/OOD metric record. A mere dataset-version ref
        #     is metadata, not gap evidence -> reject the claim.
        if candidate.generation_source == "id_ood_gap":
            if not any(r.startswith(ID_OOD_METRIC_REF_PREFIX)
                       and _resolve_ref(r, snapshot) for r in cands):
                errors.append(
                    "id_ood_gap claim cites no comparable ID/OOD metric record "
                    "(a dataset version alone is not gap evidence)")
        # (b) No candidate may use a frozen/final split as selection evidence.
        #     Frozen splits are reserved for independent final confirmation.
        reserved = _reserved_final_splits(snapshot)
        for r in cands:
            parts = r.split(":")
            if (r.startswith(ID_OOD_METRIC_REF_PREFIX) and len(parts) == 6
                    and parts[3].lower() in reserved):
                errors.append(
                    f"frozen final split {parts[3]!r} cannot be used as "
                    f"objective-selection evidence")
                break

        # verifiability
        sc = candidate.success_criteria or {}
        criterion = sc.get("criterion")
        if not criterion or not str(criterion).strip():
            errors.append("success_criteria has no measurable criterion")
        elif not sc.get("measurable"):
            errors.append("success_criteria not marked measurable")

        # intervention (directional hint, engine-agnostic)
        inter = candidate.proposed_intervention or {}
        if not isinstance(inter, dict) or not inter:
            errors.append("proposed_intervention is empty")
        elif not any(k for k in inter if not k.startswith("_")):
            errors.append("proposed_intervention has no usable key")

        # duplicate existing objective
        norm = _normalise(candidate.description)
        for obj in (snapshot.objectives or []):
            if (obj.get("domain") == candidate.domain
                    and _normalise(obj.get("description", "")) == norm):
                errors.append(
                    f"duplicate of existing objective "
                    f"{obj.get('objective_id')!r}")
                break

        return (len(errors) == 0), errors
