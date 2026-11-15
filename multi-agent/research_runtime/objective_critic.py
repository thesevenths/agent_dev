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
from .objective_candidate_generator import ObjectiveCandidate

# Refs we accept as resolvable, keyed by their colon-delimited prefix.
_RESOLVABLE_PREFIXES = ("sha256:", "domain:", "session:", "objective:",
                        "experience:")


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
    if prefix == "domain:":
        # domain:<name>:... -> provenance.domains must contain <name>
        parts = ref.split(":")
        return parts[1] in (snapshot.evidence_view or {}).get(
            "provenance", {}).get("domains", {})
    if prefix in ("session:", "objective:"):
        ids = set()
        for s in snapshot.sessions:
            ids.add(s.get("session_id") or "")
            ids.add(s.get("objective_id") or "")
        for lst in (snapshot.sessions_by_objective or {}).values():
            for s in lst:
                ids.add(s.get("session_id") or "")
                ids.add(s.get("objective_id") or "")
        return prefix + ":" in ref and any(
            ref.startswith(f"{prefix}:{i}:") or ref == f"{prefix}:{i}"
            for i in ids if i)
    if prefix == "experience:":
        # An experience ref resolves only if a real Experience is recorded.
        for s in snapshot.sessions:
            if s.get("experience_available"):
                return True
        for lst in (snapshot.sessions_by_objective or {}).values():
            if any(s.get("experience_available") for s in lst):
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
