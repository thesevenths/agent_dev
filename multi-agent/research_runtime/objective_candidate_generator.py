"""M4-E2E-01A -- Objective Candidate generation from a DiscoverySnapshot.

READ-ONLY consumer of the DiscoverySnapshot produced by M4.1-00. It proposes
*Objective Candidates* -- NOT formal Objectives and NOT scientific verdicts. The
LLM (if injected) may only append a non-authoritative ``hypothesis_hint``; it can
never decide scientific value, statistical conclusion or execution permission.

Candidate sources (spec M4-E2E-01A.2), derived deterministically from REAL data
in the snapshot:

  * historical weak / negative-verdict evidence   (REJECTED / INCONCLUSIVE)
  * ID / OOD performance gaps                      (provenance.domains OOD versions)
  * historical failures / Experience               (session failure signals / exp)

Every candidate carries the required fields
(domain, description, evidence_refs, source_objective_ids, hypothesis_hint,
proposed_intervention, success_criteria, budget, provenance). No ACCEPT / REJECT
/ INCONCLUSIVE is produced here -- that is the Critic's and the gate's job.

Determinism: same Snapshot + same config => same candidates, same order
(evidence is sorted by identity; OOD domains and sessions are sorted by key).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .discovery_context_provider import DiscoverySnapshot

# Raw verdict strings (kept verbatim by the read model) that signal a direction
# worth re-investigating. We never recompute a verdict, only recognise a stored
# negative signal.
_NEGATIVE_VERDICTS = {"rejected", "inconclusive", "fail", "failed"}


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _candidate_id(source: str, key: str) -> str:
    return f"cand_{source}_{_short_hash(source + '|' + key)}"


@dataclass
class ObjectiveCandidate:
    """A proposed research direction. NOT a committed Objective, NOT a verdict.

    ``proposed_intervention`` is a *directional* hint (axis + rationale). The
    concrete, Critic-approved config override is resolved later by the Hypothesis
    layer -- this generator must not invent engine-specific config keys.
    """

    candidate_id: str
    domain: str
    description: str
    evidence_refs: list[str]
    source_objective_ids: list[str]
    hypothesis_hint: str
    proposed_intervention: dict[str, Any]
    success_criteria: dict[str, Any]
    budget: dict[str, Any]
    provenance: dict[str, Any]
    generation_source: str = "unknown"
    # Filled by the Critic / Ranker; never used to decide scientific value.
    critic_verdict: dict[str, Any] | None = None
    rank_score: float | None = None
    rank_breakdown: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "domain": self.domain,
            "description": self.description,
            "evidence_refs": list(self.evidence_refs),
            "source_objective_ids": list(self.source_objective_ids),
            "hypothesis_hint": self.hypothesis_hint,
            "proposed_intervention": dict(self.proposed_intervention),
            "success_criteria": dict(self.success_criteria),
            "budget": dict(self.budget),
            "provenance": dict(self.provenance),
            "generation_source": self.generation_source,
            "critic_verdict": self.critic_verdict,
            "rank_score": self.rank_score,
            "rank_breakdown": self.rank_breakdown,
        }


class ObjectiveCandidateGenerator:
    """Deterministic, rule-driven generator. LLM is OPTIONAL and advisory only."""

    def __init__(self, *, domain: str = "scinet",
                 llm_generate: Callable[[str], Any] | None = None,
                 default_budget: dict[str, Any] | None = None):
        self.domain = domain
        self.llm_generate = llm_generate
        self.default_budget = dict(default_budget or {
            "max_campaign_rounds": 1,
            "max_candidates": 1,
            "paired_seeds": 5,
            "note": "single-domain MVP default; bounded by the ResearchService gate",
        })

    # -- public ----------------------------------------------------------
    def generate(self, snapshot: DiscoverySnapshot) -> list[ObjectiveCandidate]:
        """Generate candidates from all four required sources, deterministically.

        Sources (spec M4-E2E-01A.2 + M4-E2E-01D):
          * historical weak / negative-verdict evidence   (REJECTED / INCONCLUSIVE)
          * ID / OOD performance gaps                      (provenance.domains OOD)
          * historical failures / Experience signals       (session failure signals)
          * recorded Experience                            (session experience rows)

        Adding the Experience source lets round-2 Discovery prove that newly
        written Experience changes its output (spec M4-E2E-01D) without ever
        treating a historical failure as a success or a historical success as
        proof of a new intervention.
        """
        candidates: list[ObjectiveCandidate] = []
        candidates += self._from_negative_verdicts(snapshot)
        candidates += self._from_ood_gaps(snapshot)
        candidates += self._from_failures(snapshot)
        candidates += self._from_experience(snapshot)
        # Stable de-duplication by candidate_id (generation is deterministic, so
        # the same source key can never collide; this guards against future
        # multi-source overlap without reordering).
        seen: dict[str, ObjectiveCandidate] = {}
        for c in candidates:
            seen.setdefault(c.candidate_id, c)
        return list(seen.values())

    # -- source 1: negative / weak verdict evidence ----------------------
    def _from_negative_verdicts(self, snapshot: DiscoverySnapshot
                                ) -> list[ObjectiveCandidate]:
        out: list[ObjectiveCandidate] = []
        view = snapshot.evidence_view or {}
        for ent in sorted(view.get("evidence", []),
                         key=lambda e: e.get("evidence_id", "")):
            eid = ent.get("evidence_id", "")
            verdicts = {str(v).lower() for v in ent.get("verdicts", [])}
            if not (verdicts & _NEGATIVE_VERDICTS):
                continue
            cid = _candidate_id("neg_verdict", eid)
            out.append(ObjectiveCandidate(
                candidate_id=cid,
                domain=self.domain,
                description=(
                    f"Investigate negative evidence {eid} "
                    f"(verdicts={sorted(verdicts)}) and propose a direction that "
                    f"resolves it."),
                evidence_refs=[eid, *[s for s in ent.get("sources", []) if s]],
                source_objective_ids=list(snapshot.objectives
                                          and [o.get("objective_id") for o in
                                               snapshot.objectives] or []),
                hypothesis_hint=(
                    f"Negative result {eid} indicates a failed direction; "
                    f"form a falsifiable hypothesis about WHY and a candidate "
                    f"intervention to test it."),
                proposed_intervention={
                    "axis": "derive_from_failure_category",
                    "rationale": f"resolve negative verdict for {eid}",
                },
                success_criteria={
                    "criterion": f"resolve or explain negative verdict for {eid}",
                    "measurable": True,
                    "basis_evidence": eid,
                },
                budget=dict(self.default_budget),
                provenance={
                    "generation_source": "negative_verdict_evidence",
                    "evidence_id": eid,
                    "verdicts": sorted(verdicts),
                    "snapshot_run_id": snapshot.run_id,
                },
                generation_source="negative_verdict_evidence",
            ))
        return out

    # -- source 2: ID / OOD performance gap -----------------------------
    def _from_ood_gaps(self, snapshot: DiscoverySnapshot
                       ) -> list[ObjectiveCandidate]:
        out: list[ObjectiveCandidate] = []
        domains = (snapshot.evidence_view or {}).get("provenance", {}).get("domains", {})
        for dname in sorted(domains):
            d = domains[dname] or {}
            ood_version = d.get("ood_dataset_version")
            if not ood_version:
                continue
            cid = _candidate_id("ood_gap", dname)
            ref = f"domain:{dname}:ood_dataset_version:{ood_version}"
            out.append(ObjectiveCandidate(
                candidate_id=cid,
                domain=self.domain,
                description=(
                    f"Close the ID/OOD generalisation gap for domain '{dname}' "
                    f"(OOD dataset {ood_version})."),
                evidence_refs=[ref],
                source_objective_ids=list(o.get("objective_id")
                                          for o in snapshot.objectives),
                hypothesis_hint=(
                    "A degraded OOD metric implies the ID-trained model does not "
                    "generalise; hypothesize a robustness/regularisation change "
                    "and test it on the OOD split (held-out, never the frozen "
                    "final split)."),
                proposed_intervention={
                    "axis": "generalisation_robustness",
                    "rationale": f"improve OOD generalisation for {dname}",
                },
                success_criteria={
                    "criterion": f"reduce OOD gap for {dname} on held-out split",
                    "measurable": True,
                    "basis_provenance": ref,
                },
                budget=dict(self.default_budget),
                provenance={
                    "generation_source": "id_ood_gap",
                    "domain": dname,
                    "ood_dataset_version": ood_version,
                    "snapshot_run_id": snapshot.run_id,
                },
                generation_source="id_ood_gap",
            ))
        return out

    # -- source 3: historical failures / Experience signals -----------------
    def _from_failures(self, snapshot: DiscoverySnapshot
                       ) -> list[ObjectiveCandidate]:
        out: list[ObjectiveCandidate] = []
        signals = (snapshot.failures or {}).get("session_failure_signals", [])
        for sig in sorted(signals, key=lambda s: (
                s.get("objective_id", ""), s.get("session_id") or "",
                s.get("event_type") or "", s.get("created_at") or "")):
            oid = sig.get("objective_id", "")
            sid = sig.get("session_id") or ""
            etype = sig.get("event_type", "fail")
            cid = _candidate_id("failure", f"{oid}|{sid}|{etype}")
            ref = f"session:{sid}:event:{etype}" if sid else \
                f"objective:{oid}:event:{etype}"
            out.append(ObjectiveCandidate(
                candidate_id=cid,
                domain=self.domain,
                description=(
                    f"Address research failure signal '{etype}' on objective "
                    f"'{oid}' (session {sid})."),
                evidence_refs=[ref],
                source_objective_ids=[oid] if oid else
                list(o.get("objective_id") for o in snapshot.objectives),
                hypothesis_hint=(
                    "The recorded failure event indicates a previously attempted "
                    "direction did not succeed; formulate a distinct, falsifiable "
                    "hypothesis that avoids repeating the same failure."),
                proposed_intervention={
                    "axis": "avoid_prior_failure",
                    "rationale": f"do not repeat failure '{etype}' on {oid}",
                },
                success_criteria={
                    "criterion": f"avoid repeating failure '{etype}' and make "
                                 f"measurable progress on {oid}",
                    "measurable": True,
                    "basis_session": ref,
                },
                budget=dict(self.default_budget),
                provenance={
                    "generation_source": "historical_failure",
                    "objective_id": oid,
                    "session_id": sid,
                    "event_type": etype,
                    "snapshot_run_id": snapshot.run_id,
                },
                generation_source="historical_failure",
            ))
        return out

    # -- source 4: recorded Experience (M4-E2E-01D second-round Discovery) --
    def _from_experience(self, snapshot: DiscoverySnapshot
                         ) -> list[ObjectiveCandidate]:
        """Derive candidates from REAL recorded Experience (never fabricate).

        A candidate is produced per (objective, session) that actually carries a
        non-empty Experience row. The candidate is deliberately CAUTIOUS: it
        proposes to REFINE the next intervention using what the Experience shows,
        explicitly avoiding the failed direction -- it does NOT treat a historical
        failure as a success, nor a historical (inconclusive) success as proof of
        a new intervention.
        """
        out: list[ObjectiveCandidate] = []
        seen: set[str] = set()
        for s in snapshot.sessions:
            exps = s.get("experience") or []
            if not exps:
                continue
            sid = s.get("session_id") or ""
            oid = s.get("objective_id") or ""
            key = f"{oid}|{sid}"
            if key in seen:
                continue
            seen.add(key)
            ref = (f"experience:{sid}" if sid
                   else f"experience:{oid}")
            cid = _candidate_id("experience", key)
            out.append(ObjectiveCandidate(
                candidate_id=cid,
                domain=self.domain,
                description=(
                    f"Refine the next intervention for objective '{oid}' using "
                    f"recorded Experience from session '{sid}'; do NOT repeat the "
                    f"direction the Experience shows failed."),
                evidence_refs=[ref],
                source_objective_ids=[oid] if oid else
                list(o.get("objective_id") for o in snapshot.objectives),
                hypothesis_hint=(
                    "Recorded Experience records a prior direction's outcome; form "
                    "a falsifiable hypothesis that avoids the failed path and tests "
                    "a refined variant, without assuming prior success."),
                proposed_intervention={
                    "axis": "refine_from_experience",
                    "rationale": f"leverage experience of {sid}",
                },
                success_criteria={
                    "criterion": f"improve outcome on {oid} without repeating the "
                                 f"recorded failures",
                    "measurable": True,
                    "basis_experience": ref,
                },
                budget=dict(self.default_budget),
                provenance={
                    "generation_source": "historical_experience",
                    "objective_id": oid,
                    "session_id": sid,
                    "n_experience": len(exps),
                    "snapshot_run_id": snapshot.run_id,
                },
                generation_source="historical_experience",
            ))
        return out

    # -- optional LLM advisory (NON-authoritative) ----------------------
    def _advisory_hint(self, candidate: ObjectiveCandidate) -> str:
        """If an LLM is wired, ask for a *hint* only; failures degrade to none."""
        if self.llm_generate is None:
            return candidate.hypothesis_hint
        prompt = json.dumps({
            "task": "suggest a falsifiable hypothesis hint only",
            "candidate_description": candidate.description,
            "evidence_refs": candidate.evidence_refs,
            "constraint": "return a short string; no config keys; advisory only",
        }, ensure_ascii=False, sort_keys=True)
        try:
            out = self.llm_generate(prompt)
        except Exception:
            return candidate.hypothesis_hint
        if isinstance(out, str) and out.strip():
            return out.strip()
        if isinstance(out, dict) and isinstance(out.get("hint"), str):
            return out["hint"].strip()
        return candidate.hypothesis_hint
