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
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .discovery_context_provider import DiscoverySnapshot
from .schemas import RESERVED_FINAL_SPLITS

# Raw verdict strings (kept verbatim by the read model) that signal a direction
# worth re-investigating. We never recompute a verdict, only recognise a stored
# negative signal.
_NEGATIVE_VERDICTS = {"rejected", "inconclusive", "fail", "failed"}


# ---------------------------------------------------------------------------
# Failure evidence reference contract (M4.3-00 / P0-01)
# ---------------------------------------------------------------------------
# A failure evidence ref has exactly one of these shapes:
#     session:<session_id>:event:<event_type>
#     objective:<objective_id>:event:<event_type>
# The scope prefix selects WHICH id field is authoritative (a session id must
# never be verifiable through the ``objective:`` prefix, or vice versa). Both
# the id AND the event_type must come from a REAL
# ``failures.session_failure_signals`` entry. The generator builds refs from
# those signals and the Critic resolves them against the SAME signals, so a
# fabricated event, an unknown id, or a scope-mismatched id can never be
# grounded. Plain entity refs (``session:<id>`` / ``objective:<id>``) remain
# supported for traceability to a real session/objective.
FAILURE_REF_SCOPES = ("session", "objective")
FAILURE_ID_FIELD = {"session": "session_id", "objective": "objective_id"}


def make_failure_ref(signal: dict[str, Any]) -> str:
    """Build the canonical evidence ref for ONE real failure signal."""
    sid = str(signal.get("session_id") or "")
    oid = str(signal.get("objective_id") or "")
    etype = str(signal.get("event_type") or "fail")
    if sid:
        return f"session:{sid}:event:{etype}"
    return f"objective:{oid}:event:{etype}"


def parse_failure_ref(ref: str) -> tuple[str, str, str | None] | None:
    """Parse a failure/entity ref -> ``(scope, id, event_type|None)`` or None.

    ``event_type`` is ``None`` for a plain entity ref (``session:<id>`` /
    ``objective:<id>``). Anything malformed -- wrong scope, missing id, missing
    ``event`` marker, missing/extra segments -- returns ``None`` so it can never
    be treated as grounded evidence.
    """
    if not ref or ":" not in ref:
        return None
    parts = ref.split(":")
    scope = parts[0]
    if scope not in FAILURE_REF_SCOPES:
        return None
    if len(parts) == 2 and parts[1]:                 # plain entity ref
        return (scope, parts[1], None)
    if (len(parts) == 4 and parts[2] == "event"
            and parts[1] and parts[3]):              # event ref
        return (scope, parts[1], parts[3])
    return None


# ---------------------------------------------------------------------------
# Comparable ID/OOD metric evidence contract (M4.3-00 / P0-02)
# ---------------------------------------------------------------------------
# The MERE PRESENCE of ``provenance.domains[d].ood_dataset_version`` is metadata
# about a dataset, NOT evidence that an ID/OOD generalisation GAP exists. A
# "close the gap" claim is only legitimate when the Evidence View carries a
# REAL, COMPARABLE ID/OOD metric record, i.e. two measurements that are
# comparable along ALL FOUR identity dimensions:
#   * domain                 -- the same domain as the claim
#   * dataset version       -- a real ID version AND a declared OOD version,
#                              distinct, with the OOD version matching the
#                              domain's declared version
#   * metric identity       -- ID and OOD measured on the SAME metric
#   * evaluation split      -- ID and OOD measured on the SAME split, and that
#                              split must NOT be a frozen/final split
# The record must also trace back to at least one REAL evidence entity. No
# verdict is recomputed here: we only READ stored measurements. Framework-
# agnostic by construction -- no domain name, metric name or split name is
# hardcoded; reserved final splits may be extended by the Evidence View.
ID_OOD_METRICS_KEY = "id_ood_comparable_metrics"
ID_OOD_METRIC_REF_PREFIX = "id_ood_metric:"


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _reserved_final_splits(snapshot: DiscoverySnapshot) -> set[str]:
    """Global reserved splits + any declared by this Evidence View (generic)."""
    extra = (snapshot.evidence_view or {}).get("reserved_final_splits") or []
    return {str(s).lower() for s in RESERVED_FINAL_SPLITS} | {
        str(s).lower() for s in extra}


def comparable_id_ood_records(snapshot: DiscoverySnapshot,
                              domain: str) -> list[dict[str, Any]]:
    """Return ONLY the real, comparable ID/OOD metric records for ``domain``.

    A record survives iff it passes every identity/grounding check. Rejected
    records (version mismatch, metric mismatch, split mismatch, frozen split,
    non-finite/absent values, untraceable evidence) never produce a gap claim.
    """
    view = snapshot.evidence_view or {}
    domains = (view.get("provenance") or {}).get("domains") or {}
    declared_ood = str((domains.get(domain) or {}).get("ood_dataset_version") or "")
    if not declared_ood:
        return []
    entity_ids = {str(e.get("evidence_id") or "")
                  for e in (view.get("evidence") or [])}
    entity_ids.discard("")
    reserved = _reserved_final_splits(snapshot)

    out: list[dict[str, Any]] = []
    for rec in view.get(ID_OOD_METRICS_KEY) or []:
        if not isinstance(rec, dict):
            continue
        # (1) domain identity
        if str(rec.get("domain") or "") != domain:
            continue
        # (2) dataset version identity: distinct ID/OOD, OOD == declared
        id_ver = str(rec.get("id_dataset_version") or "")
        ood_ver = str(rec.get("ood_dataset_version") or "")
        if not id_ver or not ood_ver or id_ver == ood_ver or ood_ver != declared_ood:
            continue
        # (3) metric identity: one and the SAME metric on both sides
        id_metric = str(rec.get("id_metric") or "")
        ood_metric = str(rec.get("ood_metric") or "")
        if not id_metric or id_metric != ood_metric:
            continue
        # (4) evaluation split identity: same split, never a frozen/final split
        id_split = str(rec.get("id_evaluation_split") or "")
        ood_split = str(rec.get("ood_evaluation_split") or "")
        if (not id_split or id_split != ood_split
                or id_split.lower() in reserved):
            continue
        # Real numeric measurements, not fabricated placeholders.
        if not (_finite_number(rec.get("id_value"))
                and _finite_number(rec.get("ood_value"))):
            continue
        # Traceability to REAL evidence entities (no synthetic grounding).
        refs = [str(x) for x in (rec.get("evidence_ids") or []) if str(x)]
        if not refs or not all(r in entity_ids for r in refs):
            continue
        out.append(rec)
    return out


def make_id_ood_metric_ref(rec: dict[str, Any]) -> str:
    """Canonical ref for ONE comparable record (domain:metric:split:versions)."""
    return (f"{ID_OOD_METRIC_REF_PREFIX}"
            f"{rec.get('domain')}:{rec.get('id_metric')}:"
            f"{rec.get('id_evaluation_split')}:"
            f"{rec.get('id_dataset_version')}:{rec.get('ood_dataset_version')}")


def resolve_id_ood_metric_ref(ref: str, snapshot: DiscoverySnapshot) -> bool:
    """A gap-metric ref resolves ONLY to a real, comparable record."""
    if not ref.startswith(ID_OOD_METRIC_REF_PREFIX):
        return False
    parts = ref.split(":")
    if len(parts) != 6 or not all(parts[1:]):       # prefix + 5 identity parts
        return False
    domain, metric, split, id_ver, ood_ver = parts[1:]
    for rec in comparable_id_ood_records(snapshot, domain):
        if (str(rec.get("id_metric")) == metric
                and str(rec.get("id_evaluation_split")) == split
                and str(rec.get("id_dataset_version")) == id_ver
                and str(rec.get("ood_dataset_version")) == ood_ver):
            return True
    return False


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
                # Every evidence reference must be a machine-readable
                # ``key:value`` claim (the hypothesis Critic rejects bare text).
                # A carrier file path is a REAL artefact, so it is cited as
                # ``path:<relpath>`` instead of being dropped.
                evidence_refs=[eid, *[f"path:{s}" for s in ent.get("sources", [])
                                      if s]],
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

    # -- source 2: ID / OOD gap (M4.3-00 / P0-02) ------------------------
    def _from_ood_gaps(self, snapshot: DiscoverySnapshot
                       ) -> list[ObjectiveCandidate]:
        """A "close the gap" claim requires REAL comparable ID/OOD metrics.

        M4.3-00 / P0-02: a declared ``ood_dataset_version`` is metadata, NOT a
        measured gap. Only a record that passes ALL FOUR identity checks
        (domain / dataset version / metric identity / evaluation split, on a
        NON-frozen split and traced to real evidence) may produce a gap claim.
        When only the version exists, we emit an EXPLORATORY candidate that is
        explicitly marked ``evidence_insufficient`` and never asserts a gap.
        """
        out: list[ObjectiveCandidate] = []
        domains = (snapshot.evidence_view or {}).get("provenance", {}).get("domains", {})
        for dname in sorted(domains):
            d = domains[dname] or {}
            ood_version = d.get("ood_dataset_version")
            if not ood_version:
                continue
            source_objectives = list(o.get("objective_id")
                                     for o in snapshot.objectives)
            records = comparable_id_ood_records(snapshot, dname)

            if records:
                # REAL comparable measurements exist -> a grounded gap claim.
                # No verdict is recomputed: we cite the stored measurements.
                for rec in records:
                    ref = make_id_ood_metric_ref(rec)
                    meta_ref = f"domain:{dname}:ood_dataset_version:{ood_version}"
                    cid = _candidate_id("ood_gap", ref)
                    out.append(ObjectiveCandidate(
                        candidate_id=cid,
                        domain=self.domain,
                        description=(
                            f"Investigate and reduce the measured ID/OOD "
                            f"generalisation gap for domain '{dname}' "
                            f"(metric {rec['id_metric']} on split "
                            f"{rec['id_evaluation_split']}; OOD dataset "
                            f"{ood_version})."),
                        evidence_refs=[ref, meta_ref],
                        source_objective_ids=source_objectives,
                        hypothesis_hint=(
                            "Comparable ID/OOD measurements are recorded on the "
                            "same metric and the same non-frozen split; form a "
                            "falsifiable hypothesis for a robustness/regularisation "
                            "change and re-measure BOTH sides on that same split. "
                            "Do not use the frozen final split for selection."),
                        proposed_intervention={
                            "axis": "generalisation_robustness",
                            "rationale": (
                                f"reduce measured ID/OOD gap for {dname} on "
                                f"{rec['id_evaluation_split']}"),
                        },
                        success_criteria={
                            "criterion": (
                                f"reduce {rec['id_metric']} gap for {dname} "
                                f"on {rec['id_evaluation_split']}"),
                            "measurable": True,
                            "basis_evidence": ref,
                            "evidence_insufficient": False,
                        },
                        budget=dict(self.default_budget),
                        provenance={
                            "generation_source": "id_ood_gap",
                            "domain": dname,
                            "ood_dataset_version": ood_version,
                            "id_dataset_version": rec["id_dataset_version"],
                            "metric": rec["id_metric"],
                            "evaluation_split": rec["id_evaluation_split"],
                            "evidence_ids": list(rec.get("evidence_ids") or []),
                            "evidence_insufficient": False,
                            "snapshot_run_id": snapshot.run_id,
                        },
                        generation_source="id_ood_gap",
                    ))
            else:
                # ONLY a version, no comparable metrics -> NO gap conclusion.
                # An exploratory candidate is allowed, but MUST be marked as
                # evidence_insufficient (persisted via success_criteria).
                meta_ref = f"domain:{dname}:ood_dataset_version:{ood_version}"
                cid = _candidate_id("ood_exploration", dname)
                out.append(ObjectiveCandidate(
                    candidate_id=cid,
                    domain=self.domain,
                    description=(
                        f"Investigate OOD generalization for domain '{dname}' "
                        f"(OOD dataset {ood_version} recorded; no comparable "
                        f"ID/OOD metrics available yet)."),
                    evidence_refs=[meta_ref],
                    source_objective_ids=source_objectives,
                    hypothesis_hint=(
                        "Only an OOD dataset version is recorded; there is NO "
                        "comparable ID/OOD metric evidence for a gap claim. "
                        "First establish comparable ID and OOD measurements on "
                        "the SAME metric and the SAME non-frozen split, then "
                        "decide whether a gap-closing direction is warranted."),
                    proposed_intervention={
                        "axis": "establish_comparable_ood_measurement",
                        "rationale": (
                            "gap evidence is insufficient; collect comparable "
                            "ID/OOD metrics before proposing a gap-closing change"),
                    },
                    success_criteria={
                        "criterion": (
                            f"establish comparable ID/OOD {dname} metrics on a "
                            f"non-frozen split before any gap claim"),
                        "measurable": True,
                        "basis_provenance": meta_ref,
                        "evidence_insufficient": True,
                    },
                    budget=dict(self.default_budget),
                    provenance={
                        "generation_source": "id_ood_exploration",
                        "domain": dname,
                        "ood_dataset_version": ood_version,
                        "evidence_insufficient": True,
                        "reason": "no_comparable_id_ood_metrics",
                        "snapshot_run_id": snapshot.run_id,
                    },
                    generation_source="id_ood_exploration",
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
            # M4.3-00 / P0-01: build the ref through the SAME canonical contract
            # the Critic resolves against, so the id scope + event type always
            # match a real stored failure signal.
            ref = make_failure_ref(sig)
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
