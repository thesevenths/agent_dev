"""M4-E2E-01A -- Deterministic Objective Ranker.

Assigns a reproducible score to every CRITIC-APPROVED candidate. Same Snapshot +
same config => identical scores AND identical ordering (spec M4-E2E-01A.6). The
score is a weighted decomposition so the ranking is explainable and auditable;
no statistical verdict is computed.

Score components (each in [0, 1]):
  * evidence_strength  -- fraction of the candidate's evidence_refs that resolve
                         to real snapshot data (more grounded == higher).
  * verifiability      -- 1.0 if the success_criteria is measurable, else 0.0.
  * intervention       -- 1.0 if a directional intervention hint is present.
  * no_duplicate       -- 1.0 (Critic already removed duplicates; carried for clarity).
  * source_priority    -- priority weight of the generation source
                         (historical_failure=1.0, historical_experience=0.9,
                          id_ood_gap=0.8, negative_verdict=0.6).
  * provenance_richness-- 1.0 if the candidate carries a snapshot_run_id provenance.

Weights are fixed constants (never tuned from results). The final score is the
dot product; ties broken by candidate_id ascending for determinism.
"""
from __future__ import annotations

from typing import Any

from .objective_candidate_generator import ObjectiveCandidate

RANKER_VERSION = "m4-e2e-01a-deterministic-v1"

# Fixed, documented weights. Do NOT tune these from observed outcomes.
_WEIGHTS = {
    "evidence_strength": 0.30,
    "verifiability": 0.20,
    "intervention": 0.15,
    "no_duplicate": 0.10,
    "source_priority": 0.15,
    "provenance_richness": 0.10,
}

_SOURCE_PRIORITY = {
    "historical_failure": 1.0,
    "historical_experience": 0.9,
    "id_ood_gap": 0.8,
    "negative_verdict_evidence": 0.6,
}


def _frac(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 for v in values if v) / len(values)


class ObjectiveRanker:
    """Deterministic ranking with explicit, replayable score decomposition."""

    def __init__(self, *, version: str = RANKER_VERSION, weights: dict[str, float] | None = None):
        self.version = version
        self.weights = dict(weights or _WEIGHTS)

    # -- component computation -------------------------------------------
    def _components(self, candidate: ObjectiveCandidate,
                    critiques: dict[str, tuple[bool, list[str]]],
                    *, resolvable_fn: Any = None) -> dict[str, float]:
        passed, _ = critiques.get(candidate.candidate_id, (False, []))
        # A candidate that failed the Critic is excluded before ranking; if it is
        # somehow passed here we still score it honestly (components reflect reality).
        refs = candidate.evidence_refs or []
        evidence_strength = _frac([bool(resolvable_fn(r)) for r in refs]) if \
            resolvable_fn is not None else (1.0 if refs else 0.0)
        sc = candidate.success_criteria or {}
        verifiability = 1.0 if (sc.get("criterion") and sc.get("measurable")) else 0.0
        inter = candidate.proposed_intervention or {}
        intervention = 1.0 if (isinstance(inter, dict) and
                              any(k for k in inter if not k.startswith("_"))) else 0.0
        no_duplicate = 1.0 if passed else 0.0
        source_priority = float(_SOURCE_PRIORITY.get(candidate.generation_source, 0.5))
        prov = candidate.provenance or {}
        provenance_richness = 1.0 if prov.get("snapshot_run_id") else 0.0
        return {
            "evidence_strength": round(evidence_strength, 6),
            "verifiability": verifiability,
            "intervention": intervention,
            "no_duplicate": no_duplicate,
            "source_priority": source_priority,
            "provenance_richness": provenance_richness,
        }

    def rank(self, candidates: list[ObjectiveCandidate],
             critiques: dict[str, tuple[bool, list[str]]],
             *, resolvable_fn: Any = None) -> list[ObjectiveCandidate]:
        """Return candidates sorted by descending score, deterministic tie-break.

        Only critic-approved candidates are ranked; rejected ones are dropped (their
        critiques are preserved by the orchestrator). Each ranked candidate gets
        ``rank_score`` and ``rank_breakdown`` set for full auditability.
        """
        approved = [c for c in candidates
                    if critiques.get(c.candidate_id, (False, []))[0]]
        scored: list[tuple[float, str, ObjectiveCandidate]] = []
        for c in approved:
            comp = self._components(c, critiques, resolvable_fn=resolvable_fn)
            score = round(sum(self.weights[k] * comp[k] for k in self.weights), 6)
            breakdown = {
                "ranker_version": self.version,
                "components": comp,
                "weights": dict(self.weights),
                "score": score,
            }
            c.rank_score = score
            c.rank_breakdown = breakdown
            scored.append((score, c.candidate_id, c))
        # Deterministic: score desc, then candidate_id asc.
        scored.sort(key=lambda t: (-t[0], t[1]))
        for i, (_, _, c) in enumerate(scored, start=1):
            if c.rank_breakdown is not None:
                c.rank_breakdown["rank"] = i
        return [c for _, _, c in scored]
