"""M4-E2E-01A -- Objective Discovery orchestrator.

Single deterministic entry point: generate candidates from a real DiscoverySnapshot,
critique them (rules-only), rank the approved ones, and persist the approved
candidates as *Proposed Objectives* via the existing ``ObjectiveStore``.

It does NOT:
  * produce ACCEPT / REJECT / INCONCLUSIVE,
  * decide scientific value (the LLM hint is advisory only),
  * modify historical Evidence, verdicts or metrics,
  * invent Experience.

Same Snapshot + same config + same store => identical proposed objective ids and
identical ranking (spec M4-E2E-01A.6 / M4-E2E-01A.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .discovery_context_provider import DiscoverySnapshot
from .objective_candidate_generator import (
    ObjectiveCandidate, ObjectiveCandidateGenerator,
)
from .objective_critic import ObjectiveCritic, _resolve_ref
from .objective_ranker import ObjectiveRanker
from .research_store import ObjectiveStore, ResearchStores, from_research_objective
from .schemas import ResearchObjective


def _objective_id_for(candidate: ObjectiveCandidate) -> str:
    """Deterministic, stable objective id derived from the candidate id."""
    return f"obj_disc_{candidate.candidate_id}"


@dataclass
class DiscoveryResult:
    """What one ``discover()`` produced -- fully auditable, no verdicts."""
    run_id: str
    proposed_objective_ids: list[str] = field(default_factory=list)
    ranked: list[dict[str, Any]] = field(default_factory=list)
    critiques: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    n_candidates: int = 0
    n_approved: int = 0
    ranker_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_candidates": self.n_candidates,
            "n_approved": self.n_approved,
            "proposed_objective_ids": list(self.proposed_objective_ids),
            "ranked": self.ranked,
            "critiques": self.critiques,
            "rejected": self.rejected,
            "ranker_version": self.ranker_version,
        }


class ObjectiveDiscovery:
    """Orchestrates the A(B)C pipeline against a DiscoverySnapshot."""

    def __init__(self, *, generator: ObjectiveCandidateGenerator | None = None,
                 critic: ObjectiveCritic | None = None,
                 ranker: ObjectiveRanker | None = None,
                 domain: str = "scinet",
                 llm_generate: Callable[[str], Any] | None = None,
                 default_budget: dict[str, Any] | None = None):
        self.generator = generator or ObjectiveCandidateGenerator(
            domain=domain, llm_generate=llm_generate, default_budget=default_budget)
        self.critic = critic or ObjectiveCritic()
        self.ranker = ranker or ObjectiveRanker()

    # -- main -------------------------------------------------------------
    def discover(self, snapshot: DiscoverySnapshot, *,
                 objective_store: ObjectiveStore,
                 domain: str | None = None) -> DiscoveryResult:
        domain = domain or self.generator.domain
        candidates = self.generator.generate(snapshot)
        critiques: dict[str, tuple[bool, list[str]]] = {
            c.candidate_id: self.critic.critique(c, snapshot=snapshot)
            for c in candidates
        }

        result = DiscoveryResult(run_id=snapshot.run_id,
                                n_candidates=len(candidates),
                                ranker_version=self.ranker.version)

        for c in candidates:
            passed, errors = critiques[c.candidate_id]
            result.critiques[c.candidate_id] = {
                "candidate_id": c.candidate_id,
                "passed": passed,
                "errors": errors,
                "generation_source": c.generation_source,
            }
            if not passed:
                result.rejected.append({
                    "candidate_id": c.candidate_id,
                    "description": c.description,
                    "errors": errors,
                    "generation_source": c.generation_source,
                })

        ranked = self.ranker.rank(
            candidates, critiques,
            resolvable_fn=lambda r: _resolve_ref(r, snapshot))
        result.n_approved = len(ranked)

        # Persist each approved candidate as a Proposed Objective (idempotent).
        for c in ranked:
            oid = _objective_id_for(c)
            obj = ResearchObjective(
                objective_id=oid,
                domain=domain,
                description=c.description,
                success_criteria=dict(c.success_criteria),
                constraints={
                    "evidence_refs": list(c.evidence_refs),
                    "source_objective_ids": list(c.source_objective_ids),
                    "hypothesis_hint": c.hypothesis_hint,
                    "proposed_intervention": dict(c.proposed_intervention),
                    "budget": dict(c.budget),
                    "generation_source": c.generation_source,
                },
                created_at=snapshot.generated_at or "",
            )
            record = objective_store.save(
                from_research_objective(
                    obj, status="proposed",
                    provenance={
                        "candidate_id": c.candidate_id,
                        "generation_source": c.generation_source,
                        "rank_score": c.rank_score,
                        "ranker_version": self.ranker.version,
                        "discovery_run_id": snapshot.run_id,
                    }))
            result.proposed_objective_ids.append(oid)
            result.ranked.append({
                "objective_id": oid,
                "candidate_id": c.candidate_id,
                "generation_source": c.generation_source,
                "rank_score": c.rank_score,
                "rank_breakdown": c.rank_breakdown,
                "description": c.description,
                "evidence_refs": list(c.evidence_refs),
                "persist_status": record.get("status"),
            })
        return result

    # -- reconciliation helper (used by M4-E2E-01D second-round discovery) --
    @staticmethod
    def load_store(db_path: Any) -> ObjectiveStore:
        return ResearchStores(db_path).objectives
