"""M7 -- evidence-grounded research questions + multi-objective value assessment.

Questions must cite literature/experience gaps and carry falsifiability
requirements; value uses SEPARATE scored dimensions (no single opaque score)
and an INFORMATION-GAIN term computed over competing hypotheses:
    IG(E) = H(H) - E[H(H|E)]
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

VALUE_DIMENSIONS = ("scientific_significance", "novelty_gap",
                    "falsifiability", "feasibility", "expected_impact",
                    "reproducibility", "resource_cost")


@dataclass(frozen=True)
class ResearchQuestion:
    question_id: str
    title: str
    domain: str
    motivation: str
    literature_gap: str
    existing_claims: list[str] = field(default_factory=list)
    related_experiences: list[str] = field(default_factory=list)
    expected_contribution: str = ""
    falsifiability_requirements: list[str] = field(default_factory=list)
    estimated_budget: dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


def generate_questions(*, domain: str,
                       novelty: dict[str, Any],
                       literature_gaps: list[str],
                       failures: list[dict[str, Any]] | None = None,
                       experiences: list[str] | None = None,
                       max_questions: int = 6) -> list[ResearchQuestion]:
    """Generate questions grounded in REAL gaps/failures; no free-floating topics."""
    out: list[ResearchQuestion] = []
    sources: list[tuple[str, str]] = [("literature_gap", g)
                                      for g in literature_gaps]
    for f in failures or []:
        sources.append(("experiment_failure", str(f.get("reason", ""))
                        + ": " + str(f.get("detail", ""))))
    for e in experiences or []:
        sources.append(("verified_experience", str(e)))
    for verdict in novelty.get("verdicts", []):
        if verdict.get("status") in ("UNCERTAIN", "POTENTIALLY_NOVEL"):
            sources.append(("novelty_gap", str(verdict.get("claim_id", ""))))
    for i, (kind, text) in enumerate(sources[:max_questions]):
        if len(text.strip()) < 8:
            continue
        qid = "rq_" + hashlib.sha256(
            f"{domain}|{kind}|{text}".encode()).hexdigest()[:16]
        out.append(ResearchQuestion(
            question_id=qid,
            title=f"{domain}: {kind.replace('_',' ')} #{i+1}",
            domain=domain,
            motivation=f"Derived from real {kind} evidence: {text[:180]}",
            literature_gap=text if kind in ("literature_gap", "novelty_gap") else "",
            existing_claims=[v.get("claim_id", "") for v in
                             novelty.get("verdicts", [])
                             if v.get("status") == "KNOWN"],
            related_experiences=list(experiences or []),
            expected_contribution=f"Resolve {kind} with a falsifiable experiment",
            falsifiability_requirements=[
                "state observable prediction with metric and direction",
                "state result that would refute the proposed mechanism"],
            estimated_budget={"max_campaign_rounds": 1, "max_jobs": 8},
            uncertainty=0.5,
            evidence_refs=[f"{kind}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"]))
    return out


@dataclass(frozen=True)
class ValueAssessment:
    question_id: str
    scores: dict[str, float]
    uncertainties: dict[str, float]
    rationales: dict[str, str]
    evidence_refs: list[str]
    evaluator_version: str
    expected_information_gain: float
    budget_efficiency: float
    priority: float
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(max(p, 1e-12)) for p in probs)


def expected_information_gain(h_prior: list[float],
                              h_posterior: list[float]) -> float:
    """IG(E)=H(H)-E[H(H|E)]. Inputs are normalised hypothesis probabilities."""
    if not h_prior or len(h_prior) != len(h_posterior):
        return 0.0
    sp, sq = sum(h_prior), sum(h_posterior)
    if sp <= 0 or sq <= 0:
        return 0.0
    before = _entropy([p / sp for p in h_prior])
    after = _entropy([q / sq for q in h_posterior])
    return max(0.0, before - after)


EVALUATOR_VERSION = "m7_value_v1"


def assess_value(q: ResearchQuestion, *, novelty: dict[str, Any],
                 cost_units: float = 1.0,
                 h_prior: list[float] | None = None,
                 h_posterior: list[float] | None = None) -> ValueAssessment:
    """Deterministic multi-dimensional scoring with evidence refs and rationale."""
    statuses = [v.get("status") for v in novelty.get("verdicts", [])]
    novel_gap = 1.0 if "POTENTIALLY_NOVEL" in statuses else (
        0.5 if "UNCERTAIN" in statuses else 0.1)
    scores = {
        "scientific_significance": 0.8 if q.expected_contribution else 0.3,
        "novelty_gap": novel_gap,
        "falsifiability": (1.0 if len(q.falsifiability_requirements) >= 2
                           else 0.2),
        "feasibility": 0.7 if q.estimated_budget else 0.3,
        "expected_impact": 0.6 if q.related_experiences else 0.4,
        "reproducibility": 0.6 if q.evidence_refs else 0.2,
        "resource_cost": max(0.0, 1.0 - min(1.0, cost_units / 10.0)),
    }
    ig = expected_information_gain(h_prior or [0.5, 0.5],
                                   h_posterior or [0.9, 0.1])
    efficiency = (sum(v for k, v in scores.items() if k != "resource_cost")
                  / len(scores)) / max(0.1, cost_units)
    priority = 0.5 * ig + 0.3 * scores["novelty_gap"] + 0.2 * efficiency
    decision = ("PROMISING" if priority >= 0.55 else "EXPLORATORY"
                if priority >= 0.3 else "DEPRIORITIZED")
    return ValueAssessment(
        question_id=q.question_id, scores=scores,
        uncertainties={k: 0.2 for k in scores},
        rationales={k: f"rule-based from {q.question_id} evidence"
                    for k in scores},
        evidence_refs=list(q.evidence_refs),
        evaluator_version=EVALUATOR_VERSION,
        expected_information_gain=round(ig, 4),
        budget_efficiency=round(efficiency, 4),
        priority=round(priority, 4), decision=decision)


__all__ = ["VALUE_DIMENSIONS", "EVALUATOR_VERSION", "ResearchQuestion",
           "ValueAssessment", "generate_questions", "assess_value",
           "expected_information_gain"]
