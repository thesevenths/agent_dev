"""ARR-03 M3 -- verified research knowledge with provenance for transfer.

Only knowledge that PASSED M1/M2 verification may be transferred; never ground
truth labels, hidden failure IDs or exact benchmark answers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .integrity import assert_no_ground_truth, sha256_json


@dataclass
class ResearchKnowledge:
    knowledge_id: str
    scientific_mechanism: str
    evidence: list[str]
    intervention: dict[str, Any]
    before_after: dict[str, float]
    confidence: float
    verification_status: str
    applicability: str
    limitations: list[str] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self):
        if self.provenance_hash == "":
            body = {k: v for k, v in asdict(self).items()
                    if k != "provenance_hash"}
            object.__setattr__(self, "provenance_hash", sha256_json(body))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def agent_view(self) -> dict[str, Any]:
        """Sanitized view: mechanism + intervention + evidence, no hidden IDs."""
        view = {
            "knowledge_id": self.knowledge_id,
            "scientific_mechanism": self.scientific_mechanism,
            "evidence": self.evidence,
            "intervention": self.intervention,
            "before_after": self.before_after,
            "confidence": self.confidence,
            "applicability": self.applicability,
            "limitations": self.limitations,
        }
        assert_no_ground_truth(view)
        return view


def build_knowledge_from_m2(*, verified_units: list[dict[str, Any]],
                            min_effect: float = 0.05) -> list[ResearchKnowledge]:
    """Distill only statistically verified intervention effects from M2.

    Each verified unit is: {cause, intervention, mean_improvement, ci_low,
    source_hash, evidence_refs}. Units whose CI lower bound <= 0 or whose
    improvement < MME are NOT transferred.
    """
    out: list[ResearchKnowledge] = []
    for unit in verified_units:
        eff = float(unit.get("mean_improvement", 0.0))
        lo = float(unit.get("ci_low", -1.0))
        if eff < min_effect or lo <= 0:
            continue
        out.append(ResearchKnowledge(
            knowledge_id="K_" + sha256_json(unit)[:10],
            scientific_mechanism=str(unit.get("cause", "")),
            evidence=[str(x) for x in unit.get("evidence_refs", [])],
            intervention=dict(unit.get("intervention") or {}),
            before_after={"before": 1.0, "after": 1.0 - eff},
            confidence=float(unit.get("confidence", 0.6)),
            verification_status="M2_VERIFIED",
            applicability="state-based action-conditioned dynamics models",
            limitations=["transfer validity must be re-tested in M3 regime"],
        ))
    return out


def knowledge_digest(items: list[ResearchKnowledge]) -> str:
    return sha256_json([i.to_dict() for i in items])


__all__ = ["ResearchKnowledge", "build_knowledge_from_m2", "knowledge_digest"]
