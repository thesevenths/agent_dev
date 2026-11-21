"""M7 -- claim decomposition, prior-art matching and HONEST novelty status.

Novelty is NEVER claimed from semantic similarity alone. Each atomic claim is
matched against retrieved papers; if literature coverage is insufficient, the
status is NOT_ASSESSED / UNCERTAIN. LLM may supply hints but cannot assert
global novelty (no literature index = cannot assert).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

NOVELTY_STATUSES = ("KNOWN", "INCREMENTAL", "POTENTIALLY_NOVEL",
                    "UNCERTAIN", "NOT_ASSESSED")
RELATIONS = ("SUPPORTS", "CONTRADICTS", "EXTENDS", "REPRODUCES",
             "USES", "SUPERSEDES", "UNRELATED")

_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|;\s*")


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    question_id: str
    kind: str  # mechanism | observation | application | method | evaluation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriorArtRelation:
    claim_id: str
    paper_id: str
    relation: str
    evidence: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NoveltyVerdict:
    claim_id: str
    status: str
    rationale: str
    relations: list[PriorArtRelation] = field(default_factory=list)
    literature_check: str = "performed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["relations"] = [r.to_dict() for r in self.relations]
        return d


def extract_claims(question_id: str, question: str,
                   target_claims: list[str] | None = None) -> list[AtomicClaim]:
    text = " ".join([question] + list(target_claims or []))
    parts = [p.strip() for p in _CLAIM_SPLIT.split(text) if len(p.strip()) > 12]
    if not parts:
        parts = [text.strip()]
    out: list[AtomicClaim] = []
    for i, p in enumerate(parts):
        lower = p.lower()
        kind = ("mechanism" if any(k in lower for k in ("because", "mechanism",
                "caused by", "due to")) else
                "evaluation" if any(k in lower for k in ("metric", "benchmark",
                "evaluat", "measure")) else
                "method" if any(k in lower for k in ("model", "algorithm",
                "method", "architecture")) else
                "application" if "apply" in lower or "application" in lower
                else "observation")
        cid = "claim_" + hashlib.sha256(
            f"{question_id}|{p}".encode()).hexdigest()[:14]
        out.append(AtomicClaim(cid, p, question_id, kind))
    return out


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9]{3,}", (text or "").lower())
            if t not in {"this", "that", "with", "from", "than", "have",
                         "improves", "improve", "across", "using"}}


def match_prior_art(claim: AtomicClaim, papers: list[Any],
                    *, threshold: float = 0.28) -> list[PriorArtRelation]:
    """Deterministic lexical prior-art match; always bound to a real Paper."""
    ct = _tokens(claim.text)
    rels: list[PriorArtRelation] = []
    for p in papers:
        pt = _tokens(" ".join([str(getattr(p, "title", "")),
                               str(getattr(p, "abstract", ""))]))
        if not ct or not pt:
            continue
        j = len(ct & pt) / len(ct | pt)
        if j < threshold:
            continue
        text = str(getattr(p, "title", ""))
        relation = "SUPPORTS" if j >= 0.45 else "EXTENDS"
        if any(w in text.lower() for w in ("fails", "negative", "does not")):
            relation = "CONTRADICTS"
        rels.append(PriorArtRelation(
            claim_id=claim.claim_id,
            paper_id=str(getattr(p, "paper_id", "")),
            relation=relation,
            evidence=f"token_jaccard={j:.3f}; title={text[:120]}",
            confidence=round(j, 4)))
    return sorted(rels, key=lambda r: -r.confidence)


def assess_novelty(claims: list[AtomicClaim], papers: list[Any],
                   *, coverage: dict[str, Any],
                   llm_hints: list[str] | None = None) -> dict[str, Any]:
    """Return per-claim novelty. Abstain when coverage/providers insufficient."""
    cov_ok = bool(coverage.get("status") == "OK"
                  and coverage.get("n_unique_papers", 0) >= 3
                  and len(coverage.get("providers", [])) >= 1)
    verdicts: list[NoveltyVerdict] = []
    for c in claims:
        rels = match_prior_art(c, papers)
        if not cov_ok:
            status = "NOT_ASSESSED"
            rationale = "literature coverage insufficient; cannot assess novelty"
        elif rels and rels[0].confidence >= 0.45:
            status = "KNOWN"
            rationale = "strong prior-art overlap with retrieved papers"
        elif rels:
            status = "INCREMENTAL"
            rationale = "partial overlap; extension of known work"
        elif len(papers) < 5:
            status = "UNCERTAIN"
            rationale = "too few retrieved papers to rule out prior art"
        else:
            status = "POTENTIALLY_NOVEL"
            rationale = ("no prior-art match in retrieved corpus; "
                         "NOT a claim of global novelty")
        verdicts.append(NoveltyVerdict(
            claim_id=c.claim_id, status=status, rationale=rationale,
            relations=rels,
            literature_check="performed" if cov_ok else "not_performed"))
    return {"version": "m7_novelty_v1",
            "literature_check": "performed" if cov_ok else "not_performed",
            "claims": [c.to_dict() for c in claims],
            "verdicts": [v.to_dict() for v in verdicts],
            "llm_hints": list(llm_hints or []),
            "note": "LLM hints are advisory only; never assert global novelty"}


__all__ = ["NOVELTY_STATUSES", "RELATIONS", "AtomicClaim",
           "PriorArtRelation", "NoveltyVerdict", "extract_claims",
           "match_prior_art", "assess_novelty"]
