"""M7.1 -- prior-art assessment with evidence-limited novelty states.

Token similarity is a retrieval signal only. It is never treated as proof of
novelty. Abstract-only coverage can downgrade a claim but cannot produce
LIKELY_NOVEL; absence in a small corpus is not absence in the world.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

NOVELTY_STATUSES = ("LIKELY_NOVEL", "POSSIBLE_PRIOR_ART", "NOT_NOVEL",
                    "NOVELTY_UNCERTAIN")
RELATIONS = ("SUPPORTS", "CONTRADICTS", "EXTENDS", "REPRODUCES",
             "USES", "SUPERSEDES", "UNRELATED")
_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|;\s*")


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    question_id: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriorArtRelation:
    claim_id: str
    paper_id: str
    relation: str
    evidence: str
    confidence: float
    method: str = "lexical_jaccard_screening"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NoveltyVerdict:
    claim_id: str
    status: str
    rationale: str
    relations: list[PriorArtRelation] = field(default_factory=list)
    literature_check: str = "not_performed"
    source_coverage: str = "ABSTRACT_ONLY"
    limitations: list[str] = field(default_factory=list)
    # M7.6 P1-F: a lexical screen can only ever be a SCREENING signal. Global
    # novelty is never CONFIRMED from metadata/abstracts alone.
    evidence_grade: str = "SCREENING_ONLY"
    confirmed_novelty: bool = False

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
    for p in parts:
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
    """Lexical SCREENING only; every relation binds to a real retrieved paper."""
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
                   llm_hints: list[str] | None = None,
                   source_coverage: str = "ABSTRACT_ONLY") -> dict[str, Any]:
    """Return evidence-limited novelty verdicts for each atomic claim."""
    cov_ok = bool(coverage.get("status") == "OK"
                  and coverage.get("n_unique_papers", 0) >= 3
                  and len(coverage.get("providers", [])) >= 1)
    multi_source = len(coverage.get("providers", [])) >= 2
    full_text = source_coverage.upper() == "FULL_TEXT"
    verdicts: list[NoveltyVerdict] = []
    for c in claims:
        rels = match_prior_art(c, papers)
        limits = [f"source_coverage={source_coverage}",
                  "lexical screening is not semantic or experimental review"]
        if not cov_ok:
            status, rationale = ("NOVELTY_UNCERTAIN",
                                 "literature coverage insufficient")
        elif rels and rels[0].confidence >= 0.45:
            # P1-F: lexical overlap is a prior-art SCREENING HIT, never proof
            # of NOT_NOVEL. Downgraded from NOT_NOVEL to POSSIBLE_PRIOR_ART,
            # which honestly signals "needs semantic/experimental review".
            status, rationale = ("POSSIBLE_PRIOR_ART",
                                 "strong lexical prior-art overlap (screening only; "
                                 "NOT a confirmed non-novelty verdict)")
        elif rels:
            status, rationale = ("POSSIBLE_PRIOR_ART",
                                 "partial prior-art screening hit")
        elif full_text and multi_source and len(papers) >= 10:
            status, rationale = ("LIKELY_NOVEL",
                                 "no prior-art hit in full-text multi-source corpus")
        else:
            status, rationale = ("NOVELTY_UNCERTAIN",
                                 "no screening hit but coverage cannot support novelty")
        verdicts.append(NoveltyVerdict(
            claim_id=c.claim_id, status=status, rationale=rationale,
            relations=rels,
            literature_check="performed" if cov_ok else "not_performed",
            source_coverage=source_coverage, limitations=limits,
            evidence_grade="SCREENING_ONLY",
            confirmed_novelty=False))
    return {"version": "m7_1_novelty_v1",
            "literature_check": "performed" if cov_ok else "not_performed",
            "source_coverage": source_coverage,
            "claims": [c.to_dict() for c in claims],
            "verdicts": [v.to_dict() for v in verdicts],
            "llm_hints": list(llm_hints or []),
            "note": "Absence from a screened corpus is not global novelty."}


__all__ = ["NOVELTY_STATUSES", "RELATIONS", "AtomicClaim",
           "PriorArtRelation", "NoveltyVerdict", "extract_claims",
           "match_prior_art", "assess_novelty"]
