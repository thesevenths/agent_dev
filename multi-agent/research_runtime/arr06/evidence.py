"""ARR-06 P0: explicit Evidence / Claim / Hypothesis system.

A Claim cannot become VERIFIED merely because an LLM generated it. Every
Claim must reference immutable raw evidence. Verification is evidence-driven
and reproducible, independent of the generating agent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import core

STATUSES = ("PROVISIONAL", "VERIFIED", "CONTRADICTED", "REJECTED", "INCONCLUSIVE")


@dataclass
class Evidence:
    evidence_id: str
    experiment_id: str
    claim_text: str
    metric: str
    value: float
    raw_record_path: str
    raw_sha256: str
    protocol_hash: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    statement: str
    source: str                       # "LLM" | "ANALYSIS" | "PROTOCOL"
    hypothesis_id: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    status: str = "PROVISIONAL"
    confidence: float = 0.0
    verification_note: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = core.sha256_json(
                {k: v for k, v in asdict(self).items() if k != "provenance_hash"})
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    competing: list[str] = field(default_factory=list)
    falsifiable_prediction: str = ""
    discriminative_experiment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceStore:
    """Immutable evidence registry backed by raw records on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.path = self.root / "evidence_system" / "evidence_store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) \
            if self.path.is_file() else {"evidence": {}, "claims": {},
                                         "hypotheses": {}, "verification_log": []}

    def add_evidence(self, ev: Evidence) -> None:
        raw = Path(ev.raw_record_path)
        if not raw.is_file():
            raise RuntimeError(f"evidence raw record missing: {raw}")
        if core.sha256_file(raw) != ev.raw_sha256:
            raise RuntimeError(f"evidence raw hash mismatch: {raw}")
        self.data["evidence"][ev.evidence_id] = ev.to_dict()
        self.save()

    def register_claim(self, c: Claim) -> None:
        if c.source == "LLM" and c.status == "VERIFIED":
            raise RuntimeError("LLM-generated claim cannot be self-VERIFIED")
        self.data["claims"][c.claim_id] = c.to_dict()
        self.save()

    def verify_claim(self, claim_id: str, *, min_supporting: int = 1,
                     require_no_contradiction: bool = True) -> dict[str, Any]:
        """Evidence-driven verification, independent of claim generator."""
        c = self.data["claims"][claim_id]
        sup = [self.data["evidence"][e] for e in c["supporting_evidence"]
               if e in self.data["evidence"]]
        con = [self.data["evidence"][e] for e in c["contradicting_evidence"]
               if e in self.data["evidence"]]
        ok = len(sup) >= min_supporting
        if require_no_contradiction and con:
            ok = False
        new_status = "VERIFIED" if ok else (
            "CONTRADICTED" if con and not sup else "INCONCLUSIVE")
        self.data["claims"][claim_id]["status"] = new_status
        self.data["claims"][claim_id]["verification_note"] = (
            f"evidence-driven: {len(sup)} supporting, {len(con)} contradicting")
        self.data["verification_log"].append({
            "claim_id": claim_id, "from": c["status"], "to": new_status,
            "timestamp": core.now_iso(), "source": c["source"]})
        self.save()
        return {"claim_id": claim_id, "status": new_status,
                "supporting": len(sup), "contradicting": len(con)}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True,
                                        ensure_ascii=False, default=str),
                             encoding="utf-8")


__all__ = ["STATUSES", "Evidence", "Claim", "Hypothesis", "EvidenceStore"]
