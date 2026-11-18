"""M7 -- Verified Research Memory with a separate Candidate Memory tier.

M7 reuses the M6 Experience Bank as the raw experience source, but adds a
claim-level, evidence-bound memory. Unverified speculation is stored only in
Candidate Memory and can never be retrieved as verified knowledge.

All writes are append-only JSONL with a hash chain. A verified record must have:
  * a claim whose graph lifecycle is not REFUTED/INCONCLUSIVE/SUPERSEDED,
  * at least one real evidence SHA (optionally re-read from disk),
  * an experiment id, source SHA, domain, applicability and limitations,
  * a verification status from VERIFIED_STATUSES.

Retrieval scores evidence strength, domain similarity, stated applicability,
matched experimental conditions and prior failure history. Refuted/superseded
claims are always excluded.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ...harness_rsi.identity import freeze_json, is_sha256, sha256_json

VERIFIED_STATUSES = ("VERIFIED", "REPLICATED", "FAILED_WITH_CAUSE",
                     "BOUNDED_METHOD")
CANDIDATE_STATUSES = ("PROPOSED", "UNVERIFIED", "AWAITING_REPLICATION")
# ARR v1.0 WP-05: six memory CLASSES (orthogonal to candidate/verified tier),
# and the full lifecycle. Old records default to CANDIDATE-like classes so the
# append-only hash chain and stored artifacts stay compatible.
MEMORY_CLASSES = ("CANDIDATE", "VERIFIED", "FAILED_EXPERIMENT",
                  "METHODOLOGY", "LIMITATION", "CONTRADICTION")
MEMORY_LIFECYCLE = ("CANDIDATE", "EVIDENCE_ATTACHED", "UNDER_VERIFICATION",
                    "VERIFIED", "CONTRADICTED", "DEPRECATED", "REVOKED")
MemoryResolver = Callable[[str], Path | None]


class MemoryError(ValueError):
    """Raised when a memory record fails its evidence/verification gate."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class ResearchMemoryRecord:
    memory_id: str
    claim_id: str
    experiment_id: str
    experience_id: str
    evidence_ids: list[str] = field(default_factory=list)
    source_sha: str = ""
    verification_status: str = "UNVERIFIED"
    domain: str = ""
    applicability: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    experimental_conditions: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
    created_at: str = field(default_factory=_now)
    # ---- ARR v1.0 WP-05/§12.3 additional fields (defaulted for compatibility)
    memory_class: str = "CANDIDATE"
    lifecycle: str = "CANDIDATE"
    evidence_shas: list[str] = field(default_factory=list)
    source_round: str = ""
    method_sha: str = ""
    contradictions: list[str] = field(default_factory=list)
    supersedes: str = ""
    last_verified_at: str = ""

    def __post_init__(self) -> None:
        # Backward compatibility: derive class from verification_status for
        # legacy records constructed without an explicit memory_class.
        if self.memory_class == "CANDIDATE" and self.verification_status in (
                "VERIFIED", "REPLICATED", "FAILED_WITH_CAUSE", "BOUNDED_METHOD"):
            self.memory_class = (
                "FAILED_EXPERIMENT" if self.verification_status == "FAILED_WITH_CAUSE"
                else "METHODOLOGY" if self.verification_status == "BOUNDED_METHOD"
                else "VERIFIED")
            self.lifecycle = "VERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


class _AppendOnlyLedger:
    def __init__(self, path: Path, kind: str):
        self.path = path
        self.kind = kind
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        if self.path.is_file():
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    self.events.append(json.loads(raw))
            self._replay()

    def _replay(self) -> None:
        for event in self.events:
            rec = event["record"]
            self.records[rec["memory_id"]] = rec

    def append(self, record: ResearchMemoryRecord) -> dict[str, Any]:
        if record.memory_id in self.records:
            existing = self.records[record.memory_id]
            if existing != record.to_dict():
                raise MemoryError("memory identity changed (append-only)")
            return {"ok": True, "duplicate": True}
        event = {"event_type": f"{self.kind}_memory_written",
                 "record": record.to_dict(),
                 "record_sha": record.sha256,
                 "prev_sha": self.events[-1]["event_sha"]
                 if self.events else "0" * 64}
        event["event_sha"] = sha256_json(event)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self.events.append(event)
        self.records[record.memory_id] = record.to_dict()
        return {"ok": True, "record_sha": record.sha256,
                "event_sha": event["event_sha"]}

    def verify_integrity(self) -> dict[str, Any]:
        prev = "0" * 64
        reasons: list[str] = []
        for event in self.events:
            body = {k: v for k, v in event.items() if k != "event_sha"}
            if sha256_json(body) != event["event_sha"]:
                reasons.append("event_sha_mismatch")
            if event["prev_sha"] != prev:
                reasons.append("hash_chain_break")
            if sha256_json(event["record"]) != event["record_sha"]:
                reasons.append("record_sha_mismatch")
            prev = event["event_sha"]
        return {"ok": not reasons, "reasons": reasons,
                "n_events": len(self.events)}


def _tokens(text: str) -> set[str]:
    return {t for t in str(text).lower().replace("_", " ").split()
            if len(t) >= 3}


def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


class VerifiedResearchMemory:
    """Two-tier, claim-level research memory backed by one append-only store."""

    def __init__(self, out_dir: str | Path, *, graph: Any = None,
                 evidence_resolver: MemoryResolver | None = None,
                 db: Any = None):
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.graph = graph
        self.resolver = evidence_resolver
        self.db = db
        self.verified = _AppendOnlyLedger(root / "verified_memory.jsonl",
                                          "verified")
        self.candidate = _AppendOnlyLedger(root / "candidate_memory.jsonl",
                                           "candidate")

    def _validate(self, rec: ResearchMemoryRecord, *, verified: bool
                  ) -> dict[str, Any]:
        problems: list[str] = []
        if not rec.memory_id or not rec.claim_id or not rec.experiment_id:
            problems.append("missing_binding")
        if not rec.source_sha or not is_sha256(rec.source_sha):
            problems.append("invalid_source_sha")
        if not rec.evidence_ids or any(not is_sha256(e)
                                       for e in rec.evidence_ids):
            problems.append("invalid_evidence_sha")
        if not rec.domain or not rec.applicability or not rec.limitations:
            problems.append("missing_scope")
        if not 0.0 <= rec.confidence <= 1.0:
            problems.append("invalid_confidence")
        if verified and rec.verification_status not in VERIFIED_STATUSES:
            problems.append("unverified_cannot_be_promoted")
        if not verified and rec.verification_status in VERIFIED_STATUSES:
            problems.append("verified_record_belongs_in_verified_tier")
        if self.resolver is not None:
            for evidence_sha in rec.evidence_ids:
                path = self.resolver(evidence_sha)
                if path is None or not Path(path).is_file():
                    problems.append(f"evidence_missing:{evidence_sha[:12]}")
                    continue
                digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                if digest != evidence_sha:
                    problems.append(f"evidence_sha_mismatch:{evidence_sha[:12]}")
        if verified and self.graph is None:
            problems.append("claim_graph_missing")
        if verified and self.graph is not None:
            gate = self._claim_gate(rec)
            if not gate["ok"]:
                problems.append(gate["reason"])
        return {"ok": not problems, "reasons": sorted(set(problems))}

    def _claim_gate(self, rec: ResearchMemoryRecord) -> dict[str, Any]:
        graph = self.graph
        node = next((n for n in graph.obj["nodes"]
                     if n["node_id"] == rec.claim_id), None)
        if node is None:
            return {"ok": False, "reason": "claim_missing_from_graph"}
        status = node.get("status", "PROPOSED")
        if status in ("REFUTED", "INCONCLUSIVE", "SUPERSEDED"):
            return {"ok": False, "reason": "claim_invalidated"}
        if rec.verification_status == "REPLICATED":
            can = graph.can_accept(rec.claim_id)
            if not can["ok"]:
                return {"ok": False,
                        "reason": "replication_gate_failed:" + can["reason"]}
        if rec.verification_status == "VERIFIED" and status not in (
                "SUPPORTED", "REPLICATED", "ACCEPTED"):
            return {"ok": False, "reason": "claim_not_evidence_reviewed"}
        if rec.verification_status == "FAILED_WITH_CAUSE" and not rec.rationale:
            return {"ok": False, "reason": "failure_cause_missing"}
        if rec.verification_status == "BOUNDED_METHOD" and not rec.applicability:
            return {"ok": False, "reason": "applicability_missing"}
        return {"ok": True, "reason": "claim_gate_passed"}

    def add_candidate(self, rec: ResearchMemoryRecord) -> dict[str, Any]:
        v = self._validate(rec, verified=False)
        if not v["ok"]:
            raise MemoryError("|".join(v["reasons"]))
        return self.candidate.append(rec)

    def add_verified(self, rec: ResearchMemoryRecord) -> dict[str, Any]:
        v = self._validate(rec, verified=True)
        if not v["ok"]:
            raise MemoryError("|".join(v["reasons"]))
        result = self.verified.append(rec)
        if self.db is not None:
            self._write_db(rec)
        return result

    def promote(self, candidate_id: str, *, verification_status: str,
                confidence: float, rationale: str = "") -> dict[str, Any]:
        """Create a NEW verified record; the candidate record is never edited."""
        raw = self.candidate.records.get(candidate_id)
        if raw is None:
            raise MemoryError("candidate_memory_missing")
        promoted = ResearchMemoryRecord(
            **{**raw, "verification_status": verification_status,
               "confidence": confidence,
               "rationale": rationale or raw.get("rationale", ""),
               "created_at": _now()})
        return self.add_verified(promoted)

    def _write_db(self, rec: ResearchMemoryRecord) -> None:
        with self.db.session() as conn:
            conn.execute("""INSERT INTO m7_verified_memory (
                memory_id, claim_id, experiment_id, evidence_shas,
                verification_status, source_sha, confidence, domain,
                applicability, limitations, provenance, status, version,
                created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    rec.memory_id, rec.claim_id, rec.experiment_id,
                    json.dumps(rec.evidence_ids, sort_keys=True),
                    rec.verification_status, rec.source_sha, rec.confidence,
                    rec.domain, json.dumps(rec.applicability, sort_keys=True),
                    json.dumps(rec.limitations, sort_keys=True),
                    json.dumps(rec.to_dict(), sort_keys=True),
                    "ACTIVE", "m7_verified_memory_v1", rec.created_at))

    def retrieve(self, *, domain: str | None = None,
                 problem: str = "", conditions: dict[str, Any] | None = None,
                 top: int = 5, include_failures: bool = True
                 ) -> list[dict[str, Any]]:
        q_tokens = _tokens(problem) | _tokens(domain or "")
        cond = conditions or {}
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self.verified.records.values():
            if self.graph is not None:
                node = next((n for n in self.graph.obj["nodes"]
                             if n["node_id"] == rec["claim_id"]), None)
                if node is None or node.get("status") in (
                        "REFUTED", "INCONCLUSIVE", "SUPERSEDED"):
                    continue
            if domain and rec["domain"] != domain:
                continue
            if not include_failures and rec["verification_status"] == \
                    "FAILED_WITH_CAUSE":
                continue
            status_weight = {"REPLICATED": 1.0, "VERIFIED": 0.8,
                             "BOUNDED_METHOD": 0.7,
                             "FAILED_WITH_CAUSE": 0.6}.get(
                                 rec["verification_status"], 0.0)
            evidence_strength = min(1.0, len(rec["evidence_ids"]) / 3.0)
            domain_sim = 1.0 if domain == rec["domain"] else (
                _overlap(q_tokens, _tokens(rec["domain"])))
            applicability = _overlap(q_tokens, rec["applicability"])
            condition_fit = 1.0 if not cond else _overlap(
                cond.keys(), rec["experimental_conditions"].keys())
            score = (0.25 * status_weight + 0.2 * evidence_strength
                     + 0.15 * domain_sim + 0.15 * applicability
                     + 0.15 * condition_fit + 0.1 * rec["confidence"])
            scored.append((score, {**rec, "retrieval_score": round(score, 4)}))
        scored.sort(key=lambda sr: (-sr[0], sr[1]["memory_id"]))
        return [rec for _, rec in scored[:top]]

    def verify_integrity(self) -> dict[str, Any]:
        verified = self.verified.verify_integrity()
        candidate = self.candidate.verify_integrity()
        return {"ok": verified["ok"] and candidate["ok"],
                "verified": verified, "candidate": candidate}

    def finalize(self) -> dict[str, Any]:
        integrity = self.verify_integrity()
        if not integrity["ok"]:
            return {"ok": False, "reasons": ["memory_integrity_failed"]}
        frozen = freeze_json({
            "version": "m7_memory_v1",
            "verified": list(self.verified.records.values()),
            "candidate": list(self.candidate.records.values()),
            "integrity": integrity}, self.root / "memory_snapshot.json")
        return {"ok": True, "memory_sha": frozen["sha256"],
                "n_verified": len(self.verified.records),
                "n_candidate": len(self.candidate.records)}


__all__ = ["VERIFIED_STATUSES", "CANDIDATE_STATUSES", "MEMORY_CLASSES",
           "MEMORY_LIFECYCLE", "MemoryError",
           "ResearchMemoryRecord", "VerifiedResearchMemory"]
