"""M7 -- Claim-Evidence Knowledge Graph (append-only, SHA-bound).

Nodes: ResearchQuestion/Paper/ScientificClaim/Hypothesis/Experiment/Dataset/
Model/Method/Result/Evidence/CounterEvidence/Replication/ResearchExperience.
Edges: SUPPORTS/REFUTES/CONTRADICTS/DERIVED_FROM/TESTED_BY/REPLICATES/
USES_DATASET/USES_METHOD/EXTENDS/SUPERSEDES.

Original claims and evidence are NEVER deleted or rewritten: a later refutation
is a NEW REFUTES edge, not a mutation of the old SUPPORTS edge. Lifecycle:
PROPOSED->LITERATURE_CHECKED->HYPOTHESIS_REGISTERED->PREREGISTERED->
EXPERIMENTED->EVIDENCE_REVIEWED->SUPPORTED->REPLICATED->ACCEPTED; other
terminals REFUTED/INCONCLUSIVE/SUPERSEDED. SUPPORTED != ACCEPTED.

Status on a node is a *mutable projection* of the append-only lifecycle event
log. Node identity deliberately excludes that projection so replay/integrity
checks remain meaningful.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...harness_rsi.identity import freeze_json, is_sha256, sha256_json

NODE_TYPES = ("ResearchQuestion", "Paper", "ScientificClaim", "Hypothesis",
              "Experiment", "Dataset", "Model", "Method", "Result",
              "Evidence", "CounterEvidence", "Replication",
              "ResearchExperience")
EDGE_TYPES = ("SUPPORTS", "REFUTES", "CONTRADICTS", "DERIVED_FROM",
              "TESTED_BY", "REPLICATES", "USES_DATASET", "USES_METHOD",
              "EXTENDS", "SUPERSEDES",
              # ARR v1.0 ADR-003: an INCONCLUSIVE verdict is a NEUTRAL evidence
              # link -- it must NEVER become a REFUTES/REPLICATES edge (that
              # would turn uncertainty into a contradiction and poison the
              # memory promotion gate). The Evidence node is still preserved.
              "INCONCLUSIVE_EVIDENCE")
LIFECYCLE = ("PROPOSED", "LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED",
             "PREREGISTERED", "EXPERIMENTED", "EVIDENCE_REVIEWED",
             "SUPPORTED", "REPLICATED", "ACCEPTED", "REFUTED",
             "INCONCLUSIVE", "SUPERSEDED")

_TRANSITIONS: dict[str, set[str]] = {
    "PROPOSED": {"LITERATURE_CHECKED", "REFUTED", "INCONCLUSIVE"},
    "LITERATURE_CHECKED": {"HYPOTHESIS_REGISTERED", "REFUTED",
                           "INCONCLUSIVE"},
    "HYPOTHESIS_REGISTERED": {"PREREGISTERED", "REFUTED", "INCONCLUSIVE"},
    "PREREGISTERED": {"EXPERIMENTED", "INCONCLUSIVE"},
    "EXPERIMENTED": {"EVIDENCE_REVIEWED", "REFUTED", "INCONCLUSIVE"},
    "EVIDENCE_REVIEWED": {"SUPPORTED", "REFUTED", "INCONCLUSIVE"},
    "SUPPORTED": {"REPLICATED", "REFUTED", "INCONCLUSIVE", "SUPERSEDED"},
    "REPLICATED": {"ACCEPTED", "SUPERSEDED"},
    "ACCEPTED": {"SUPERSEDED", "REFUTED"},
    "REFUTED": set(),
    "INCONCLUSIVE": set(),
    "SUPERSEDED": set(),
}
_TERMINAL = {"REFUTED", "INCONCLUSIVE", "SUPERSEDED"}


class GraphError(ValueError):
    """Raised on illegal node/edge or lifecycle transition."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class KGNode:
    node_id: str
    node_type: str
    payload: dict[str, Any]
    source_sha: str
    status: str = "PROPOSED"
    version: str = "m7_kg_v1"
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_payload(self) -> dict[str, Any]:
        """Immutable identity: lifecycle status is a projection, not identity."""
        return {"node_id": self.node_id, "node_type": self.node_type,
                "payload": self.payload, "source_sha": self.source_sha,
                "version": self.version, "created_at": self.created_at}

    @property
    def sha256(self) -> str:
        return sha256_json(self.identity_payload())


@dataclass(frozen=True)
class KGEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    evidence_sha: str
    rationale: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


class ClaimEvidenceGraph:
    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir) / "claim_evidence_graph.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.obj: dict[str, Any] = (json.loads(self.path.read_text(
            encoding="utf-8")) if self.path.is_file() else {
                "version": "m7_kg_v1", "nodes": [], "edges": [],
                "lifecycle_events": []})
        for key in ("nodes", "edges", "lifecycle_events"):
            self.obj.setdefault(key, [])

    def _node(self, node_id: str) -> dict[str, Any]:
        node = next((n for n in self.obj["nodes"]
                     if n["node_id"] == node_id), None)
        if node is None:
            raise GraphError(f"node not found: {node_id}")
        return node

    def add_node(self, node: KGNode) -> dict[str, Any]:
        if node.node_type not in NODE_TYPES:
            raise GraphError(f"unknown node type {node.node_type}")
        if node.status != "PROPOSED":
            raise GraphError("new nodes must start at PROPOSED")
        if not is_sha256(node.source_sha):
            raise GraphError("node source_sha must be 64-hex SHA")
        existing = next((n for n in self.obj["nodes"]
                         if n["node_id"] == node.node_id), None)
        if existing is not None:
            if existing.get("node_sha") != node.sha256:
                raise GraphError("node identity changed (append-only)")
            return {"ok": True, "duplicate": True,
                    "node_sha": existing["node_sha"]}
        stored = {**node.to_dict(), "node_sha": node.sha256}
        self.obj["nodes"].append(stored)
        return {"ok": True, "node_sha": node.sha256}

    def add_edge(self, *, source_id: str, target_id: str, relation: str,
                 evidence_sha: str, rationale: str = "") -> dict[str, Any]:
        if relation not in EDGE_TYPES:
            raise GraphError(f"unknown edge type {relation}")
        if not is_sha256(evidence_sha):
            raise GraphError("edge evidence_sha must be 64-hex SHA")
        ids = {n["node_id"] for n in self.obj["nodes"]}
        if source_id not in ids or target_id not in ids:
            raise GraphError("edge endpoints must be existing nodes")
        source = self._node(source_id)
        target = self._node(target_id)
        if relation == "SUPERSEDES" and not (
                source["node_type"] == "ScientificClaim"
                and target["node_type"] == "ScientificClaim"):
            raise GraphError("SUPERSEDES must link two ScientificClaim nodes")
        eid = "edge_" + hashlib.sha256(
            f"{source_id}|{target_id}|{relation}|{evidence_sha}".encode()
        ).hexdigest()[:16]
        edge = KGEdge(eid, source_id, target_id, relation, evidence_sha,
                      rationale, _now())
        stored = {**edge.to_dict(), "edge_sha": edge.sha256}
        if any(e["edge_id"] == eid for e in self.obj["edges"]):
            if any(e.get("edge_sha") == edge.sha256 for e in self.obj["edges"]):
                return {"ok": True, "duplicate": True, "edge_id": eid}
            raise GraphError("edge_id collision with different content")
        self.obj["edges"].append(stored)
        return {"ok": True, "edge_id": eid, "edge_sha": edge.sha256}

    def supports(self, claim_id: str) -> list[dict[str, Any]]:
        self._node(claim_id)
        return [e for e in self.obj["edges"]
                if e["target_id"] == claim_id
                and e["relation"] in ("SUPPORTS", "REPLICATES")]

    def replications(self, claim_id: str) -> list[dict[str, Any]]:
        return [e for e in self.supports(claim_id)
                if e["relation"] == "REPLICATES"]

    def contradictions(self, claim_id: str) -> list[dict[str, Any]]:
        self._node(claim_id)
        return [e for e in self.obj["edges"]
                if e["target_id"] == claim_id
                and e["relation"] in ("REFUTES", "CONTRADICTS")]

    def superseded_by(self, claim_id: str) -> list[dict[str, Any]]:
        return [e for e in self.obj["edges"]
                if e["target_id"] == claim_id and e["relation"] == "SUPERSEDES"]

    def history(self, claim_id: str) -> list[dict[str, Any]]:
        return [e for e in self.obj["lifecycle_events"]
                if e["claim_id"] == claim_id]

    def transition(self, claim_id: str, to_state: str,
                   *, evidence_sha: str, reason: str = "") -> dict[str, Any]:
        if to_state not in LIFECYCLE:
            raise GraphError(f"unknown lifecycle state {to_state}")
        if not is_sha256(evidence_sha):
            raise GraphError("transition evidence_sha must be 64-hex SHA")
        node = self._node(claim_id)
        if node["node_type"] != "ScientificClaim":
            raise GraphError("transition target is not a ScientificClaim")
        current = node["status"]
        allowed = _TRANSITIONS[current]
        if to_state == "SUPERSEDED" and current not in _TERMINAL:
            allowed = set(allowed) | {"SUPERSEDED"}
        if to_state not in allowed:
            raise GraphError(f"illegal lifecycle transition {current}->{to_state}")
        if to_state == "ACCEPTED":
            gate = self.can_accept(claim_id)
            if not gate["ok"]:
                raise GraphError(f"acceptance gate failed: {gate['reason']}")
        # Append-only: record the transition event; do NOT mutate originals.
        event = {"claim_id": claim_id, "from": current, "to": to_state,
                 "evidence_sha": evidence_sha, "reason": reason,
                 "at": _now()}
        event["event_sha"] = sha256_json(event)
        self.obj["lifecycle_events"].append(event)
        node["status"] = to_state
        node["updated_at"] = event["at"]
        return {"ok": True, "event_sha": event["event_sha"]}

    def supersede(self, *, old_claim_id: str, new_claim_id: str,
                  evidence_sha: str, rationale: str = "") -> dict[str, Any]:
        """Append a SUPERSEDES edge and retire the old claim (never delete it)."""
        old = self._node(old_claim_id)
        new = self._node(new_claim_id)
        if old["node_type"] != "ScientificClaim" or \
                new["node_type"] != "ScientificClaim":
            raise GraphError("supersession requires ScientificClaim nodes")
        edge = self.add_edge(source_id=new_claim_id, target_id=old_claim_id,
                             relation="SUPERSEDES",
                             evidence_sha=evidence_sha, rationale=rationale)
        if old["status"] not in _TERMINAL:
            self.transition(old_claim_id, "SUPERSEDED",
                            evidence_sha=evidence_sha,
                            reason=rationale or "superseded by newer claim")
        return edge

    def can_accept(self, claim_id: str, *, min_replications: int = 1) -> dict:
        node = next((n for n in self.obj["nodes"]
                     if n["node_id"] == claim_id), None)
        if not node:
            return {"ok": False, "reason": "claim_missing"}
        if node["status"] not in ("REPLICATED", "ACCEPTED"):
            return {"ok": False, "reason": "not_replicated"}
        if self.contradictions(claim_id):
            return {"ok": False, "reason": "unresolved_contradiction"}
        if self.superseded_by(claim_id):
            return {"ok": False, "reason": "superseded"}
        n_rep = len({e["evidence_sha"] for e in self.replications(claim_id)})
        if n_rep < min_replications:
            return {"ok": False, "reason": "insufficient_replication"}
        return {"ok": True, "reason": "acceptance_conditions_met",
                "n_independent_replications": n_rep}

    def verify_integrity(self) -> dict[str, Any]:
        reasons: list[str] = []
        for node in self.obj["nodes"]:
            recomputed = sha256_json({
                "node_id": node["node_id"], "node_type": node["node_type"],
                "payload": node["payload"], "source_sha": node["source_sha"],
                "version": node.get("version", "m7_kg_v1"),
                "created_at": node["created_at"]})
            if recomputed != node.get("node_sha"):
                reasons.append(f"node_sha_mismatch:{node['node_id']}")
        ids = {n["node_id"] for n in self.obj["nodes"]}
        for edge in self.obj["edges"]:
            recomputed = sha256_json({k: v for k, v in edge.items()
                                      if k != "edge_sha"})
            if recomputed != edge.get("edge_sha"):
                reasons.append(f"edge_sha_mismatch:{edge['edge_id']}")
            if edge["source_id"] not in ids or edge["target_id"] not in ids:
                reasons.append(f"edge_dangling:{edge['edge_id']}")
        replay: dict[str, str] = {}
        for event in self.obj["lifecycle_events"]:
            body = {k: v for k, v in event.items() if k != "event_sha"}
            if sha256_json(body) != event.get("event_sha"):
                reasons.append(f"event_sha_mismatch:{event['event_sha']}")
            if event["claim_id"] not in ids:
                reasons.append(f"event_unknown_claim:{event['claim_id']}")
            replay[event["claim_id"]] = event["to"]
        for node in self.obj["nodes"]:
            if node["node_type"] == "ScientificClaim":
                expected = replay.get(node["node_id"], "PROPOSED")
                if node["status"] != expected:
                    reasons.append(f"status_projection_mismatch:{node['node_id']}")
        return {"ok": not reasons, "reasons": reasons,
                "n_nodes": len(self.obj["nodes"]),
                "n_edges": len(self.obj["edges"]),
                "n_events": len(self.obj["lifecycle_events"])}

    def finalize(self) -> dict[str, Any]:
        integrity = self.verify_integrity()
        if not integrity["ok"]:
            return {"ok": False, "reasons": integrity["reasons"]}
        frozen = freeze_json(self.obj, self.path)
        return {"ok": True, "graph_sha": frozen["sha256"],
                **integrity, "path": frozen["path"]}


__all__ = ["NODE_TYPES", "EDGE_TYPES", "LIFECYCLE", "GraphError",
           "KGNode", "KGEdge", "ClaimEvidenceGraph"]
