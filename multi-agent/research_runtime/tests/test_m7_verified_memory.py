"""M7 G10: Verified Memory never promotes unverified or invalidated claims."""
from __future__ import annotations

import hashlib

import pytest

from research_runtime.discovery.knowledge import KGNode, ClaimEvidenceGraph
from research_runtime.discovery.memory import (MemoryError,
                                               ResearchMemoryRecord,
                                               VerifiedResearchMemory)
from research_runtime.discovery.schema import install_m7_schema
from research_runtime.research_store.db import ResearchDatabase


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def graph_with_claim(tmp_path, status_chain: list[str] | None = None):
    g = ClaimEvidenceGraph(tmp_path)
    g.add_node(KGNode("c1", "ScientificClaim", {"text": "claim"}, sha("src")))
    g.add_node(KGNode("rep", "Replication", {}, sha("rep-src")))
    g.add_edge(source_id="rep", target_id="c1", relation="REPLICATES",
               evidence_sha=sha("rep-evidence"))
    for state in status_chain or ["LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED",
                                  "PREREGISTERED", "EXPERIMENTED",
                                  "EVIDENCE_REVIEWED", "SUPPORTED",
                                  "REPLICATED"]:
        g.transition("c1", state, evidence_sha=sha(state))
    return g


def record(**kw):
    base = dict(memory_id="mem1", claim_id="c1", experiment_id="exp1",
                experience_id="expbank1", evidence_ids=[sha("evidence")],
                source_sha=sha("source"), domain="embodied",
                applicability=["ood_dynamics"],
                limitations=["single benchmark"], confidence=0.8,
                rationale="real paired experiment")
    base.update(kw)
    return ResearchMemoryRecord(**base)


def test_unverified_cannot_enter_verified_tier(tmp_path):
    g = graph_with_claim(tmp_path, status_chain=[
        "LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED", "PREREGISTERED",
        "EXPERIMENTED", "EVIDENCE_REVIEWED", "SUPPORTED"])
    mem = VerifiedResearchMemory(tmp_path, graph=g)
    candidate = record(verification_status="UNVERIFIED", confidence=0.4)
    assert mem.add_candidate(candidate)["ok"]
    with pytest.raises(MemoryError, match="unverified_cannot_be_promoted"):
        mem.add_verified(candidate)
    with pytest.raises(MemoryError, match="replication_gate_failed"):
        mem.promote("mem1", verification_status="REPLICATED", confidence=0.9)


def test_verified_record_requires_replicated_claim_and_real_evidence(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("real result", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    g = graph_with_claim(tmp_path)
    mem = VerifiedResearchMemory(
        tmp_path, graph=g, evidence_resolver=lambda h: evidence if h == digest
        else None)
    ok = record(verification_status="REPLICATED", evidence_ids=[digest])
    assert mem.add_verified(ok)["ok"]
    assert len(mem.retrieve(domain="embodied")) == 1

    forged = record(memory_id="mem2", verification_status="REPLICATED",
                    evidence_ids=[sha("forged")])
    with pytest.raises(MemoryError, match="evidence_missing"):
        mem.add_verified(forged)
    assert mem.verify_integrity()["ok"]


def test_refuted_or_superseded_claims_are_never_retrieved(tmp_path):
    g = graph_with_claim(tmp_path)
    mem = VerifiedResearchMemory(tmp_path, graph=g)
    assert mem.add_verified(record(verification_status="REPLICATED"))["ok"]
    assert len(mem.retrieve()) == 1

    g.transition("c1", "ACCEPTED", evidence_sha=sha("accept"))
    g.transition("c1", "SUPERSEDED", evidence_sha=sha("newer"))
    assert mem.retrieve() == []


def test_memory_can_persist_to_shared_database(tmp_path):
    g = graph_with_claim(tmp_path)
    db = ResearchDatabase(tmp_path / "research.sqlite")
    install_m7_schema(db)
    mem = VerifiedResearchMemory(tmp_path / "memory", graph=g, db=db)
    assert mem.add_verified(record(verification_status="REPLICATED"))["ok"]
    rows = db.rows("SELECT memory_id, domain FROM m7_verified_memory")
    assert len(rows) == 1
    assert rows[0]["domain"] == "embodied"
