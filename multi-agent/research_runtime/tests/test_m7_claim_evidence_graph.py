"""M7 G8/G9: Claim-Evidence Graph append-only and claim lifecycle tests."""
from __future__ import annotations

import hashlib

import pytest

from research_runtime.discovery.knowledge import (GraphError, KGNode,
                                                   ClaimEvidenceGraph)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def claim(graph, claim_id: str) -> None:
    graph.add_node(KGNode(claim_id, "ScientificClaim",
                          {"text": claim_id}, sha("source")))


def advance(graph, claim_id: str, states: list[str], tag: str) -> None:
    for state in states:
        graph.transition(claim_id, state, evidence_sha=sha(tag + state),
                         reason="test")


def test_graph_history_is_append_only_and_tamper_evident(tmp_path):
    g = ClaimEvidenceGraph(tmp_path)
    claim(g, "c1")
    advance(g, "c1", ["LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED",
                      "PREREGISTERED", "EXPERIMENTED", "EVIDENCE_REVIEWED",
                      "SUPPORTED"], "flow")
    assert len(g.history("c1")) == 6
    assert g.verify_integrity()["ok"]
    frozen = g.finalize()
    assert frozen["ok"] and frozen["n_nodes"] == 1

    g.obj["lifecycle_events"][0]["reason"] = "tampered"
    assert not g.verify_integrity()["ok"]


def test_supported_is_not_accepted_and_replication_is_dependent(tmp_path):
    g = ClaimEvidenceGraph(tmp_path)
    claim(g, "c1")
    g.add_node(KGNode("ev1", "Evidence", {"result": "single run"}, sha("r1")))
    g.add_edge(source_id="ev1", target_id="c1", relation="SUPPORTS",
               evidence_sha=sha("result1"))
    advance(g, "c1", ["LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED",
                      "PREREGISTERED", "EXPERIMENTED", "EVIDENCE_REVIEWED",
                      "SUPPORTED"], "flow")
    assert g.can_accept("c1")["reason"] == "not_replicated"
    with pytest.raises(GraphError, match="illegal lifecycle"):
        g.transition("c1", "ACCEPTED", evidence_sha=sha("bad"))

    g.add_node(KGNode("rep1", "Replication", {"run": "independent"},
                      sha("rep")))
    g.add_edge(source_id="rep1", target_id="c1", relation="REPLICATES",
               evidence_sha=sha("rep-result"))
    advance(g, "c1", ["REPLICATED", "ACCEPTED"], "confirm")
    assert g.can_accept("c1")["ok"]
    assert g.verify_integrity()["ok"]


def test_contradiction_blocks_acceptance_without_deleting_history(tmp_path):
    g = ClaimEvidenceGraph(tmp_path)
    claim(g, "c1")
    g.add_node(KGNode("ev1", "Evidence", {"result": "support"}, sha("r1")))
    g.add_node(KGNode("ev2", "CounterEvidence", {"result": "refutes"},
                      sha("r2")))
    g.add_edge(source_id="ev1", target_id="c1", relation="SUPPORTS",
               evidence_sha=sha("support"))
    g.add_edge(source_id="ev2", target_id="c1", relation="REFUTES",
               evidence_sha=sha("refute"))
    assert len(g.supports("c1")) == 1
    assert len(g.contradictions("c1")) == 1
    advance(g, "c1", ["LITERATURE_CHECKED", "HYPOTHESIS_REGISTERED",
                      "PREREGISTERED", "EXPERIMENTED", "EVIDENCE_REVIEWED",
                      "SUPPORTED"], "flow")
    assert g.can_accept("c1")["reason"] == "not_replicated"
    g.add_node(KGNode("rep1", "Replication", {}, sha("rep")))
    g.add_edge(source_id="rep1", target_id="c1", relation="REPLICATES",
               evidence_sha=sha("rep-result"))
    g.transition("c1", "REPLICATED", evidence_sha=sha("rep-evidence"))
    assert g.can_accept("c1")["reason"] == "unresolved_contradiction"
    with pytest.raises(GraphError, match="acceptance gate"):
        g.transition("c1", "ACCEPTED", evidence_sha=sha("bad"))
    assert any(e["relation"] == "SUPPORTS" for e in g.supports("c1"))


def test_supersession_is_new_edge_and_retires_old_claim(tmp_path):
    g = ClaimEvidenceGraph(tmp_path)
    claim(g, "old")
    claim(g, "new")
    g.add_edge(source_id="new", target_id="old", relation="EXTENDS",
               evidence_sha=sha("extends"))
    g.supersede(old_claim_id="old", new_claim_id="new",
                evidence_sha=sha("supersede"), rationale="better explanation")
    old = next(n for n in g.obj["nodes"] if n["node_id"] == "old")
    assert old["status"] == "SUPERSEDED"
    assert any(e["relation"] == "SUPERSEDES" for e in g.obj["edges"])
    assert g.verify_integrity()["ok"]


def test_edge_requires_real_sha_and_existing_nodes(tmp_path):
    g = ClaimEvidenceGraph(tmp_path)
    claim(g, "c1")
    with pytest.raises(GraphError, match="64-hex"):
        g.add_edge(source_id="c1", target_id="c1", relation="SUPPORTS",
                   evidence_sha="not-a-sha")
    with pytest.raises(GraphError, match="existing nodes"):
        g.add_edge(source_id="c1", target_id="missing", relation="SUPPORTS",
                   evidence_sha=sha("ok"))
