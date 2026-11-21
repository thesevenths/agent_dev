"""M7 engine-free coverage + adversarial tests (no network, no engine)."""
from __future__ import annotations

import hashlib
import json
import urllib.error

import pytest

from research_runtime.discovery.domain_packs import load_pack
from research_runtime.discovery.experiment.designer import (
    design_experiment, freeze_preregistration, verify_preregistration)
from research_runtime.discovery.hypotheses.engine import (
    IndependentCritic, ScientificHypothesis, generate_competing_hypotheses,
    review_all)
from research_runtime.discovery.literature import (LiteratureStore, Paper,
                                                    ResearchQuestionInput,
                                                    SearchQuery, expand_query)
from research_runtime.discovery.literature.base import validate_paper
from research_runtime.discovery.literature.providers import OpenAlexProvider
from research_runtime.discovery.novelty.assessor import (assess_novelty,
                                                         extract_claims)
from research_runtime.discovery.questions import (ResearchQuestion, assess_value,
                                                   expected_information_gain)
from research_runtime.harness_rsi.benchmark_v2 import (BenchmarkSpecV2,
                                                       BenchmarkTask,
                                                       HoldoutAccessVerifier)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def paper(pid: str, *, doi: str = "", title: str = "world model study",
          abstract: str = "action conditioned multi-step prediction") -> Paper:
    return Paper(paper_id=pid, doi=doi, title=title, authors=["A"],
                 year=2020, abstract=abstract, source="test", url="",
                 citations=1, references=[], retrieved_at="2026-01-01",
                 query_id="q1", content_sha=sha("full " + pid),
                 metadata_sha=sha("meta " + pid))


def test_query_expansion_covers_six_strategies():
    inp = ResearchQuestionInput(question="world model", domain="embodied",
                                concepts=["world model", "generalization"],
                                target_claims=["action conditioning fails"])
    queries = expand_query(inp)
    strategies = {q.strategy for q in queries}
    assert {"exact_phrase", "synonym", "technical", "related_method",
            "alternative_explanation", "contradictory"} <= strategies
    assert len({q.query_id for q in queries}) == len(queries)


def test_paper_doi_dedup_and_metadata_tampering(tmp_path):
    store = LiteratureStore(tmp_path)
    p1 = paper("p1", doi="10.1000/abc", title="Original title")
    p2 = paper("p2", doi="10.1000/abc", title="Tampered title")
    assert p1.metadata_sha != p2.metadata_sha
    assert store.add_papers([p1, p2])["n_dropped"] == 1
    assert len(store._all_papers()) == 1


def test_fake_doi_and_false_full_text_are_rejected(tmp_path):
    bad_doi = paper("bad", doi="not-a-doi")
    bad_full = paper("full", doi="10.1000/full")
    object.__setattr__(bad_full, "full_text_available", True)
    object.__setattr__(bad_full, "content_sha", bad_full.content_digest)
    assert "invalid_doi" in validate_paper(bad_doi)["reasons"]
    assert "full_text_claim_without_full_text" in validate_paper(
        bad_full)["reasons"]
    store = LiteratureStore(tmp_path)
    assert store.add_papers([bad_doi, bad_full])["n_invalid"] == 2
    assert store._all_papers() == []


def test_retry_and_rate_limit_are_recorded():
    calls = {"n": 0}

    def transport(url, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                url, 429, "rate limited", {"Retry-After": "0"}, None)
        return json.dumps({"results": [{"id": "https://openalex.org/W1",
                                        "doi": "https://doi.org/10.1/x",
                                        "title": "World model",
                                        "publication_year": 2024}],
                           "meta": {}})

    provider = OpenAlexProvider(transport=transport)
    budget = {"max_requests": 4, "requests_used": 0}
    query = SearchQuery("q1", "world model", "exact_phrase", "openalex", "")
    out = provider.search(query, limit=1, budget=budget)
    assert out and out[0].source == "openalex"
    assert any(e.get("status") == "rate_limited" for e in provider.events)
    assert budget["requests_used"] == 1


def test_novelty_abstention_and_known_claim(tmp_path):
    claims = extract_claims("q1", "action conditioned world model improves "
                            "multi-step prediction", ["action conditioning fails"])
    coverage_zero = {"status": "INSUFFICIENT", "n_unique_papers": 0,
                     "providers": []}
    abstain = assess_novelty(claims, [], coverage=coverage_zero)
    assert abstain["literature_check"] == "not_performed"
    assert all(v["status"] == "NOT_ASSESSED" for v in abstain["verdicts"])
    papers = [paper("p1", title=claims[0].text,
                    abstract=claims[0].text)]
    cov = {"status": "OK", "n_unique_papers": 3, "providers": ["openalex"]}
    known = assess_novelty(claims, papers, coverage=cov)
    assert any(v["status"] == "KNOWN" for v in known["verdicts"])
    for verdict in known["verdicts"]:
        for rel in verdict["relations"]:
            assert rel["paper_id"] == "p1"  # never invented prior art


def test_value_assessment_and_information_gain():
    q = ResearchQuestion(question_id="q1", title="t", domain="embodied",
                         motivation="m", literature_gap="g",
                         falsifiability_requirements=["a", "b"],
                         evidence_refs=["paper:p1"])
    novelty = {"verdicts": [{"status": "POTENTIALLY_NOVEL"}]}
    v = assess_value(q, novelty=novelty, cost_units=2.0)
    assert len(v.scores) == 7 and len(v.rationales) == 7
    assert v.expected_information_gain > 0
    assert expected_information_gain([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_competing_hypotheses_and_unfalsifiable_rejection():
    q = ResearchQuestion(question_id="q1", title="t", domain="embodied",
                         motivation="m", literature_gap="g",
                         evidence_refs=["paper:p1"])
    hs = review_all(generate_competing_hypotheses(q))
    assert {h.kind for h in hs} == {"H1", "H2", "H3", "H0"}
    assert all(h.status == "ACCEPTED" for h in hs)
    bad = ScientificHypothesis(
        hypothesis_id="bad", question_id="q1", kind="H1",
        mechanism="magic", assumptions=[], predictions=["good things"],
        falsifying_observations=[], confounders=[], required_data=[],
        required_experiments=[], literature_refs=["paper:p1"], confidence=0.5)
    assert IndependentCritic().review(bad)[0] == "REJECTED"


def test_preregistration_is_immutable_and_metric_change_fails(tmp_path):
    q = ResearchQuestion(question_id="q1", title="t", domain="embodied",
                         motivation="m", literature_gap="g")
    h = generate_competing_hypotheses(q)[0]
    protocol = design_experiment(
        question=q, hypotheses=[h], family_manifest_sha=sha("fam"),
        split_manifest_sha=sha("split"), dataset_manifest_sha=sha("data"),
        primary_metric="state_mse@1", mme=0.01)
    frozen = freeze_preregistration(protocol, tmp_path / "pre.json")
    assert verify_preregistration(protocol, tmp_path / "pre.json")["ok"]
    raw = json.loads((tmp_path / "pre.json").read_text(encoding="utf-8"))
    raw["primary_metric"] = "state_mse@5"
    (tmp_path / "pre.json").write_text(json.dumps(raw), encoding="utf-8")
    assert not verify_preregistration(protocol, tmp_path / "pre.json")["ok"]
    assert frozen["preregistration_sha"] == protocol.sha256


def test_seeds_are_not_families_and_holdout_quarantine_is_enforced(tmp_path):
    pack = load_pack("embodied_world_model")
    ident = pack.dataset_identity()
    fam = ident["family_manifest"]
    assert fam["seed_is_not_family"] is True
    assert fam["splits"]["validation"]["families"] != \
        fam["splits"]["test"]["families"]
    tasks = [BenchmarkTask(
        task_id="m7_test_ep1", family_id="m7::test:ep1", domain="embodied",
        objective="o", task_config={}, task_sha=sha("task"), split="holdout",
        source={}, budget={}, seed=42, primary_metric="state_mse@1",
        direction="minimize")]
    spec = BenchmarkSpecV2(version="v1", tasks=tasks)
    verifier = HoldoutAccessVerifier(spec, tmp_path / "holdout_access.json")
    denied = verifier.authorize(purpose="candidate_tuning",
                                experiment_id="e",
                                requested_task_ids=["m7_test_ep1"],
                                actor="generator")
    assert not denied["ok"] and denied["reason"] == "HOLDOUT_LEAKAGE"
    allowed = verifier.authorize(purpose="final_holdout_evaluation",
                                 experiment_id="e",
                                 requested_task_ids=["m7_test_ep1"],
                                 actor="coordinator")
    assert allowed["ok"] and verifier.verify_chain()["ok"]
