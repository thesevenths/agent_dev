"""M7.1 regression tests for audited P0/P1 defects."""
from __future__ import annotations

import hashlib

from research_runtime.discovery.hypotheses.engine import (
    IndependentCritic, generate_competing_hypotheses, review_all)
from research_runtime.discovery.literature.base import Paper
from research_runtime.discovery.novelty.assessor import (assess_novelty,
                                                         extract_claims)
from research_runtime.discovery.questions import (NOT_ASSESSED,
                                                  ResearchQuestion,
                                                  assess_value,
                                                  expected_information_gain_from_model)
from research_runtime.harness_rsi.statistics_v2 import (PreregistrationV2,
                                                        analyse_family_confirmation)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_abstract_only_no_match_is_novelty_uncertain():
    claims = extract_claims("q", "action conditioned world model improves "
                            "multi step prediction")
    cov = {"status": "OK", "n_unique_papers": 6, "providers": ["openalex"]}
    papers = [Paper(paper_id=f"p{i}", doi=f"10.1/x{i}", title="unrelated topic",
                    authors=["A"], year=2024, abstract="chemistry catalysis",
                    source="openalex", url="", citations=0, references=[],
                    retrieved_at="", query_id="q", content_sha=sha("c"),
                    metadata_sha=sha("m")) for i in range(6)]
    verdict = assess_novelty(claims, papers, coverage=cov)
    assert verdict["source_coverage"] == "ABSTRACT_ONLY"
    assert all(v["status"] == "NOVELTY_UNCERTAIN" for v in verdict["verdicts"])


def test_full_text_multi_source_can_support_likely_novel():
    claims = extract_claims("q", "action conditioned world model improves "
                            "multi step prediction")
    cov = {"status": "OK", "n_unique_papers": 12,
           "providers": ["openalex", "crossref"]}
    papers = [Paper(paper_id=f"p{i}", doi=f"10.1/y{i}", title="unrelated topic",
                    authors=["A"], year=2024, abstract="chemistry catalysis",
                    source="x", url="", citations=0, references=[],
                    retrieved_at="", query_id="q", content_sha=sha("c"),
                    metadata_sha=sha("m")) for i in range(12)]
    verdict = assess_novelty(claims, papers, coverage=cov,
                             source_coverage="FULL_TEXT")
    assert all(v["status"] == "LIKELY_NOVEL" for v in verdict["verdicts"])


def test_value_does_not_invent_eig_or_priority():
    q = ResearchQuestion(question_id="q", title="t", domain="d",
                         motivation="m", literature_gap="g")
    result = assess_value(q, novelty={"verdicts": [],
                                      "literature_check": "performed"})
    assert result.status == NOT_ASSESSED
    assert result.priority is None
    assert result.expected_information_gain == NOT_ASSESSED
    bad = expected_information_gain_from_model(
        hypotheses=["H0", "H1"], priors=[0.5, 0.5], likelihoods={})
    assert bad["status"] == NOT_ASSESSED


def test_stronger_critic_checks_counterexample_and_confounders():
    q = ResearchQuestion(question_id="q", title="t", domain="d",
                         motivation="m", literature_gap="g",
                         evidence_refs=["paper:p1"])
    reviewed = review_all(generate_competing_hypotheses(q))
    assert len(reviewed) == 4
    assert all(h.status == "ACCEPTED" for h in reviewed)
    assert all(h.counterexamples for h in reviewed)
    assert IndependentCritic().review_competition(reviewed)["ok"]


def test_family_confirmation_with_no_families_is_inconclusive_not_rejected():
    pre = PreregistrationV2(
        experiment_id="e", primary_metric="m", primary_component="c",
        metric_direction="minimize",
        minimum_practical_effect=0.01, alpha=0.05, target_power=0.8,
        minimum_families=4, seed=42)
    result = analyse_family_confirmation(prereg=pre, results=[])
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["reason"] == "insufficient_independent_families"
