"""M7.6 P1 regressions: benchmark integrity + evidence-driven discovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_runtime.capability_bench import integrated_bench as ib
from research_runtime.capability_bench import memory_bench as mb
from research_runtime.capability_bench import scientific_judgment as sj
from research_runtime.capability_bench import rsi_bench as rb
from research_runtime.discovery.novelty.assessor import (
    assess_novelty, extract_claims)
from research_runtime.discovery.questions import (
    ResearchQuestion, generate_questions, assess_value)
from research_runtime.harness_rsi.identity import freeze_json


# --- P1-B: scientific judgment multi-primary gate -------------------------

def _sj_setup(tmp_path, *, n=34, expert="PERFORMED", evidence_correct=0.9,
              novelty_acc=1.0):
    root = tmp_path / "sj"
    (root / "raw_results").mkdir(parents=True)
    for i in range(n):
        row = {"case_id": f"c{i}", "category": "KNOWN", "prediction": {
            "novelty": "LIKELY_NOVEL" if i < int(n * novelty_acc) else "NOT_NOVEL",
            "falsifiability": "FALSIFIABLE",
            "evidence_sufficiency": "SUFFICIENT" if i < int(n * evidence_correct)
                                    else "INSUFFICIENT",
            "abstain": False}}
        freeze_json(row, root / "raw_results" / f"c{i}.json")
    gt = {f"c{i}": {"novelty": "LIKELY_NOVEL", "falsifiability": "FALSIFIABLE",
                    "evidence_sufficiency": "SUFFICIENT",
                    "abstention_expected": False} for i in range(n)}
    manifest = {"expert_blind_review": expert, "min_cases": 30}
    return sj.compute_metrics(results_dir=root, ground_truth=gt,
                              benchmark_manifest=manifest)


def test_novelty_alone_cannot_pass(tmp_path):
    m = _sj_setup(tmp_path, novelty_acc=1.0, evidence_correct=0.4)
    assert m["novelty_accuracy"] == 1.0
    assert m["verdict"] in ("FAIL", "INCONCLUSIVE")
    assert "evidence_sufficiency_accuracy" in m["preregistered_outcomes"]["primary"]


def test_bad_expert_review_forces_inconclusive(tmp_path):
    m = _sj_setup(tmp_path, expert="NOT_PERFORMED", evidence_correct=1.0)
    assert m["verdict"] == "INCONCLUSIVE"


# --- P1-C: memory uses ground truth ---------------------------------------

def test_memory_repeat_failure_uses_ground_truth(tmp_path):
    root = tmp_path / "mem" / "data"
    out = tmp_path / "mem" / "run"
    (root / "ground_truth").mkdir(parents=True)
    (out / "raw_results").mkdir(parents=True)
    # Ground truth: s01 actually repeats (YES).
    freeze_json({"expected_repeat_failure": "YES", "relevant_memory_ids": [],
                 "shuffled_memory_ids": []}, root / "ground_truth" / "s01.json")
    # Model PREDICTS no repeat -> accuracy 0, confusion FN.
    for arm in mb.ARMS:
        freeze_json({"scenario_id": "s01", "arm": arm, "context_chars": 100,
                     "prediction": {"repeat_failure": False,
                                    "used_memory_ids": [],
                                    "rejected_memory_ids": []}},
                    out / "raw_results" / f"s01_{arm}.json")
    m = mb.compute_metrics(root=root, results_dir=out,
                           manifest={"expert_review": "NOT_PERFORMED",
                                     "min_results": 4})
    on = m["per_arm"]["ON"]
    assert on["repeat_failure_accuracy"] == 0.0
    assert on["repeat_failure_confusion"]["fn"] == 1
    assert m["verdict"] == "INCONCLUSIVE"  # unreviewed cannot PASS


# --- P1-D: exact sampling + family-level stats ----------------------------

def test_select_tasks_returns_exact_n_and_unique():
    cases = [{"case_id": f"c{i}", "category": f"cat{i % 7}"} for i in range(7)]
    selected = ib.select_tasks(cases, n=8)
    # only 7 eligible -> cannot fabricate 8; must not silently duplicate.
    assert len(selected) == 7
    assert len({c["case_id"] for c in selected}) == 7
    cases20 = [{"case_id": f"c{i}", "category": f"cat{i % 7}"} for i in range(20)]
    selected = ib.select_tasks(cases20, n=8, seed=1)
    assert len(selected) == 8
    assert len({c["case_id"] for c in selected}) == 8
    assert [c["case_id"] for c in ib.select_tasks(cases20, 8, seed=1)] == \
           [c["case_id"] for c in ib.select_tasks(cases20, 8, seed=1)]


def test_integrated_counts_unique_families_not_rows(tmp_path):
    out = tmp_path / "results"
    (out / "raw_results").mkdir(parents=True)
    # 12 cases all from ONE family -> 1 independent family, not 12.
    for i in range(12):
        for group in ib.GROUPS:
            freeze_json({"case_id": f"c{i}", "category": "KNOWN",
                         "family_id": "F1", "group": group, "parse_ok": True,
                         "scores": {"evidence": 1, "novelty": 1,
                                    "falsifiability": 1},
                         "prompt_tokens": 10, "completion_tokens": 5,
                         "context_chars": 0},
                        out / "raw_results" / f"c{i}_{group}.json")
    m = ib.metrics(results_dir=out, mme=0.1, min_families=8)
    assert m["n_unique_families"] == 1
    assert m["verdict"] == "INCONCLUSIVE"
    assert m["per_group"]["A_BASE"]["n_cases"] == 12


# --- P1-E: bottleneck from real evidence ----------------------------------

def test_bottleneck_diagnosis_uses_real_failures():
    rows = []
    for i in range(6):
        rows.append({"method": "base", "parse_ok": True,
                     "case_id": f"c{i}",
                     "scores": {"evidence": 0 if i < 5 else 1,
                                "novelty": 1, "falsifiability": 1}})
    d = rb.diagnose_bottleneck_from_rows(rows)
    assert d["category"] == "EVIDENCE_SUFFICIENCY_ERROR"
    assert d["candidate"] == "candidate_evidence_gate"
    assert d["n_failures"] == 5
    # No failures -> no assumed bottleneck.
    clean = [{**r, "scores": {"evidence": 1, "novelty": 1,
                              "falsifiability": 1}} for r in rows]
    assert rb.diagnose_bottleneck_from_rows(clean)["candidate"] is None


# --- P1-F: novelty screen is not proof ------------------------------------

def test_jaccard_hit_is_not_confirmed_novelty():
    claims = extract_claims("q", "action conditioned world model prediction")
    papers = [type("P", (), {"paper_id": "p1",
                             "title": claims[0].text,
                             "abstract": claims[0].text})()]
    cov = {"status": "OK", "n_unique_papers": 5, "providers": ["openalex"]}
    out = assess_novelty(claims, papers, coverage=cov)
    v = out["verdicts"][0]
    assert v["status"] in ("POSSIBLE_PRIOR_ART", "NOVELTY_UNCERTAIN")
    assert v["confirmed_novelty"] is False
    assert v["evidence_grade"] == "SCREENING_ONLY"


# --- P1-A: multi-paper synthesis ------------------------------------------

def test_questions_synthesize_multiple_papers():
    gaps = [
        {"text": "action conditioned world model improves multi-step dynamics "
                 "prediction on unseen physical conditions", "paper_id": "p1",
         "doi": "10.1/a", "title": "A", "evidence_type": "ABSTRACT_SNIPPET"},
        {"text": "action conditioned latent dynamics improves multi-step "
                 "prediction under unseen conditions and limited data",
         "paper_id": "p2", "doi": "10.1/b", "title": "B",
         "evidence_type": "ABSTRACT_SNIPPET"},
    ]
    qs = generate_questions(domain="embodied", novelty={"verdicts": []},
                            literature_gaps=gaps, max_questions=4)
    assert qs
    any_multi = any(q.n_supporting_papers >= 2
                    and q.synthesis_method == "MULTI_PAPER_THEMATIC_TOKEN_SYNTHESIS"
                    for q in qs)
    assert any_multi
    assert all(q.evidence_refs for q in qs)


def test_value_not_assessed_has_no_priority():
    q = ResearchQuestion(question_id="q", title="t", domain="d",
                         motivation="m", literature_gap="g",
                         evidence_refs=[])
    v = assess_value(q, novelty={"verdicts": [], "literature_check": "not_performed"})
    assert v.priority is None
