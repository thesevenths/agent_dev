"""ARR v1.0 WP-03/04/05/06 tests: judgment + memory + transfer.

Engine-free; verifies the NEW contracts with source-level behavior.
"""
from __future__ import annotations

import json

import pytest

from research_runtime.capability_bench.memory_attribution import (
    ATTRIBUTION_STATES, MemoryAttributionLedger, negative_transfer_report)
from research_runtime.discovery.experiment.designer import (
    discrimination_matrix, design_experiment)
from research_runtime.discovery.memory import (
    MEMORY_CLASSES, MEMORY_LIFECYCLE, ResearchMemoryRecord)


# --- WP-04: experiment x hypothesis discrimination matrix ------------------

class _H:
    def __init__(self, hid, preds):
        self.hypothesis_id = hid
        self.predictions = preds


def test_discrimination_matrix_detects_indistinguishable_hypotheses():
    h1 = _H("H1", ["state_mse improves with action conditioning"])
    h2 = _H("H2", ["state_mse improves with matched data"])
    out = discrimination_matrix(hypotheses=[h1, h2],
                                arms=["baseline", "action_conditioned"])
    assert out["status"] == "ASSESSED"
    assert "H1|H2" in out["pairs"]
    # Both predict "improve" everywhere -> NOT distinguishable; system must say so
    assert out["all_pairs_distinguishable"] is False


def test_discrimination_matrix_distinguishable_when_predictions_differ():
    h1 = _H("H1", ["state_mse improves with action conditioning"])
    h0 = _H("H0", ["no difference beyond MME; effect vanishes"])
    out = discrimination_matrix(hypotheses=[h1, h0],
                                arms=["baseline", "action_conditioned"])
    assert out["all_pairs_distinguishable"] is True


def test_design_experiment_embeds_discrimination(tmp_path):
    q = type("Q", (), {"question_id": "q1"})()
    hs = [_H("H1", ["state_mse improves"]), _H("H0", ["no difference"])]
    p = design_experiment(
        question=q, hypotheses=hs, family_manifest_sha="f" * 64,
        split_manifest_sha="s" * 64, dataset_manifest_sha="d" * 64,
        primary_metric="state_mse@1", mme=0.01,
        pilot_effects={"e1": 0.1, "e2": 0.12, "e3": 0.11, "e4": 0.13})
    assert p.discrimination["status"] == "ASSESSED"
    assert "all_pairs_distinguishable" in p.discrimination


# --- WP-05: six memory classes + lifecycle ---------------------------------

def test_six_memory_classes_and_lifecycle():
    assert len(MEMORY_CLASSES) == 6
    assert set(MEMORY_CLASSES) == {"CANDIDATE", "VERIFIED",
                                   "FAILED_EXPERIMENT", "METHODOLOGY",
                                   "LIMITATION", "CONTRADICTION"}
    assert "UNDER_VERIFICATION" in MEMORY_LIFECYCLE
    assert "REVOKED" in MEMORY_LIFECYCLE


def test_memory_record_class_and_backward_compat():
    legacy = ResearchMemoryRecord(
        memory_id="m1", claim_id="c1", experiment_id="e1", experience_id="x1",
        verification_status="VERIFIED")
    assert legacy.memory_class == "VERIFIED"  # derived for legacy compatibility
    method = ResearchMemoryRecord(
        memory_id="m2", claim_id="c2", experiment_id="e2", experience_id="x2",
        memory_class="METHODOLOGY", lifecycle="EVIDENCE_ATTACHED",
        verification_status="UNVERIFIED", source_round="P1_r1",
        method_sha="a" * 64)
    assert method.memory_class == "METHODOLOGY"
    assert method.source_round == "P1_r1"
    assert method.to_dict()["method_sha"] == "a" * 64


# --- WP-06: attribution seven states + negative transfer -------------------

def test_attribution_tracks_seven_distinct_states(tmp_path):
    led = MemoryAttributionLedger(root=tmp_path, run_id="r1")
    for s in ATTRIBUTION_STATES[:5]:
        led.record(scenario_id="s1", arm="ON", memory_id="mem1", state=s)
    led.record(scenario_id="s1", arm="ON", memory_id="mem1",
               state="memory_improved_outcome")
    assert led.trace(scenario_id="s1", memory_id="mem1") == list(
        ATTRIBUTION_STATES[:5]) + ["memory_improved_outcome"]
    sm = led.summary()
    assert sm["retrieved"] == 1 and sm["applied"] == 1
    assert sm["improved_outcome"] == 1
    assert sm["usage_rate"] == 1.0


def test_retrieval_is_not_equated_with_improvement(tmp_path):
    led = MemoryAttributionLedger(root=tmp_path, run_id="r2")
    led.record(scenario_id="s1", arm="ON", memory_id="m", state="memory_retrieved")
    sm = led.summary()
    assert sm["retrieved"] == 1
    assert sm["improved_outcome"] == 0
    assert sm["usage_rate"] == 0.0  # retrieved but never applied


def test_negative_transfer_report_uses_gt(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rec = {"scenario_id": "s1", "arm": "ON",
           "prediction": {"used_memory_ids": ["shuf_1", "stale_1"]}}
    (raw / "s1_ON.json").write_text(json.dumps(rec), encoding="utf-8")
    gt = {"s1": {"shuffled_memory_ids": ["shuf_1"],
                 "stale_memory_ids": ["stale_1"],
                 "contradictory_memory_ids": []}}
    out = negative_transfer_report(raw_results_dir=raw, ground_truth=gt)
    assert out["wrong_domain_use"] == 1 and out["stale_use"] == 1
    assert out["wrong_domain_rate"] == 1.0
