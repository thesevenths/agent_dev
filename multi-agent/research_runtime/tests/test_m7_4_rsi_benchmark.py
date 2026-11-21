"""M7.4 RSI pilot audit tests."""
from __future__ import annotations

import json
from pathlib import Path


def test_preregistration_precedes_results_and_contains_mme():
    pre = json.loads(Path(
        "M7_4_RSI/pre_registration/preregistration.json").read_text(
            encoding="utf-8"))
    assert pre["mme"] == 0.10
    assert pre["min_families"] == 8
    assert "holdout_case_ids" in pre["splits"]
    assert pre["primary_outcome"] == "evidence_sufficiency_accuracy"


def test_no_adoption_when_development_gate_fails():
    log = json.loads(Path(
        "M7_4_RSI/promotion/promotion_log.json").read_text(encoding="utf-8"))
    decision = json.loads(Path(
        "M7_4_RSI/promotion/adoption_decision.json").read_text(
            encoding="utf-8"))
    assert log["promotion_decision"] in ("INCONCLUSIVE",
                                         "REJECTED_HOLDOUT_FAILED")
    assert decision["adoption"] == "NOT_ADOPTED"
    assert decision["human_approval"] == "NOT_PERFORMED"
