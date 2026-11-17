"""M7.5 integrated benchmark and final verdict tests."""
from __future__ import annotations

import json
from pathlib import Path


def test_integrated_pilot_is_inconclusive_and_groups_are_sealed():
    metrics = json.loads(Path(
        "M7_5_INTEGRATED/metrics/metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads(Path(
        "M7_5_INTEGRATED/protocols/experiment_protocol.json").read_text(
            encoding="utf-8"))
    assert metrics["verdict"] == "INCONCLUSIVE"
    assert set(metrics["per_group"]) == {"A_BASE", "B_MEMORY", "C_RSI",
                                         "D_FULL"}
    assert protocol["mme"] == 0.10
    assert protocol["min_families"] == 8


def test_final_verdict_does_not_overclaim():
    verdict = json.loads(Path("ARR_CAPABILITY_VERDICT.json").read_text(
        encoding="utf-8"))
    for key in ("engineering_complete", "scientific_judgment",
                "research_memory", "recursive_improvement",
                "integrated_capability"):
        assert verdict[key] == "INCONCLUSIVE"
    assert verdict["confirmed_new_findings"] == []
    assert "M7.1-D10b formal Dispatcher/Campaign lifecycle remains partial" \
        in verdict["unresolved_issues"]
