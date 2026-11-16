"""M5 INTEGRATED -- REAL three-layer evidence (OPT-IN).

Set M5_INTEGRATED_RUN_REAL=1 only after running the real drivers:
  python run_m5_integrated.py --phase research
  python run_m5_rrsi.py
  python run_m5_validation.py
Without the flag this module SKIPS; engine-free CI is never a real-engine pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "M5_INTEGRATED_runs"


def _load(p: Path) -> dict:
    assert p.is_file(), f"missing real artifact {p}; run the real driver first"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.skipif(os.environ.get("M5_INTEGRATED_RUN_REAL") != "1",
                    reason="M5 integrated real run is opt-in")
def test_research_round1_three_domains_and_experience_reflow():
    d = _load(RUN / "evidence" / "phase_research_evidence.json")
    assert d["run"]["state"] in ("ROUND2_CROSS_DOMAIN", "COMPLETED")
    extra = d["extra"]
    assert len(extra["round1"]["results"]) == 3
    assert all(r.get("ok") for r in extra["round1"]["results"]), extra["round1"]
    assert len(extra["experience"]) == 3
    assert set(extra["cross_domain"]["source_domains"]) == {
        "scinet", "embodied", "quant"}
    assert extra["cross_domain"]["experience_available"] is True
    # The key research-quality claim: next-round objectives are EXperience-
    # driven, not only bootstrap evidence.
    assert any("experience" in str(s) or s == "historical_experience"
               for s in extra["generation_sources"]), extra["generation_sources"]
    assert d["journal"]["ok"] is True
    assert d["global_budget"]["consumed"] == 3


@pytest.mark.skipif(os.environ.get("M5_INTEGRATED_RUN_REAL") != "1",
                    reason="M5 integrated real run is opt-in")
def test_rrsi_two_layers_structured_holdout_and_next_round_quality():
    d = _load(RUN / "rrsi" / "rrsi_result.json")
    assert d["state"] == "COMPLETED"
    dom = d["domain_model"]
    assert dom["layer"] == "DOMAIN_MODEL"
    assert dom["n_candidates"] >= 2               # real improvement + failure
    assert dom["n_retained_failures"] >= 1         # failed candidate retained
    assert dom["outcome"]["status"] == "PROMOTED"  # independent holdout passed
    assert dom["candidate"]["target"] == "model.drift_correction"
    assert dom["real_metrics"]["holdout_direction_fraction"] >= 0.6
    harness = d["harness"]
    assert harness["layer"] == "RESEARCH_HARNESS"
    assert harness.get("candidate_sha")
    # The harness layer must have a REAL evaluation source; a live SciNet++
    # ingest may be blocked by environment limits and honestly fall back to a
    # persisted real artifact (which is NOT a fake). Promotion is not claimed
    # unless the structured holdout gate actually passed.
    assert harness.get("evaluation_source") in (
        "live_real_scinet_ingest", "replayed_real_persisted_artifact")
    if harness.get("status") == "PROMOTED":
        assert harness["outcome"]["gate_passed"] is True
    # Next round is measurably better on REAL holdout of the domain engine.
    q = d["next_round_quality"]
    assert q["promoted"] is True
    assert q["holdout_improvement"] is not None and q["holdout_improvement"] > 0
    rb = d["rollback"]
    assert set(rb["layers"]) == {"DOMAIN_MODEL", "RESEARCH_HARNESS"}


@pytest.mark.skipif(os.environ.get("M5_INTEGRATED_RUN_REAL") != "1",
                    reason="M5 integrated real run is opt-in")
def test_production_validation_honest_results():
    d = _load(RUN / "validation" / "validation_result.json")
    assert d["n_checks"] >= 10
    assert d["n_pass"] == d["n_checks"]
    names = {c["name"] for c in d["checks"]}
    assert any("crash-injection" in n for n in names)
    assert any("restart" in n for n in names)
    assert any("24h" in n for n in names)
