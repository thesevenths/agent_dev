"""M5-00 -- REAL two-round E2E acceptance (OPT-IN).

Set M5_00_RUN_REAL=1 to run the real SciNet++ loop. Without it this module
SKIPS: engine-free CI must never be reported as a real-engine pass.

The real driver writes ``M5_00_runs/e2e/m5_e2e_result.json``; this test validates
the produced evidence chain, not a fabricated fixture.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "M5_00_runs" / "e2e"


@pytest.mark.skipif(os.environ.get("M5_00_RUN_REAL") != "1",
                    reason="real-engine E2E is opt-in (M5_00_RUN_REAL=1)")
def test_two_real_rounds_with_experience_chain():
    result_path = RUN / "m5_e2e_result.json"
    assert result_path.is_file(), (
        "run `python run_m5_00_e2e.py` first; no result is fabricated here")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rounds = result["rounds"]
    assert len(rounds) >= 2
    r1, r2 = rounds[0], rounds[1]
    # each round has a full trusted chain
    for rd in (r1, r2):
        for key in ("objective_id", "plan_id", "approval_id", "campaign_id",
                    "report_id", "experience_id"):
            assert rd.get(key), f"round {rd['round_no']} missing {key}"
        assert rd.get("gate_passed") is True
        assert rd.get("director_state") == "COMPLETED"
    # Round 2 explicitly cites round-1 Experience
    assert r1["experience_id"]
    assert r2["prior_experience_ref"] == r1["experience_id"]
    assert r2["objective_id"] != r1["objective_id"]
    selected = [x for x in r2["discovery_source"]["ranked"]
                if x.get("objective_id") == r2["objective_id"]]
    assert selected and selected[0]["generation_source"] == "historical_experience"
    assert r2.get("reused_prior_objective_ids"), (
        "later Discovery must reuse prior objectives without overwriting them")
    assert result["integrity_ok"] is True
    # per-round evidence files persisted
    for n in (1, 2):
        assert (RUN / "rounds" / f"round_{n:02d}.json").is_file()
    # event-log round tags point at the EXECUTING round
    with sqlite3.connect(RUN / "research.sqlite") as conn:
        events = conn.execute(
            "SELECT event, round_no, payload_json FROM m5_loop_events "
            "ORDER BY seq").fetchall()
    assert events
    for event, round_no, payload_json in events:
        if event not in ("create", "preflight"):
            assert round_no >= 1, (event, round_no)
        payload = json.loads(payload_json)
        if "round_no" in payload:
            assert payload["round_no"] == round_no, (event, payload, round_no)
