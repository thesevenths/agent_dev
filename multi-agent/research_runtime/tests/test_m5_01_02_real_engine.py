"""M5-01/02 -- REAL three-domain smoke evidence (OPT-IN).

Set M5_01_02_RUN_REAL=1 only after running ``python run_m5_01_02_smoke.py``.
Without the flag this module SKIPS; engine-free CI must never be reported as a
real-engine pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "M5_01_02_runs" / "e2e"


@pytest.mark.skipif(os.environ.get("M5_01_02_RUN_REAL") != "1",
                    reason="three-domain real smoke is opt-in")
def test_three_real_domains_and_cross_domain_experience():
    path = RUN / "m5_01_02_result.json"
    assert path.is_file(), "run run_m5_01_02_smoke.py first; no fake evidence"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["state"] == "COMPLETED"
    results = {r.get("domain_id"): r for r in result["summary"]["results"]}
    assert set(results) == {"scinet", "embodied", "quant"}
    for domain, rec in results.items():
        assert rec.get("campaign_id"), (domain, rec)
        assert rec.get("director_state") == "COMPLETED", (domain, rec)
        assert rec.get("budget", {}).get("consumed") == 1, (domain, rec)
    assert result["preflight"]["quant"]["detail"]["synthetic_data"] is True
    assert {r["source_domain"] for r in result["experience"]} == set(results)
    assert {r["domain_id"] for r in result["domain_records"]} == set(results)
    assert result["cross_domain_snapshot"]["experience_available"] is True
