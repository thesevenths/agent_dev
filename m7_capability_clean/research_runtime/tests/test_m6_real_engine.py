"""M6 -- opt-in REAL-engine integration evidence test.

Gated by M6_RUN_REAL=1. Reads the artifacts of a real full run (SciNet++/Embodied/
Quant adapters), verifies that real engine Q values and job directories exist,
and checks the complete promotion->adoption->rollback chain. No FakeRunner and
no synthetic success is accepted here.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("M6_RUN_REAL") != "1",
    reason="opt-in real-engine M6 evidence test (set M6_RUN_REAL=1)")

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get(
    "M6_REAL_RUN_DIR", str(ROOT / "M6_EVIDENCE" / "run_real_v3")))
if not RUN.is_absolute():
    RUN = ROOT / RUN


def _load(rel: str) -> dict:
    p = RUN / rel
    assert p.is_file(), f"missing real-run artifact: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def test_real_full_run_is_complete():
    d = _load("final_result.json")
    assert d["overall"] == "COMPLETE", d.get("reason", "")
    assert d["journal_ok"] is True
    assert d["promotion"]["status"] == "PROMOTED"
    assert d["promotion"]["verification"]["verified"] is True
    assert d["rollback"]["status"] == "ROLLED_BACK"
    assert d["rollback"]["to_sha"] == d["base_sha"]
    assert d["adoption"]["record"]["loaded_sha"] == d["candidate_sha"]
    assert d["code_versions"]["agent_dev"]["head"]


def test_real_engines_produced_finite_q_and_job_artifacts():
    results = []
    for p in RUN.rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict) and "real_engine" in obj:
            results.append((p, obj["real_engine"]))
    assert results, "no real-engine evidence found in result artifacts"
    domains = set()
    for p, ev in results:
        q = ev.get("engine_Q")
        assert isinstance(q, (int, float)) and math.isfinite(q), (p, q)
        assert ev.get("domain") in ("scinet", "embodied", "quant")
        domains.add(ev["domain"])
        job = Path(ev["job_dir"])
        assert job.is_dir(), f"engine job dir missing: {job}"
    assert domains == {"scinet", "embodied", "quant"}, domains


def test_real_holdout_and_post_adoption_are_genuine():
    d = _load("final_result.json")
    hold = d["holdout"]
    assert hold["decision"] in ("ACCEPTED", "INCONCLUSIVE")
    assert hold["primary_meets_mme"] is True
    primary = next(c for c in hold["contrasts"] if c["is_primary"])
    assert primary["mean_difference"] > 0
    assert "post_adoption_metrics" in d
    assert "ablation" in d


def test_real_journal_contains_all_m6_stages():
    from research_runtime.research_store.db import ResearchDatabase
    from research_runtime.runtime_state import RuntimeState
    rid = _load("final_result.json")["run_id"]
    state = RuntimeState(db=ResearchDatabase(RUN / "runtime.sqlite"))
    events = {e["event"] for e in state.journal(rid)}
    for stage in ("m6_budget_reserved", "m6_benchmark_ready",
                  "m6_preregistered", "m6_failure_analysis",
                  "m6_candidate_frozen", "m6_paired_evaluated",
                  "m6_holdout_evaluated", "m6_promotion_recorded",
                  "m6_adoption_verified", "m6_new_round_verified",
                  "m6_rollback_tested", "m6_finalized"):
        assert stage in events, stage
    assert state.verify_journal(rid)["ok"]
