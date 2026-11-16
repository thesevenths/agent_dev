"""M4.3-02 -- REAL SciNet++ A/B/C pilot E2E (OPT-IN, skipped in CI).

This is the ONLY M4.3-02 test that claims a REAL ENGINE run. It never uses a
FakeRunner and never synthesises evidence: when the environment is incomplete it
SKIPS explicitly (an honest skip, never a fabricated pass).

What it proves (one block is enough for the mechanism):
  1. the real preflight passes only with checkout + engine python + real evidence;
  2. three REAL arm chains run through ONE ResearchService and the 12-step gate;
  3. every arm's approval is granted OUT OF BAND by the ``approval_cli`` process;
  4. outcomes are DERIVED from persisted Campaign/Report/Experience records;
  5. the registry is append-only and integrity-verifiable;
  6. a SEPARATE process recovers the same database (no duplicate campaign);
  7. the objectives are queued for independent human blind review (pending);
  8. an evidence bundle is written.

Run manually (trains real models; minutes)::

    M4_3_RUN_REAL=1 python -m pytest \\
        research_runtime/tests/test_m4_3_02_real_engine.py -q -rs
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "arr_m34_001"


def _gate_state() -> dict:
    from research_runtime.settings import DEFAULT_SCINET_DIR, DEFAULT_SCINET_PYTHON
    scinet = Path(DEFAULT_SCINET_DIR) / "research_api" / "cli.py"
    optin = os.environ.get("M4_3_RUN_REAL") == "1"
    evidence_ok = False
    n_evidence = 0
    try:
        from research_runtime.discovery_context_provider import (
            DiscoveryContextProvider,
        )
        from research_runtime.research_store import ResearchStores
        probe_db = Path(tempfile.mkdtemp()) / "probe.sqlite"
        snap = DiscoveryContextProvider(stores=ResearchStores(probe_db)) \
            .build_snapshot(RUN_ID, objective_ids=[])
        n_evidence = int(snap.evidence_view.get("n_evidence") or 0)
        evidence_ok = n_evidence > 0
    except Exception:  # noqa: BLE001 - a gating probe must never raise
        evidence_ok = False
    return {"scinet_checkout": scinet.is_file(),
            "engine_python": Path(DEFAULT_SCINET_PYTHON).is_file(),
            "real_evidence": evidence_ok, "n_evidence": n_evidence,
            "operator_optin": optin,
            "ready": bool(scinet.is_file() and Path(DEFAULT_SCINET_PYTHON).is_file()
                          and evidence_ok and optin)}


_GATE = _gate_state()
SKIP_REASON = (
    "M4.3-02 REAL SciNet++ pilot E2E is opt-in and requires ALL of: "
    "M4_3_RUN_REAL=1, a SciNet++ checkout with research_api/cli.py, the engine "
    f"python, and REAL evidence (checkout={_GATE['scinet_checkout']}, "
    f"engine_python={_GATE['engine_python']}, "
    f"evidence={_GATE['n_evidence']}, optin={_GATE['operator_optin']}). "
    "Explicit SKIP -- never a FakeRunner, never a synthetic Snapshot.")


def test_gate_is_honest_and_engine_free():
    assert _GATE["ready"] == (_GATE["scinet_checkout"]
                              and _GATE["engine_python"]
                              and _GATE["real_evidence"]
                              and _GATE["operator_optin"])
    assert "FakeRunner" in SKIP_REASON and "synthetic" in SKIP_REASON


def _subprocess_recover(db_path: Path) -> dict:
    code = (
        "import json\n"
        "from research_runtime.m4_3.registry_store import RegistryStore\n"
        f"s=RegistryStore(db_path=r'{db_path}')\n"
        "print(json.dumps({'integrity':s.verify_integrity(),"
        "'n_runs':len(s.load_runs()),'n_blocks':len(s.load_blocks()),"
        "'campaigns':sorted({r['campaign_id'] for r in s.load_runs()})}))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"recovery subprocess failed:\n{r.stderr}"
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not _GATE["ready"], reason=SKIP_REASON)
def test_real_pilot_one_block_e2e(tmp_path):
    from research_runtime.domains.scinet import scinet_research_policy
    from research_runtime.m4_3.experiment_controller import (
        BudgetGuard, ExperimentController,
    )
    from research_runtime.m4_3.pilot_runner import (
        PilotConfig, PilotRunner, preflight,
    )
    from research_runtime.m4_3.review_queue import ReviewQueue
    from research_runtime.research_environment import build_default_research_service

    db = tmp_path / "research.sqlite"
    service = build_default_research_service(
        "scinet", db_path=db,
        domain_policy=scinet_research_policy(paired_seeds=(42, 43)))
    cfg = PilotConfig(experiment_id="pytest_real_pilot", run_id=RUN_ID,
                      n_blocks=1, paired_seeds=(42, 43), candidate_budget=3,
                      campaign_budget={"max_jobs": 8, "max_seeds": 2,
                                       "max_candidates": 2},
                      approver_id="pytest-operator", max_campaigns=3)
    rep = preflight(cfg, service=service)
    assert rep.ready, rep.to_dict()

    queue = ReviewQueue(db=service.stores.db)
    controller = ExperimentController.create(
        experiment_id=cfg.experiment_id, db=service.stores.db,
        config=cfg.to_dict(), budget=BudgetGuard(max_campaigns=3))
    controller.preflight(detail=rep.to_dict())
    controller.freeze_snapshots([f"{RUN_ID}::block_1"])
    runner = PilotRunner(service=service, config=cfg, out_dir=tmp_path / "pilot",
                         controller=controller, review_queue=queue)
    out = runner.run()

    assert out["n_runs"] == 3
    block = out["blocks"][0]
    assert set(block["outcomes"]) == {"A", "B", "C"}
    assert set(block["gate_refs"]) == {"A", "B", "C"}
    for arm, ref in block["gate_refs"].items():
        assert ref["campaign_id"] and ref["report_id"]
        assert ref["plan_id"] and ref["approval_id"]
    # Distinct real campaigns (one campaign can never serve two runs).
    assert len({r["campaign_id"] for r in block["gate_refs"].values()}) == 3
    # Real Report + Experience writeback for every arm.
    records = runner.harness.registry.all_runs()
    assert len(records) == 3
    for rec in records:
        rep = service.get_research_report(session_id=rec.gate_evidence["ref"][
            "session_id"])
        assert rep["ok"] and rep["report"]["experience_refs"]
        assert rep["report"]["frozen_test_used"] is False
    # Human blind review is QUEUED, never fabricated.
    status = queue.queue_status(cfg.experiment_id)
    assert status["n_queued"] == 3
    assert status["n_pending"] == 3
    assert runner.harness.verify_integrity() is True

    recovered = _subprocess_recover(db)
    assert recovered["integrity"] is True
    assert recovered["n_blocks"] == 1 and recovered["n_runs"] == 3
    assert len(recovered["campaigns"]) == 3

    bundle = {"stage": "M4.3-02 pilot", "engine": "REAL SciNet++",
              "preflight": rep.to_dict(), "block": block,
              "outcomes": block["outcomes"], "recovery": recovered,
              "review_status": status}
    out_dir = Path(os.environ.get("M4_3_EVIDENCE_DIR", str(tmp_path)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pilot_real_e2e_evidence.json"
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    assert path.stat().st_size > 0
