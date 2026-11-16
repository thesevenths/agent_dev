"""M5-00 E2E driver: TWO real SciNet++ rounds (thin autonomous loop).

Run from the project root. Uses a REAL ResearchService + ResearchDirector + the
original 12-step gate; approvals are granted by a SEPARATE approval_cli subprocess
(the same mechanism the pilot used), never by the loop itself.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from research_runtime.autonomous_research_loop import (
    AutonomousResearchLoop, LoopConfig,
)
from research_runtime.domains.scinet import scinet_research_policy
from research_runtime.evidence_store import EvidenceStore
from research_runtime.research_environment import build_default_research_service

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "M5_00_runs" / "e2e"
DB = OUT / "research.sqlite"


def resolve_source_experiment() -> str:
    """Pick a REAL evaluate experiment with metrics + a diagnosis."""
    from research_runtime.research_service import default_evidence_provider
    events = EvidenceStore().query(event_type="result")
    last: dict[str, tuple[str, dict]] = {}
    for e in events:
        p = e.get("payload") or {}
        r = p.get("result") or {}
        eid = str(p.get("experiment_id") or r.get("experiment_id") or "")
        if eid:
            last[eid] = (str(e.get("at") or ""), r)
    usable = [(at, eid) for eid, (at, r) in last.items()
              if str(r.get("stage")) == "evaluate"
              and any(k.startswith("latent_mse") or k.startswith("state_mse")
                      for k in (r.get("metrics") or {}))]
    strong: list[str] = []
    for _at, eid in sorted(usable, reverse=True):
        try:
            _res, reports = default_evidence_provider(eid)
        except Exception:
            continue
        if reports:
            strong.append(eid)
        if len(strong) >= 4:
            break
    if not strong:
        raise SystemExit("no real experiment with a diagnosis; refusing E2E")
    return strong[0]


def make_approver(db: str):
    def grant(approval_id: str, plan_id: str) -> None:
        r = subprocess.run(
            [sys.executable, "-m",
             "research_runtime.research_store.approval_cli",
             "--db", db, "grant", approval_id,
             "--approved-by", "m5_e2e_operator", "--yes"],
            cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"approval_cli grant failed rc={r.returncode}\n{r.stdout}\n"
                f"{r.stderr}")
        print(f"  [external approval] {approval_id} granted via CLI subprocess",
              flush=True)
    return grant


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source = resolve_source_experiment()
    print(f"source experiment: {source}", flush=True)
    service = build_default_research_service(
        "scinet", db_path=DB,
        domain_policy=scinet_research_policy(paired_seeds=(42, 43)))
    config = LoopConfig(
        loop_id="m5_e2e_loop", domain="scinet", run_id="arr_m34_001",
        max_rounds=2, source_experiment_id=source)
    loop = AutonomousResearchLoop(
        service=service, config=config, out_dir=OUT,
        approver=make_approver(str(DB)))
    t0 = time.time()
    result = loop.run()
    result["elapsed_s"] = round(time.time() - t0, 2)
    (OUT / "m5_e2e_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    print(json.dumps({
        "state": result["state"], "n_rounds": result["n_rounds"],
        "integrity_ok": result["integrity_ok"],
        "elapsed_s": result["elapsed_s"]}, indent=2), flush=True)
    for rd in result["rounds"]:
        print(json.dumps({
            "round": rd["round_no"], "objective_id": rd["objective_id"],
            "campaign_id": rd["campaign_id"],
            "experience_id": rd["experience_id"],
            "prior_experience_ref": rd["prior_experience_ref"],
            "gate_passed": rd["gate_passed"],
            "director_state": rd["director_state"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
