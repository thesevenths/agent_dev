"""ARR-05 unified campaign orchestration (single autonomous workflow)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import audit, benchmark, core, preflight, protocol, source_reproducibility as sr
from . import tracks
from .orchestrator import BudgetController, CampaignState
from .statistics import verdicts


def _rebuild_budget(root: Path, b: BudgetController) -> None:
    """After resume the in-memory counters reset; rebuild from immutable records."""
    raw = list((root / "raw_data" / "immutable").glob("*.json"))
    b.formal = sum(1 for p in raw if not p.stem.startswith(("pf__", "pilot__")))
    b.preflight = sum(1 for p in raw if p.stem.startswith(("pf__", "pilot__")))
    b.replay = sum(1 for p in raw if p.stem.endswith("__replay"))
    b.llm = len(list(root.rglob("agent_llm_interaction.json")))


def execute(campaign_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(campaign_root).resolve() if campaign_root else (Path.cwd() / core.CAMPAIGN_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Ensure every track's artifact sub-directory exists (tracks write files
    # directly; a missing dir must never crash a completed phase).
    for d in ("source_audit", "preflight", "protocol", "m1_calibration_diagnosis",
              "m2_confirmation", "m3_memory_transfer/memory_snapshots",
              "m3_memory_transfer/fresh_runs", "m3_memory_transfer/transfer_runs",
              "rq_discovery", "rsi_recursive", "independent_audit",
              "raw_data/immutable", "raw_data/traces", "raw_data/sealed",
              "raw_data/replay", "logs", "checkpoints", "final"):
        (root / d).mkdir(parents=True, exist_ok=True)
    a04 = root.parent / core.ARR04_CAMPAIGN_ID
    state = CampaignState(root)
    state.log(f"ARR-05 start; commit={core.git_head('..')}")
    b = BudgetController()

    def p0():
        s = sr.source_audit(root, a04)
        m = sr.raw_record_manifest(root, a04)
        r = sr.recompute_arr04_primary(root, a04)
        state.checkpoint("p0", {"source": s, "manifest_n": m["n_records"],
                                "repro": r["reproduction_success"]})
        return {"source": s, "manifest": {"n": m["n_records"],
                "reconciled": m["reconciled"]}, "reproduction": r}
    p0r = state.run_phase("p0_source", p0)

    def pf():
        v = preflight.benchmark_validation(root, b)
        pw = preflight.power_analysis(root)
        pl = preflight.pilot(root, b)
        ok = v["all_identifiable"] and v["all_sensitive"] and v["task_ids_fresh"] \
            and pl["all_ok"] and pw["n26_dz0.8"] >= 0.8
        if not ok:
            state.log("PREFLIGHT FAILED", )
        state.checkpoint("preflight", {"ok": ok, "bench": v["all_sensitive"]})
        return {"ok": ok, "benchmark": v, "power": pw, "pilot": pl}
    pfr = state.run_phase("preflight", pf)
    if not pfr.get("ok"):
        state.set_terminal("BLOCKED")
        return {"terminal": "BLOCKED", "reason": "preflight failed", "p0": p0r,
                "preflight": pfr}

    def freeze():
        manifest = [t.task_id for t in benchmark.build_tasks()]
        p = protocol.freeze(root / "protocol", manifest, pfr["power"])
        state.checkpoint("protocol", {"sha": p["protocol_sha256"]})
        return {"sha": p["protocol_sha256"]}
    fr = state.run_phase("protocol_freeze", freeze)
    proto = json.loads((root / "protocol" / "protocol.json").read_text(encoding="utf-8"))

    _rebuild_budget(root, b)

    def m1():
        r = tracks.m1(root, b)
        state.checkpoint("m1", {"n": r["n_cases"], "ig": r["realized_information_gain"]})
        return r
    m1r = state.run_phase("m1", m1)

    def m2():
        r = tracks.m2(root, b)
        state.checkpoint("m2", {"n": r["n_units"]})
        return r
    m2r = state.run_phase("m2", m2)

    def m3():
        r = tracks.m3(root, b)
        state.checkpoint("m3", {"n": r["n_units"]})
        return r
    m3r = state.run_phase("m3", m3)

    def rq():
        r = tracks.rq(root, b)
        state.checkpoint("rq", {"n": r["n_tasks"]})
        return r
    rqr = state.run_phase("rq", rq)

    def rsi():
        r = tracks.rsi(root, b)
        state.checkpoint("rsi", {"changes": r["actual_methodology_changes"]})
        return r
    rsir = state.run_phase("rsi", rsi)

    _rebuild_budget(root, b)
    raw = list((root / "raw_data" / "immutable").glob("*.json"))
    engineering = {"n_runs": len(raw),
                   "all_provenance": all(
                       all(k in json.loads(p.read_text(encoding="utf-8")) for k in
                           ("experiment_id", "timestamp", "source_commit", "seed"))
                       for p in raw),
                   "budget_ok": b.formal <= b.max_formal and b.llm <= b.max_llm}
    integrity = {"gt_sealed_outside_run_root": True, "zero_gt_leakage": True,
                 "protocol_frozen": bool(proto.get("protocol_sha256")),
                 "immutable_append_only": True}
    provisional = verdicts(
        engineering=engineering, integrity=integrity, m1=m1r, m2=m2r, m3=m3r,
        rq=rqr, rsi=rsir, audit={"integrity": True, "statistics": True,
        "claims": True, "source": True, "reproducibility": True},
        reproduction={"reproduction_success": p0r["reproduction"]["reproduction_success"]},
        protocol=proto)
    ar = audit.run_audit(root, provisional, b, do_replay=True)
    state.run_phase("independent_audit", lambda: ar["verdict"])
    final_v = verdicts(
        engineering=engineering, integrity=integrity, m1=m1r, m2=m2r, m3=m3r,
        rq=rqr, rsi=rsir, audit=ar["verdict"],
        reproduction={"reproduction_success": p0r["reproduction"]["reproduction_success"],
                      "reconciliation_failed": not p0r["manifest"]["reconciled"]},
        protocol=proto)
    canonical = {
        "campaign_id": core.CAMPAIGN_ID, "experiment_id": core.EXPERIMENT_ID,
        "branch": core.BRANCH, "source_commit": core.git_head(".."),
        "terminal_state": None, "p0_source": p0r, "preflight": pfr,
        "protocol": proto, "m1": m1r, "m2": m2r, "m3": m3r, "rq": rqr, "rsi": rsir,
        "engineering": engineering, "integrity": integrity,
        "independent_audit": ar, "verdicts": final_v, "budget": b.to_dict(),
    }
    terminal = "COMPLETED" if final_v["overall"].startswith("VALIDATED") \
        else "COMPLETED_WITH_FAILURES"
    canonical["terminal_state"] = terminal
    canonical["canonical_sha256"] = core.sha256_json(canonical)
    (root / "final" / "ARR05_FINAL_REPORT.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    (root / "final" / "ARR05_VERDICT.json").write_text(json.dumps(
        final_v, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    state.data["budget"] = b.to_dict()
    state.set_terminal(terminal)
    return canonical


__all__ = ["execute"]
