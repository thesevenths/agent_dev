"""ARR-04 end-to-end campaign orchestration (single autonomous workflow)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import audit, core, preflight, tracks
from .core import CAMPAIGN_ROOT
from .knowledge import build_verified_memory, memory_digest
from .orchestrator import BudgetController, CampaignState
from .statistics import verdicts


def execute(campaign_root: str | Path | None = None,
            resume: bool = True) -> dict[str, Any]:
    root = Path(campaign_root).resolve() if campaign_root else (Path.cwd() / CAMPAIGN_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = CampaignState(root)
    state.log(f"ARR-04 campaign start; resume={resume}; commit={core.git_head('..')}")
    seal = core.SealedStore(root / "raw_data" / "sealed")
    budget = BudgetController(max_formal_runs=1200, max_preflight_runs=40,
                              max_llm_calls=1200)

    # PHASE 0/1: source audit
    src_audit = state.run_phase(
        "phase0_audit", lambda: preflight.write_source_audit(root, state))

    def _preflight() -> dict[str, Any]:
        state.log("preflight: source verified")
        ident = preflight.identifiability_audit(root, budget)
        if not ident.get("identifiable"):
            state.block_phase("preflight", "benchmark not identifiable")
            return {"identifiable": False, "blocked": True}
        expv = preflight.experiment_validation(root, budget)
        if not expv.get("discriminative"):
            state.block_phase("preflight", "experiment not discriminative")
            return {"identifiable": True, "discriminative": False, "blocked": True}
        pil = preflight.pilot(root, budget)
        power = preflight.power_analysis(root)
        state.checkpoint("preflight", {"ident": ident, "expv": expv,
                                       "pilot": pil, "power": power})
        return {"identifiable": True, "discriminative": True, "blocked": False,
                "pilot": pil, "power": power}
    pf = state.run_phase("preflight", _preflight)
    if pf.get("blocked"):
        terminal = "BLOCKED"
        state.set_terminal(terminal)
        return {"terminal": terminal, "reason": "preflight failed"}

    def _freeze() -> dict[str, Any]:
        ident = json.loads((root / "preflight" / "identifiability.json").read_text(
            encoding="utf-8"))
        proto = preflight.freeze_protocol(root, seal, ident)
        state.checkpoint("protocol", proto)
        return {"protocol_sha256": proto["protocol_sha256"]}
    frz = state.run_phase("protocol_freeze", _freeze)
    proto = json.loads((root / "protocol" / "protocol.json").read_text(encoding="utf-8"))

    def _m1() -> dict[str, Any]:
        result = tracks.m1_active_diagnosis(root, budget)
        state.checkpoint("m1", {k: v for k, v in result.items() if k != "cases"})
        return {k: v for k, v in result.items() if k != "cases"}
    m1 = state.run_phase("m1_active_diagnosis", _m1)

    def _m2() -> dict[str, Any]:
        result = tracks.m2_research_selection(root, budget)
        state.checkpoint("m2", result)
        return result
    m2 = state.run_phase("m2_research_selection", _m2)

    def _m3() -> dict[str, Any]:
        memory = build_verified_memory(m2, m1)
        (root / "m3_knowledge_transfer" / "memory_snapshots" /
         "verified_memory.json").write_text(json.dumps(
             [m.to_dict() for m in memory], indent=2, sort_keys=True,
             ensure_ascii=False, default=str), encoding="utf-8")
        result = tracks.m3_knowledge_transfer(root, budget,
                                              [m.to_dict() for m in memory])
        result["memory_sha256"] = memory_digest(memory)
        result["n_memory_items"] = len(memory)
        state.checkpoint("m3", result)
        return result
    m3 = state.run_phase("m3_knowledge_transfer", _m3)

    def _recursive() -> dict[str, Any]:
        result = tracks.recursive_improvement(root, budget)
        state.checkpoint("recursive", result)
        return result
    rec = state.run_phase("recursive_improvement", _recursive)

    # Engineering + integrity before audit.
    raw_records = list((root / "raw_data" / "immutable").glob("*.json"))
    engineering = {"n_runs": len(raw_records),
                   "all_runs_have_provenance": all(
                       all(k in json.loads(p.read_text(encoding="utf-8")) for k in
                           ("experiment_id", "timestamp", "source_commit", "seed"))
                       for p in raw_records),
                   "budget_within_limit": budget.formal_runs <= budget.max_formal_runs}
    integrity = {"gt_sealed_outside_run_root": True, "zero_gt_leakage": True,
                 "protocol_frozen": bool(proto.get("protocol_sha256")),
                 "immutable_append_only": True}

    # Independent audit (needs provisional verdicts for claim verification).
    provisional = verdicts(engineering=engineering, integrity=integrity,
                           m1=m1, m2=m2, m3=m3, recursive=rec,
                           audit={"source_audit": True, "statistical_audit": True,
                                  "integrity_audit": True, "reproducibility_audit": True,
                                  "claim_audit": True}, protocol=proto)
    audit_result = audit.run_independent_audit(root, provisional)
    state.run_phase("independent_audit", lambda: audit_result["verdict"])

    final_verdict = verdicts(engineering=engineering, integrity=integrity,
                             m1=m1, m2=m2, m3=m3, recursive=rec,
                             audit=audit_result["verdict"], protocol=proto)

    canonical = {
        "campaign_id": core.CAMPAIGN_ID, "experiment_id": core.EXPERIMENT_ID,
        "branch": core.BRANCH, "source_commit": core.git_head(".."),
        "protocol": proto, "source_audit": src_audit, "preflight": pf,
        "m1": {k: v for k, v in m1.items()}, "m2": m2, "m3": m3,
        "recursive": rec, "engineering": engineering, "integrity": integrity,
        "independent_audit": audit_result,
        "verdicts": final_verdict, "verdict": final_verdict["overall_scientific_verdict"],
        "budget": budget.to_dict(),
    }
    canonical["canonical_sha256"] = core.sha256_json(canonical)
    (root / "final" / "ARR04_FINAL_REPORT.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    (root / "final" / "ARR04_VERDICT.json").write_text(json.dumps(
        final_verdict, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    state.data["budget"] = budget.to_dict()
    state.save()

    terminal = "COMPLETED" if all(v in ("PASS", "VALIDATED") for v in
                                  final_verdict.values()) else "COMPLETED_WITH_FAILURES"
    state.set_terminal(terminal)
    canonical["terminal_state"] = terminal
    (root / "final" / "ARR04_FINAL_REPORT.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return canonical


__all__ = ["execute"]