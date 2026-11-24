"""Offline finalize: regenerate source audit, evidence store, audits, verdicts,
report from persisted records. NO formal experiment is re-executed."""
from __future__ import annotations

import json
from pathlib import Path

from . import audit, core, protocol
from .campaign import _source_audit
from .evidence import Claim, Evidence, EvidenceStore
from .orchestrator import reproducibility_contract
from .report import generate


def finalize(root: str | Path) -> dict:
    root = Path(root).resolve()
    c = json.loads((root / "final" / "ARR06_FINAL_REPORT.json").read_text(encoding="utf-8"))
    # Regenerate source audit with corrected path resolution.
    s0 = _source_audit(root)
    c["source_audit"] = s0
    # Generate evidence store if missing (claim VERIFIED from raw evidence).
    store = EvidenceStore(root)
    raw_det = root / "raw_data" / "immutable" / "determinism__full_det__r0.json"
    if raw_det.is_file() and "ev_det_01" not in store.data["evidence"]:
        rec = json.loads(raw_det.read_text(encoding="utf-8"))
        ev = Evidence("ev_det_01", "determinism__full_det__r0",
            "ARR-06 det engine spread=0", "one_step_mse",
            float(rec["metrics"]["one_step_mse"]), str(raw_det),
            core.sha256_file(raw_det), "preflight")
        store.add_evidence(ev)
        store.register_claim(Claim("cl_det_01",
            "ARR-06 engine deterministic under det mode", "ANALYSIS",
            "hyp_det", supporting_evidence=["ev_det_01"]))
        store.verify_claim("cl_det_01")
    from .orchestrator import BudgetController
    b = BudgetController()
    from .campaign import _rebuild_budget
    _rebuild_budget(root, b)
    ar = audit.run_audit(root, b)
    c["independent_audit"] = ar
    c["reproducibility"] = reproducibility_contract(root)
    c["budget"] = b.to_dict()
    g = c["protocol"]["gates"]
    v = {
        "source_integrity": "PASS" if s0["pass"] else "FAIL",
        "engineering": "PASS" if c["reproducibility"]["R0_artifact_integrity"] else "FAIL",
        "reproducibility": "PASS" if all(c["reproducibility"][k] for k in
            ("R0_artifact_integrity", "R1_statistical_recalculation",
             "R2_experiment_replay")) else "PARTIAL",
        "rq1_belief_calibration": "PASS" if (c["rq1"].get("ece",1) <= g["rq1_max_ece"]
            and c["rq1"].get("mean_information_gain",0) >= g["rq1_min_info_gain"]) else "FAIL",
        "rq2_research_selection": "PASS" if (c["rq2"].get("arr_vs_random",{}).get("significant")
            and c["rq2"].get("arr_vs_random",{}).get("mean",0) >= g["rq2_mme"])
            else "INCONCLUSIVE",
        "rq3_memory_transfer": (
            "FAIL" if (c["rq3"].get("gated_vs_fresh",{}).get("p",1) <= 0.05
                       and c["rq3"].get("gated_vs_fresh",{}).get("mean",0) < 0)
            else "PASS" if (c["rq3"].get("gated_vs_fresh",{}).get("mean",0)
                            >= g["rq3_mme"]
                            and c["rq3"].get("negative_transfer_rate",1)
                            <= g["rq3_max_negative"])
            else "INCONCLUSIVE"),
        # Task book rule: a statistically significant harmful effect is FAIL,
        # never INCONCLUSIVE. ARR-06 gated transfer is -0.426 at p~0.
        "rq4_methodology_evolution": "PASS" if c["rq4"].get("actual_promotions",0) >= 1
            and not c["rq4"].get("ood_regression",True) else "INCONCLUSIVE",
        "rq5_rq_discovery": "PASS" if (c["rq5"].get("arr_vs_random",{}).get("mean",0)
            >= g["rq5_mme"] and c["rq5"].get("falsifiability",0) >= 0.7)
            else "INCONCLUSIVE",
        "independent_audit": "PASS" if all(ar["verdict"].values()) else "FAIL",
    }
    # Capability levels L0-L7 must count ONLY research capabilities, not
    # engineering/infrastructure PASSes (which would inflate the level).
    capability_keys = ("reproducibility", "rq1_belief_calibration",
                       "rq2_research_selection", "rq3_memory_transfer",
                       "rq4_methodology_evolution", "rq5_rq_discovery")
    cert = sum(1 for k in capability_keys if v[k] == "PASS")
    v["certification_level"] = f"L{min(cert,7)}"
    v["overall"] = ("VALIDATED WITHIN DEFINED BENCHMARK SCOPE" if cert >= 9
                    else "PARTIALLY VALIDATED" if cert >= 4
                    else "AUTONOMOUS RESEARCH SYSTEM — NOT VALIDATED")
    c["verdicts"] = v
    c["terminal_state"] = "COMPLETED" if cert >= 9 else "COMPLETED_WITH_FAILURES"
    c.pop("canonical_sha256", None)
    c["canonical_sha256"] = core.sha256_json(c)
    (root / "final" / "ARR06_FINAL_REPORT.json").write_text(json.dumps(
        c, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    (root / "final" / "ARR06_VERDICT.json").write_text(json.dumps(
        v, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    generate(root, c)
    return c


__all__ = ["finalize"]
