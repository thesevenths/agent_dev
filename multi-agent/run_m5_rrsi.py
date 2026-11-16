"""M5 INTEGRATED -- REAL dual-layer RRSI evaluation + promotion proof.

Layer 1 (Domain Model): real embodied engine runner via domain_model_evolver;
    paired seeds on decision split + INDEPENDENT holdout seeds.
Layer 2 (Harness): real SciNet++ paired+holdout via research_ingest.ingest(real=True).

Both layers use RRSICoordinator with require_structured_holdout=True: promotion
requires a structured Holdout Evidence object cross-checked against the exact
candidate/parent SHA, dataset manifest SHA, split ids, task ids (disjoint from
selection) and result SHAs. A bare boolean never promotes.

Proves:
  * two layers have independent versions + evidence, never mixed
  * rejected/inconclusive candidates are retained with their SHA
  * promotion needs independent holdout (structured evidence)
  * "next round is better": promoted candidate's REAL holdout metric vs baseline
  * rollback: parent SHA inheritance lets a run revert to the prior version
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "M5_INTEGRATED_runs" / "rrsi"

from research_runtime.research_store.db import ResearchDatabase  # noqa: E402
from research_runtime.rrsi import (  # noqa: E402
    RRSICoordinator, EvolutionRegistry, DOMAIN_MODEL_LAYER,
    RESEARCH_HARNESS_LAYER, build_holdout_evidence,
)
from research_runtime.cli import _build_real_domain_runner  # noqa: E402
from research_runtime.domains.embodied import (  # noqa: E402
    BASELINE_CONFIG as EMB_BC, PROJECT_DIR as EMB_DIR, read_dataset_version,
)
from research_runtime.evolution.candidate import DomainModelCandidate  # noqa: E402
from research_runtime.evolution.domain_evolver import (  # noqa: E402
    generate_domain_candidates, evaluate_domain_candidate,
)

MANIFEST_EMB = EMB_DIR / "datasets" / "robot_sim_v1" / "manifest.json"
DECISION_SEEDS = (42, 43)
HOLDOUT_SEEDS = (60, 61, 62)
DOMINANT_DOMAIN = "DECISION_UNCERTAINTY"  # -> train.lr (embodied whitelist)


def _log(*args: Any) -> None:
    print(*args, flush=True)


def _grow_registry(out: Path) -> tuple[RRSICoordinator, EvolutionRegistry]:
    reg = EvolutionRegistry(db=ResearchDatabase(out / "evolution.sqlite"))
    return RRSICoordinator(reg), reg


def phase_domain_model(out: Path) -> dict[str, Any]:
    """Real embodied domain-model candidate evaluation + structured holdout.

    Two REAL candidates are evaluated:
      * a search-space candidate (DECISION_UNCERTAINTY -> train.lr), which may
        be REJECTED -- retained as a failed-candidate evidence record; and
      * a diagnosis-targeted intervention (dynamics_error_accumulation ->
        model.drift_correction) from the domain's real INTERVENTION_POLICY,
        which is the genuinely improving candidate expected to PROMOTE.
    Both go through the SAME real engine + paired + independent holdout gate.
    """
    _log("\n--- RRSI Layer 1: real Domain Model candidates (embodied) ---")
    coord, reg = _grow_registry(out)
    candidates = generate_domain_candidates(domain="embodied",
                                             dominant=DOMINANT_DOMAIN, limit=1)
    # Diagnosis-targeted candidate from the domain's REAL intervention policy.
    from research_runtime.domains.embodied import INTERVENTION_POLICY
    drift_val = (INTERVENTION_POLICY.get("dynamics_error_accumulation", {})
                 .get("model.drift_correction"))
    targeted = None
    if drift_val is not None:
        targeted = DomainModelCandidate(
            candidate_id=f"dom_drift_{str(drift_val).replace('.', '_')}",
            domain="embodied", target="model.drift_correction", old=None,
            new=drift_val, taxonomy="dynamics_error_accumulation",
            hypothesis="diagnosis-targeted drift correction from real domain policy")
    eval_candidates = candidates + ([targeted] if targeted is not None else [])
    if not eval_candidates:
        raise RuntimeError("no real domain candidate generated for embodied")
    sha_by_id = {c.candidate_id: coord._candidate_sha(c.to_dict())
                 for c in eval_candidates}
    for c in eval_candidates:
        _log(f"candidate: {c.target} {c.old}->{c.new} "
             f"sha={sha_by_id[c.candidate_id][:12]}")

    runner = _build_real_domain_runner("embodied", project_dir=EMB_DIR)
    dataset_version = read_dataset_version()
    campaign_id = "rrsi_m5i_domain_embodied"
    base_cfg = str(EMB_DIR / EMB_BC)

    result_holder: dict[str, Any] = {}

    def generator(**kwargs):
        return [c.to_dict() for c in eval_candidates]

    def evaluator(**kwargs):
        candidate_dict = kwargs["candidate"]
        cid = candidate_dict["candidate_id"]
        cand = next(c for c in eval_candidates if c.candidate_id == cid)
        cand_sha = sha_by_id[cid]
        rec = evaluate_domain_candidate(
            domain="embodied", candidate=cand, base_config=base_cfg,
            project_dir=EMB_DIR, campaign_id=campaign_id,
            seeds=DECISION_SEEDS, dataset_version=dataset_version,
            runner=runner, holdout_seeds=HOLDOUT_SEEDS)
        direction = rec.get("metric_direction")
        ho_imp = rec.get("holdout_improvements") or []
        ho_frac = rec.get("holdout_direction_fraction")
        n_valid = rec.get("holdout_n_valid", 0)
        # decision = direction-normalized mean improvement on decision split
        mean_imp = rec.get("mean_improvement")
        decision = ("ACCEPTED" if (mean_imp is not None and mean_imp > 0
                                   and n_valid >= 2)
                    else "REJECTED" if (mean_imp is not None and mean_imp < 0)
                    else "INCONCLUSIVE")
        holdout_ok = bool(n_valid >= 2 and ho_frac is not None
                          and ho_frac >= 0.6 and all(x >= 0 for x in ho_imp))
        he = build_holdout_evidence(
            manifest_path=str(MANIFEST_EMB), split_id="holdout_seed",
            selection_split_id="decision_seed",
            holdout_seeds=list(HOLDOUT_SEEDS),
            selection_seeds=list(DECISION_SEEDS),
            candidate_sha=cand_sha,
            holdout_metrics={"holdout_improvement": rec.get("holdout_improvement"),
                             "holdout_direction_fraction": ho_frac,
                             "mean_improvement": mean_imp},
            n_holdout_seeds=n_valid, direction_fraction=ho_frac)
        result_holder[cid] = rec
        return {"layer": DOMAIN_MODEL_LAYER, "candidate_id": cid,
                "candidate_sha": cand_sha, "decision": decision,
                "holdout_confirmed": holdout_ok,
                "frozen_test_used": False, "training_only": False,
                "primary_metric": rec.get("primary_metric"),
                "metric_direction": direction,
                "mean_improvement": mean_imp,
                "holdout_improvement": rec.get("holdout_improvement"),
                "holdout_direction_fraction": ho_frac,
                "holdout_evidence": he}

    out = coord.run(
        experiment_id="M5I_RRSI_DOMAIN", dominant_failure=DOMINANT_DOMAIN,
        evidence={"changed_layers": [DOMAIN_MODEL_LAYER],
                  "source_domains": ["embodied"]},
        generator=generator, evaluator=evaluator,
        layer=DOMAIN_MODEL_LAYER, domain="embodied",
        require_structured_holdout=True)
    for o in out["outcomes"]:
        _log(f"domain layer: {o['decision']} gate={o['gate_passed']} "
             f"status={o['status']} ({o.get('candidate_id', '?')})")
    promoted = next((o for o in out["outcomes"]
                     if o["status"] == "PROMOTED"), None)
    best = promoted or out["outcomes"][-1]
    best_cid = best.get("candidate_id", "")
    best_rec = result_holder.get(best_cid, {})
    return {"layer": DOMAIN_MODEL_LAYER, "outcomes": out["outcomes"],
            "candidate": next((c.to_dict() for c in eval_candidates
                               if c.candidate_id == best_cid), {}),
            "candidate_sha": sha_by_id.get(best_cid, ""),
            "outcome": best,
            "n_candidates": len(out["outcomes"]),
            "n_retained_failures": sum(1 for o in out["outcomes"]
                                       if o["status"] != "PROMOTED"),
            "real_metrics": {k: best_rec.get(k) for k in (
                "primary_metric", "metric_direction", "mean_improvement",
                "holdout_improvement", "holdout_direction_fraction",
                "holdout_base_Q", "holdout_cand_Q", "holdout_n_valid")},
            "promotion_reasons": best["record"].get("evidence", {}).get(
                "promotion_gate", {}).get("reasons", [])}


def phase_harness(out: Path) -> dict[str, Any]:
    """Real SciNet++ harness candidate evaluation + structured holdout."""
    _log("\n--- RRSI Layer 2: real Harness candidate (SciNet++ ingest) ---")
    coord, reg = _grow_registry(out)
    from adaptive_runtime.research_ingest import (
        ingest, scinet_available, export_research_trace,
    )
    if not scinet_available():
        return {"layer": RESEARCH_HARNESS_LAYER, "status": "blocked",
                "reason": "SciNet++ engine unavailable"}
    # Load a REAL persisted report (never synthesise one) from research phase
    # output or the canonical reports dir.
    candidates = sorted((ROOT / "research_runtime" / "_data" / "reports"
                         ).glob("research_report_*.json"))
    if not candidates:
        candidates = sorted((ROOT / "research_runtime" / "_data" / "reports"
                             ).glob("*.json"))
    if not candidates:
        return {"layer": RESEARCH_HARNESS_LAYER, "status": "blocked",
                "reason": "no real ResearchReport available"}
    report_path = candidates[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _log(f"harness source report: {report_path.name}")

    harness_dir = out / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = ingest(
        report,
        ledger_path=harness_dir / "ledger.jsonl",
        experience_path=harness_dir / "experience.jsonl",
        sandbox_root=harness_dir / "sandbox",
        real=True, paired_seeds=list(DECISION_SEEDS),
        decision_split="validation", holdout_split="test", epochs=1)
    _log(f"harness ingest: decision={res.decision} status={res.ledger_status} "
         f"behavior_changed={res.behavior_changed} "
         f"holdout_consistent={res.holdout_consistent} "
         f"({time.time()-t0:.1f}s)")

    real_ev = res.real_evidence or {}
    artifact_sha = res.artifact_sha256 or ""
    cand_key = f"harness::{real_ev.get('key', 'scinet_lr')}::{artifact_sha[:12]}"
    cand = {"candidate_id": cand_key, "target": real_ev.get("key", "train.lr"),
            "old": real_ev.get("old"), "new": real_ev.get("new"),
            "candidate_sha": artifact_sha}
    cand_sha = artifact_sha or coord._candidate_sha(cand)

    from research_runtime.settings import DEFAULT_SCINET_DIR
    manifest = Path(DEFAULT_SCINET_DIR) / "datasets" / "pendulum_smoke_v1" / "manifest.json"
    holdout_ds = real_ev.get("holdout_deltas") or []
    ho_frac = real_ev.get("holdout_direction_agreement")
    holdout_ok = bool(res.holdout_consistent is True)
    decision = {"ACCEPT": "ACCEPTED", "REJECT": "REJECTED"}.get(
        res.decision, "INCONCLUSIVE")

    he = build_holdout_evidence(
        manifest_path=str(manifest), split_id="test",
        selection_split_id="validation",
        holdout_seeds=list(DECISION_SEEDS),      # same seeds, DIFFERENT split
        selection_seeds=list(DECISION_SEEDS),
        candidate_sha=cand_sha,
        holdout_metrics={"holdout_direction_agreement": ho_frac,
                         "mean_delta": res.evidence.get("mean_delta"),
                         "holdout_deltas": holdout_ds},
        n_holdout_seeds=len(holdout_ds) or len(DECISION_SEEDS),
        direction_fraction=ho_frac)

    def generator(**kwargs):
        return [cand]

    def evaluator(**kwargs):
        return {"layer": RESEARCH_HARNESS_LAYER, "decision": decision,
                "holdout_confirmed": holdout_ok,
                "frozen_test_used": False, "training_only": False,
                "mean_delta": res.evidence.get("mean_delta"),
                "holdout_direction_agreement": ho_frac,
                "behavior_changed": res.behavior_changed,
                "real_evidence": real_ev,
                "holdout_evidence": he}

    out = coord.run(
        experiment_id="M5I_RRSI_HARNESS", dominant_failure="EXPERIMENT_DESIGN_FAILURE",
        evidence={"changed_layers": [RESEARCH_HARNESS_LAYER],
                  "source_domains": ["scinet"]},
        generator=generator, evaluator=evaluator,
        layer=RESEARCH_HARNESS_LAYER, require_structured_holdout=True)
    oc = out["outcomes"][0]
    _log(f"harness layer: decision={oc['decision']} gate={oc['gate_passed']} "
         f"status={oc['status']}")
    return {"layer": RESEARCH_HARNESS_LAYER, "candidate": cand,
            "candidate_sha": cand_sha, "outcome": oc,
            "real_evidence": {k: real_ev.get(k) for k in (
                "key", "old", "new", "behavior_changed", "holdout_consistent",
                "holdout_deltas", "holdout_direction_agreement",
                "decision_split", "holdout_split", "seeds")},
            "decision": res.decision, "ledger_status": res.ledger_status,
            "promotion_reasons": oc["record"].get("evidence", {}).get(
                "promotion_gate", {}).get("reasons", [])}


def phase_rollback(out: Path, domain_sha: str,
                   harness_sha: str) -> dict[str, Any]:
    """Prove version inheritance + rollback to the prior promoted SHA."""
    reg = EvolutionRegistry(db=ResearchDatabase(out / "evolution.sqlite"))
    result = {}
    for layer, promoted in ((DOMAIN_MODEL_LAYER, domain_sha),
                            (RESEARCH_HARNESS_LAYER, harness_sha)):
        parent = reg.parent_for(layer)
        result[layer] = {"current_parent": parent,
                         "promoted_sha": promoted,
                         "rollback_target": parent or "",
                         "n_records": len(reg.all(layer))}
    return {"layers": result,
            "note": ("parent_for() returns the most recent PROMOTED candidate SHA; "
                     "rollback restores that version. Failed candidates remain "
                     "in the registry but are never used as rollback targets.")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-harness", action="store_true",
                    help="skip the slow real SciNet++ harness layer")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"run_id": "m5_integrated_v1", "phase": "rrsi"}
    try:
        r1 = phase_domain_model(out)
        result["domain_model"] = r1
        r2 = ({"layer": RESEARCH_HARNESS_LAYER, "status": "blocked",
               "reason": "--skip-harness"}
              if args.skip_harness else phase_harness(out))
        result["harness"] = r2
        result["rollback"] = phase_rollback(
            out, r1.get("candidate_sha", ""), r2.get("candidate_sha", ""))
        # "next round better": real holdout improvement of promoted candidate
        rmet = r1.get("real_metrics", {})
        result["next_round_quality"] = {
            "metric": rmet.get("primary_metric"),
            "direction": rmet.get("metric_direction"),
            "baseline_holdout_Q": (rmet.get("holdout_base_Q") or [None])[0]
            if rmet.get("holdout_base_Q") else None,
            "promoted_holdout_Q": (rmet.get("holdout_cand_Q") or [None])[0]
            if rmet.get("holdout_cand_Q") else None,
            "holdout_improvement": rmet.get("holdout_improvement"),
            "promoted": r1.get("outcome", {}).get("status") == "PROMOTED",
        }
        result["state"] = "COMPLETED"
    except Exception as exc:
        result["state"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _log("RRSI FAILED:", result["error"])
    (out / "rrsi_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    _log("\nRRSI RESULT:", json.dumps(
        {"state": result["state"],
         "domain_status": (result.get("domain_model", {}).get("outcome", {})
                           or {}).get("status"),
         "harness_status": result.get("harness", {}).get("status") or
                           (result.get("harness", {}).get("outcome", {}) or {}).get("status")},
        ensure_ascii=False))
    return 0 if result["state"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
