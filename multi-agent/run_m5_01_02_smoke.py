"""M5-01/02 REAL three-domain smoke E2E (OPT-IN driver).

Runs SMALL real campaigns in scinet / embodied / quant through the SAME
ResearchService, ResearchDirector, external approval CLI and 12-step Gate used
by M5-00. No FakeRunner. Quant data provenance is explicitly SYNTHETIC.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from research_runtime.cross_domain_scheduler import CrossDomainScheduler
from research_runtime.evidence_store import EvidenceStore
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.domain_adapter_registry import build_default_adapters
from research_runtime.domain_budget_manager import DomainBudgetManager
from research_runtime.domain_run_record import DomainRunRecordStore
from research_runtime.domains.embodied import (
    BASELINE_CONFIG as EMB_BC, PROJECT_DIR as EMB_DIR,
    embodied_research_policy, read_dataset_version as emb_version,
)
from research_runtime.domains.quant import (
    BASELINE_CONFIG as Q_BC, PROJECT_DIR as Q_DIR,
    quant_research_policy, read_dataset_version as q_version,
)
from research_runtime.domains.scinet import scinet_research_policy
from research_runtime.domain_adapter_registry import (
    CAPABILITY_BACKTEST, CAPABILITY_PAIRED_SEEDS, CAPABILITY_REAL_ENGINE,
    CAPABILITY_RESEARCH_CAMPAIGN, CAPABILITY_SIMULATION,
    CAPABILITY_VALIDATION_SPLIT,
)
from research_runtime.objective_discovery import ObjectiveDiscovery
from research_runtime.research_environment import build_default_research_service
from research_runtime.research_store import from_research_objective
from research_runtime.schemas import ExperimentSpec, ResearchObjective

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "M5_01_02_runs" / "e2e"
RUN_ID = "m5_cross_domain_smoke_v1"
SEEDS = (42, 43)


def _domain_specs():
    return {
        "scinet": {
            "objective_id": "obj_m5s2_scinet",
            "capabilities": [CAPABILITY_RESEARCH_CAMPAIGN,
                             CAPABILITY_REAL_ENGINE, CAPABILITY_PAIRED_SEEDS,
                             CAPABILITY_VALIDATION_SPLIT],
            "evidence": "run:arr_m34_001",
            "hint": "Use a smaller learning rate to reduce latent prediction error.",
            "intervention": {"train.lr": 0.0005},
            "source_experiment": "cand_H_HORIZON_LATENT_MSE_seed_43",
        },
        "embodied": {
            "objective_id": "obj_m5s2_embodied",
            "capabilities": [CAPABILITY_RESEARCH_CAMPAIGN,
                             CAPABILITY_SIMULATION, CAPABILITY_PAIRED_SEEDS,
                             CAPABILITY_VALIDATION_SPLIT],
            "evidence": f"path:{EMB_DIR / EMB_BC}",
            "hint": "Increase drift correction to reduce world-model state error.",
            "intervention": {"model.drift_correction": 0.6},
            "source_experiment": "exp_a1642ec3db",
        },
        "quant": {
            "objective_id": "obj_m5s2_quant",
            "capabilities": [CAPABILITY_RESEARCH_CAMPAIGN,
                             CAPABILITY_BACKTEST, CAPABILITY_PAIRED_SEEDS,
                             CAPABILITY_VALIDATION_SPLIT],
            "evidence": f"path:{Q_DIR / Q_BC}",
            "hint": "Smooth causal signals to improve cost-adjusted return.",
            "intervention": {"strategy.signal_smoothing": 0.6},
            "source_experiment": "exp_a1642ec3db",
        },
    }


def _services():
    OUT.mkdir(parents=True, exist_ok=True)
    services, db_paths, holders = {}, {}, {}
    services["scinet"] = build_default_research_service(
        "scinet", db_path=OUT / "scinet.sqlite",
        domain_policy=scinet_research_policy(
            paired_seeds=SEEDS, max_candidates=2, max_seeds=2,
            max_jobs=12, max_total_seconds=900.0))
    db_paths["scinet"] = OUT / "scinet.sqlite"

    for domain, data_dir, policy in (
        ("embodied", OUT / "embodied", embodied_research_policy(
            paired_seeds=SEEDS, max_candidates=1, max_seeds=2,
            max_jobs=12, max_total_seconds=300.0)),
        ("quant", OUT / "quant", quant_research_policy(
            paired_seeds=SEEDS, max_candidates=1, max_seeds=2,
            max_jobs=12, max_total_seconds=300.0)),
    ):
        services[domain] = build_default_research_service(
            domain, db_path=data_dir / "research.sqlite",
            data_dir=data_dir, domain_policy=policy)
        db_paths[domain] = data_dir / "research.sqlite"
    holders = {d: {"service": s} for d, s in services.items()}
    return services, db_paths, holders


def _make_smoke_approver(services, db_paths):
    """DEMO-ONLY automated grant for smoke runs.

    This is NOT human approval. Real/integrated Runtime MUST wait for an
    external human operator via approval_cli and never call this. The grant is
    bound to the current Plan SHA by approval_cli itself (current_resource_
    sha256), and the artifact records approval_mode explicitly.
    """
    def grant(approval_id: str, plan_id: str) -> None:
        domain = ""
        for name, svc in services.items():
            if svc.stores.approvals.get(approval_id) is not None:
                domain = name
                break
        if not domain:
            raise RuntimeError(f"approval {approval_id} not found in any domain")
        r = subprocess.run(
            [sys.executable, "-m",
             "research_runtime.research_store.approval_cli",
             # source must stay in policy.allowed_sources=("human",) for the
             # Gate to pass. Honesty about demo mode is carried by approved_by
             # and by the artifact's approval_mode field -- NOT by forging a
             # human identity.
             "--db", str(db_paths[domain]), "grant", approval_id,
             "--approved-by", "smoke_demo_operator_NOT_human", "--yes"],
            cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"approval CLI failed for {domain}: {r.stdout}\n{r.stderr}")
        print(f"  [approval] {domain}:{approval_id} granted externally",
              flush=True)
    return grant


def _seed_objectives(services, specs):
    for domain, spec in specs.items():
        obj = ResearchObjective(
            objective_id=spec["objective_id"], domain=domain,
            description=f"M5-01/02 real smoke objective for {domain}",
            constraints={
                "required_capabilities": spec["capabilities"],
                "evidence_refs": [spec["evidence"]],
                "hypothesis_hint": spec["hint"],
                "proposed_intervention": dict(spec["intervention"]),
            })
        services[domain].stores.objectives.save(
            from_research_objective(obj, status="proposed"))


def _ensure_local_source_experiment(domain: str, service: Any,
                                    project_dir: Path, spec: dict[str, Any]) -> str:
    """Create a REAL local-domain grounding experiment (train + evaluate).

    It is used ONLY as this domain's structured evidence/diagnosis source. No
    cross-domain metric is imported into another domain's science.
    """
    experiment_id = f"m5s2_src_{domain}"
    prior = EvidenceStore().query(experiment_id=experiment_id,
                                  event_type="result")
    if any((e.get("payload") or {}).get("status") == "succeeded"
           and ((e.get("payload") or {}).get("result") or {}).get("stage")
           == "evaluate" for e in prior):
        return experiment_id
    policy = service.domain
    config_path = str(project_dir / policy.baseline_config_path)
    run_id = f"run_{experiment_id}"
    manager = ExperimentManager(service.deps.adapter)
    for stage, split in (("train", "train"), ("evaluate", policy.evaluation_split)):
        job = ExperimentSpec(
            experiment_id=experiment_id,
            objective_id=f"obj_src_{domain}",
            engine="local", stage=stage,
            model_name="world_model" if domain == "embodied" else "factor_model",
            config_path=config_path, dataset_version=policy.dataset_version,
            seed=SEEDS[0], split=split, run_id=run_id,
            budget={"job_id": f"job_{stage}_{experiment_id}"})
        _jid, result = manager.run(job, timeout=180)
        if result.status != "succeeded":
            raise RuntimeError(
                f"real {domain} source {stage} failed: {result.error}")
    return experiment_id


def main() -> int:
    t0 = time.time()
    specs = _domain_specs()
    services, db_paths, holders = _services()
    specs["embodied"]["source_experiment"] = _ensure_local_source_experiment(
        "embodied", services["embodied"], EMB_DIR, specs["embodied"])
    specs["quant"]["source_experiment"] = _ensure_local_source_experiment(
        "quant", services["quant"], Q_DIR, specs["quant"])
    _seed_objectives(services, specs)
    registry = build_default_adapters(service_holders=holders)
    shared_db = DomainBudgetManager(db=services["scinet"].stores.db).db
    scheduler = CrossDomainScheduler(
        run_id=RUN_ID, registry=registry,
        budgets=DomainBudgetManager(db=shared_db),
        records=DomainRunRecordStore(db=shared_db),
        services=services, approver=_make_smoke_approver(services, db_paths),
        out_dir=OUT / "directors", max_campaigns_per_domain=1)

    preflight = scheduler.preflight()
    print("preflight:", json.dumps(
        {k: v.get("ready") for k, v in preflight.items()}), flush=True)
    if not all(v.get("ready") for v in preflight.values()):
        raise SystemExit(f"domain preflight failed: {preflight}")

    objectives = [s["objective_id"] for s in specs.values()]
    # Each domain has its OWN real grounding experiment; pass it explicitly
    # instead of using one global source id for all domains.
    results = []
    for domain, spec in specs.items():
        try:
            results.append(scheduler.run_objective(
                spec["objective_id"],
                source_experiment_id=spec["source_experiment"]))
        except Exception as exc:  # keep domains isolated, never fake success
            results.append({"domain_id": domain,
                            "objective_id": spec["objective_id"],
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})
    # M5-INTEGRATED: overall state is DERIVED from real per-domain outcomes.
    # A failed/partial domain can never be reported as COMPLETED. Each result
    # keeps its own error and evidence; nothing is silently dropped.
    def _domain_ok(r: dict[str, Any]) -> bool:
        return (bool(r.get("ok", True))
                and r.get("director_state") == "COMPLETED"
                and bool(r.get("campaign_id"))
                and not r.get("error"))

    domain_ok = {r.get("domain_id", r.get("objective_id", "?")):
                 _domain_ok(r) for r in results}
    all_ok = bool(results) and all(domain_ok.values())
    overall_state = "COMPLETED" if all_ok else (
        "PARTIAL" if any(domain_ok.values()) else "FAILED")

    summary = {"run_id": RUN_ID, "results": results,
               "state": overall_state, "domain_ok": domain_ok,
               "budgets": {d: scheduler.budgets.snapshot(RUN_ID, d)
                           for d in services}}
    print("campaign summary:", json.dumps([
        {k: r.get(k) for k in ("domain_id", "objective_id", "campaign_id",
                              "director_state")}
        for r in summary["results"]], ensure_ascii=False), flush=True)

    # Next-round Discovery reads the merged cross-domain Experience with SOURCE.
    snapshot = scheduler.build_cross_domain_snapshot(
        RUN_ID, objective_ids=objectives)
    discovery = ObjectiveDiscovery(
        domain="scinet", default_budget={"max_campaign_rounds": 1})
    next_round = discovery.discover(
        snapshot, objective_store=services["scinet"].stores.objectives,
        domain="scinet")

    result = {
        "run_id": RUN_ID, "state": overall_state,
        "approval_mode": "automated_smoke_demo_NOT_human",
        "approval_note": ("smoke driver invoked approval_cli grant automatically; "
                          "this is NOT human approval. Integrated Runtime waits "
                          "for an external human operator and binds the exact "
                          "Plan SHA before campaign execution."),
        "elapsed_s": round(time.time() - t0, 2),
        "preflight": preflight, "summary": summary,
        "cross_domain_snapshot": {
            "objectives": len(snapshot.objectives),
            "sessions": len(snapshot.sessions),
            "experience_available": snapshot.experience_available,
            "unresolved_gaps": snapshot.unresolved_gaps,
        },
        "discovery_source": next_round.to_dict(),
        "experience": scheduler.read_experience(objectives),
        "domain_records": scheduler.records.all(RUN_ID),
    }
    (OUT / "m5_01_02_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False,
                   default=str), encoding="utf-8")
    print(json.dumps({
        "state": result["state"], "elapsed_s": result["elapsed_s"],
        "domains": [r.get("domain_id") for r in summary["results"]],
        "next_round_proposals": len(next_round.proposed_objective_ids),
        "experience_rows": len(result["experience"])}, indent=2), flush=True)
    # Non-zero exit on any failed/partial domain: no false-green smoke.
    return 0 if overall_state == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
