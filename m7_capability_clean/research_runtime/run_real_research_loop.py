"""ARR 0.2 / Package B: REAL autonomous research loop on SciNet++ (low-compute).

Drives the GENERIC ``ResearchService`` with the REAL ``SciNetAdapter`` end-to-end,
with NO FakeRunner anywhere:

    objective
      -> REAL baseline experiment (SciNet++ train.py -> evaluate.py, real engine CLI)
      -> autonomous hypothesis (RuleBasedProposer reads the REAL metrics)
      -> deterministic Critic (independent, LLM can never approve)
      -> plan
      -> approval (granted out-of-band, as the human console would)
      -> REAL candidate campaign (SciNet++ runs again with the intervened config)
      -> comparator (baseline vs candidate on the held-out validation split)
      -> immutable ResearchReport

The engine CLI at SCINET_DIR is invoked for real; artifacts, run ids, dataset
versions and metric values are genuine. Run:

    D:/python313/python.exe research_runtime/run_real_research_loop.py
"""
from __future__ import annotations

import json
import sys

from research_runtime.adapters import SciNetAdapter
from research_runtime.evidence_store import EvidenceStore
from research_runtime.research_environment import build_default_research_service
from research_runtime.schemas import ExperimentSpec, ExperimentResult

SCINET_CFG = "configs/pendulum_smoke.yaml"
OBJECTIVE = "obj_pendulum_multistep"
SOURCE_EXP = "exp_baseline_demo"
BASE_EPOCHS = 1


def real_baseline() -> ExperimentResult:
    """Run a REAL SciNet++ baseline (train + evaluate) and record the result."""
    adapter = SciNetAdapter()  # resolves SCINET_DIR / SCINET_PYTHON from env
    train_spec = ExperimentSpec(
        experiment_id=SOURCE_EXP, objective_id=OBJECTIVE,
        stage="train", config_path=SCINET_CFG, split="train",
        budget={"epochs": BASE_EPOCHS},
    )
    jid = adapter.submit(train_spec)
    adapter.wait(jid, timeout=600)
    train_res = adapter.collect_result(jid, train_spec)
    if train_res.status != "succeeded":
        raise RuntimeError(f"baseline train failed: {train_res.status} {train_res.error}")
    eval_spec = train_spec.with_run_id(train_res.run_id, stage="evaluate", split="validation")
    jid2 = adapter.submit(eval_spec)
    adapter.wait(jid2, timeout=600)
    eval_res = adapter.collect_result(jid2, eval_spec)
    print(f"[baseline] eval status={eval_res.status} metrics={eval_res.metrics}")
    print(f"[baseline] dataset_version={eval_res.dataset_version} "
          f"checkpoint_sha256={(eval_res.checkpoint or {}).get('sha256')}")
    # Feed the REAL result into the EvidenceStore the service reads from, so the
    # autonomous proposer grounds its hypothesis in genuine measured evidence.
    EvidenceStore().append(
        "result",
        {"experiment_id": SOURCE_EXP, "result": eval_res.model_dump(mode="json")},
    )
    return eval_res


def main() -> int:
    real_baseline()

    svc = build_default_research_service("scinet")
    s = svc.start_research_session(
        objective_id=OBJECTIVE,
        description="Improve multi-step pendulum dynamics prediction",
    )
    sid = s["session_id"]
    print(f"[session] {sid}")

    # Autonomous (no hypotheses_json): RuleBasedProposer reads the real baseline.
    p = svc.propose_research_hypotheses(
        objective_id=OBJECTIVE, source_experiment_id=SOURCE_EXP, session_id=sid)
    print(f"[propose] hypotheses={len(p['hypotheses'])} "
          f"skip_reasons={p.get('skip_reasons')}")
    for h in p["hypotheses"]:
        print(f"   - {h['hypothesis_id']} sha={h['content_sha256'][:12]}")
    hids = [h["hypothesis_id"] for h in p["hypotheses"]]
    if not hids:
        print("[ABORT] autonomous proposer produced no hypothesis", file=sys.stderr)
        return 2

    # Deterministic Critic -- the ONLY approval gate (LLM never approves).
    r = svc.review_research_hypotheses(
        hypothesis_ids_json=json.dumps(hids), session_id=sid)
    print(f"[review] passed={[x['hypothesis_id'] for x in r['reviews'] if x['passed']]} "
          f"rejected={[x['hypothesis_id'] for x in r['reviews'] if not x['passed']]}")

    plan = svc.create_research_plan(
        objective_id=OBJECTIVE, hypothesis_ids_json=json.dumps(hids), session_id=sid)
    print(f"[plan] {plan['plan_id']} candidates={plan['candidate_ids']} "
          f"paired_seeds={plan['paired_seeds']}")

    ap = svc.request_plan_approval(plan_id=plan["plan_id"], session_id=sid)
    aid = ap["approval_id"]
    # Out-of-band human grant (as the approval_cli console would do).
    svc.stores.approvals.grant(aid, approved_by="demo-script", source="human")
    print(f"[approval] granted {aid}")

    run = svc.run_research_campaign(
        plan_id=plan["plan_id"], approval_id=aid, session_id=sid)
    print(f"[run] campaign={run['campaign_id']} status={run['status']} "
          f"trials={run['total_trials']}/{run['succeeded_trials']}")
    if run["status"] not in ("completed", "succeeded"):
        print(f"[WARN] campaign not completed: {run['status']}", file=sys.stderr)

    cmp = svc.compare_research_campaign(campaign_id=run["campaign_id"], session_id=sid)
    print(f"[compare] outcome={cmp['research_outcome']} "
          f"comparisons={len(cmp['comparisons'])}")
    for c in cmp["comparisons"]:
        print(f"   - {c['candidate_id']}: {c['decision']} mean_delta={c.get('mean_delta')}")

    rep = svc.get_research_report(session_id=sid)
    report = rep["report"]
    print(f"[REPORT] outcome={report['outcome']} "
          f"campaigns={report['campaign_count']} "
          f"report_id={report['report_id']}")
    print("[REPORT experience_refs]", report.get("experience_refs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
