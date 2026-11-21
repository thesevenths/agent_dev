"""ARR v1.0 WP-09: REAL multi-round research program E2E.

Scope honesty:
  * L1 production execution (formal Runtime/approval/harness/engine) was
    proven by M7.6 real E2E (M7_6_EVIDENCE/run2). This driver does NOT repeat
    that; it exercises the NEW WP-02 program layer across rounds.
  * Each round uses the REAL LLM (vLLM) for scientific judgment on a REAL
    literature snapshot, and a frozen DomainPack verdict mapping. Round N+1
    decisions are derived from round N artifacts (never hardcoded).
  * No human approval is fabricated; the program runs in `research_judgment`
    mode (no production campaign execution) and records that explicitly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research_runtime.discovery.llm_agent import LLMAgentAdapter, is_available
from research_runtime.discovery.literature import OpenAlexProvider, expand_query
from research_runtime.discovery.domain_packs import load_pack
from research_runtime.harness_rsi.identity import freeze_json, git_fingerprint
from research_runtime.program import ResearchProgramDriver
from research_runtime.capability_bench.memory_attribution import MemoryAttributionLedger


def _real_round(*, program_id, round, planning_digest, root):
    """One research round with REAL LLM judgment on a REAL literature snapshot."""
    rid = round["round_id"]
    out = Path(root) / rid
    out.mkdir(parents=True, exist_ok=True)
    pack = load_pack("embodied_world_model")
    inp = pack.literature_input()
    provider = OpenAlexProvider(cache_dir=out / "cache")
    budget = {"max_requests": 3, "requests_used": 0, "timeout_s": 25.0}
    papers = []
    for q in expand_query(inp)[:3]:
        try:
            papers.extend(provider.search(q, limit=4, budget=budget))
        except Exception:  # noqa: BLE001 - partial coverage recorded
            continue
        if budget["requests_used"] >= budget["max_requests"]:
            break
    prior = planning_digest["digest"]
    agent = LLMAgentAdapter(max_retries=2, timeout_s=90.0)
    prompt = json.dumps({
        "task": "research_round_judgment",
        "round_id": rid, "family_id": round["family_id"],
        "objective": round.get("objective", ""),
        "n_papers": len(papers),
        "paper_titles": [p.title for p in papers[:6]],
        "prior_round_verdicts": prior.get("verdicts", {}),
        "prior_failures": prior.get("failures", [])[:3],
        "input_memory_refs": round.get("input_memory_refs", [])[:5],
        "required_output": {"question": "...", "hypotheses": ["H1", "H0"],
                            "evidence_sufficiency": "SUFFICIENT|INSUFFICIENT",
                            "verdict": "SUPPORTED|REFUTED|INCONCLUSIVE",
                            "used_memory_ids": [], "reason": "..."}},
        ensure_ascii=False, sort_keys=True)
    traj = agent.call(prompt=prompt,
                      system="You are a rigorous scientist. Be conservative: "
                             "if evidence is insufficient return INCONCLUSIVE.",
                      toolset=[], response_format_json=True, temperature=0.0,
                      max_tokens=700)
    raw = (traj.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    s, e = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[s:e + 1]) if s >= 0 and e > s else {}
    verdict = str(obj.get("verdict", "INCONCLUSIVE")).upper()
    if verdict not in ("SUPPORTED", "REFUTED", "INCONCLUSIVE"):
        verdict = "INCONCLUSIVE"  # fail closed
    # Memory attribution: record retrieved -> injected -> applied if used.
    led = MemoryAttributionLedger(root=out, run_id=rid)
    used = [str(x) for x in (obj.get("used_memory_ids") or [])]
    for mid in round.get("input_memory_refs", [])[:5]:
        led.record(scenario_id=rid, arm="ON", memory_id=mid,
                   state="memory_retrieved")
        if mid in used:
            led.record(scenario_id=rid, arm="ON", memory_id=mid,
                       state="memory_injected")
            led.record(scenario_id=rid, arm="ON", memory_id=mid,
                       state="memory_applied")
    freeze_json({"trajectory": traj.to_dict(), "parsed": obj,
                 "n_papers": len(papers), "verdict": verdict},
                out / "round_judgment.json")
    return {
        "status": "COMPLETED", "scientific_verdict": verdict,
        "question_id": str(obj.get("question", ""))[:200],
        "hypothesis_ids": list(obj.get("hypotheses") or []),
        "preregistration_sha": freeze_json(
            {"round": rid, "family": round["family_id"],
             "prompt_sha": traj.prompt_sha256}, out / "preregistration.json")["sha256"],
        "method_sha": "", "harness_sha": "",
        "dataset_sha": freeze_json({"paper_titles": [p.title for p in papers[:6]]},
                                   out / "literature_snapshot.json")["sha256"],
        "campaign_ids": [],  # research_judgment mode: no production campaign
        "evidence_refs": [str(out / "round_judgment.json")],
        "claim_ids": [f"claim_{rid}"],
        "budget_used": {"rounds": 1, "experiments": 0,
                        "tokens": traj.total_tokens or 0, "llm_calls": 1},
        "trace_ref": str(out / "round_judgment.json"),
        "memory_used": bool(used),
        "execution_mode": "RESEARCH_JUDGMENT_NO_PRODUCTION_CAMPAIGN",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ARR_V1_EVIDENCE/program_e2e")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    families = ["pendulum_dynamics", "mass_force_estimation",
                "friction_identification", "latent_dynamics",
                "multi_step_prediction", "ood_generalization",
                "competing_physical_hypotheses", "data_efficiency"]
    driver = ResearchProgramDriver(
        root=root, program_id="ARR_V1_PROG_1",
        objective={"question": "Improve multi-step physical dynamics prediction "
                               "under limited data and unseen conditions while "
                               "holding model capacity and compute fixed",
                   "domain": "embodied_world_model"},
        domain_scope="embodied_world_model",
        research_families=families,
        resource_policy={"round": {"experiments": 50, "tokens": 100000,
                                   "llm_calls": 50}},
        max_rounds=args.rounds, round_fn=_real_round)
    report = driver.run()
    report["repository"] = git_fingerprint(Path(__file__).resolve().parents[1])
    report["execution_mode"] = "REAL_LLM_RESEARCH_JUDGMENT_PROGRAM"
    report["l1_production_evidence"] = "M7_6_EVIDENCE/run2 (separate real E2E)"
    freeze_json(report, root / "program_e2e_result.json")
    print(json.dumps({k: report.get(k) for k in
                      ("ok", "status", "rounds", "verdicts", "budget")},
                     ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
