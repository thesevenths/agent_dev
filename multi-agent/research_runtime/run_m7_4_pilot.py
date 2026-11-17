"""Run the M7.4 Recursive Improvement pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_runtime.capability_bench.rsi_bench import (
    diagnose_bottleneck_from_rows, run_round, split_tasks, summarise)
from research_runtime.capability_bench.scientific_judgment import (
    load_cases, load_ground_truth)
from research_runtime.discovery.llm_agent import LLMAgentAdapter, is_available
from research_runtime.harness_rsi.identity import freeze_json, git_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="M7_4_RSI")
    ap.add_argument("--mme", type=float, default=0.10)
    ap.add_argument("--min-families", type=int, default=8)
    args = ap.parse_args()
    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED")
    root = Path(args.out_dir)
    (root / "pre_registration").mkdir(parents=True, exist_ok=True)
    (root / "base_candidate").mkdir(parents=True, exist_ok=True)
    (root / "holdout").mkdir(parents=True, exist_ok=True)
    (root / "promotion").mkdir(parents=True, exist_ok=True)
    (root / "post_adoption").mkdir(parents=True, exist_ok=True)
    tasks = split_tasks(load_cases("M7_2_BENCHMARK"))
    gt = load_ground_truth("M7_2_BENCHMARK")
    prereg = {
        "version": "m7_4_prereg_v1",
        "primary_outcome": "evidence_sufficiency_accuracy",
        "secondary": ["novelty_accuracy", "falsifiability_accuracy"],
        "mme": args.mme, "alpha": 0.05, "target_power": 0.8,
        "primary_test": "paired_family_sign_test",
        "min_families": args.min_families,
        "multiple_comparison": "holm",
        "missing_data_policy": "parse_failure_scores_zero",
        "stopping_rule": "stop after pre-registered rounds; no early stopping",
        "splits": {"development_case_ids": [c["case_id"]
                                            for c in tasks["development"]],
                   "holdout_case_ids": [c["case_id"]
                                        for c in tasks["holdout"]],
                   "ood_case_ids": [c["case_id"] for c in tasks["ood"]]},
        "source": git_fingerprint(Path(__file__).resolve().parents[1])}
    freeze_json(prereg, root / "pre_registration" / "preregistration.json")
    agent = LLMAgentAdapter(max_retries=2, timeout_s=90.0)
    # P1-E: diagnose the bottleneck from REAL base trajectories on development
    # tasks BEFORE choosing a Candidate. No prewritten "correct" candidate.
    import json as _json
    from research_runtime.capability_bench.rsi_bench import METHODS, _case_prompt, _parse, _score
    probe_rows = []
    for task in tasks["development"]:
        traj = agent.call(prompt=_case_prompt(task),
                          system=METHODS["base"]["system"], toolset=[],
                          response_format_json=True, temperature=0.0,
                          max_tokens=500)
        try:
            pred, ok = _parse(traj.content), True
        except (ValueError, TypeError):
            pred, ok = None, False
        probe_rows.append({"case_id": task["case_id"], "category": task["category"],
                           "method": "base", "parse_ok": ok, "prediction": pred,
                           "scores": _score(pred, gt[task["case_id"]])})
    diagnosis = diagnose_bottleneck_from_rows(probe_rows)
    freeze_json({"bottleneck": diagnosis, "base_rows": probe_rows,
                 "note": "candidate is derived from these REAL base failures"},
                root / "bottleneck_diagnosis.json")
    chosen = diagnosis.get("candidate")
    candidates = [chosen] if chosen else []
    rounds = []
    for idx, candidate in enumerate(candidates, 1):
        rounds.append(run_round(
            round_no=idx, candidate_method=candidate,
            tasks=tasks["development"], ground_truth=gt, agent=agent,
            out_dir=root / "base_candidate"))
    summary = summarise(rounds=rounds, mme=args.mme,
                        min_families=args.min_families)
    # M7.6 P1-E: bind the Candidate's creation to the REAL diagnosis so the
    # improvement is demonstrably evidence-driven, not a prewritten answer.
    summary["bottleneck_provenance"] = diagnosis
    summary["candidate_generation"] = "FROM_REAL_BASE_FAILURES"
    best = summary.get("best_candidate")
    if best and best["candidate"]:
        holdout = run_round(
            round_no=99, candidate_method=best["candidate"],
            tasks=tasks["holdout"], ground_truth=gt, agent=agent,
            out_dir=root / "holdout")
        summary["holdout_delta"] = holdout["candidate_minus_base"]
        summary["holdout_n_pairs"] = holdout["n_pairs"]
        if holdout["candidate_minus_base"] < args.mme:
            summary["promotion_decision"] = "REJECTED_HOLDOUT_FAILED"
        # P1-E: also execute the sealed OOD split (previously only recorded).
        ood = run_round(
            round_no=100, candidate_method=best["candidate"],
            tasks=tasks["ood"], ground_truth=gt, agent=agent,
            out_dir=root / "holdout")
        summary["ood_delta"] = ood["candidate_minus_base"]
        summary["ood_n_pairs"] = ood["n_pairs"]
        summary["ood_result"] = "EXECUTED"
    freeze_json(summary, root / "promotion" / "promotion_log.json")
    freeze_json({"human_approval": "NOT_PERFORMED",
                 "adoption": "NOT_ADOPTED",
                 "rollback": "NOT_APPLICABLE_NO_ADOPTION",
                 "reason": summary["promotion_decision"]},
                root / "promotion" / "adoption_decision.json")
    freeze_json({"status": "NOT_RUN",
                 "reason": "no candidate adopted in pilot"},
                root / "post_adoption" / "post_adoption_result.json")
    freeze_json({"model_id": agent.cfg["model"], "provider": agent._provider(),
                 "n_calls": len(agent.calls)}, root / "run_manifest.json")
    print(json.dumps({"decision": summary["promotion_decision"],
                      "rounds": summary["rounds"],
                      "holdout_delta": summary.get("holdout_delta")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
