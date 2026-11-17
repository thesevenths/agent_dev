"""Run the M7.4 Recursive Improvement pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_runtime.capability_bench.rsi_bench import run_round, split_tasks, summarise
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
    candidates = ["candidate_evidence_gate", "candidate_adversarial"]
    rounds = []
    for idx, candidate in enumerate(candidates, 1):
        rounds.append(run_round(
            round_no=idx, candidate_method=candidate,
            tasks=tasks["development"], ground_truth=gt, agent=agent,
            out_dir=root / "base_candidate"))
    summary = summarise(rounds=rounds, mme=args.mme,
                        min_families=args.min_families)
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
