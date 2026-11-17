"""Run the M7.5 integrated A/B/C/D capability pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_runtime.capability_bench.integrated_bench import (metrics, run,
                                                                select_tasks)
from research_runtime.capability_bench.scientific_judgment import (
    load_cases, load_ground_truth)
from research_runtime.discovery.llm_agent import LLMAgentAdapter, is_available
from research_runtime.harness_rsi.identity import freeze_json, git_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="M7_5_INTEGRATED")
    ap.add_argument("--mme", type=float, default=0.10)
    ap.add_argument("--min-families", type=int, default=8)
    args = ap.parse_args()
    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED")
    root = Path(args.out_dir)
    for sub in ("protocols", "raw_results", "trajectories", "metrics",
                "cross_round"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cases = select_tasks(load_cases("M7_2_BENCHMARK"), n=8)
    gt = load_ground_truth("M7_2_BENCHMARK")
    memories = json.loads(Path(
        "M7_3_MEMORY/memories/m7_verified_failures.json").read_text(
            encoding="utf-8"))["memories"]
    protocol = {"version": "m7_5_protocol_v1",
                "primary_outcome": "evidence_sufficiency_accuracy",
                "mme": args.mme, "min_families": args.min_families,
                "groups": ["A_BASE", "B_MEMORY", "C_RSI", "D_FULL"],
                "case_ids": [c["case_id"] for c in cases],
                "model_control": "same agent/model/token budget for all groups",
                "source": git_fingerprint(Path(__file__).resolve().parents[1])}
    freeze_json(protocol, root / "protocols" / "experiment_protocol.json")
    agent = LLMAgentAdapter(max_retries=2, timeout_s=90.0)
    executed = run(cases=cases, ground_truth=gt, memories=memories,
                   agent=agent, out_dir=root,
                   run_id="m7_5_integrated_pilot")
    result = metrics(results_dir=root, mme=args.mme,
                     min_families=args.min_families)
    freeze_json(result, root / "metrics" / "metrics.json")
    freeze_json({"n_calls": len(agent.calls), "model_id": agent.cfg["model"],
                 "provider": agent._provider()}, root / "run_manifest.json")
    print(json.dumps({"verdict": result["verdict"],
                      "n_results": executed["n_results"],
                      "per_group": result["per_group"],
                      "deltas": result["deltas_vs_base"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
