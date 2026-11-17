"""Run ONE experience-ablation arm in an isolated process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from research_runtime.harness_rsi.experience import ExperienceBank
from research_runtime.harness_rsi.experience_ablation_v2 import run_ablation_v2
from research_runtime.harness_rsi.benchmark_v2 import build_benchmark_v2
from research_runtime.harness_rsi.harness import base_harness
from research_runtime.harness_rsi.real_trajectory import run_real_trajectory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    bank = ExperienceBank(root / "bank.json")
    bank.add(__import__(
        "research_runtime.harness_rsi.experience",
        fromlist=["ResearchExperience"]).ResearchExperience(
        experience_id="exp_real_rootcause", kind="FAILURE",
        problem="falsifiability critic",
        claim="require metric-anchored falsification",
        component="hypothesis_critic", source_run_id="m6_1_acceptance_v1",
        source_task_ids=["m6_bench_v2_embodied_selection_sel_dynamics_000"],
        evidence_shas=["a" * 64],
        artifact_shas=["6c1979100dda02f035a0ee7160580cdc6d4f29500f71ec064e20cdd1df6b000d"]))
    bench = build_benchmark_v2(root / "bench")
    task = [t for t in bench.by_split("dev") if t.domain == "embodied"][0]

    def runner(ctx, t):
        return run_real_trajectory(
            benchmark_task=t, harness=base_harness(), arm=("on" if ctx["arm"]=="with_retrieval" else "off"),
            out_dir=root / "traj",
            component_config={"require_falsifying_observation":
                              bool(ctx["experience_context"])},
            seed=t.seed)

    def grader(raw, t):
        return {"metrics": {"downstream_experiment_success":
                            1.0 if raw.get("ok") else 0.0},
                "result_sha": raw.get("trajectory_sha", "")}

    res = run_ablation_v2(bank=bank, problem="falsifiability critic",
                          tasks=[task], runner=runner, grader=grader,
                          out_dir=root)
    print("verdict", res["verdict"], "on",
          res["arms"]["with_retrieval"]["mean_research_quality"],
          "off", res["arms"]["without_retrieval"]["mean_research_quality"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
