"""Post-commit acceptance: one REAL, reproducible Agent -> SciNet++ run.

Run AFTER committing BOTH repositories (engine first, runtime second)::

    D:/python313/python.exe tests/rt_tests/acceptance.py

Unlike the unit tests (which use temporary directories), this writes a real
report into ``research_runtime/_data/reports/`` and verifies that the report
can be tied to CLEAN, COMMITTED code in BOTH repositories. A report produced
from a dirty worktree is explicitly rejected -- that was the reproducibility
gap: code_version=HEAD while the run used uncommitted fixes.

Also prints the A08 two-scoreboard contrast, so a reader can see that:
  * an empty objective is 'undefined', never a success;
  * a declared, decidable target is the only thing that can be 'achieved'.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research_runtime.adapters import SciNetAdapter
from research_runtime.experiment_manager import ExperimentManager
from research_runtime.orchestrator import ResearchOrchestrator, evaluate_objective
from research_runtime.schemas import ResearchObjective

CONFIG = "configs/pendulum_smoke.yaml"
# Regression gate derived from the frozen smoke baseline (~0.026 MSE / ~0.54
# cosine at 1 step). A bound like 10.0 would also pass a collapsed model and
# therefore proves nothing about the model at all.
CRITERIA = {"latent_mse@1": {"max": 0.05},
            "cosine@1": {"min": 0.40}}


def _head(root: str) -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=root, capture_output=True, text=True, timeout=10)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    adapter = SciNetAdapter()
    manager = ExperimentManager(adapter)
    orchestrator = ResearchOrchestrator(manager)

    objective = ResearchObjective(
        objective_id="post_commit_acceptance",
        domain="science",
        description="Committed-code acceptance: frozen data, test split, "
                    "decidable two-metric regression gate",
        success_criteria=CRITERIA,
    )
    report = orchestrator.run_baseline(
        objective, config_path=CONFIG, epochs=1, wait_timeout=300,
    )

    metrics = (report.get("evaluate") or {}).get("metrics") or {}
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "report_path": report["report_path"],
        "pipeline_success": report["pipeline_success"],
        "objective_status": report["objective_status"],
        "objective_success": report["objective_success"],
        "success": report["success"],
        "success_criteria": report["success_criteria"],
        "code_versions": report["code_versions"],
        "metrics": metrics,
        "a08_contrast": {
            "declared_target": evaluate_objective(
                CRITERIA, metrics, pipeline_success=True)["status"],
            "no_target": evaluate_objective(
                {}, metrics, pipeline_success=True)["status"],
            "broken_criterion": evaluate_objective(
                {"latent_mse@1": None}, metrics,
                pipeline_success=True)["status"],
            "pipeline_down": evaluate_objective(
                CRITERIA, metrics, pipeline_success=False)["status"],
        },
    }, indent=2, ensure_ascii=False, default=str))

    checks: list[tuple[str, bool]] = [
        ("train and evaluate both ran", report["pipeline_success"] is True),
        ("declared target achieved", report["objective_success"] is True),
        ("objective_status == achieved", report["objective_status"] == "achieved"),
        ("metrics are structured", bool(metrics)),
    ]
    for label, info in (report["code_versions"] or {}).items():
        if not info:
            checks.append((f"{label} repo is a git checkout", False))
            continue
        checks.append((f"{label} HEAD matches this commit",
                       bool(info.get("head")) and info["head"] == _head(info["root"])))
        checks.append((f"{label} worktree is clean", info.get("dirty") is False))

    mv = report.get("model_version") or {}
    engine = (report["code_versions"] or {}).get("engine") or {}
    runtime = (report["code_versions"] or {}).get("runtime") or {}
    checks.append(("lineage records engine HEAD",
                   mv.get("code_version") == engine.get("head")))
    checks.append(("lineage records runtime HEAD",
                   mv.get("runtime_code_version") == runtime.get("head")))
    checks.append(("lineage says code was clean", mv.get("code_dirty") is False))

    print("\n--- acceptance checks ---")
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    failed = [label for label, ok in checks if not ok]
    if failed:
        print(f"\nACCEPTANCE FAILED: {len(failed)} check(s): {failed}")
        return 1
    print(f"\nACCEPTANCE PASSED: {len(checks)}/{len(checks)} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
