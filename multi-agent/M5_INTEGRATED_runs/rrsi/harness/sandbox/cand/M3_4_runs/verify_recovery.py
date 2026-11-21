"""P0-3 -- real recovery evidence WITHOUT touching historical run artifacts.

The original ``arr_m34_001`` directory is the immutable evidence of a 78-minute
real-engine run. To prove recovery semantics without risking that evidence, we
run the recovery invocation on a COPY and compare file hashes.

The engine runner is mocked to "fail if called": a correct recovery must invoke
it ZERO times and must not rewrite any domain_summary.json / registry.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_runtime.scheduler import UnifiedResearchScheduler  # noqa: E402
from M3_4_runs.run_unified import RUN_ID, _specs  # noqa: E402

SOURCE = REPO / "M3_4_runs" / RUN_ID
COPY = REPO / "M3_4_runs" / "recovery_verification" / f"{RUN_ID}_resume_copy"


def _hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def main() -> dict:
    if not SOURCE.is_dir():
        raise SystemExit(f"source run missing: {SOURCE}")
    if COPY.exists():
        shutil.rmtree(COPY)
    COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, COPY)

    before_original = _hashes(SOURCE)
    before_copy = _hashes(COPY)

    calls: list[dict] = []

    def _must_not_run(**kw):
        calls.append(kw)
        raise AssertionError("engine must NOT be invoked on a full resume")

    with mock.patch("research_runtime.scheduler.run_dual_layer",
                    side_effect=_must_not_run):
        sched = UnifiedResearchScheduler(
            run_id=RUN_ID, run_root=COPY, repo_root=REPO,
            domains=_specs(), strict_resume=False)
        out = sched.run()

    after_original = _hashes(SOURCE)
    after_copy = _hashes(COPY)
    # Files the scheduler is ALLOWED to write during a recovery invocation.
    mutable = {"scheduler_state.json", "evidence_index.json",
               "run_history.jsonl"}
    overwritten = sorted(k for k in before_copy
                         if k not in mutable and after_copy.get(k) != before_copy[k])
    result = {
        "run_id": RUN_ID,
        "verification": "recovery on a COPY; original run untouched",
        "engine_calls": len(calls),
        "completed_this_invocation": out["completed_this_invocation"],
        "completed_total": out["completed_total"],
        "resumed_domains": out["resumed_domains"],
        "failed_domains": out["failed_domains"],
        "status": out["status"],
        "resume_validation": {
            d: {"status": v["status"], "checks": v["checks"]}
            for d, v in out["resume_validation"].items()},
        "historical_files_overwritten": overwritten,
        "original_run_unchanged": before_original == after_original,
        "append_only_history_lines": len(
            (COPY / "run_history.jsonl").read_text(encoding="utf-8").splitlines()),
    }
    (COPY.parent / "recovery_verification_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8")
    return result


if __name__ == "__main__":
    r = main()
    for k in ("engine_calls", "completed_this_invocation", "completed_total",
              "resumed_domains", "failed_domains", "status",
              "historical_files_overwritten", "original_run_unchanged",
              "append_only_history_lines"):
        print(f"{k}: {r[k]}")
    print("resume_validation:")
    for d, v in r["resume_validation"].items():
        print(f"  {d}: {v['status']} {v['checks']}")
