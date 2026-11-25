"""ARR-07 SciNet++ campaign: autonomous loop on the REAL engine, end to end."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scinet_loop import run_scinet_loop
from . import core


def execute(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve() if root else Path("experiments/campaigns/ARR-07-SCINET-CAMPAIGN").resolve()
    for d in ("raw_data/scinet", "raw_data/configs", "raw_data/traces", "final"):
        (root / d).mkdir(parents=True, exist_ok=True)
    tasks = ["scinet_wm_capacity", "scinet_wm_target", "scinet_wm_lr"]
    results = []
    for t in tasks:
        r = run_scinet_loop(task_id=t, root=root, seed=42, max_iterations=3,
                            target=0.0, protocol_hash=core.sha256_json("arr07-scinet-v1"))
        results.append({"task_id": t, "converged": r.converged,
                        "iterations": r.iterations, "stop_reason": r.stop_reason,
                        "steps": r.steps,
                        "designs": r.designed_experiments})
    # Real engine evidence: all raw records must be ok=True.
    raw = list((root / "raw_data" / "scinet").glob("*.json"))
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in raw]
    ok = sum(int(r.get("ok", False)) for r in recs)
    improved = 0
    for r in results:
        if any((s.get("improvement") or 0) > 0 for s in r["steps"]):
            improved += 1
    c = {"campaign_id": "ARR-07-SCINET", "engine": "scinet++ (real subprocess)",
         "source_commit": core.git_head(".."), "branch": core.BRANCH,
         "n_tasks": len(tasks), "n_engine_records": len(recs),
         "n_engine_ok": ok, "n_tasks_with_improvement": improved,
         "tasks": results,
         "backend_operational": bool(recs) and ok == len(recs)}
    c["overall"] = ("SCINET_LOOP_OPERATIONAL" if c["backend_operational"]
                    else "SCINET_LOOP_FAILED")
    c["canonical_sha256"] = core.sha256_json(c)
    (root / "final" / "ARR07_SCINET_REPORT.json").write_text(json.dumps(
        c, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return c


__all__ = ["execute"]


if __name__ == "__main__":
    import json
    c = execute()
    print(json.dumps({"overall": c["overall"],
                      "engine_records": c["n_engine_records"],
                      "engine_ok": c["n_engine_ok"],
                      "tasks_improved": c["n_tasks_with_improvement"]},
                     indent=2, ensure_ascii=False))
