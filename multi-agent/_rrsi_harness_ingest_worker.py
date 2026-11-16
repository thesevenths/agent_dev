"""Worker: run ONE real SciNet++ harness ingest and emit structured evidence.

Invoked as a subprocess by run_m5_rrsi.py so the parent can enforce a hard
wall-clock timeout (the real engine can hang on a wrong-domain report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--candidate-sha", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from adaptive_runtime.research_ingest import ingest
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    t0 = time.time()
    res = ingest(
        report,
        ledger_path=out / "ledger.jsonl",
        experience_path=out / "experience.jsonl",
        sandbox_root=out / "sandbox",
        real=True, paired_seeds=list(args.seeds),
        decision_split="validation", holdout_split="test", epochs=1)
    real_ev = res.real_evidence or {}
    payload = {
        "decision": res.decision, "ledger_status": res.ledger_status,
        "behavior_changed": res.behavior_changed,
        "holdout_consistent": res.holdout_consistent,
        "mean_delta": (res.evidence or {}).get("mean_delta"),
        "task_deltas": (res.evidence or {}).get("task_deltas"),
        "evaluation_mode": res.evaluation_mode,
        "artifact_sha256": res.artifact_sha256,
        "real_evidence": real_ev,
        "elapsed_s": round(time.time() - t0, 2),
        "source_report": Path(args.report).name,
    }
    (out / "ingest_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print("INGEST_DONE", json.dumps(
        {k: payload[k] for k in ("decision", "holdout_consistent",
                                 "elapsed_s")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
