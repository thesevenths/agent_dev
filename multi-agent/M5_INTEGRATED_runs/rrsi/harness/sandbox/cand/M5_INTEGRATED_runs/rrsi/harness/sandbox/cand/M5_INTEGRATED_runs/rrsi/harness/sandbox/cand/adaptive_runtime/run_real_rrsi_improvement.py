"""ARR 0.2 / Package C: REAL RRSI self-improvement loop (no fakes).

Runs the OUTER self-improvement loop for real, driven by a REAL ResearchReport
produced by Package B (``research_runtime/run_real_research_loop.py``):

    real ResearchReport
      -> bottleneck diagnosis (export_research_trace)
      -> candidate over REAL training knobs (propose_research_candidate)
      -> patch materialised in an ISOLATED, hash-traced config
         (CandidateConfigBuilder + frozen-dataset SHA-256 verification)
      -> verify the patch changed the runnable config AND the engine's real metrics
      -> base/cand on PAIRED SEEDS, unified eval on the decision split,
         INDEPENDENT holdout as confirmation (never used to select)
      -> REAL measured deltas -> gate_a / gate_b
      -> ACCEPT / REJECT / INCONCLUSIVE -> ledger + experience + evidence artifact
      -> next round (ledger dedup makes the proposer pick the next adjacent value)

Limits: --rounds, --max-minutes, --seeds. Nothing is promoted to the live runtime:
applying a validated candidate is a separate, operator-gated step.

Run (agent-runtime interpreter, which has pydantic/annotated_types):

    D:/python313/python.exe -m adaptive_runtime.run_real_rrsi_improvement --rounds 1

The SciNet++ engine itself is invoked via SCINET_PYTHON (conda scinet_gpu) in a
subprocess by SciNetAdapter -- the two interpreters are intentionally different.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from adaptive_runtime import research_ingest as bridge

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "research_runtime" / "_data" / "reports"
OUT_DIR = ROOT / "experiments" / "arr_0_2"


def load_real_report(explicit: str | None) -> tuple[dict, Path]:
    """Load a REAL persisted ResearchReport (never synthesise one)."""
    if explicit:
        path = Path(explicit)
    else:
        candidates = sorted(REPORT_DIR.glob("research_report_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"no real ResearchReport in {REPORT_DIR}; run Package B first: "
                "D:/python313/python.exe -m research_runtime.run_real_research_loop")
        path = candidates[0]
    report = json.loads(path.read_text(encoding="utf-8"))
    for field in ("report_id", "objective_id", "comparisons", "outcome"):
        if field not in report:
            raise ValueError(f"not a real ResearchReport (missing {field!r}): {path}")
    return report, path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=None,
                    help="path to a REAL persisted ResearchReport JSON")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--decision-split", default="validation")
    ap.add_argument("--holdout-split", default="test")
    args = ap.parse_args()

    report, report_path = load_real_report(args.report)
    print(f"[report] {report['report_id']} outcome={report['outcome']} "
          f"sha={report.get('report_sha256', '')[:12]} file={report_path.name}")

    if not bridge.scinet_available():
        print("[ABORT] SciNet++ engine unreachable; Package C requires a REAL run "
              "(it will not silently fake). Set SCINET_DIR / SCINET_PYTHON.",
              file=sys.stderr)
        return 2

    trace = bridge.export_research_trace(report)
    print(f"[diagnose] dominant={trace['dominant_signal']} "
          f"quality={trace['quality_score']} n_decisions={trace['n_decisions']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%dT%H%M%S")
    ledger_path = OUT_DIR / f"rrsi_ledger_{run_tag}.jsonl"
    experience_path = OUT_DIR / f"rrsi_experience_{run_tag}.jsonl"
    rounds: list[dict] = []
    t_start = time.time()

    for rnd in range(1, args.rounds + 1):
        elapsed_min = (time.time() - t_start) / 60.0
        if elapsed_min >= args.max_minutes:
            print(f"[stop] time budget reached after {elapsed_min:.1f} min")
            break
        print(f"\n=== ROUND {rnd}/{args.rounds} (elapsed {elapsed_min:.1f} min) ===")
        res = bridge.ingest(
            report,
            ledger_path=ledger_path,
            experience_path=experience_path,
            sandbox_root=OUT_DIR / f"sandbox_{run_tag}_r{rnd}",
            real=True,
            paired_seeds=list(args.seeds),
            decision_split=args.decision_split,
            holdout_split=args.holdout_split,
            epochs=args.epochs,
        )
        pr = res.real_evidence
        print(f"[candidate] {pr.get('key')} {pr.get('old')} -> {pr.get('new')} "
              f"seeds={pr.get('seeds')} campaign={pr.get('campaign_id')}")
        print(f"[verify] config_changed={pr.get('config_changed')} "
              f"override_effective={pr.get('override_effective')} "
              f"behavior_changed={pr.get('behavior_changed')} "
              f"holdout_consistent={pr.get('holdout_consistent')} "
              f"missing_runs={pr.get('missing_runs')}")
        print(f"[measure] base_Q={res.base_measure['Q']:.6f} "
              f"cand_Q={res.cand_measure['Q']:.6f} "
              f"mean_delta={res.evidence['mean_delta']:.6f} "
              f"task_deltas={res.evidence['task_deltas']}")
        print(f"[gate] a={res.decision_a} b={res.decision_b} "
              f"status={res.ledger_status} DECISION={res.decision}")
        print(f"[safety] live_runtime_untouched={res.live_runtime_untouched} "
              f"artifact_sha256={res.artifact_sha256[:16]}")

        payload = res.to_dict()
        payload["source_report"] = {
            "report_id": report.get("report_id"),
            "report_sha256": report.get("report_sha256"),
            "file": report_path.name,
        }
        payload["round"] = rnd
        name = (f"rrsi_real_{run_tag}_r{rnd}_{pr.get('key', 'knob').replace('.', '-')}_"
                f"{str(pr.get('dataset_version', 'na'))[:20]}_"
                f"{res.artifact_sha256[:12]}.json")
        out_path = OUT_DIR / name
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                       default=str), encoding="utf-8")
        print(f"[artifact] {out_path}")
        rounds.append({
            "round": rnd, "candidate": pr.get("key"),
            "old": pr.get("old"), "new": pr.get("new"),
            "mean_delta": res.evidence["mean_delta"],
            "decision": res.decision, "ledger_status": res.ledger_status,
            "artifact": out_path.name,
        })

    (OUT_DIR / f"rrsi_summary_{run_tag}.json").write_text(
        json.dumps({"source_report": report.get("report_id"),
                    "dominant_signal": trace["dominant_signal"],
                    "seeds": args.seeds, "rounds": rounds},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] rounds={len(rounds)} "
          f"decisions={[r['decision'] for r in rounds]} "
          f"elapsed={(time.time()-t_start)/60:.1f} min "
          f"(no promote() called; application remains operator-gated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
