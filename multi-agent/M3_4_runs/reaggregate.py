"""Offline re-aggregation for the M3.4 unified run.

The real-engine E2E wrote ``domain_summary.json`` and ``evidence_index.json``
with a buggy aggregation layer (decision verdicts are lower-case strings but
the scheduler compared against upper-case constants, and the evidence index
keyed decisions by ``target`` so the three ``train.lr`` candidates collapsed
into one entry). The RAW per-candidate records produced by the real statistical
gate were correct and are preserved unchanged here.

This script ONLY re-derives the aggregation (counts / promotion / evidence
index) from those already-correct raw records. It does NOT re-run any
experiment and does NOT alter any engine output.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_runtime.scheduler import UnifiedResearchScheduler  # noqa: E402


def _mini_spec(metric_direction: str, synthetic: bool, dataset_version: str):
    class _S:
        pass
    s = _S()
    s.metric_direction = metric_direction
    s.synthetic_data = synthetic
    s.dataset_version = dataset_version
    s.domain = None
    s.paired_seeds = []
    s.holdout_seeds = []
    s.ood_config_path = None
    return s


def main(run_root: Path):
    summaries = {}
    for d in sorted(run_root.iterdir()):
        sp = d / "domain_summary.json"
        if not sp.exists():
            continue
        summary = json.loads(sp.read_text(encoding="utf-8"))
        records = summary.get("records", [])
        norm = UnifiedResearchScheduler._norm_decision
        n_acc = sum(1 for r in records if norm(r.get("decision")) == "ACCEPTED")
        n_rej = sum(1 for r in records if norm(r.get("decision")) == "REJECTED")
        n_inc = sum(1 for r in records if norm(r.get("decision")) == "INCONCLUSIVE")
        md = records[0].get("metric_direction", "maximize") if records else "maximize"
        spec = _mini_spec(md, summary.get("synthetic_data", False),
                          summary.get("dataset_version", ""))
        promo = UnifiedResearchScheduler._select_promotion(spec, records)
        summary["n_accepted"] = n_acc
        summary["n_rejected"] = n_rej
        summary["n_inconclusive"] = n_inc
        summary["promotion_recommendation"] = promo
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[d.name] = summary
        print(f"{d.name}: accepted={n_acc} rejected={n_rej} "
              f"inconclusive={n_inc} promotion={promo.get('chosen')}")

    # Rebuild evidence index from the corrected summaries.
    state_path = run_root / "scheduler_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    evidence = UnifiedResearchScheduler._build_evidence_index(None, state, summaries)
    (run_root / "evidence_index.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    print("evidence_index.json rebuilt")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("M3_4_runs/arr_m34_001")
    main(Path(root))
