"""P0-2 -- offline, READ-ONLY re-derivation of the M3.4 evidence index.

What this does
--------------
The real-engine run ``arr_m34_001`` (78 min, 3 domains) wrote correct RAW
per-candidate records, but its aggregation layer never surfaced the OOD
traceability chain: ``ood_dataset_version`` was ``null`` even though the
SciNet++ engine had written 40 real ``ood_metrics.json`` artifacts containing
dataset_version / checkpoint_sha256 / run_id / raw OOD metrics, and the frozen
OOD manifest has SHA-256 pins for the config and all four splits.

This script re-derives ONLY the index/aggregation layer from those unchanged
raw records plus the REAL on-disk artifacts. It does NOT:

  * re-run any experiment,
  * change any raw per-candidate record, decision, p-value or metric,
  * compute or invent a metric / hash that is not already on disk,
  * lower any statistical threshold.

Before touching the main index, the original v1 files are copied to
``*_v1_original.json`` so no historical evidence is destroyed.

Usage: python M3_4_runs/rederive_evidence_index.py [run_root]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_runtime.scheduler import (  # noqa: E402
    UnifiedResearchScheduler, collect_domain_provenance,
)
from M3_4_runs.run_unified import RUN_ID, _specs  # noqa: E402


def _backup(path: Path) -> Path:
    dst = path.with_name(path.stem + "_v1_original.json")
    if not dst.is_file():
        shutil.copy2(path, dst)
    return dst


def main(run_root: Path) -> dict:
    if not run_root.is_dir():
        raise SystemExit(f"run_root missing: {run_root}")
    specs = {s.domain: s for s in _specs()}
    state_path = run_root / "scheduler_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["run_id"] == RUN_ID, state["run_id"]

    summaries: dict[str, dict] = {}
    report: dict[str, dict] = {}
    for domain, spec in specs.items():
        sp = run_root / domain / "domain_summary.json"
        if not sp.is_file():
            raise SystemExit(f"missing domain_summary.json for {domain}")
        summary = json.loads(sp.read_text(encoding="utf-8"))
        prov = collect_domain_provenance(
            spec=spec, dr_root=run_root / domain,
            campaign_id=f"{RUN_ID}_{domain}")
        # Keep the RAW records and every decision / metric byte-for-byte.
        # Only the provenance chain and the OOD dataset identity (previously
        # null) are added, and both come from real files on disk.
        summary["provenance"] = prov
        disk_version = prov["ood_dataset"]["dataset_version"]
        if disk_version and not summary.get("ood_dataset_version"):
            summary["ood_dataset_version"] = disk_version
        _backup(sp)
        sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        summaries[domain] = summary
        report[domain] = {
            "provenance_status": prov["status"],
            "missing": prov["missing"],
            "ood_dataset_version": summary.get("ood_dataset_version"),
            "ood_manifest_sha256": prov["ood_dataset"]["manifest"].get("manifest_sha256"),
            "ood_checkpoint_count": len(prov["checkpoint_sha256"]["ood"]),
            "ood_evaluation_run_count": len(prov["evaluation_run_ids"]["ood"]),
            "ood_metric_artifacts": prov["ood_metrics"]["n_artifacts"],
            "raw_records_unchanged": True,
        }

    ev_path = run_root / "evidence_index.json"
    _backup(ev_path)
    evidence = UnifiedResearchScheduler._build_evidence_index(
        None, state, summaries)
    ev_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    (run_root / "provenance_rederivation_report.json").write_text(
        json.dumps({"run_id": RUN_ID, "source": "real on-disk artifacts, no rerun",
                    "schema_version": evidence["schema_version"],
                    "domains": report}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return report


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "M3_4_runs" / RUN_ID
    out = main(root)
    for name, d in out.items():
        print(f"{name}: status={d['provenance_status']} "
              f"ood_version={d['ood_dataset_version']} "
              f"ckpts={d['ood_checkpoint_count']} "
              f"runs={d['ood_evaluation_run_count']} "
              f"ood_artifacts={d['ood_metric_artifacts']} missing={d['missing']}")
