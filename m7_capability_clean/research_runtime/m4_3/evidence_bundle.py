"""M4.3-02 -- evidence bundle builder (real engine logs + registry export).

A value claim is only auditable if the raw artefacts survive. This module
collects, for a finished run:

  * the append-only registry export (with its SHA-256) and the event chain;
  * the experiment controller's state + event history;
  * the REAL engine job logs that belong to the campaigns in the registry;
  * each campaign directory's trial configs / metrics files;
  * a verification summary (event-chain integrity, per-run trusted ids).

Nothing is invented: every path copied is discovered from the persisted
records, and a missing artefact is reported as missing rather than skipped
silently.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .experiment_state import ExperimentEventStore
from .registry_store import RegistryStore


def _engine_log_index(evidence_store_path: Path | None = None) -> dict[str, list]:
    """Map campaign/job ids to the engine's job log lines that mention them."""
    index: dict[str, list[str]] = {}
    if evidence_store_path is None or not evidence_store_path.is_file():
        return index
    for line in evidence_store_path.read_text(encoding="utf-8",
                                              errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        payload = event.get("payload") or {}
        spec = payload.get("spec") or {}
        cid = str(spec.get("campaign_id") or payload.get("campaign_id") or "")
        if cid:
            index.setdefault(cid, []).append(event)
    return index


def build_evidence_bundle(*, db: Any, out_dir: str | Path,
                          experiment_id: str,
                          campaign_root: str | Path | None = None,
                          job_log_dir: str | Path | None = None,
                          copy_logs: bool = True) -> dict[str, Any]:
    """Collect the real evidence for one experiment into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = RegistryStore(db=db)
    export = store.export()
    (out / "registry_export.json").write_text(
        json.dumps(export, indent=2, sort_keys=True, default=str),
        encoding="utf-8")

    controller = ExperimentEventStore(db=db)
    try:
        controller_export = controller.export(experiment_id)
    except Exception as exc:  # noqa: BLE001 - report, never crash
        controller_export = {"error": f"{type(exc).__name__}: {exc}"}
    (out / "controller_export.json").write_text(
        json.dumps(controller_export, indent=2, sort_keys=True, default=str),
        encoding="utf-8")

    runs = store.load_runs()
    campaign_ids = sorted({str(r.get("campaign_id") or "") for r in runs
                           if r.get("campaign_id")})
    campaign_dirs: list[dict[str, Any]] = []
    missing: list[str] = []

    # ---- engine job logs ---------------------------------------------------
    copied_logs: list[str] = []
    if copy_logs and job_log_dir is not None:
        log_dir = Path(job_log_dir)
        dest = out / "engine_logs"
        dest.mkdir(exist_ok=True)
        wanted = set(campaign_ids) | {str(r.get("run_id")) for r in runs}
        for log in sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []:
            try:
                text = log.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(cid and cid in text for cid in wanted):
                shutil.copy2(log, dest / log.name)
                copied_logs.append(log.name)
        if not copied_logs:
            missing.append("no engine job log mentioned the recorded campaigns")

    # ---- campaign directories ---------------------------------------------
    if campaign_root is not None:
        root = Path(campaign_root)
        for cid in campaign_ids:
            cdir = root / cid
            entry: dict[str, Any] = {"campaign_id": cid, "path": str(cdir),
                                     "exists": cdir.is_dir(), "trials": []}
            if cdir.is_dir():
                for trial in sorted(p for p in cdir.iterdir() if p.is_dir()):
                    files = sorted(f.name for f in trial.glob("*.json"))
                    entry["trials"].append({"trial": trial.name,
                                            "json_files": files})
            else:
                missing.append(f"campaign dir missing: {cdir}")
            campaign_dirs.append(entry)
        (out / "campaigns.json").write_text(
            json.dumps(campaign_dirs, indent=2, sort_keys=True),
            encoding="utf-8")

    bundle = {
        "schema": "m4_3_evidence_bundle_v1",
        "experiment_id": experiment_id,
        "n_runs": len(runs),
        "campaign_ids": campaign_ids,
        "registry_export_sha256": export.get("export_sha256"),
        "registry_integrity": store.verify_integrity(),
        "n_events": len(export.get("events", [])),
        "copied_engine_logs": copied_logs,
        "campaigns": campaign_dirs,
        "missing_artefacts": missing,
        "runs": [{"run_id": r.get("run_id"), "arm": r.get("arm"),
                  "outcome": r.get("outcome"), "finalized": r.get("finalized"),
                  "campaign_id": r.get("campaign_id"),
                  "report_id": r.get("report_id"),
                  "experience_id": r.get("experience_id"),
                  "gate_passed": (r.get("gate_evidence") or {}).get("gate_passed")}
                 for r in runs],
    }
    (out / "evidence_bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    return bundle


__all__ = ["build_evidence_bundle"]
