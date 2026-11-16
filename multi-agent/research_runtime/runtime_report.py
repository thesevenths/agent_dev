"""M5 Integrated -- reproducible EVIDENCE PACKAGE writer.

Collects the real IDs, SHAs and summaries needed to independently re-check a
run, without shipping bulky raw logs. Every field is derived from REAL
persisted state (RuntimeState) or caller-supplied real artifacts. No verdict is
fabricated; if a domain failed, the package says so.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _sha_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")).hexdigest()


class RuntimeReport:
    def write(self, *, coordinator: Any, out_dir: str | Path,
              filename: str = "runtime_evidence_package.json",
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
        state = coordinator.state
        run_id = coordinator.config.run_id
        status = coordinator.status()
        integ = status["journal"]
        package = {
            "schema": "m5-integrated-evidence-v1",
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run": status["run"],
            "rounds": status["rounds"],
            "global_budget": status["budget"],
            "journal": {"ok": integ["ok"], "n_events": integ.get("n_events", 0),
                        "head_sha": integ.get("head_sha", "")},
            "recovery": status["recovery"],
            "config": coordinator.config.__dict__,
            "dataset_shas": coordinator.dataset_shas,
            "approval_mode": "external_human_injected" if coordinator.approver
                             else "NONE_FAIL_CLOSED",
            "extra": extra or {},
        }
        pkg_sha = _stable_sha(package)
        package["package_sha256"] = pkg_sha
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(json.dumps(package, indent=2, sort_keys=True,
                                   ensure_ascii=False, default=str),
                        encoding="utf-8")
        package["artifact_path"] = str(path)
        package["artifact_sha256"] = _sha_file(path)
        return package


__all__ = ["RuntimeReport"]
