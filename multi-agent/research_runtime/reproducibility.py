"""ARR v1.0 WP-10: export a reproducibility package for a research program.

Bundles ONLY on-disk artifacts with canonical SHA. Fails closed if any required
artifact is missing or its SHA cannot be recomputed. No report text is generated
from memory; everything is re-read from disk.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .harness_rsi.identity import freeze_json, sha256_file


def export(*, program_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    src = Path(program_dir).resolve()
    dst = Path(out_dir).resolve()
    if not src.is_dir():
        return {"ok": False, "reason": "program_dir_missing"}
    dst.mkdir(parents=True, exist_ok=True)
    required = ["program_state.json", "program_report.json"]
    for name in required:
        if not (src / name).is_file():
            return {"ok": False, "reason": f"missing:{name}"}
    # Copy the whole program evidence tree (append-only, never overwrite source)
    manifest_files = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            manifest_files.append({
                "path": str(rel).replace("\\", "/"),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size})
    program = json.loads((src / "program_state.json").read_text(encoding="utf-8"))
    report = json.loads((src / "program_report.json").read_text(encoding="utf-8"))
    pkg = {
        "version": "arr_v1_reproducibility_v1",
        "program_id": program.get("program_id", ""),
        "program_status": program.get("status", ""),
        "program_manifest_sha": program.get("program_manifest_sha", ""),
        "report_verdicts": report.get("verdicts", {}),
        "n_files": len(manifest_files),
        "files": manifest_files,
        "verification": {
            "program_integrity": (program.get("program_manifest_sha", "")
                                  == report.get("program_manifest_sha", "")),
            "n_rounds": len(program.get("rounds", [])),
            "all_files_have_sha": all(f["sha256"] for f in manifest_files)},
        "note": "all artifacts re-read from disk; no model-generated SHA trusted",
    }
    meta = freeze_json(pkg, dst / "REPRODUCIBILITY_MANIFEST.json")
    return {"ok": True, "out": str(dst), "manifest_sha": meta["sha256"],
            "n_files": len(manifest_files),
            "integrity": pkg["verification"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = export(program_dir=args.program_dir, out_dir=args.out_dir)
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
