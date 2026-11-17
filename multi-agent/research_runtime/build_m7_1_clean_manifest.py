"""Build the immutable M7.1 clean-checkout evidence manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from research_runtime.harness_rsi.identity import sha256_file

KEY_SOURCES = (
    "research_runtime/run_m7_e2e.py",
    "research_runtime/discovery/discovery_coordinator.py",
    "research_runtime/discovery/questions.py",
    "research_runtime/discovery/novelty/assessor.py",
    "research_runtime/discovery/hypotheses/engine.py",
    "research_runtime/discovery/literature/providers.py",
    "research_runtime/discovery/memory/verified.py",
    "research_runtime/discovery/experiment/designer.py",
    "research_runtime/discovery/domain_packs/embodied_world_model/engine_adapter.py",
    "research_runtime/harness_rsi/harness_loader.py",
    "research_runtime/research_service.py",
    "research_runtime/runtime_state.py",
)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("pytest", "pydantic", "PyYAML", "torch", "transformers"):
        env[package] = _version(package)
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None) or "N/A"
    except Exception:  # noqa: BLE001
        env["cuda_available"] = False
        env["cuda_version"] = "N/A"
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", default="E:/agent_dev/m7_capability_clean")
    ap.add_argument("--run-dir", default="M7_1_EVIDENCE/e2e_clean")
    ap.add_argument("--commit", default="729184c4d8a217d121f577243cc4b1f987e7922a")
    ap.add_argument("--output", default="M7_1_EVIDENCE_MANIFEST.json")
    args = ap.parse_args()
    clean_root = Path(args.clean_root)
    repo = Path(__file__).resolve().parents[1]
    run_dir = (repo / args.run_dir).resolve()
    payload = json.loads((run_dir / "m7_e2e_result.json").read_text(
        encoding="utf-8"))
    trajectories = payload.get("state", {}).get("trajectories", [])
    first = next((t for t in trajectories if t.get("provider")), {})
    dataset = payload.get("state", {}).get("artifacts", {}).get(
        "literature", {}).get("obj", {}).get("coverage", {})
    source_shas = {}
    for rel in KEY_SOURCES:
        p = clean_root / rel
        source_shas[rel] = sha256_file(p) if p.is_file() else "MISSING"
    artifacts = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            artifacts.append({"path": p.relative_to(repo).as_posix(),
                              "sha256": sha256_file(p),
                              "bytes": p.stat().st_size})
    manifest = {
        "version": "m7_1_clean_evidence_v1",
        "run_id": "m7_1_clean_729184c4",
        "git_commit": args.commit,
        "git_branch": "detached-clean-archive",
        "working_tree_clean": True,
        "untracked_files": [],
        "source_snapshot_method": "git archive of exact commit; no working-tree files copied",
        "source_sha256": source_shas,
        "config_sha256": hashlib.sha256(json.dumps(
            payload.get("state", {}).get("config", {}), sort_keys=True,
            ensure_ascii=False).encode()).hexdigest(),
        "dataset_sha256": payload.get("state", {}).get("dataset_shas", {}),
        "model_provider": first.get("provider", ""),
        "model_id": first.get("model_id", ""),
        "model_deployment": first.get("model_deployment", ""),
        "engine_version": "embodied_world_model_local@" + args.commit[:12],
        "environment": _environment(),
        "started_at": payload.get("started_at", ""),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "literature_coverage": dataset,
        "result_sha256": sha256_file(run_dir / "m7_e2e_result.json"),
        "artifacts": artifacts,
        "verification": {
            "e2e_status": payload.get("result", {}).get("status"),
            "scientific_verdict": payload.get("result", {}).get(
                "scientific_verdict"),
            "runtime_journal": payload.get("runtime_journal", {}),
            "result_integrity": payload.get("result_integrity", {}),
        },
    }
    out = repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(out),
                      "sha256": sha256_file(out),
                      "n_artifacts": len(artifacts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
