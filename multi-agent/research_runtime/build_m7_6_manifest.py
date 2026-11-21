"""Build the M7.6 evidence manifest from on-disk artifacts (no fabrication)."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from research_runtime.harness_rsi.identity import sha256_file


def _sha(path: Path) -> dict:
    if path.is_file():
        return {"path": str(path), "sha256": sha256_file(path),
                "bytes": path.stat().st_size}
    return {"path": str(path), "sha256": "", "bytes": 0, "missing": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="M7_6_EVIDENCE/run2")
    ap.add_argument("--out", default="M7_6_EVIDENCE_MANIFEST.json")
    args = ap.parse_args()
    root = Path(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()

    e2e = json.loads((root / "m7_e2e_result.json").read_text(encoding="utf-8"))
    prep = json.loads((root / "m7_6_prepare_result.json").read_text(
        encoding="utf-8"))
    state = e2e["state"]
    approval = state.get("approval", {})
    evidence_map = json.loads((repo / "M7_6_EVIDENCE" /
                               "value_evidence.json").read_text(encoding="utf-8"))

    # LLM call summary from the REAL E2E trajectories.
    llm_calls = [{"stage": t.get("stage"), "provider": t.get("provider"),
                  "model_id": t.get("model_id"),
                  "total_tokens": t.get("total_tokens"), "ok": t.get("ok")}
                 for t in e2e.get("llm", [])]
    literature = state["artifacts"]["literature"]["obj"]["coverage"]

    manifests = {
        "version": "m7_6_evidence_manifest_v1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": {"commit_sha": head, "branch": branch,
                "working_tree_clean": False,
                "note": "manifest built on a validation branch with evidence "
                        "artifacts intentionally untracked/pending commit"},
        "identities": {
            "harness_sha": state.get("identity_separation", {}).get("harness_sha"),
            "git_commit_sha_at_run": state.get("identity_separation", {}).get(
                "git_commit_sha"),
            "runtime_integration": state.get("runtime_integration"),
            "harness_source": "M6 human-approved harness 6c197910... via "
                              "real HarnessLoader; NEVER git HEAD"},
        "approval": {
            "approval_id": approval.get("approval_id"),
            "protocol_sha_sha256_bound": approval.get("protocol_sha"),
            "trust": approval.get("trust"),
            "source": "out-of-band approval_cli (separate process)",
            "approved_by": approval.get("approved_by"),
            "trust_model": "unauthenticated operator string (M4.3 boundary); "
                           "approved via explicit user AskUserQuestion 2026-11-17",
            "ok": approval.get("ok")},
        "model": {"provider": "vllm-openai-compatible",
                  "model_id": "/models/Qwen3.8-27B-FP8",
                  "calls": llm_calls,
                  "total_tokens": sum(c["total_tokens"] or 0 for c in llm_calls)},
        "academic_api": {
            "providers_enabled": literature.get("providers", []),
            "n_unique_papers": literature.get("n_unique_papers"),
            "n_queries": literature.get("n_queries"),
            "coverage_status": literature.get("status"),
            "source_coverage": state["artifacts"]["novelty"]["obj"].get(
                "source_coverage")},
        "value_evidence": {
            "sha256": sha256_file(repo / "M7_6_EVIDENCE" /
                                  "value_evidence.json"),
            "reviewer": evidence_map.get("trust", {}).get("reviewer_identity"),
            "used_for": evidence_map.get("trust", {}).get("used_for"),
            "not_used_for": evidence_map.get("trust", {}).get("not_used_for")},
        "runtime": {
            "journal": e2e.get("runtime_journal"),
            "reservations": e2e.get("runtime_reservations"),
            "result_integrity": e2e.get("result_integrity")},
        "experiments": {
            "campaigns": [r.get("campaign_id")
                          for r in e2e.get("runtime_reservations", [])],
            "selection_verdict": state.get("selection_verdict"),
            "replication_verdict": state.get("replication_verdict"),
            "scientific_verdict": state.get("scientific_verdict"),
            "power_gate": state.get("power_gate"),
            "memory_tier": state.get("artifacts", {}).get("memory", {}).get(
                "obj", {}).get("tier")},
        "benchmarks": {
            "M7_2": _sha(repo / "M7_2_BENCHMARK" / "pilot_run" / "metrics.json"),
            "M7_3": _sha(repo / "M7_3_MEMORY" / "pilot_run" / "metrics.json"),
            "M7_4": _sha(repo / "M7_4_RSI" / "promotion" / "promotion_log.json"),
            "M7_5": _sha(repo / "M7_5_INTEGRATED" / "metrics" / "metrics.json"),
        },
        "artifacts": {
            "prepare_result": _sha(root / "m7_6_prepare_result.json"),
            "e2e_result": _sha(root / "m7_e2e_result.json"),
            "source_audit": _sha(repo / "M7_6_SOURCE_AUDIT.md"),
            "implementation_report": _sha(repo / "M7_6_IMPLEMENTATION_REPORT.md"),
            "test_report": _sha(repo / "M7_6_TEST_REPORT.md"),
            "real_e2e_report": _sha(repo / "M7_6_REAL_E2E_REPORT.md"),
            "multi_round_report": _sha(repo / "M7_6_MULTI_ROUND_CAPABILITY_REPORT.md"),
            "verdict": _sha(repo / "M7_6_VERDICT.json"),
        },
        "holdout": {
            "ledger": _sha(root / "claims" / "holdout_access.json"),
            "authorization_purpose": "final_holdout_evaluation"},
        "replication": {
            "artifact": _sha(root / "artifacts" / "replication.json")},
    }
    out = repo / args.out
    meta = __import__("research_runtime.harness_rsi.identity", fromlist=[
        "freeze_json"]).freeze_json(manifests, out)
    print(json.dumps({"ok": True, "manifest": str(out),
                      "sha256": meta["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
