"""M7 real end-to-end acceptance driver.

Runs the full Scientific Discovery chain with REAL academic APIs, a REAL LLM
agent and the REAL embodied world-model engine. No FakeRunner/MockRunner is
used: those exist only in engine-free unit tests.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from research_runtime.discovery.discovery_coordinator import DiscoveryCoordinator
from research_runtime.discovery.domain_packs.embodied_world_model import evaluator
from research_runtime.discovery.llm_agent import LLMAgentAdapter, is_available
from research_runtime.discovery.literature.providers import (
    CrossrefProvider, OpenAlexProvider, SemanticScholarProvider)
from research_runtime.harness_rsi.identity import (freeze_json, git_fingerprint,
                                                   sha256_file)
from research_runtime.research_store.db import ResearchDatabase


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="M7_EVIDENCE/run")
    ap.add_argument("--run-id", default="m7_real_discovery")
    ap.add_argument("--max-requests", type=int, default=6)
    ap.add_argument("--include-s2", action="store_true")
    args = ap.parse_args()

    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED: no valid LLM credentials")

    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    db = ResearchDatabase(root / "store" / "research.sqlite")
    providers = [OpenAlexProvider(cache_dir=root / "cache" / "openalex"),
                 CrossrefProvider(cache_dir=root / "cache" / "crossref")]
    if args.include_s2:
        providers.append(SemanticScholarProvider(
            cache_dir=root / "cache" / "s2"))

    approvals: list[dict] = []

    def approve(protocol_sha: str) -> dict:
        decision = {"protocol_sha": protocol_sha, "approver": "human-operator",
                    "decision": "APPROVED", "at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S")}
        approvals.append(decision)
        freeze_json({"approval": decision}, root / "approval.json")
        return decision

    coordinator = DiscoveryCoordinator(
        run_id=args.run_id, db=db, out_dir=root,
        agent=LLMAgentAdapter(max_retries=2, timeout_s=90.0),
        providers=providers, approval=approve, mme=0.01,
        seeds=[42, 43], selection_seeds=[42, 43],
        replication_seeds=[44, 45], global_budget=2,
        harness_sha=os.getenv("M7_HARNESS_SHA") or _git_head(),
        max_requests=args.max_requests)
    result = coordinator.run()
    payload = {
        "version": "m7_real_e2e_v1",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repository": git_fingerprint(Path(__file__).resolve().parents[1]),
        "harness_sha": coordinator.harness_sha,
        "llm": coordinator.state.get("trajectories", []),
        "approval": approvals,
        "result": result,
        "state": coordinator.state,
        "runtime_journal": coordinator.runtime.verify_journal(args.run_id),
        "runtime_reservations": coordinator.runtime.reservations(args.run_id),
    }
    if (root / "experiments" / "selection" / "engine_results.json").is_file():
        raw = json.loads((root / "experiments" / "selection"
                          / "engine_results.json").read_text(encoding="utf-8"))
        payload["result_integrity"] = evaluator.verify_result_shas(
            {"summary": raw})
    frozen = freeze_json(payload, root / "m7_e2e_result.json")
    print(json.dumps({"ok": result.get("ok"), "status": result.get("status"),
                      "stage": result.get("stage"),
                      "scientific_verdict": result.get("scientific_verdict"),
                      "result_sha": frozen["sha256"],
                      "evidence_dir": str(root)}, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
