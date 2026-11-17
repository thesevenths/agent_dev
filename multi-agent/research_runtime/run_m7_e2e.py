"""M7.6 REAL end-to-end driver -- two-phase, out-of-band approval, formal Runtime.

NO auto-approval. NO FakeRunner. NO git-HEAD-as-harness-SHA. NO direct executor
in production.

Flow
----
  prepare : real academic APIs + real LLM; discover -> design -> freeze
            preregistration -> STOP at HUMAN_APPROVAL. A pending approval bound
            to the frozen protocol SHA is written to the trusted ApprovalStore.

  (human): operator runs approval_cli grant <id> --approved-by <name> in a
            SEPARATE process. The driver never grants an approval.

  resume  : re-read the preregistration, re-verify the out-of-band approval,
            then execute through the FORMAL M5/M6 Runtime seam.

All three capabilities (scientific judgment / memory / RSI) are exercised by
the multi-round driver; this script runs one production research round.
"""
from __future__ import annotations

import argparse
import json
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
from research_runtime.m7_production_assembly import build_production_runtime
from research_runtime.research_store import ResearchStores
from research_runtime.research_store.db import ResearchDatabase


def _git_head() -> str:
    """Git commit identity -- RECORDED separately, NEVER used as harness SHA."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _load_evidence_map(args) -> dict:
    """Load REAL value-assessment evidence from an operator-supplied file.

    M7.6 P1-A/P1-D: independent review and an explicit EIG observation model are
    required for a production value ranking. The driver NEVER fabricates them;
    without this file the run fails closed at VALUE_ASSESSMENT.
    """
    if not args.evidence_map:
        return {}
    p = Path(args.evidence_map).resolve()
    if not p.is_file():
        raise SystemExit(f"EVIDENCE_MAP_NOT_FOUND: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("EVIDENCE_MAP_INVALID: expected a JSON object")
    return obj


def _build_coordinator(*, args, root: Path, db, providers, stores,
                       approval_id: str, evidence_map: dict) -> tuple[DiscoveryCoordinator, dict]:
    assembly = build_production_runtime(
        repo_root=Path(__file__).resolve().parents[1], run_id=args.run_id,
        domain="embodied_world_model",
        backend=_domain_backend(root), out_dir=root / "production")
    return DiscoveryCoordinator(
        run_id=args.run_id, db=db, out_dir=root,
        agent=LLMAgentAdapter(max_retries=2, timeout_s=90.0),
        providers=providers, mme=0.01,
        seeds=[42, 43], selection_seeds=[42, 43],
        replication_seeds=[44, 45], global_budget=2,
        harness_sha=assembly["harness_sha"],          # from REAL M6 loader
        git_commit_sha=_git_head(),                   # separate identity
        formal_loader=assembly["loader"],
        production_runtime=assembly["adapter"],
        approval_store=stores.approvals, approval_id=approval_id,
        evidence_map=evidence_map,
        production_mode=True, max_requests=args.max_requests), assembly


def _domain_backend(root: Path):
    """The REAL DomainPack engine, invoked ONLY via the formal adapter."""
    from research_runtime.discovery.domain_packs import load_pack
    pack = load_pack("embodied_world_model")
    return pack.run_experiment


def _providers(root: Path, args):
    providers = [OpenAlexProvider(cache_dir=root / "cache" / "openalex"),
                 CrossrefProvider(cache_dir=root / "cache" / "crossref")]
    if args.include_s2:
        providers.append(SemanticScholarProvider(cache_dir=root / "cache" / "s2"))
    return providers


def _prepare(args, root: Path) -> int:
    db = ResearchDatabase(root / "store" / "research.sqlite")
    stores = ResearchStores(root / "store" / "trusted.sqlite")
    coordinator, assembly = _build_coordinator(
        args=args, root=root, db=db, providers=_providers(root, args),
        stores=stores, approval_id="",
        evidence_map=_load_evidence_map(args))
    result = coordinator.run()
    state = coordinator.state
    if result.get("stage") != "HUMAN_APPROVAL" or result.get("status") != "BLOCKED":
        # Only a clean STOP at the approval gate is a successful prepare.
        freeze_json({"phase": "prepare", "result": result,
                     "state": state, "harness_sha": assembly["harness_sha"],
                     "git_commit_sha": coordinator.git_commit_sha,
                     "llm": state.get("trajectories", [])},
                    root / "m7_6_prepare_result.json")
        print(json.dumps({"ok": False, "reason": "prepare_did_not_stop_at_approval",
                          "result": result}, ensure_ascii=False))
        return 2
    protocol_sha = state["preregistration_sha"]
    rec = stores.approvals.request(
        resource_type="m7_preregistration",
        resource_id=f"{args.run_id}:{protocol_sha}",
        resource_sha256=protocol_sha, requested_by="m7_discovery_coordinator",
        notes="M7.6 preregistration; requires out-of-band human approval")
    freeze_json({
        "phase": "prepare", "status": "AWAITING_HUMAN_APPROVAL",
        "approval_id": rec.approval_id, "protocol_sha": protocol_sha,
        "harness_sha": assembly["harness_sha"],
        "git_commit_sha": coordinator.git_commit_sha,
        "approval_cli": (
            f"py -m research_runtime.research_store.approval_cli --db "
            f"{root / 'store' / 'trusted.sqlite'} grant {rec.approval_id} "
            f"--approved-by '<operator>' --yes"),
        "trust_model": ("out-of-band operator approval; approved_by is an "
                        "unauthenticated operator string (M4.3 boundary)"),
        "result": result, "state": state,
        "llm": state.get("trajectories", []),
    }, root / "m7_6_prepare_result.json")
    print(json.dumps({"ok": True, "status": "AWAITING_HUMAN_APPROVAL",
                      "approval_id": rec.approval_id,
                      "protocol_sha": protocol_sha,
                      "result_sha": sha256_file(root / "m7_6_prepare_result.json")},
                     ensure_ascii=False))
    return 0


def _resume(args, root: Path) -> int:
    if not args.approval_id:
        raise SystemExit("RESUME_REQUIRES_APPROVAL_ID: out-of-band grant first")
    db = ResearchDatabase(root / "store" / "research.sqlite")
    stores = ResearchStores(root / "store" / "trusted.sqlite")
    coordinator, assembly = _build_coordinator(
        args=args, root=root, db=db, providers=_providers(root, args),
        stores=stores, approval_id=args.approval_id,
        evidence_map=_load_evidence_map(args))
    result = coordinator.run()
    payload = {
        "version": "m7_6_real_e2e_v1", "phase": "resume",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repository": git_fingerprint(Path(__file__).resolve().parents[1]),
        "harness_sha": coordinator.harness_sha,
        "git_commit_sha": coordinator.git_commit_sha,
        "identity_separation": coordinator.state.get("identity_separation", {}),
        "runtime_integration": coordinator.state.get("runtime_integration"),
        "approval_id": args.approval_id,
        "approval_verdict": coordinator.state.get("approval", {}),
        "llm": coordinator.state.get("trajectories", []),
        "result": result, "state": coordinator.state,
        "runtime_journal": coordinator.runtime.verify_journal(args.run_id),
        "runtime_reservations": coordinator.runtime.reservations(args.run_id),
    }
    engine = root / "experiments" / "selection" / "engine_results.json"
    if engine.is_file():
        raw = json.loads(engine.read_text(encoding="utf-8"))
        payload["result_integrity"] = evaluator.verify_result_shas(
            {"summary": raw})
    frozen = freeze_json(payload, root / "m7_e2e_result.json")
    print(json.dumps({"ok": result.get("ok"), "status": result.get("status"),
                      "stage": result.get("stage"),
                      "scientific_verdict": result.get("scientific_verdict"),
                      "runtime_integration": payload["runtime_integration"],
                      "result_sha": frozen["sha256"], "evidence_dir": str(root)},
                     ensure_ascii=False))
    return 0 if result.get("ok") else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prepare", "resume"], required=True)
    ap.add_argument("--out-dir", default="M7_EVIDENCE/m7_6_real")
    ap.add_argument("--run-id", default="arr_m76_real")
    ap.add_argument("--max-requests", type=int, default=6)
    ap.add_argument("--approval-id", default="")
    ap.add_argument("--evidence-map", default="",
                    help="JSON with real independent_review + EIG observation "
                         "model; required for a production value ranking")
    ap.add_argument("--include-s2", action="store_true")
    args = ap.parse_args()
    if not is_available():
        raise SystemExit("REAL_LLM_BLOCKED: no valid LLM credentials")
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        return _prepare(args, root)
    return _resume(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
