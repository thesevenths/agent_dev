"""M7.6 production assembly: real M6 HarnessLoader + M5 RuntimeState seam.

Reuses the ACTUAL human-approved M6 harness artifacts (approval captured
out-of-band in session 2026-11-16; SHA 6c197910...) -- it does NOT fabricate a
harness identity, and it never derives the harness SHA from Git HEAD.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .discovery.formal_runtime import FormalRuntimeAdapter
from .harness_rsi.adoption import AdoptionManager
from .harness_rsi.harness_loader import CampaignVersionBinding, HarnessLoader
from .harness_rsi.identity import freeze_json, sha256_json
from .research_store.db import ResearchDatabase
from .rrsi import EvolutionRegistry
from .runtime_state import RuntimeState

APPROVED_SHA = "6c1979100dda02f035a0ee7160580cdc6d4f29500f71ec064e20cdd1df6b000d"
PARENT_SHA = "9d8bae03fb9354489c3f3ac855cabc1d38a97b883948dfe1e600e6e454b25f6e"


def build_production_runtime(*, repo_root: str | Path, run_id: str,
                             domain: str, backend: Any,
                             out_dir: str | Path,
                             global_budget: int = 4) -> dict[str, Any]:
    """Assemble the formal seam on REAL approved harness artifacts.

    Returns the loader, adapter, RuntimeState and the resolved harness SHA.
    Fails closed if any real artifact is missing or the approval is not bound
    to the approved candidate SHA.
    """
    repo = Path(repo_root)
    approval_path = repo / "M6_EVIDENCE" / "human_approval.json"
    bundle_path = (repo / "M6_EVIDENCE" / "run_final2" /
                   "paired_selection" / "candidate_harness.json")
    promotion_path = repo / "M6_EVIDENCE" / "run_final2" / "promotion_verdict.json"
    for p in (approval_path, bundle_path, promotion_path):
        if not p.is_file():
            raise RuntimeError(f"production_artifact_missing:{p.name}")
    approval = __import__("json").loads(approval_path.read_text(encoding="utf-8"))
    bundle = __import__("json").loads(bundle_path.read_text(encoding="utf-8"))
    approved_sha = str(bundle.get("candidate_sha", ""))
    if approved_sha != APPROVED_SHA:
        raise RuntimeError("bundle_sha_not_approved")
    if approval.get("reviewed_candidate_sha") != approved_sha:
        raise RuntimeError("human_approval_not_bound_to_approved_sha")
    if approval.get("approved") is not True:
        raise RuntimeError("human_approval_not_granted")
    if not str(approval.get("approved_by", "")).strip():
        raise RuntimeError("human_approval_missing_approver")

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = RuntimeState(db=ResearchDatabase(out / "production_runtime.sqlite"))
    if not state.run(run_id):
        state.initialize(run_id, config={"domain": domain, "phase": "m7_6"},
                         dataset_shas={}, max_rounds=6,
                         global_budget=global_budget)

    adoption = AdoptionManager(
        state=state, run_id=run_id, round_no=1,
        record_path=out / "activation.json")
    activation = {
        "loaded_sha": approved_sha,
        "approved_sha": approved_sha,
        "loaded_artifact_path": str(bundle_path),
        "loaded_file_sha256": __import__("hashlib").sha256(
            bundle_path.read_bytes()).hexdigest(),
        "dispatch_stopped": False,
        "human_approval": approval,
    }
    freeze_json(activation, adoption.path)

    registry = EvolutionRegistry(
        db=ResearchDatabase(out / "evolution.sqlite"))
    promotion = __import__("json").loads(promotion_path.read_text(encoding="utf-8"))
    registry.record(
        experiment_id="m7_6_production", layer="research_harness",
        candidate_id="M6_HYPOTHESIS_CRITIC_V2", candidate_sha=approved_sha,
        parent_sha=str(bundle.get("parent_sha") or PARENT_SHA),
        status="PROMOTED", gate_passed=True,
        evidence={"human_approval": approval,
                  "source_promotion": sha256_json(promotion)})
    bindings = CampaignVersionBinding(
        db=ResearchDatabase(out / "bindings.sqlite"))
    loader = HarnessLoader(
        registry=registry, adoption=adoption, bindings=bindings,
        artifact_index={approved_sha: str(bundle_path)})

    adapter = FormalRuntimeAdapter(
        runtime=state, harness_loader=loader, backend=backend,
        domain=domain, run_id=run_id)
    return {"loader": loader, "adapter": adapter, "runtime": state,
            "harness_sha": approved_sha, "bindings": bindings,
            "adoption": adoption, "approval": approval}


__all__ = ["APPROVED_SHA", "PARENT_SHA", "build_production_runtime"]
