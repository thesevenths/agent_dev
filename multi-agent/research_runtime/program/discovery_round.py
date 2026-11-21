"""ARR v1.0 WP-02: adapter that runs ONE DiscoveryCoordinator round.

Bridges the ResearchProgramDriver's round_fn contract to the M7
DiscoveryCoordinator (single-round engine). It does NOT change the coordinator;
it only maps its state artifacts into a ResearchRound result. In production a
coordinator_factory wires real LLM/approval/formal_runtime; in engine-free tests
a stub factory is injected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..harness_rsi.identity import sha256_json


class DiscoveryRoundAdapter:
    def __init__(self, *, coordinator_factory: Callable[..., Any],
                 require_verified_memory: bool = False):
        self.factory = coordinator_factory
        self.require_verified_memory = require_verified_memory

    def __call__(self, *, program_id: str, round: dict[str, Any],
                 planning_digest: dict[str, Any], root: str) -> dict[str, Any]:
        rid = round["round_id"]
        run_id = f"{program_id}_{rid}"
        out_dir = str(Path(root) / rid)
        coord = self.factory(run_id=run_id, out_dir=out_dir,
                             family_id=round.get("family_id", ""),
                             program_id=program_id,
                             input_memory_refs=round.get("input_memory_refs", []))
        result = coord.run()
        st = coord.state
        # Map REAL coordinator artifacts into the round record (no invention).
        hyp = st.get("artifacts", {}).get("hypotheses", {}).get("obj", {})
        claim = st.get("artifacts", {}).get("claim_graph", {}).get("obj", {})
        ev = st.get("artifacts", {}).get("evidence", {}).get("obj", {})
        return {
            "status": result.get("status", "NOT_RUN"),
            "scientific_verdict": st.get("scientific_verdict", "INCONCLUSIVE"),
            "question_id": st.get("selected_question", {}).get("question_id", ""),
            "hypothesis_ids": [h.get("kind", "") for h in hyp.get("hypotheses", [])],
            "preregistration_sha": st.get("preregistration_sha", ""),
            "approval_id": (st.get("approval") or {}).get("approval_id", ""),
            "harness_sha": getattr(coord, "harness_sha", ""),
            "method_sha": (st.get("approval") or {}).get("method_sha", ""),
            "dataset_sha": (st.get("dataset_shas") or {}).get(
                "dataset_manifest_sha", ""),
            "campaign_ids": [r.get("campaign_id") for r in
                             (result.get("runtime_reservations", []) or [])
                             if r.get("campaign_id")],
            "evidence_refs": [ev.get("evidence_sha", "")] if ev.get("evidence_sha") else [],
            "claim_ids": [claim.get("claim_id", "")] if claim.get("claim_id") else [],
            "budget_used": {"rounds": 1, "experiments": 2,
                            "tokens": sum(t.get("total_tokens", 0)
                                          for t in st.get("trajectories", []))},
            "trace_ref": str(Path(root) / rid / "discovery_state.json"),
            "memory_used": bool(st.get("question_memory_trace", {}).get("retrieved_ids")),
            "round_result_sha": sha256_json(result),
        }


__all__ = ["DiscoveryRoundAdapter"]
