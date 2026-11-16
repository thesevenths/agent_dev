"""M5-02 -- lightweight RRSI Evolution Coordinator.

The coordinator identifies the bottleneck layer, runs ONLY that layer's
candidate experiment, and promotes only when that layer's own gate passes with
independent hold-out confirmation. It never updates both layers in one
experiment and never lowers a statistical threshold.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from .domain_model_evolver import DOMAIN_MODEL_LAYER
from .evolution_experiment import (
    RESEARCH_HARNESS_LAYER, EvolutionExperiment, LayerMixError,
    assert_single_layer, gate_allows_promotion,
)
from .evolution_registry import EvolutionRegistry
from .holdout_evidence import gate_allows_promotion_verified


class RRSICoordinator:
    def __init__(self, registry: EvolutionRegistry):
        self.registry = registry

    @staticmethod
    def _candidate_sha(candidate: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(
            candidate, sort_keys=True, ensure_ascii=False,
            default=str).encode("utf-8")).hexdigest()

    def select_layer(self, *, dominant_failure: str,
                     domain: str | None = None,
                     domain_candidate_available: bool = True) -> str:
        """Bottleneck -> layer. Shared Experience may inform diagnosis only."""
        if domain and domain_candidate_available and dominant_failure:
            return DOMAIN_MODEL_LAYER
        return RESEARCH_HARNESS_LAYER

    def run(self, *, experiment_id: str, dominant_failure: str,
            evidence: dict[str, Any],
            generator: Callable[..., list[dict[str, Any]]],
            evaluator: Callable[..., dict[str, Any]],
            layer: str | None = None,
            domain: str | None = None,
            domain_candidate_available: bool = True,
            require_structured_holdout: bool = False,
            holdout_artifact_reader: Any = None,
            holdout_manifest_path: str = "") -> dict[str, Any]:
        chosen = layer or self.select_layer(
            dominant_failure=dominant_failure, domain=domain,
            domain_candidate_available=domain_candidate_available)
        changed = set(evidence.get("changed_layers") or ())
        assert_single_layer(chosen, changed)

        # Generation receives diagnosis ONLY -- never held-out metrics.
        candidates = generator(dominant_failure=dominant_failure,
                               source_layer=chosen)
        parent_sha = self.registry.parent_for(chosen)
        outcomes: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            eid = f"{experiment_id}_{chosen.lower()}_{index}"
            exp = EvolutionExperiment(
                experiment_id=eid, layer=chosen, candidate=candidate,
                parent_sha=parent_sha,
                evidence={"source_experiment_id": experiment_id,
                          "source_layer": chosen,
                          "diagnosis": dominant_failure,
                          "source_domains": evidence.get("source_domains", [])})
            result = evaluator(experiment=eid, layer=chosen,
                               candidate=candidate,
                               dominant_failure=dominant_failure)
            decision = str(result.get("decision") or "INCONCLUSIVE").upper()
            holdout_ok = bool(result.get("holdout_confirmed"))
            frozen = bool(result.get("frozen_test_used"))
            training_only = bool(result.get("training_only"))
            sha = self._candidate_sha(candidate)
            if require_structured_holdout:
                # Real promotion: a bare boolean is NOT proof. Cross-check the
                # structured holdout evidence against THIS candidate + parent.
                gate, gate_reasons = gate_allows_promotion_verified(
                    decision, holdout_confirmed=holdout_ok,
                    frozen_test_used=frozen, training_only=training_only,
                    holdout_evidence=result.get("holdout_evidence"),
                    expected_candidate_sha=sha, expected_parent_sha=parent_sha,
                    artifact_reader=holdout_artifact_reader,
                    manifest_path=holdout_manifest_path)
            else:
                gate = gate_allows_promotion(
                    decision, holdout_confirmed=holdout_ok,
                    frozen_test_used=frozen, training_only=training_only)
                gate_reasons = [] if gate else ["base_gate_denied"]
            exp.decision = decision
            exp.gate_passed = gate
            exp.evidence.update(result)
            exp.evidence["promotion_gate"] = {
                "structured_required": bool(require_structured_holdout),
                "passed": bool(gate), "reasons": gate_reasons}
            status = "PROMOTED" if gate else decision
            record = self.registry.record(
                experiment_id=eid, layer=chosen,
                candidate_id=str(candidate.get("candidate_id", "")),
                candidate_sha=sha, parent_sha=parent_sha,
                status=status, gate_passed=gate,
                evidence=exp.evidence)
            outcomes.append({"experiment_id": eid, "layer": chosen,
                             "candidate_id": str(candidate.get("candidate_id", "")),
                             "candidate_sha": sha,
                             "decision": decision, "gate_passed": gate,
                             "status": status, "record": record})
        return {"experiment_id": experiment_id, "selected_layer": chosen,
                "parent_sha": parent_sha, "n_candidates": len(candidates),
                "outcomes": outcomes}

    def layers(self) -> dict[str, list[dict[str, Any]]]:
        return {DOMAIN_MODEL_LAYER: self.registry.all(DOMAIN_MODEL_LAYER),
                RESEARCH_HARNESS_LAYER: self.registry.all(RESEARCH_HARNESS_LAYER)}


__all__ = ["RRSICoordinator", "LayerMixError"]
