"""M5-02 -- dual-layer RRSI isolation and promotion gates (ENGINE-FREE)."""
from __future__ import annotations

import pytest

from research_runtime.research_store.db import (
    ResearchDatabase, ResearchStoreConflict,
)
from research_runtime.rrsi import (
    RRSICoordinator, gate_allows_promotion,
)
from research_runtime.rrsi.domain_model_evolver import DOMAIN_MODEL_LAYER
from research_runtime.rrsi.evolution_experiment import LayerMixError
from research_runtime.rrsi.evolution_registry import EvolutionRegistry
from research_runtime.rrsi.harness_evolver import RESEARCH_HARNESS_LAYER


def _coordinator(tmp_path):
    return RRSICoordinator(
        EvolutionRegistry(db=ResearchDatabase(tmp_path / "rrsi.sqlite")))


def test_layer_selection_and_single_layer_guard(tmp_path):
    coord = _coordinator(tmp_path)
    assert coord.select_layer(dominant_failure="dynamics_error_accumulation",
                             domain="scinet") == DOMAIN_MODEL_LAYER
    assert coord.select_layer(dominant_failure="DECISION_UNCERTAINTY",
                             domain=None) == RESEARCH_HARNESS_LAYER
    with pytest.raises(LayerMixError):
        coord.run(
            experiment_id="mixed", dominant_failure="x",
            evidence={"changed_layers": [RESEARCH_HARNESS_LAYER]},
            generator=lambda **kw: [{"candidate_id": "c"}],
            evaluator=lambda **kw: {"decision": "ACCEPTED",
                                    "holdout_confirmed": True},
            layer=DOMAIN_MODEL_LAYER)


def test_holdout_never_reaches_generation_and_only_gated_candidate_promotes(tmp_path):
    coord = _coordinator(tmp_path)
    seen = []

    def generator(**kwargs):
        seen.append(kwargs)
        return [{"candidate_id": "dom_1", "target": "train.lr", "new": 0.001}]

    def evaluator(**kwargs):
        return {"decision": "ACCEPTED", "holdout_confirmed": True,
                "frozen_test_used": False, "training_only": False,
                "primary_metric": "m"}

    out = coord.run(
        experiment_id="EXP1", dominant_failure="dynamics_error_accumulation",
        evidence={"changed_layers": [DOMAIN_MODEL_LAYER],
                  "source_domains": ["scinet"]},
        generator=generator, evaluator=evaluator,
        layer=DOMAIN_MODEL_LAYER)
    assert out["outcomes"][0]["status"] == "PROMOTED"
    assert "holdout" not in seen[0]
    assert seen[0]["source_layer"] == DOMAIN_MODEL_LAYER

    # Rejected candidates are retained, and promotion requires held-out proof.
    def reject_eval(**kwargs):
        return {"decision": "REJECTED", "holdout_confirmed": True}

    coord.run(experiment_id="EXP2", dominant_failure="x",
              evidence={"changed_layers": [DOMAIN_MODEL_LAYER]},
              generator=lambda **kw: [{"candidate_id": "dom_2"}],
              evaluator=reject_eval, layer=DOMAIN_MODEL_LAYER)
    records = coord.registry.all(DOMAIN_MODEL_LAYER)
    assert {r["status"] for r in records} == {"PROMOTED", "REJECTED"}
    assert coord.registry.all(RESEARCH_HARNESS_LAYER) == []


def test_version_inheritance_uses_promoted_parent_sha(tmp_path):
    coord = _coordinator(tmp_path)
    cand = {"candidate_id": "c1"}
    for eid in ("V1", "V2"):
        out = coord.run(
            experiment_id=eid, dominant_failure="x",
            evidence={"changed_layers": [RESEARCH_HARNESS_LAYER]},
            generator=lambda **kw: [cand],
            evaluator=lambda **kw: {"decision": "ACCEPTED",
                                    "holdout_confirmed": True},
            layer=RESEARCH_HARNESS_LAYER)
        if eid == "V1":
            first = out["outcomes"][0]["record"]
        else:
            second = out
    assert first["candidate_sha"]
    assert second["parent_sha"] == first["candidate_sha"]


def test_gate_never_accepts_training_only_or_frozen_test():
    assert gate_allows_promotion("ACCEPTED", holdout_confirmed=True) is True
    assert gate_allows_promotion("ACCEPTED", holdout_confirmed=False) is False
    assert gate_allows_promotion("ACCEPTED", holdout_confirmed=True,
                                 training_only=True) is False
    assert gate_allows_promotion("ACCEPTED", holdout_confirmed=True,
                                 frozen_test_used=True) is False
    assert gate_allows_promotion("REJECTED", holdout_confirmed=True) is False


def test_experiment_id_cannot_be_reused_across_layers(tmp_path):
    registry = EvolutionRegistry(db=ResearchDatabase(tmp_path / "cross.sqlite"))
    registry.record(experiment_id="E", layer=DOMAIN_MODEL_LAYER,
                    candidate_id="d", candidate_sha="a", status="REJECTED")
    with pytest.raises(ResearchStoreConflict):
        registry.record(experiment_id="E", layer=RESEARCH_HARNESS_LAYER,
                        candidate_id="h", candidate_sha="b", status="REJECTED")
