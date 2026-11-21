"""M7 Domain Pack: embodied world model (real local engine)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ...literature.base import ResearchQuestionInput
from . import dataset_registry, engine_adapter, evaluator, experiment_templates
from . import metrics as domain_metrics

ROOT = Path(__file__).resolve().parent


class EmbodiedWorldModelPack:
    name = "embodied_world_model"

    def __init__(self) -> None:
        self.config = yaml.safe_load((ROOT / "domain.yaml").read_text(
            encoding="utf-8"))
        self.literature_config = yaml.safe_load(
            (ROOT / "literature_queries.yaml").read_text(encoding="utf-8"))

    def literature_input(self, *, question: str | None = None,
                         concepts: list[str] | None = None,
                         target_claims: list[str] | None = None
                         ) -> ResearchQuestionInput:
        cfg = self.literature_config
        return ResearchQuestionInput(
            question=question or cfg["question"],
            domain=cfg["domain"],
            concepts=concepts or list(cfg["concepts"]),
            target_claims=target_claims or list(cfg["target_claims"]))

    def dataset_identity(self) -> dict[str, Any]:
        ident = dataset_registry.dataset_identity()
        ident["family_manifest"] = dataset_registry.family_manifest()
        ident["family_isolation"] = dataset_registry.validate_family_isolation()
        return ident

    def make_configs(self, out_dir: str | Path) -> dict[str, Any]:
        return experiment_templates.make_configs(out_dir)

    def run_experiment(self, *, experiment_id: str, arms: list[str],
                       seeds: list[int], splits: list[str],
                       out_dir: str | Path) -> dict[str, Any]:
        return engine_adapter.run_experiment(
            experiment_id=experiment_id, arms=arms, seeds=seeds,
            splits=splits, out_dir=out_dir)

    def evaluate(self, *, results: dict[str, Any],
                 primary_metric: str | None = None,
                 mme: float = 0.01, direction: str | None = None
                 ) -> dict[str, Any]:
        return evaluator.evaluate(
            results=results,
            primary_metric=primary_metric or self.config["primary_metric"],
            mme=mme, direction=direction or self.config["metric_direction"])

    def safety_check(self, *, results: dict[str, Any]) -> dict[str, Any]:
        return evaluator.safety_check(results=results)


__all__ = ["EmbodiedWorldModelPack", "ROOT", "dataset_registry",
           "engine_adapter", "evaluator", "experiment_templates",
           "domain_metrics"]
