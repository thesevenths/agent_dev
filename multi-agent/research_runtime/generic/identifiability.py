"""Domain-agnostic experiment identifiability audit.

Problem abstracted from ARR-07: a benchmark registers several experiments, each
claiming to discriminate a different mechanism. If two experiments produce an
IDENTICAL effective configuration on a task, no outcome can tell the mechanisms
apart. A "mechanism-correct" verdict against such a pair is unfalsifiable — a
benchmark artifact, not an agent failure.

This module knows nothing about Reacher, JEPA, or science domains. It operates on
plain dicts: a context config and an experiment's override. It is reusable by any
benchmark that represents interventions as config overrides (including
domain-pack engine knobs, SciNet++ experiment specs, etc.).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class RegisteredExperiment:
    experiment_id: str
    claimed_mechanism: str
    overrides: dict[str, Any]          # nested dict, dot-free keys per section

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id,
                "claimed_mechanism": self.claimed_mechanism,
                "overrides": self.overrides}


@dataclass
class Collision:
    task_id: str
    experiment_ids: list[str]
    claimed_mechanisms: list[str]
    effective_config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id,
                "experiment_ids": self.experiment_ids,
                "claimed_mechanisms": self.claimed_mechanisms,
                "effective_config_hash": self.effective_config_hash}


@dataclass
class IdentifiabilityReport:
    n_tasks: int
    n_experiments: int
    collisions: list[Collision] = field(default_factory=list)
    equivalence_classes: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"n_tasks": self.n_tasks, "n_experiments": self.n_experiments,
                "n_collisions": len(self.collisions),
                "collisions": [c.to_dict() for c in self.collisions],
                "equivalence_classes": self.equivalence_classes,
                "all_identifiable": not self.collisions}

    @property
    def all_identifiable(self) -> bool:
        return not self.collisions


def apply_override(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base; returns a new dict (no mutation)."""
    out = json.loads(json.dumps(base))
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = apply_override(out[k], v)
        else:
            out[k] = v
    return out


def effective_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def audit(contexts: Iterable[tuple[str, dict[str, Any]]],
          experiments: Iterable[RegisteredExperiment]) -> IdentifiabilityReport:
    """Find experiments that are indistinguishable on any context.

    contexts: (task_id, base_config) pairs. Experiments are applied to each
    base config; those yielding identical effective configs collide.
    """
    ctxs = list(contexts)
    exps = list(experiments)
    collisions: list[Collision] = []
    classes: dict[str, set[str]] = defaultdict(set)
    for task_id, base in ctxs:
        by_hash: dict[str, list[RegisteredExperiment]] = defaultdict(list)
        for e in exps:
            h = effective_config_hash(apply_override(base, e.overrides))
            by_hash[h].append(e)
        for h, group in by_hash.items():
            if len(group) > 1:
                ids = sorted(e.experiment_id for e in group)
                collisions.append(Collision(
                    task_id=task_id, experiment_ids=ids,
                    claimed_mechanisms=sorted(e.claimed_mechanism for e in group),
                    effective_config_hash=h))
                classes[" ≡ ".join(ids)].update(ids)
    return IdentifiabilityReport(
        n_tasks=len(ctxs), n_experiments=len(exps), collisions=collisions,
        equivalence_classes={k: sorted(v) for k, v in sorted(classes.items())})


def mechanism_label_is_falsifiable(report: IdentifiabilityReport,
                                   experiment_id: str) -> bool:
    """A mechanism label is falsifiable only if no other experiment collides with it."""
    return all(experiment_id not in c.experiment_ids for c in report.collisions)


__all__ = ["RegisteredExperiment", "Collision", "IdentifiabilityReport",
           "apply_override", "effective_config_hash", "audit",
           "mechanism_label_is_falsifiable"]
