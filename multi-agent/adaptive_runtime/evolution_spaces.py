"""ARR 1.0 / D2 -- two DISTINCT evolution targets, one shared RRSI substrate.

Package C optimised ``train.lr`` / ``weight_decay``. Those are DOMAIN MODEL
knobs: they improve a scientific model, NOT the research system's ability to do
science. Confusing the two is the P0-2 architectural defect.

This module declares the separation explicitly and reuses the identical
Sandbox / Gate / Ledger / Experience / Version primitives for both::

    DOMAIN_MODEL      optimises the object of study (model, loss, factor, policy)
    RESEARCH_HARNESS  optimises the research system (planner / designer /
                      reviewer prompts, workflow, memory retrieval)

They have DIFFERENT candidate types, DIFFERENT configuration spaces, DIFFERENT
evaluation targets and NAME-SPACED ledger fingerprints, so a domain knob can
never be mistaken for a harness improvement (or vice versa).

The evaluation protocol (D1 ``ResearchQualityEvaluator``, gate_a/gate_b,
holdout splits) is shared and OUTSIDE both spaces -- see QUALITY_FROZEN_FIELDS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ledger import fingerprint as ledger_fingerprint


class CandidateType(str, Enum):
    DOMAIN_MODEL = "domain_model"
    RESEARCH_HARNESS = "research_harness"


@dataclass(frozen=True)
class EvolutionSpace:
    candidate_type: CandidateType
    search_space: dict[str, dict[str, Any]]
    frozen: frozenset[str]
    dominant_to_target: dict[str, str]
    optimization_object: str
    evaluation_target: str
    description: str = ""


#: Package C's REAL, verified SciNet++ training knobs. KEPT INTACT (reviewer:
#: "保留当前 Package C，它是领域实验优化器，不能删除").
DOMAIN_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "train.lr": {
        "values": ["0.003", "0.005", "0.01"], "level": 1,
        "rationale": "改变优化步长，直接影响多步预测误差的收敛",
        "expectation": "improve",
        "dominant": "DECISION_UNCERTAINTY",
    },
    "train.weight_decay": {
        "values": ["0.001", "0.0"], "level": 1,
        "rationale": "正则强度影响小样本动力学拟合的泛化",
        "expectation": "improve",
        "dominant": "ROBUST_DECISION",
    },
    "jepa.vicreg_var_weight": {
        "values": ["10.0", "40.0"], "level": 1,
        "rationale": "表征方差正则影响潜空间折叠与长时预测稳定性",
        "expectation": "improve",
        "dominant": "LOW_YIELD",
    },
}

DOMAIN_FROZEN = frozenset({
    "RESEARCH_EVIDENCE_STRICT", "RESEARCH_FROZEN_DATASET_CHECK",
    "HYPOTHESIS_CRITIC_REJECT_UNKNOWN", "QUALITY_EVALUATOR",
    "EVALUATION_SPLIT", "GATE_A", "GATE_B",
})

DOMAIN_TO_TARGET = {
    "DECISION_UNCERTAINTY": "train.lr",
    "LOW_YIELD": "jepa.vicreg_var_weight",
    "ROBUST_DECISION": "train.weight_decay",
}

#: Harness knobs. Each is a REAL, applicable change to the research workflow,
#: not a marker file. ``apply_harness_patch`` in harness_evolution.py writes
#: them into an isolated sandbox and the research loop actually consumes them.
HARNESS_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "planner.strategy": {
        "values": ["diagnosis_targeted", "fixed_first_knob", "repeat_refuted"],
        "level": 2,
        "rationale": "实验设计策略：按诊断选择干预 vs 固定改第一个旋钮 vs 复用已被证伪的干预",
        "expectation": "improve",
        "dominant": "EXPERIMENT_DESIGN_FAILURE",
    },
    "planner.hypothesis_prompt": {
        "values": ["evidence_grounded_v1", "terse_v0"],
        "level": 2,
        "rationale": "Planner 提示是否要求每个假设显式写出证据引用与可证伪判据",
        "expectation": "improve",
        "dominant": "HYPOTHESIS_QUALITY_FAILURE",
    },
    "memory.experience_top_k": {
        "values": ["3", "0"], "level": 1,
        "rationale": "是否检索并注入历史失败经验，避免重复已被证伪的干预",
        "expectation": "improve",
        "dominant": "MEMORY_FAILURE",
    },
    "workflow.max_candidates": {
        "values": ["2", "1"], "level": 1,
        "rationale": "每轮竞争性假设数量：影响能否区分竞争性假设与实验成本",
        "expectation": "improve",
        "dominant": "PLANNING_FAILURE",
    },
    "workflow.campaign_rounds": {
        "values": ["2", "1"], "level": 1,
        "rationale": "允许多轮研究（下一轮假设基于上一轮证据）",
        "expectation": "improve",
        "dominant": "WORKFLOW_FAILURE",
    },
    "reviewer.require_falsification": {
        "values": ["1", "0"], "level": 1,
        "rationale": "Reviewer 是否要求显式的可证伪判据才允许进入实验",
        "expectation": "improve",
        "dominant": "EVALUATION_FAILURE",
    },
}

#: What a Harness candidate may NEVER touch. This is the F4 safety boundary:
#: the system being optimised cannot rewrite its own ruler.
HARNESS_FROZEN = frozenset({
    "evaluation.protocol", "evaluation.weights", "evaluation.split",
    "evaluation.primary_metric", "evaluation.holdout", "evaluation.ood",
    "gate.a", "gate.b", "evidence.integrity", "evidence.hashes",
    "version.promote", "sandbox.escape", "quality_evaluator",
    "QUALITY_EVALUATOR", "EVALUATION_SPLIT", "GATE_A", "GATE_B",
    "RESEARCH_EVIDENCE_STRICT", "RESEARCH_FROZEN_DATASET_CHECK",
    "HYPOTHESIS_CRITIC_REJECT_UNKNOWN", "frozen_test_used", "scoring",
})

HARNESS_TO_TARGET = {
    "HYPOTHESIS_QUALITY_FAILURE": "planner.hypothesis_prompt",
    "EXPERIMENT_DESIGN_FAILURE": "planner.strategy",
    "TOOL_USAGE_FAILURE": "planner.strategy",
    "PLANNING_FAILURE": "workflow.max_candidates",
    "WORKFLOW_FAILURE": "workflow.campaign_rounds",
    "MEMORY_FAILURE": "memory.experience_top_k",
    "EVALUATION_FAILURE": "reviewer.require_falsification",
    "COST_FAILURE": "workflow.campaign_rounds",
    "DECISION_UNCERTAINTY": "planner.strategy",
    "LOW_YIELD": "planner.hypothesis_prompt",
    "ROBUST_DECISION": "memory.experience_top_k",
}

SPACES: dict[CandidateType, EvolutionSpace] = {
    CandidateType.DOMAIN_MODEL: EvolutionSpace(
        candidate_type=CandidateType.DOMAIN_MODEL,
        search_space=DOMAIN_SEARCH_SPACE, frozen=DOMAIN_FROZEN,
        dominant_to_target=DOMAIN_TO_TARGET,
        optimization_object="the scientific model / factor / policy under study",
        evaluation_target="domain metric delta on paired seeds + holdout",
        description="ARR 0.2 Package C: real SciNet++ training-knob optimisation.",
    ),
    CandidateType.RESEARCH_HARNESS: EvolutionSpace(
        candidate_type=CandidateType.RESEARCH_HARNESS,
        search_space=HARNESS_SEARCH_SPACE, frozen=HARNESS_FROZEN,
        dominant_to_target=HARNESS_TO_TARGET,
        optimization_object="the Research Harness (planner / designer / reviewer / workflow / memory)",
        evaluation_target="D1 ResearchQualityEvaluator on the SAME fixed task "
                          "+ independent held-out / OOD tasks",
        description="ARR 1.0 Package F: genuine self-improvement of research ability.",
    ),
}


def space_for(candidate_type: CandidateType | str) -> EvolutionSpace:
    key = CandidateType(candidate_type)
    return SPACES[key]


def target_for(candidate_type: CandidateType | str, dominant: str) -> str:
    space = space_for(candidate_type)
    return space.dominant_to_target.get(dominant) or next(iter(space.search_space))


def is_frozen(candidate_type: CandidateType | str, target: str) -> bool:
    """Reject any candidate that tries to rewrite the evaluation substrate."""
    space = space_for(candidate_type)
    if target in space.frozen:
        return True
    lowered = target.lower()
    return any(token in lowered for token in ("evaluat", "gate", "scor",
                                              "quality", "holdout", "promote",
                                              "evidence_integrity"))


def namespaced_fingerprint(candidate_type: CandidateType | str, *,
                           target: str, old: Any, new: Any,
                           taxonomy: str) -> str:
    """Ledger key is name-spaced so the two spaces can never collide/mix."""
    key = CandidateType(candidate_type)
    return f"{key.value}::{ledger_fingerprint(target, old, new, taxonomy)}"


def candidate_type_of(fingerprint: str) -> CandidateType:
    for key in CandidateType:
        if fingerprint.startswith(key.value + "::"):
            return key
    raise ValueError(f"fingerprint {fingerprint!r} has no evolution namespace")


def describe() -> dict[str, Any]:
    return {
        key.value: {
            "optimization_object": s.optimization_object,
            "evaluation_target": s.evaluation_target,
            "knobs": sorted(s.search_space),
            "frozen_count": len(s.frozen),
        } for key, s in SPACES.items()
    }
