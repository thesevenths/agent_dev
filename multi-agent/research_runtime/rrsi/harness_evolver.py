"""M5-02 Layer 2 -- harness evolution via the EXISTING harness machinery."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evolution_experiment import RESEARCH_HARNESS_LAYER


def generate(*, dominant_failure: str, base_harness: Any, ledger: Any,
             limit: int = 3) -> list[dict[str, Any]]:
    """Propose harness-only candidates; diagnosis contains NO held-out metrics."""
    from adaptive_runtime.harness_evolution import generate_harness_candidates
    if base_harness is None or ledger is None:
        raise RuntimeError("real base harness and ledger are required")
    candidates = generate_harness_candidates(
        dominant=dominant_failure, base_harness=base_harness,
        ledger=ledger, limit=limit)
    return [c.to_dict() for c in candidates]


def evaluate(*, base_harness: Any, candidate_harness: Any,
             candidate: dict[str, Any], tasks: list[Any],
             heldout_task: Any, run_root: str | Path,
             repo_root: str | Path, experience_path: str | Path,
             cfg: Any | None = None) -> dict[str, Any]:
    """Paired harness evaluation with fixed tasks + independent held-out."""
    from adaptive_runtime.harness_evolution import (
        EvolutionRecord, evaluate_candidate,
    )
    from adaptive_runtime.evolution_spaces import HarnessCandidate
    model = HarnessCandidate(**candidate) if not isinstance(
        candidate, HarnessCandidate) else candidate
    record: EvolutionRecord = evaluate_candidate(
        base_harness=base_harness, cand_harness=candidate_harness,
        candidate=model, tasks=tasks, heldout_task=heldout_task,
        run_root=Path(run_root), experience_path=Path(experience_path),
        repo_root=Path(repo_root), cfg=cfg)
    return {"layer": RESEARCH_HARNESS_LAYER,
            "record": record.model_dump(mode="json")
            if hasattr(record, "model_dump") else dict(record)}


__all__ = ["generate", "evaluate", "RESEARCH_HARNESS_LAYER"]
