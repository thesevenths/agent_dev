"""ARR 1.1 / G3 -- one scientific research cycle (one round of the loop).

A cycle is a single invocation of the dual-layer RRSI controller (Package F),
pointed at the CURRENT inherited baseline and base harness. It is intentionally
thin: all the real science lives in ``run_dual_layer`` (reused, not rewritten).
The cycle only wires the loop's accumulated state into that call and returns the
payload, so the loop controller stays declarative about *what* runs while the
controller owns *when* and *with what inherited state*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

#: The contract the loop depends on. The concrete default is
#: ``research_runtime.evolution.run_dual_layer``; tests inject a stub.
DualLayerFn = Callable[..., dict[str, Any]]


def run_research_cycle(*, round_no: int, state: Any,
                        run_dual_layer_fn: DualLayerFn,
                        project_dir: str | Path,
                        dataset_version: str,
                        domain_seeds: tuple[int, ...],
                        domain_runner: Callable[[dict[str, Any]], float] | None,
                        # M2.7 P0-2: was a REQUIRED arg; now defaults ON so every
                        # Python API layer is fail-safe (not just the CLI). Every
                        # existing caller passes it explicitly, so behaviour is
                        # unchanged for them.
                        domain_require_frozen_manifest: bool = True,
                        domain_holdout_seeds: tuple[int, ...] = (),
                        harness_evolve: Callable[..., dict[str, Any]] | None,
                        base_harness: Any = None,
                        harness_tasks: list[Any] | None = None,
                        harness_heldout_task: Any = None,
                        harness_rounds: int = 1,
                        harness_max_minutes: float = 1.0,
                        # M2.7 statistical gate (P0-1): threaded verbatim to the
                        # dual-layer decision so a Python API caller cannot silently
                        # lower the ACCEPT bar below the formal 5-seed / holdout gate.
                        min_seeds_for_accept: int = 5,
                        practical_improvement_floor_rel: float = 0.01,
                        require_holdout_confirmation: bool = True,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one round and return the dual-layer payload.

    ``state`` is the live :class:`~research_runtime.loop.loop_state.LoopState`;
    its ``baseline_config_path`` and ``base_harness_tag`` are used as the round's
    starting point so version inheritance is visible end-to-end.
    """
    kwargs: dict[str, Any] = {
        # The harness layer runs the REAL research-harness evolution. A genuine
        # run supplies a concrete base_harness object and real research tasks plus
        # an independent held-out task; a recording-only / stub caller passes None.
        "base_harness": base_harness,
        "tasks": list(harness_tasks) if harness_tasks else [],
        "heldout_task": harness_heldout_task,
        "run_root": str(Path(project_dir) / f"round_{round_no}"),
        "repo_root": str(Path(project_dir)),
        "rounds": harness_rounds,
        "max_minutes": harness_max_minutes,
        "domain": state.domain,
        "domain_base_config": state.baseline_config_path,
        "project_dir": str(project_dir),
        "dataset_version": dataset_version,
        "domain_seeds": domain_seeds,
        "domain_require_frozen_manifest": domain_require_frozen_manifest,
        "domain_holdout_seeds": domain_holdout_seeds,
        "min_seeds_for_accept": min_seeds_for_accept,
        "practical_improvement_floor_rel": practical_improvement_floor_rel,
        "require_holdout_confirmation": require_holdout_confirmation,
        "domain_runner": domain_runner,
        # Each round is an isolated experiment: namespace the domain campaign dir
        # by round so a changed baseline across rounds (domain overrides baked in)
        # never collides with the previous round's baseline/candidate artifacts.
        "domain_campaign_id": f"dual_layer_r{round_no}",
    }
    if harness_evolve is not None:
        kwargs["harness_evolve"] = harness_evolve
    if extra:
        kwargs.update(extra)
    return run_dual_layer_fn(**kwargs)
