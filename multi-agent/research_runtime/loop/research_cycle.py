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
                        domain_require_frozen_manifest: bool,
                        domain_holdout_seeds: tuple[int, ...] = (),
                        harness_evolve: Callable[..., dict[str, Any]] | None,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one round and return the dual-layer payload.

    ``state`` is the live :class:`~research_runtime.loop.loop_state.LoopState`;
    its ``baseline_config_path`` and ``base_harness_tag`` are used as the round's
    starting point so version inheritance is visible end-to-end.
    """
    kwargs: dict[str, Any] = {
        "base_harness": state.base_harness_tag or None,
        "tasks": [],
        "heldout_task": None,
        "run_root": str(Path(project_dir) / f"round_{round_no}"),
        "repo_root": str(Path(project_dir)),
        "rounds": 1,
        "max_minutes": 1.0,
        "domain": state.domain,
        "domain_base_config": state.baseline_config_path,
        "project_dir": str(project_dir),
        "dataset_version": dataset_version,
        "domain_seeds": domain_seeds,
        "domain_require_frozen_manifest": domain_require_frozen_manifest,
        "domain_holdout_seeds": domain_holdout_seeds,
        "domain_runner": domain_runner,
    }
    if harness_evolve is not None:
        kwargs["harness_evolve"] = harness_evolve
    if extra:
        kwargs.update(extra)
    return run_dual_layer_fn(**kwargs)
