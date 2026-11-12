"""ARR 1.1 / G4 -- the autonomous research-loop controller.

Owns *when* rounds run and *with what inherited state*. It does not invent any
science: each round is delegated to the dual-layer controller (Package F), and
each accepted result is inherited via :mod:`improvement_cycle`. The controller
guarantees three properties the acceptance criteria demand:

1. ONE command runs many rounds (``--rounds``); the loop advances
   ``current_round`` itself.
2. The next round inherits the accepted version (domain config baked into a new
   baseline; harness patches applied via ``harness_inherit_fn``).
3. A crashed process resumes from the last COMPLETED round because a checkpoint
   is persisted after every round (see :mod:`recovery`).

A ``stop_on_no_accept`` policy (default True) ends the loop early when a full
round accepts nothing in either layer -- a real no-gain stop, not a fake loop
that spins forever.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .improvement_cycle import derive_next_version
from .loop_report import build_loop_report, summarise_text
from .loop_state import LoopState, save
from .recovery import load_or_init, mark_completed, mark_failed
from .research_cycle import run_research_cycle
from .version_inheritance import default_harness_inherit


def run_loop(*, state_path: str | Path,
             domain: str,
             baseline_config_path: str | Path,
             project_dir: str | Path,
             max_rounds: int = 3,
             dataset_version: str = "test-v1",
             domain_seeds: tuple[int, ...] = (50, 51),
             domain_runner: Callable[[dict[str, Any]], float] | None = None,
             domain_require_frozen_manifest: bool = False,
             domain_holdout_seeds: tuple[int, ...] = (),
             harness_evolve: Callable[..., dict[str, Any]] | None = None,
             harness_inherit_fn: Callable[[list[dict[str, Any]], Any], Any]
             | None = None,
             stop_on_no_accept: bool = True,
             base_harness_tag: Any = None,
             loop_id: str = "",
             run_dual_layer_fn: Callable[..., dict[str, Any]] | None = None,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run (or resume) a multi-round autonomous research loop.

    Returns the loop report. The real dual-layer function is injected via
    ``run_dual_layer_fn`` (default the packaged ``run_dual_layer``); tests pass a
    stub. ``harness_inherit_fn`` defaults to the real
    :func:`default_harness_inherit`.
    """
    from research_runtime.evolution import run_dual_layer
    rdl = run_dual_layer_fn or run_dual_layer
    inherit = harness_inherit_fn or default_harness_inherit

    # A safe default checkpoint location: a dedicated file under the project dir.
    # An empty/None path must never resolve to the CWD (which is unsafe and easy
    # to clobber) -- this mirrors the CLI's default and protects library callers.
    if not state_path:
        state_path = str(Path(project_dir) / "state" / "loop_state.json")

    state, resumed = load_or_init(
        state_path=state_path,
        config={"domain": domain, "max_rounds": max_rounds,
                "baseline_config_path": str(baseline_config_path),
                "base_harness_tag": "" if base_harness_tag is None else "",
                "loop_id": loop_id,
                "dataset_version": dataset_version,
                "domain_seeds": list(domain_seeds)})
    if base_harness_tag is not None and not resumed:
        state.base_harness_tag = base_harness_tag

    if state.status == "completed":
        return build_loop_report(state)
    if state.status == "failed":
        # A failed loop is not auto-restarted; the operator must start anew.
        return build_loop_report(state)

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    try:
        while state.current_round <= state.max_rounds:
            payload = run_research_cycle(
                round_no=state.current_round, state=state,
                run_dual_layer_fn=rdl, project_dir=project_dir,
                dataset_version=dataset_version, domain_seeds=domain_seeds,
                domain_runner=domain_runner,
                domain_require_frozen_manifest=domain_require_frozen_manifest,
                domain_holdout_seeds=domain_holdout_seeds,
                harness_evolve=harness_evolve, extra=extra)
            _record_payload_sha(state, payload)

            # Ingest this round's accepted results (bakes domain overrides into a
            # new baseline; threads harness patches) BEFORE advancing the counter.
            state, record, _next_baseline = derive_next_version(
                state=state, payload=payload, project_dir=project_dir,
                harness_inherit_fn=inherit)
            # Tally accepted totals for the report (recomputed from the version
            # history so a resume does not double-count).
            state.n_domain_accepted = sum(
                len(v.accepted_domain_overrides) for v in state.version_history)
            state.n_harness_accepted = sum(
                len(v.accepted_harness_targets) for v in state.version_history)

            # The round is now FULLY ingested (including the inherited version).
            # Advance to the next round and atomically persist. The checkpoint
            # therefore always points at the NEXT round to run, so a crash during
            # round N+1 re-runs N+1 -- never the already-completed, already-
            # -inherited round N.
            state.last_completed_round = state.current_round
            state.current_round += 1
            save(state, state_path)

            if stop_on_no_accept and record is None:
                # No-gain stop: a complete round accepted nothing. End the loop
                # rather than spin. The checkpoint still records the attempt.
                break

        mark_completed(state, state_path)
    except Exception as exc:  # noqa: BLE001 -- record the failure, do not hang
        mark_failed(state, state_path)
        raise

    report = build_loop_report(state)
    report["resumed"] = resumed
    report["_text"] = summarise_text(report)
    return report


def _record_payload_sha(state: LoopState, payload: dict[str, Any]) -> None:
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        state.last_round_payload_sha256 = hashlib.sha256(blob).hexdigest()
    except (TypeError, ValueError):
        state.last_round_payload_sha256 = ""
