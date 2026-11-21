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
from .recovery import (LoopRecoveryError, load_or_init, mark_completed,
                        mark_failed)
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
             # M2.7 P0-2: frozen-manifest enforcement is a RUNTIME safety property,
             # not a CLI convenience. Default ON so any direct Python-API caller
             # (not just the CLI) is protected; dev mode must opt OUT explicitly.
             domain_require_frozen_manifest: bool = True,
             domain_holdout_seeds: tuple[int, ...] = (),
             # M2.7 P0-1 statistical gate (threaded verbatim to the decision).
             min_seeds_for_accept: int = 5,
             practical_improvement_floor_rel: float = 0.01,
             require_holdout_confirmation: bool = True,
             harness_evolve: Callable[..., dict[str, Any]] | None = None,
             harness_inherit_fn: Callable[[list[dict[str, Any]], Any], Any]
             | None = None,
             base_harness: Any = None,
             harness_tasks: list[Any] | None = None,
             harness_heldout_task: Any = None,
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

    # Derive a stable string identifier for the harness layer from a supplied
    # concrete harness object (the loop tracks the object in memory; only its
    # fingerprint is persisted for recovery checks).
    if base_harness_tag is None and base_harness is not None and hasattr(
            base_harness, "fingerprint"):
        base_harness_tag = base_harness.fingerprint()

    state, resumed = load_or_init(
        state_path=state_path,
        config={"domain": domain, "max_rounds": max_rounds,
                "baseline_config_path": str(baseline_config_path),
                "base_harness_tag": "" if base_harness_tag is None else str(base_harness_tag),
                "loop_id": loop_id,
                "dataset_version": dataset_version,
                "domain_seeds": list(domain_seeds)})
    if base_harness_tag is not None and not resumed:
        state.base_harness_tag = base_harness_tag

    # M2.7 P0-2: on RESUME, re-verify the frozen dataset bound to the inherited
    # baseline before trusting a prior run's checkpoint. A changed/corrupt dataset
    # must fail closed, never silently continue. (Per-round evaluation also
    # re-verifies, but the explicit resume check is the gate ChatGPT required.)
    if resumed and domain_require_frozen_manifest:
        _reverify_frozen_manifest(
            project_dir=project_dir,
            baseline_config_path=state.baseline_config_path,
            dataset_version=dataset_version,
            domain=domain)

    if state.status == "completed":
        return build_loop_report(state)
    if state.status == "failed":
        # A failed loop is not auto-restarted; the operator must start anew.
        return build_loop_report(state)

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Concrete base harness object, tracked in memory across rounds (JSON state
    # stores only its fingerprint string). None means a recording-only / stub
    # caller -- harness inheritance then stays a string tag.
    current_base_harness = base_harness

    try:
        while state.current_round <= state.max_rounds:
            payload = run_research_cycle(
                round_no=state.current_round, state=state,
                run_dual_layer_fn=rdl, project_dir=project_dir,
                dataset_version=dataset_version, domain_seeds=domain_seeds,
                domain_runner=domain_runner,
                domain_require_frozen_manifest=domain_require_frozen_manifest,
                domain_holdout_seeds=domain_holdout_seeds,
                min_seeds_for_accept=min_seeds_for_accept,
                practical_improvement_floor_rel=practical_improvement_floor_rel,
                require_holdout_confirmation=require_holdout_confirmation,
                harness_evolve=harness_evolve,
                base_harness=current_base_harness,
                harness_tasks=harness_tasks,
                harness_heldout_task=harness_heldout_task,
                extra=extra)
            _record_payload_sha(state, payload)

            # Ingest this round's accepted results (bakes domain overrides into a
            # new baseline; threads harness patches) BEFORE advancing the counter.
            state, record, _next_baseline, next_harness_obj = derive_next_version(
                state=state, payload=payload, project_dir=project_dir,
                harness_inherit_fn=inherit,
                base_harness_object=current_base_harness)
            # Genuine harness inheritance: carry the concrete patched harness
            # object returned by derive_next_version into the next round (never a
            # phantom / recording-only tag). The inherit fn is invoked exactly
            # ONCE per round, inside derive_next_version -- we must not call it
            # again here.
            if record is not None and next_harness_obj is not None:
                current_base_harness = next_harness_obj
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


def _reverify_frozen_manifest(*, project_dir: str | Path,
                              baseline_config_path: str | Path,
                              dataset_version: str, domain: str) -> None:
    """Fail-closed re-check of the frozen dataset bound to the INHERITED baseline
    on loop RESUME (M2.7 P0-2).

    Reuses the exact on-disk verification the per-round evaluation performs, so a
    changed / corrupt dataset cannot silently continue a prior run's checkpoint.
    Raises :class:`LoopRecoveryError` on any mismatch (the caller's ``load_or_init``
    then archives the old checkpoint rather than reusing it).
    """
    from research_runtime.candidate_config import (
        CandidateConfigBuilder, CandidateConfigError,
    )
    from research_runtime.evolution.domain_evolver import domain_whitelist
    import yaml

    cfg_path = Path(baseline_config_path)
    # A checkpoint may still store a RELATIVE baseline path (older runs). Anchor
    # it to project_dir, exactly as version inheritance does (M2.6 F4), so resume
    # never silently resolves against the operator's CWD.
    if not cfg_path.is_absolute():
        cfg_path = Path(project_dir) / cfg_path
    if not cfg_path.is_file():
        raise LoopRecoveryError(
            f"resume re-verify: inherited baseline config missing: {cfg_path}")
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except (ValueError, OSError) as exc:
        raise LoopRecoveryError(
            f"resume re-verify: unreadable inherited baseline: {exc}") from exc
    try:
        allowed, rules = domain_whitelist(domain)
        builder = CandidateConfigBuilder(
            project_dir=Path(project_dir), allowed_keys=allowed,
            value_rules=rules, require_frozen_manifest=True)
        builder.verify_frozen_dataset(cfg, dataset_version)
    except (CandidateConfigError, ValueError, KeyError) as exc:
        raise LoopRecoveryError(
            f"resume re-verify: frozen dataset manifest no longer valid: {exc}"
        ) from exc
