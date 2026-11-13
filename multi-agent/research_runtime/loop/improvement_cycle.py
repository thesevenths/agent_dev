"""ARR 1.1 / G3 -- turn one round's payload into the next round's version.

This is the IMPROVEMENT half of the loop: read what the dual-layer actually
ACCEPTED and convert it into the inherited state for the next round.

  * Accepted DOMAIN candidates (from ``domain_layer.records`` where
    ``decision == ACCEPTED``) become dotted config overrides that will be baked
    into the next baseline.
  * Accepted HARNESS candidates (from ``harness_layer.accepted``) are handed to
    the injected ``harness_inherit_fn`` to produce the next base harness.
  * A :class:`~research_runtime.loop.loop_state.VersionRecord` records exactly
    what was accepted and where the new baseline lives, so the loop is auditable.

Nothing here decides acceptance -- that is the dual-layer gate's job. This module
only inherits what was already, independently, accepted.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .loop_state import LoopState, VersionRecord
from .version_inheritance import bake_baseline


def derive_next_version(*, state: LoopState, payload: dict[str, Any],
                        project_dir: str | Path,
                        harness_inherit_fn: Callable[[list[dict[str, Any]], Any], Any]
                        | None = None,
                        base_harness_object: Any = None
                        ) -> tuple[LoopState, VersionRecord | None, str | None]:
    """Update ``state`` with the accepted version and return (state, record, next_baseline).

    Returns ``(state, None, None)`` when the round accepted nothing, so the
    controller can apply its no-gain stop policy without faking a version.
    """
    domain_layer = payload.get("domain_layer") or {}
    harness_layer = payload.get("harness_layer") or {}

    accepted_domain: list[dict[str, Any]] = [
        r for r in (domain_layer.get("records") or [])
        if r.get("decision") == "accepted"]
    accepted_harness: list[dict[str, Any]] = list(
        harness_layer.get("accepted") or [])

    if not accepted_domain and not accepted_harness:
        return state, None, None, None

    # Resolve the baseline config against project_dir: bake_baseline reads it
    # relative to CWD, so a relative --baseline-config (which the CLI stores
    # verbatim) must be made absolute here. This is the data-root isolation
    # principle (M2.5 P1) applied to version inheritance -- the next round must
    # inherit the SAME project root, never the operator's shell CWD.
    baseline_src = Path(state.baseline_config_path)
    if not baseline_src.is_absolute():
        baseline_src = Path(project_dir) / baseline_src

    # ---- accumulate domain overrides (union; later rounds stack on earlier) ----
    new_overrides: dict[str, Any] = {}
    for r in accepted_domain:
        new_overrides[r["target"]] = r["new"]

    # ---- harness inheritance (real, via injected fn) ----
    next_base_harness = state.base_harness_tag
    accepted_harness_targets: list[str] = []
    patched_object: Any = None
    if accepted_harness and harness_inherit_fn is not None:
        # Prefer the concrete in-memory harness object (the loop tracks it across
        # rounds); fall back to the persisted string tag for recording-only /
        # stub callers that never supplied a real harness.
        inherit_src = (base_harness_object
                       if base_harness_object is not None else state.base_harness_tag)
        patched = harness_inherit_fn(accepted_harness, inherit_src)
        patched_object = patched  # returned to the controller for in-memory threading
        # The inherit fn returns either the concrete patched harness object (real
        # harnesses) or a string tag (recording-only). Persist a stable string
        # identifier for recovery / reporting.
        next_base_harness = (patched.fingerprint() if hasattr(patched, "fingerprint")
                             else str(patched))
        accepted_harness_targets = [
            (rec.get("candidate") or {}).get("target", "") for rec in accepted_harness]

    # ---- bake a new baseline that carries the accepted domain overrides ----
    next_baseline: str | None = None
    if new_overrides:
        out = (Path(project_dir)
               / f"baseline_v{state.current_round + 1}.yaml")
        bake_baseline(baseline_config_path=str(baseline_src),
                      overrides=new_overrides, out_path=str(out))
        next_baseline = str(out)
        # The accumulated overrides persist across rounds so a later acceptance
        # composes on top of an earlier one.
        merged = dict(state.accepted_domain_overrides)
        merged.update(new_overrides)
        state.accepted_domain_overrides = merged
    else:
        # No domain change this round -> the next round keeps the same baseline
        # (absolute, project-root-anchored, so the next round resolves it too).
        next_baseline = str(baseline_src)

    if next_base_harness is not None:
        state.base_harness_tag = next_base_harness

    record = VersionRecord(
        round_no=state.current_round,
        baseline_config_path=next_baseline or str(baseline_src),
        accepted_domain_overrides=new_overrides,
        accepted_harness_targets=accepted_harness_targets,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        note=("inherited {d} domain override(s) and {h} harness target(s)".format(
            d=len(new_overrides), h=len(accepted_harness_targets))),
    )
    state.version_history.append(record)
    if next_baseline is not None:
        state.baseline_config_path = next_baseline
    return state, record, next_baseline, patched_object
