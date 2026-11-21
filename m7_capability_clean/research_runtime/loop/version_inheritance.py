"""ARR 1.1 / G2 -- version inheritance: the next round starts from the version
the previous round ACCEPTED.

Two layers are inherited, both REAL (no fabricated promotion):

* DOMAIN MODEL layer -- accepted domain candidates are whitelisted, dotted config
  overrides (e.g. ``train.lr``). They are BAKED into a new baseline config file
  via the SAME dotted setter the campaign builder uses, so the next round's
  experiments literally run on the improved config. This is the concrete
  "version inheritance" the acceptance criteria ask for.

* RESEARCH HARNESS layer -- accepted harness patches change the research system
  itself. Applying a patch needs the REAL harness object, so it is delegated to
  an injected ``harness_inherit_fn`` (the default uses
  ``adaptive_runtime.harness_evolution.apply_harness_patch`` when concrete
  ``HarnessCandidate`` objects are available). A loop that cannot supply real
  harness objects simply records the accepted targets and does NOT silently
  pretend to have promoted anything.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

import yaml

from research_runtime.candidate_config import _set_dotted


def bake_baseline(*, baseline_config_path: str | Path,
                  overrides: dict[str, Any],
                  out_path: str | Path) -> str:
    """Write a NEW baseline config that includes the accepted domain overrides.

    Reuses :func:`research_runtime.candidate_config._set_dotted` so the override
    is applied exactly as the campaign builder would (dotted key into a nested
    mapping, existence-checked). The original baseline is never mutated.
    Returns the SHA-256 of the new config so the loop can prove the inherited
    version is byte-accountable.
    """
    src = Path(baseline_config_path)
    if not src.is_file():
        raise FileNotFoundError(f"baseline config not found: {src}")
    try:
        base = yaml.safe_load(src.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"invalid baseline YAML: {exc}") from exc
    if not isinstance(base, dict):
        raise ValueError("baseline config must be a YAML mapping")
    baked = copy.deepcopy(base)
    for key, value in overrides.items():
        # existence-checked; an override key must already exist in the baseline.
        _set_dotted(baked, key, value)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(baked, sort_keys=False, allow_unicode=True)
    out.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: Real harness inheritance: (accepted_harness_records, current_base_harness)
#: -> next_base_harness. Records are EvolutionRecord dicts carrying a
#: ``candidate`` dict with target/old/new. The default applies them through the
#: existing harness patch primitive; a loop may override with a simpler or
#: domain-specific function.
HarnessInheritFn = Callable[[list[dict[str, Any]], Any], Any]


def default_harness_inherit(accepted_harness_records: list[dict[str, Any]],
                            current_base_harness: Any) -> Any:
    """Apply accepted harness patches through the existing real primitive.

    Reconstructs a minimal ``HarnessCandidate`` from each accepted record and
    calls ``apply_harness_patch`` so the next round's base harness is the SAME
    object the real research loop would execute. Raises loudly if a record is not
    reconstructable, so a loop never silently carries a phantom harness version.
    """
    from adaptive_runtime.harness_evolution import (
        HarnessCandidate, apply_harness_patch,
    )
    from adaptive_runtime.harness_evolution import KNOB_FIELD
    harness = current_base_harness
    sandbox_root = Path.cwd() / "_loop_harness_sandbox"
    repo_root = Path.cwd()
    for rec in accepted_harness_records:
        cand = (rec or {}).get("candidate") or {}
        target = cand.get("target")
        if target is None or target not in KNOB_FIELD:
            raise ValueError(
                f"cannot inherit harness patch for {target!r}: unknown knob")
        # Reconstruct only the fields apply_harness_patch needs.
        hc = HarnessCandidate(
            candidate_id=str(cand.get("candidate_id", "inherit")),
            target=target, old=cand.get("old"), new=cand.get("new"),
            taxonomy=str(cand.get("taxonomy", "")),
            patch=dict(cand.get("patch") or {}),
            hyp_fingerprint=str(cand.get("hyp_fingerprint", "")),
        )
        patched, _evidence = apply_harness_patch(
            harness, hc, sandbox_root=sandbox_root, repo_root=repo_root)
        harness = patched
    return harness
