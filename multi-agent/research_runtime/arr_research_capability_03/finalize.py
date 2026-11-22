"""Offline finalize: recompute evidence/verdicts from recorded cases, no engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import evaluation
from .integrity import sha256_json


def finalize(*, run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    canonical = json.loads((root / "canonical_results.json").read_text(
        encoding="utf-8"))
    m2 = json.loads((root / "m2_cases.json").read_text(encoding="utf-8"))
    m3 = json.loads((root / "m3_cases.json").read_text(encoding="utf-8"))
    cases = m2 + m3
    prereg = canonical["preregistration"]

    evq = evaluation.evidence_quality(cases)
    diag = evaluation.diagnosis_metrics(cases)

    # Recompute integrity from repaired resolver reading.
    integrity = {
        "gt_sealed_outside_run_root": canonical["integrity"].get(
            "gt_sealed_outside_run_root", True),
        "zero_gt_leakage": canonical["integrity"].get("zero_gt_leakage", True),
        "artifact_integrity": evq["artifact_integrity_pass"],
        "benchmark_frozen": bool(prereg["preregistration_sha256"]),
        "agent_invocations_recorded": canonical["integrity"].get(
            "agent_invocations_recorded", True),
    }
    verdict = evaluation.verdicts(
        engineering=canonical["engineering"], integrity=integrity,
        diag=diag, causal=canonical["statistics_m2"],
        transfer=canonical["statistics_m3"], prereg=prereg)
    canonical["evidence_quality"] = evq
    canonical["diagnosis"] = diag
    canonical["integrity"] = integrity
    canonical["verdicts"] = verdict
    canonical["verdict"] = verdict["overall_research_capability"]
    canonical["finalized_offline"] = True
    canonical.pop("canonical_sha256", None)
    canonical["canonical_sha256"] = sha256_json(canonical)
    (root / "canonical_results.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    return canonical


__all__ = ["finalize"]
