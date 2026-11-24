"""ARR-07 final report generator: all values from canonical data."""
from __future__ import annotations

import json
from pathlib import Path


def generate(root: Path, c: dict) -> dict:
    root = Path(root)
    v1, v2, v3 = c["validations"]["V1"], c["validations"]["V2"], c["validations"]["V3"]
    lines = [
        "# ARR-07 Final Report — Autonomous Research Loop Validation", "",
        "## Executive Summary", "",
        f"**Overall: {c['overall']}** · Terminal: COMPLETED_WITH_FAILURES", "",
        "| Validation | Verdict | Key data |", "|---|---|---|",
        f"| V1 closed-loop convergence | {c['verdicts']['closed_loop_convergence']} | "
        f"{v1['n_converged']}/{v1['n_tasks']} converged, "
        f"{v1['n_spurious_convergence']} spurious |",
        f"| V2 blind vs leaky | {c['verdicts']['no_config_leakage']} | "
        f"blind {v2['blind_converged']}/{v2['n_tasks']} vs "
        f"leaky {v2['leaky_converged']}/{v2['n_tasks']} |",
        f"| V3 memory gate repair | {c['verdicts']['memory_gate_repair']} | "
        f"abstain instead of placebo |", "",
        "## Architecture", "",
        "LLM does COGNITION only (observe -> hypothesise -> choose experiment -> "
        "replan from measured data). A FROZEN evaluator judges goal attainment "
        "from measured improvement AND mechanism correctness; the LLM never "
        "self-certifies success. Blind mode exposes ONLY experimental measurements.", "",
        "## The decisive control (V2)", "",
        f"Blind {v2['blind_rate']:.0%} vs leaky {v2['leaky_rate']:.0%}; gap "
        f"{v2['leaky_rate']-v2['blind_rate']:.0%}. The loop works mainly from "
        "experimental data, NOT from config leakage — but reliability is still "
        "below the bar and the verdict is therefore PARTIAL, not PASS.", "",
        "## Spurious convergence (V1)", "",
        f"{v1['n_spurious_convergence']} of {v1['n_converged']} converged runs "
        "reached the improvement target with the WRONG mechanism. Generic "
        "improvements (e.g. more data) can masquerade as scientific discovery. "
        "The frozen evaluator now catches this by requiring mechanism consistency.", "",
        "## Honest conclusion", "",
        "The loop RUNS end-to-end with real LLM planning, real experiments, and "
        "data-driven replanning. It is NOT yet a reliable autonomous researcher: "
        "convergence is partial, spurious convergence is frequent, and the "
        "hardest mechanisms (distribution shift) are not solved. "
        "NOT VALIDATED — do not upgrade to a research-system claim.", "",
        f"- Canonical SHA-256: {c['canonical_sha256']}", "",
    ]
    p = root / "final" / "ARR07_FINAL_REPORT.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return {"report": p}


__all__ = ["generate"]
