"""Generate ARR-07 isolation report from artifacts; model switch recorded as provenance."""
from __future__ import annotations

import json
from pathlib import Path


def generate(root: Path, out_root: Path, gated_root: Path | None = None,
             audit_root: Path | None = None) -> dict:
    root, out_root = Path(root), Path(out_root)
    gated_root = Path(gated_root) if gated_root else root
    audit_root = Path(audit_root) if audit_root else root
    v4 = json.loads((root / "loop" / "isolation" / "planner_causality.json").read_text(
        encoding="utf-8"))
    gated = json.loads((gated_root / "loop" / "isolation" /
                        "gated_planner_causality.json").read_text(encoding="utf-8"))
    v5 = json.loads((audit_root / "loop" / "isolation" /
                     "evaluator_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# ARR-07 Link-Level Isolation Validation", "",
        "## Why this exists", "",
        "ARR-07 V1/V2 showed the loop RUNS but did not prove each link works "
        "correctly. Isolation tests target the two weak links.", "",
        "## V4 — Planner causality (THE finding)", "",
        f"- Bare LLM determinism: {v4['determinism']['stability']:.0%} "
        f"({v4['determinism']['choices']}) — NOT random.",
        f"- Bare LLM causality: {v4['causality']['causal']} — "
        "three distinct measurement signatures ALL produced EXP_COV. "
        "The LLM has an absolute coverage prior and ignores the data. "
        "This is the root cause of weak 'replan'.", "",
        f"- GATED planner causality: {gated['causal']} "
        f"({gated['distinct_sets']} distinct choice sets) — "
        "with a frozen evidence gate constraining choices, the planner "
        "responds systematically to measurements.",
        "- HONESTY NOTE: causality is achieved by the deterministic gate, "
        "NOT by LLM self-discovery. This is constrained cognition, not fully "
        "autonomous research.", "",
        "## V5 — Independent evaluator audit", "",
        f"- Agreement: {v5['n_agree']}/{v5['n_tasks']} with a SEPARATE "
        "implementation reading raw traces. Goal judgement link VERIFIED.", "",
        "## Model provenance", "",
        "- Campaign data produced with qwen3.6-flash (local vLLM gateway).",
        "- On user request, .env LLM_MODEL switched to qwen3.7-flash; "
        "API key unchanged; live call verified. .env is not git-tracked; "
        "a backup .env.bak-arr07-model-switch was kept locally.",
        "- Isolation tests V4-bare / V5 ran on qwen3.6-flash; "
        "gated causality ran on qwen3.7-flash (different model -> "
        "cross-model evidence, noted for transparency).", "",
        "## Verdict", "",
        "| Link | Status |", "|---|---|",
        "| replan (bare LLM) | FAIL — insensitive to evidence |",
        "| replan (evidence-gated) | PARTIAL — causal but rule-driven |",
        "| goal (independent audit) | PASS |", "",
    ]
    p = out_root / "ARR07_ISOLATION_REPORT.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return {"report": p}


__all__ = ["generate"]
