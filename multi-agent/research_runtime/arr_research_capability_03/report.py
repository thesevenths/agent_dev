"""ARR-03 final report generator (from the single canonical results object)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_report(*, canonical: dict[str, Any], root: Path) -> Path:
    v = canonical["verdicts"]
    lines = [
        "# ARR_RESEARCH_CAPABILITY_03 — Final Report", "",
        "## Executive Verdict", "",
        f"**Overall Research Capability: {v['overall_research_capability']}**", "",
        "| Layer | Verdict |", "|---|---|",
    ]
    for k, val in v.items():
        lines.append(f"| {k} | {val} |")
    lines += [
        "", "## Campaign Identity", "",
        f"- Campaign ID: {canonical['campaign_id']}",
        "- Branch: feature/arr-research-capability-03",
        f"- Preregistration SHA-256: {canonical['preregistration']['preregistration_sha256']}",
        f"- Canonical results SHA-256: {canonical['canonical_sha256']}",
        f"- Ground truth seal: {canonical['ground_truth_seal']['ground_truth_sha256']}",
        "- Independent units: M2=10, M3=6; seeds are repeated measurements",
        "", "## Budget", "",
        f"- M2 model runs: {canonical['engineering']['m2_model_runs']}",
        f"- M3 model runs: {canonical['engineering']['m3_model_runs']}",
        f"- Total: {canonical['engineering']['total_model_runs']} (max 216)",
        "", "## M1 Diagnosis", "",
        "| Strategy | Accuracy | Top-2 | ECE | Brier | Abstention |",
        "|---|---|---|---|---|---|",
    ]
    for strat, d in canonical["diagnosis"].items():
        lines.append(
            f"| {strat} | {_fmt(d['accuracy'])} | {_fmt(d['top2_accuracy'])} | "
            f"{_fmt(d['ece'])} | {_fmt(d['brier'])} | "
            f"{_fmt(d['abstention_rate'])} |")
    lines += [
        "", "## M2 Causal Improvement (task-level units)", "",
        "| Contrast | Mean | 95% CI | Holm p | Significant |",
        "|---|---|---|---|---|",
    ]
    for name in ("candidate_vs_base", "candidate_vs_random", "oracle_vs_base"):
        s = canonical["statistics_m2"][name]
        lines.append(
            f"| {name} | {_fmt(s['mean'])} | "
            f"[{_fmt(s['ci95'][0])}, {_fmt(s['ci95'][1])}] | "
            f"{_fmt(s.get('holm_p', s['p']))} | {s.get('significant', False)} |")
    lines += [
        "", "## M3 Embodied Transfer (Reacher-v5)", "",
        "| Contrast | Mean | 95% CI | Holm p | Significant |",
        "|---|---|---|---|---|",
    ]
    for name in ("arr_transfer_vs_fresh", "arr_transfer_vs_random", "oracle_vs_base"):
        s = canonical["statistics_m3"][name]
        lines.append(
            f"| {name} | {_fmt(s['mean'])} | "
            f"[{_fmt(s['ci95'][0])}, {_fmt(s['ci95'][1])}] | "
            f"{_fmt(s.get('holm_p', s['p']))} | {s.get('significant', False)} |")
    nk = len(canonical["knowledge_items"])
    lines += [
        "", f"Verified knowledge items transferred: {nk} "
        f"(SHA-256 {canonical['knowledge_sha256'][:16]})",
        "", "## Evidence Integrity", "",
        json.dumps(canonical["evidence_quality"], indent=2, ensure_ascii=False),
        "", "## Limitations", "",
        "- State-based dynamics (SciNet++) and Reacher-v5 dynamics are simulator "
        "experiments; no visual JEPA, no real robot, no action-conditioned "
        "representation learning beyond a small MLP dynamics model.",
        "- Independent units are task regimes (10 + 6); seeds are repeated measurements.",
        "- Overall PASS requires M1, M2 and M3 each PASS; engineering PASS "
        "cannot compensate a scientific FAIL.",
        "",
    ]
    path = root / "ARR_RESEARCH_CAPABILITY_03_FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_report"]
