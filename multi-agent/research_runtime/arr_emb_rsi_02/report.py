"""ARR_EMB_RSI_02 report generator — uses ONE canonical results object."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(*, canonical: dict[str, Any], root: Path,
                 preflight: dict[str, Any] | None = None) -> Path:
    v = canonical["verdicts"]
    st = canonical["statistics"]
    dg = canonical["diagnosis"]
    ev = canonical["evidence_audit"]
    pr = canonical["preregistration"]
    lines = [
        "# ARR_EMB_RSI_02 — Final Report", "",
        "Experiment: `ARR_EMB_RSI_02` · Branch: "
        "`feature/arr-emb-rsi-02-blind-validation`", "",
        "## Executive verdict", "",
        f"Scientific claim: **{'SUPPORTED' if canonical.get('scientific_claim') else 'NOT SUPPORTED'}**",
        "", "| Layer | Verdict |", "|---|---|",
    ]
    for k, val in v.items():
        lines.append(f"| {k} | {val} |")
    lines += ["", "## Design", "",
              f"- Independent units: {pr['independent_units']} task regimes; "
              f"seeds {pr['seeds']} are repeated measurements.",
              f"- Arms: {', '.join(pr['arms'])}; model runs: {pr['budget']['model_runs']}.",
              f"- MME (normalised): {pr['mme_normalized']}; alpha: {pr['alpha']}.",
              f"- Power: n={pr['power_analysis']['n_units']}, "
              f"dz={pr['power_analysis']['dz']}, "
              f"power={pr['power_analysis']['computed_power']}.",
              f"- Excluded regimes: {pr['excluded_regimes']}.",
              "", "## Diagnosis benchmark", "",
              "| Controller | Accuracy | 95% CI | Mean conf | ECE | Brier |",
              "|---|---|---|---|---|---|"]
    for name, row in dg.items():
        lines.append(
            f"| {name} | {row['accuracy']:.3f} | "
            f"[{row['accuracy_ci95'][0]:.3f}, {row['accuracy_ci95'][1]:.3f}] | "
            f"{row['mean_confidence']:.3f} | {row['ece']:.3f} | "
            f"{row['brier']:.3f} |")
    lines += ["", "### Reliability diagram (agent)", "",
              "| bin | n | mean confidence | empirical accuracy |",
              "|---|---|---|---|"]
    for row in dg["agent"]["reliability"]:
        lines.append(f"| {row['bin']} | {row['n']} | "
                     f"{row['mean_confidence']:.3f} | {row['empirical']:.3f} |")
    lines += ["", "## Causal improvement", "",
              "| Contrast | Mean normalized Δ | 95% CI | p | Holm p | Bootstrap CI |",
              "|---|---|---|---|---|---|"]
    for name, row in st.items():
        lines.append(
            f"| {name} | {row['mean']:.4f} | "
            f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | "
            f"{row['p']:.4g} | {row.get('holm_p', row['p']):.4g} | "
            f"[{row['bootstrap_ci95'][0]:.4f}, {row['bootstrap_ci95'][1]:.4f}] |")
    lines += ["", "## Evidence audit", "",
              f"- Artifact integrity pass: {all(a['artifact_integrity'] for a in ev)}",
              f"- Mean citation validity: "
              f"{sum(a['citation_validity'] for a in ev) / max(1, len(ev)):.3f}",
              f"- Mean evidence relevance: "
              f"{sum(a['evidence_relevance'] for a in ev) / max(1, len(ev)):.3f}",
              f"- Sufficiency pass: {all(a['evidence_sufficiency'] for a in ev)}",
              "", "## Engineering / failures", "",
              f"- Cases: {canonical['engineering']['n_cases']}; "
              f"all arms ok: {canonical['engineering']['all_arms_ok']}",
              f"- Artifact integrity: {canonical['engineering']['artifact_integrity']}",
              f"- GT outside run root / zero leakage: "
              f"{canonical['integrity']['gt_outside_run_root']} / "
              f"{canonical['integrity']['zero_gt_leakage']}",
              "", "## Scope statement", "",
              "This campaign, even if PASS, validates only controlled "
              "state-based world-model research. It does NOT validate "
              "action-conditioned robotics, Visual JEPA, Reacher, MPC or "
              "recursive self-improvement.",
              "", "## Reproducibility", "",
              "```bash",
              "python -m research_runtime.arr_emb_rsi_02.setup_preflight",
              "python -m research_runtime.arr_emb_rsi_02.cli --stage campaign "
              "--out ARR_EMB_RSI_02_FULL",
              "```", "",
              f"Canonical artifact SHA-256: `{canonical['artifact_sha256']}`",
              f"Preregistration SHA-256: `{pr['preregistration_sha256']}`", ""]
    if preflight:
        lines += ["## Preflight summary", "",
                  f"- Actual agent invocation: {preflight.get('agent_ok')}",
                  f"- Generator compatibility: {preflight.get('generators')}",
                  f"- Oracle-sensitive regimes: {preflight.get('n_valid_regimes')}",
                  ""]
    path = root / "ARR_EMB_RSI_02_FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["write_report"]
