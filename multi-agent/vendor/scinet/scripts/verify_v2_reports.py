"""
scripts/verify_v2_reports.py
============================
Strict number-traceability verification for the FOUR P1-A'/P1-B1 reports.

Same strictness as ``verify_p1a_reports.py`` v2:
  * every case binds ONE source JSON, ONE JSONPath, ONE target report, ONE
    line anchor, ONE literal and ONE explicit numeric tolerance;
  * ALL values are numerically compared (no default-True);
  * scientific notation / percent / Markdown bold are parsed;
  * Git-derived counts come from ``git_integrity.json`` (real git commands).

Also exposes ``verify(...)`` for negative tests.

Usage:
  python scripts/verify_v2_reports.py
  -> datasets/controlled_pendulum_v2_pilot/analysis/number_traceability.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_p1a_reports import (  # noqa: E402
    resolve_path, parse_literal, as_float, _NUMBER_RE,
)

DOC = ROOT / "docs" / "architecture"
V2 = ROOT / "datasets" / "controlled_pendulum_v2_pilot"
V1A = ROOT / "datasets" / "controlled_pendulum_pilot" / "analysis"

REPORTS = [
    "P1_A_REVIEW_FIX_REPORT.md",
    "P1_B1_V2_PILOT_REPORT.md",
    "P1_B1_V2_SCIENTIFIC_REVIEW.md",
    "P1_B1_V2_READINESS_REPORT.md",
]
EVIDENCE = {
    "audit_report.json": V2 / "audit_report.json",
    "manifest.json": V2 / "manifest.json",
    "trajectory_distribution.json": V2 / "analysis/trajectory_distribution.json",
    "action_information.json": V2 / "analysis/action_information.json",
    "git_integrity.json": V1A / "git_integrity.json",
    # Independent source for the v1-report verifier result; avoids a circular
    # dependency on audit_report.json (which this verifier's caller writes).
    "number_traceability_v2.json": V1A / "number_traceability_v2.json",
}


def _build_cases() -> list[Dict[str, Any]]:
    C = []

    def add(label, source, path, report, anchor, literal, tol, scale=1.0,
            token_index=-1):
        C.append({"label": label, "source": source, "jsonpath": path,
                  "target_report": report, "anchor": anchor,
                  "report_literal": literal, "tolerance": tol,
                  "scale": scale, "token_index": token_index})

    R1 = "P1_A_REVIEW_FIX_REPORT.md"
    R2 = "P1_B1_V2_PILOT_REPORT.md"
    R3 = "P1_B1_V2_SCIENTIFIC_REVIEW.md"
    R4 = "P1_B1_V2_READINESS_REPORT.md"
    A = "audit_report.json"
    M = "manifest.json"
    T = "trajectory_distribution.json"
    I = "action_information.json"
    G = "git_integrity.json"

    # --- git integrity -> fix report ---
    add("ahead_commits", G, "ahead", R1, "rev-list --left-right", "**11**", 0)
    add("behind_commits", G, "behind", R1, "rev-list --left-right", "**0**", 0)
    add("changed_files", G, "changed_files_vs_main", R1,
        "相对 main 文件变更", "**36**", 0)
    add("p2_02_evidence_files", G, "p2_02_evidence_file_count", R1,
        "P2-02 证据文件", "78", 0)

    # The two theta0=2 analytic accelerations are handled by _ANALYTIC_CASES
    # below: they are recomputed from the equation, not read from a JSON file.

    # --- v2 audit -> pilot/scientific/readiness reports ---
    add("v2_dop853_max_error", A,
        "checks[1].detail.max_abs_state_error_over_all", R3,
        "独立 DOP853 最大状态误差", "1.6515e-05", 1e-8)
    add("v2_energy_balance_residual", A,
        "checks[2].detail.max_rel_residual", R3,
        "能量平衡最大相对残差", "7.4512e-04", 1e-8)
    add("v2_max_delta_theta", A,
        "checks[4].detail.max_abs_delta_theta_per_step", R3,
        "max \\|Δθ\\| / step", "0.1390 rad", 5e-5)
    add("v2_distinct_theta0", A, "checks[6].detail.distinct_theta0", R3,
        "不同 θ0", "**20**", 0)
    add("v2_theta0_std", A, "checks[6].detail.theta0_std", R3,
        "std 2.1699", "2.1699", 5e-5)
    add("v2_audit_n_pass", A, "n_pass", R2,
        "10/10 PASS", "10/10", 0)
    add("v2_stress_err_dt005", A,
        "stress_test_q_c.max_abs_error_dt_0005", R2,
        "Q-C stress", "1.095e-05", 1e-6)
    add("v2_stress_err_dt02", A,
        "stress_test_q_c.max_abs_error_dt_002", R2,
        "v1 原始", "7.628e-03", 1e-6)
    add("v2_stress_reduction", A,
        "stress_test_q_c.error_reduction", R2,
        "误差降低", "696.8", 0.05)
    add("v2_stress_duration", A,
        "stress_test_q_c.physical_duration_s", R2,
        "物理时长", "4.0", 0)

    # --- v2 distribution -> scientific / pilot reports ---
    add("v2_theta_global_max", T, "v1_comparison.v2_theta_global_max", R3,
        "θ 全局 max", "8.819", 5e-4)
    add("v2_theta_global_std", T, "v1_comparison.v2_theta_std", R3,
        "θ 全局 std", "2.213", 5e-4)
    add("v2_top1_share_pct", T, "v1_comparison.v2_top1_share", R3,
        "单条 episode 占 \\|θ\\|", "24.0%", 0.05, scale=100.0)
    add("v1_theta_global_max", T, "v1_comparison.v1_theta_global_max", R3,
        "θ 全局 max", "82.115", 5e-4)
    add("v1_top1_share_pct", T, "v1_comparison.v1_top1_share", R3,
        "单条 episode 占 \\|θ\\|", "53.5%", 0.05, scale=100.0)
    add("v2_theta_max_global", T, "global_distribution.theta_rad.max", R3,
        "| θ (rad) |", "8.8188", 5e-5)
    add("v2_torque_min", T, "global_distribution.torque_Nm.min", R2,
        "力矩 |", "-0.9981", 5e-5)

    # --- action information (theta0=0 corrected) -> scientific report ---
    add("alpha_positive", I,
        "arms.positive_constant.initial_alpha_rad_s2", R3,
        "positive_constant (+1)", "+1.179313", 5e-7)
    add("alpha_negative", I,
        "arms.negative_constant.initial_alpha_rad_s2", R3,
        "negative_constant (-1)", "-1.179313", 5e-7)
    add("alpha_zero", I,
        "arms.zero.initial_alpha_rad_s2", R3,
        "| zero |", "0.000000", 0)
    add("analytic_alpha_positive", I, "analytic_initial_alpha.positive", R3,
        "解析值", "1.179313", 5e-7)

    # --- manifest identity -> pilot report ---
    add("v2_T", M, "T", R2, "`episode_steps`", "1000", 0)
    add("v2_n_episodes", M, "n_episodes", R2, "observations",
        "[20, 1001, 2]", 0, token_index=0)
    add("v2_split_train", M, "split_counts.train", R2,
        "episode 级划分隔离", "14/3/3", 0, token_index=0)
    add("v2_split_val", M, "split_counts.validation", R2,
        "episode 级划分隔离", "14/3/3", 0, token_index=1)

    # --- regression / verifier counts -> readiness report ---
    add("traceability_v2_pass", "number_traceability_v2.json", "n_pass", R4,
        "数字可追溯性 v2", "48/48", 0)
    return C


# Constant-source analytic cases (values come from the analytic unit test, not
# from a JSON evidence file). They are checked against the report literally AND
# against an independently recomputed analytic value.
_ANALYTIC_CASES = [
    ("theta2_alpha_pos", "P1_A_REVIEW_FIX_REPORT.md", "+2.5", "-6.420208",
     -9.81 * __import__("math").sin(2.0) + 2.5),
    ("theta2_alpha_neg", "P1_A_REVIEW_FIX_REPORT.md", "-2.5", "-11.420208",
     -9.81 * __import__("math").sin(2.0) - 2.5),
]


def verify(report_texts: Dict[str, str],
           evidence: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for c in _build_cases():
        row = {k: c[k] for k in ("label", "source", "jsonpath",
                                 "target_report", "anchor", "report_literal",
                                 "tolerance", "scale", "token_index")}
        try:
            src = as_float(resolve_path(evidence[c["source"]], c["jsonpath"]))
            row["source_value"] = src
            lit = c["report_literal"]
            text = report_texts.get(c["target_report"]) or ""
            lines = [ln for ln in text.splitlines() if lit in ln]
            anchored = [ln for ln in lines if c["anchor"] in ln]
            row["literal_present_in_target_report"] = len(anchored) >= 1
            row["occurrences_on_anchor_line"] = len(anchored)
            toks = _NUMBER_RE.findall(lit)
            idx = c.get("token_index", -1)
            parsed = (parse_literal(toks[idx]) if toks
                      and -len(toks) <= idx < len(toks) else None)
            row["parsed_report_value"] = parsed
            expected = src * c["scale"] if c["scale"] != 0 else parsed
            numeric_ok = (parsed is not None
                          and abs(parsed - expected) <= c["tolerance"])
            row["numeric_match"] = numeric_ok
            row["numeric_delta"] = (None if parsed is None
                                    else abs(parsed - expected))
            row["passed"] = bool(row["literal_present_in_target_report"]
                                 and numeric_ok)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            row.update(source_value=None, parsed_report_value=None,
                       literal_present_in_target_report=False,
                       numeric_match=False, numeric_delta=None,
                       error=repr(exc), passed=False)
        rows.append(row)

    # analytic constant cases
    for label, report, anchor, lit, value in _ANALYTIC_CASES:
        text = report_texts.get(report) or ""
        lines = [ln for ln in text.splitlines() if lit in ln]
        anchored = [ln for ln in lines if anchor in ln]
        parsed = parse_literal(lit)
        rows.append({
            "label": label, "source": "analytic_recomputation",
            "jsonpath": "-9.81*sin(2)+tau", "target_report": report,
            "anchor": anchor, "report_literal": lit,
            "tolerance": 5e-7, "scale": 1.0,
            "source_value": value,
            "parsed_report_value": parsed,
            "literal_present_in_target_report": len(anchored) >= 1,
            "occurrences_on_anchor_line": len(anchored),
            "numeric_match": abs(parsed - value) <= 5e-7,
            "numeric_delta": abs(parsed - value),
            "passed": bool(anchored) and abs(parsed - value) <= 5e-7,
        })

    n_pass = sum(1 for r in rows if r["passed"])
    return {
        "verifier_version": "v2-strict",
        "verdict": "PASS" if n_pass == len(rows) else "FAIL",
        "n_pass": n_pass, "n_total": len(rows),
        "strictness": {
            "all_cases_numerically_compared": True,
            "default_numeric_match_true": False,
            "per_report_binding": True,
            "per_line_anchor_binding": True,
        },
        "rows": rows,
    }


def load_inputs() -> tuple[Dict[str, str], Dict[str, Any]]:
    reports = {n: (DOC / n).read_text(encoding="utf-8") for n in REPORTS}
    evidence = {n: json.loads(p.read_text(encoding="utf-8"))
                for n, p in EVIDENCE.items()}
    return reports, evidence


def main() -> int:
    reports, evidence = load_inputs()
    result = verify(reports, evidence)
    out = V2 / "analysis" / "number_traceability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"v2 report number traceability: {result['verdict']} "
          f"({result['n_pass']}/{result['n_total']})")
    for r in result["rows"]:
        if not r["passed"]:
            print(f"  FAIL {r['label']}: src={r.get('source_value')} "
                  f"lit={r['report_literal']!r} in {r['target_report']} "
                  f"present={r.get('literal_present_in_target_report')} "
                  f"numeric={r.get('numeric_match')} "
                  f"delta={r.get('numeric_delta')} err={r.get('error')}")
    print(f"report: {out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
