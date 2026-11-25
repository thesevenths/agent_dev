"""
scripts/verify_p1a_reports.py  (v2 -- P1-A' review fix P2)
==========================================================
Number-traceability verification for the four P1-A review reports.

The v1 verifier had three defects (found by an independent GitHub review):
  1. ``decimals=None`` cases set ``numeric_ok = True`` unconditionally, so
     scientific-notation and integer values were NEVER numerically compared;
  2. the P2-02 evidence count ``78`` was hard-coded;
  3. all four reports were concatenated into ONE string, so a value appearing
     in the WRONG report still passed.

This v2 fixes all three:
  * Every case binds ONE source JSON, ONE JSONPath, and ONE target report.
  * Every case performs an explicit numeric comparison with an explicit
     tolerance (no default-True); scientific notation, percentages and
     Markdown bold are parsed.
  * The Git-derived count is read from ``analysis/git_integrity.json``
     (captured by ``scripts/capture_git_integrity.py`` via real git commands).
  * ``verify(report_texts, evidence)`` is injectable so unit tests can feed a
     deliberately corrupted report and assert the verifier FAILS.

Historical note: the previous ``analysis/number_traceability.json`` (28/28)
is preserved as a historical artifact. It proves literal presence for all 28
values and numeric agreement for 19 of them; it must NOT be cited as proof of
full numeric traceability.

Usage:
  python scripts/verify_p1a_reports.py
  -> datasets/controlled_pendulum_pilot/analysis/number_traceability_v2.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "datasets" / "controlled_pendulum_pilot" / "analysis"
DOC = ROOT / "docs" / "architecture"

REPORTS = [
    "P1_A_BRANCH_INTEGRITY_REPORT.md",
    "P1_A_SCIENTIFIC_REVIEW.md",
    "P1_A_TRAJECTORY_DISTRIBUTION.md",
    "P2_03A_READINESS_PLAN.md",
]
EVIDENCE_FILES = [
    "independent_verification.json",
    "step_convergence.json",
    "trajectory_distribution.json",
    "action_information.json",
    "split_proposal.json",
    "git_integrity.json",
]

# Number token: scientific notation, plain decimals, signed, optional percent.
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)
_PATH_TOKEN = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")


# --------------------------------------------------------------------------- #
# JSONPath (dot + [i] indices) resolution
# --------------------------------------------------------------------------- #
def resolve_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        m = _PATH_TOKEN.match(part)
        if not m:
            raise KeyError(f"bad JSONPath segment: {part!r}")
        key, idx = m.group(1), m.group(2)
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"JSONPath {path!r}: missing key {key!r}")
        cur = cur[key]
        if idx is not None:
            cur = cur[int(idx)]
    return cur


def parse_literal(s: str) -> float:
    """Parse a number token that may contain Markdown bold / percent / sci notation."""
    t = s.strip().replace("**", "").replace("`", "").strip()
    is_pct = t.endswith("%")
    if is_pct:
        t = t[:-1]
    return float(t)


def as_float(v: Any) -> float:
    if isinstance(v, bool):
        raise TypeError("boolean is not a numeric evidence value")
    return float(v)


# --------------------------------------------------------------------------- #
# Case specification. EVERY case MUST have source/path/report/anchor/literal/tol.
# tol is an ABSOLUTE tolerance on (parsed_literal - scale*source_value).
# anchor is a unique substring of the LINE where the value must appear; this
# prevents a value appearing elsewhere in the same report from satisfying a
# different case (the "wrong location" loophole found by negative tests).
# --------------------------------------------------------------------------- #
_ANCHORS: Dict[str, str] = {
    "scipy_dop853_max_state_error": "全局最大",
    "energy_balance_max_rel_residual": "全局最大",
    "distinct_initial_conditions": "n_distinct_initial_conditions",
    "torque_global_min": "全局 τ 范围",
    "torque_global_max": "全局 τ 范围",
    "torque_global_std": "全局 mean / std",
    "torque_out_of_bounds": "越界样本数",
    "saturation_fraction_99pct": "99% 饱和比例",
    "short_rod_I": "| short_rod | 9.81 | **0.3**",
    "short_rod_tau_over_I": "| short_rod | 9.81 | **0.3**",
    "short_rod_E_barrier": "| short_rod | 9.81 | **0.3**",
    "short_rod_E0": "| short_rod | 9.81 | **0.3**",
    "nominal_E_barrier": "| nominal | 9.81",
    "step_convergence_dt_div_1": "| 0.02000 |",
    "step_convergence_dt_div_2": "| 0.01000 |",
    "step_convergence_dt_div_4": "| 0.00500 |",
    "step_convergence_dt_div_8": "| 0.00250 |",
    "step_convergence_dt_div_16": "| 0.00125 |",
    "pos_vs_neg_final_theta_diff": "vs negative_constant",
    "pos_vs_zero_final_theta_diff": "vs zero",
    "pos_vs_shuffled_final_theta_diff": "vs **shuffled_random**",
    "n_distinct_state_final_thetas": "同一动作、不同初态",
    "theta_global_max": "| θ (rad, unwrapped) |",
    "theta_global_std": "| θ (rad, unwrapped) |",
    "theta_global_p50": "| θ (rad, unwrapped) |",
    "omega_global_max": "| ω (rad/s) |",
    "abs_delta_theta_max": "| Δθ / step (rad) |",
    "top1_episode_share_of_abs_theta_pct": "单条最大 episode 占全部",
    "n_episodes_gt_2pi": "贡献了全部",
    "regime_bounded": "| bounded_oscillation |",
    "regime_transitional": "| transitional |",
    "regime_rotation": "| continuous_rotation |",
    "S1_step_theta_final": "| **S1_step** | **rotation** |",
    "S1_step_turns": "| **S1_step** | **rotation** |",
    "S1_step_winding": "| **S1_step** | **rotation** |",
    "S1_step_fraction_above_barrier": "| **S1_step** | **rotation** |",
    "S1_sinusoid_theta_final": "| **S1_sinusoid** |",
    "S4_step_theta_final": "| **S4_step** |",
    "split_train": "| train | **14** |",
    "split_validation": "| validation | **3** |",
    "split_test": "| test | **3** |",
    "windows_train": "| train | 14 | **546** |",
    "windows_validation": "| validation | 3 | **117** |",
    "windows_total": "| 全量 | 20 | 780 |",
    "p2_02_evidence_file_count": "P2-02 证据文件数",
    "ahead_commits_current_head": "rev-list --left-right --count",
    "behind_commits_current_head": "rev-list --left-right --count",
    "changed_files_current_head": "相对 main 文件变更",
}


def _build_cases() -> list[Dict[str, Any]]:
    C = []

    def add(label, source, path, report, literal, tol, scale=1.0):
        if label not in _ANCHORS:
            raise KeyError(f"case {label!r} has no line anchor")
        C.append({
            "label": label, "source": source, "jsonpath": path,
            "target_report": report, "report_literal": literal,
            "tolerance": tol, "scale": scale, "anchor": _ANCHORS[label],
        })

    R = "P1_A_SCIENTIFIC_REVIEW.md"
    V = "independent_verification.json"
    add("scipy_dop853_max_state_error", V,
        "independent_ode_crosscheck_scipy_dop853.max_abs_state_error_over_all_episodes",
        R, "7.628e-03", 1e-6)
    add("energy_balance_max_rel_residual", V,
        "energy_balance_identity.max_rel_residual_over_all_episodes",
        R, "1.163e-03", 1e-6)
    add("distinct_initial_conditions", V,
        "initial_condition_distribution.n_distinct_initial_conditions",
        R, "= 1", 0)
    add("torque_global_min", V, "action_distribution.global_min",
        R, "-4.9863", 5e-5)
    add("torque_global_max", V, "action_distribution.global_max",
        R, "+4.9999", 5e-5)
    add("torque_global_std", V, "action_distribution.global_std",
        R, "2.714", 5e-4)
    add("torque_out_of_bounds", V, "action_distribution.n_out_of_bounds",
        R, "**0**", 0)
    add("saturation_fraction_99pct", V,
        "action_distribution.saturation_fraction_at_99pct",
        R, "0.00225", 5e-6)
    add("short_rod_I", V, "physical_parameter_units.table[1].I_kg_m2",
        R, "0.090", 5e-4)
    add("short_rod_tau_over_I", V,
        "physical_parameter_units.table[1].tau_max_over_I_rad_s2",
        R, "55.56", 5e-3)
    add("short_rod_E_barrier", V,
        "physical_parameter_units.table[1].E_barrier_J",
        R, "5.89", 5e-3)
    add("short_rod_E0", V, "physical_parameter_units.table[1].E0_J",
        R, "4.17", 5e-3)
    add("nominal_E_barrier", V,
        "physical_parameter_units.table[0].E_barrier_J",
        R, "19.62", 5e-3)

    K = "step_convergence.json"
    for i, (fac, lit, tol) in enumerate([
        (1, "7.628e-03", 1e-6), (2, "2.804e-04", 1e-7),
        (4, "1.095e-05", 1e-8), (8, "4.744e-07", 1e-9),
        (16, "2.287e-08", 1e-10),
    ]):
        add(f"step_convergence_dt_div_{fac}", K,
            f"rows[{i}].max_abs_error_vs_scipy", R, lit, tol)

    R2 = "P1_A_TRAJECTORY_DISTRIBUTION.md"
    D = "trajectory_distribution.json"
    add("theta_global_max", D, "global_distribution.theta_rad.max",
        R2, "82.115", 5e-4)
    add("theta_global_std", D, "global_distribution.theta_rad.std",
        R2, "12.685", 5e-4)
    add("theta_global_p50", D, "global_distribution.theta_rad.p50",
        R2, "0.173", 5e-4)
    add("omega_global_max", D, "global_distribution.omega_rad_s.max",
        R2, "51.264", 5e-4)
    add("abs_delta_theta_max", D,
        "global_distribution.abs_delta_theta_rad_per_step.max",
        R2, "1.0255", 5e-5)
    add("top1_episode_share_of_abs_theta_pct", D,
        "extreme_dominance.share_of_abs_theta_from_top1_episode",
        R2, "53.5%", 0.05, scale=100.0)
    add("n_episodes_gt_2pi", D,
        "extreme_dominance.n_episodes_with_abs_theta_gt_2pi",
        R2, "3 条", 0)
    add("regime_bounded", D, "regime_counts.bounded_oscillation",
        R2, "**16**", 0)
    add("regime_transitional", D, "regime_counts.transitional",
        R2, "**3**", 0)
    add("regime_rotation", D, "regime_counts.continuous_rotation",
        R2, "**1**", 0)
    # per_episode: index 5 = S1_step, 6 = S1_sinusoid, 17 = S4_step
    add("S1_step_theta_final", D, "per_episode[5].theta_final_rad",
        R2, "79.11", 5e-3)
    add("S1_step_turns", D, "per_episode[5].theta_final_turns",
        R2, "12.59", 5e-3)
    add("S1_step_winding", D, "per_episode[5].winding_number",
        R2, "12.27", 5e-3)
    add("S1_step_fraction_above_barrier", D,
        "per_episode[5].fraction_timesteps_above_barrier", R2, "0.72", 5e-3)
    add("S1_sinusoid_theta_final", D, "per_episode[6].theta_final_rad",
        R2, "-34.19", 5e-3)
    add("S4_step_theta_final", D, "per_episode[17].theta_final_rad",
        R2, "8.05", 5e-3)

    A = "action_information.json"
    add("pos_vs_neg_final_theta_diff", A,
        "pairwise_final_theta_abs_differences.positive_vs_negative_constant_final_theta_abs_diff",
        R, "22.743", 5e-4)
    add("pos_vs_zero_final_theta_diff", A,
        "pairwise_final_theta_abs_differences.positive_vs_zero_final_theta_abs_diff",
        R, "1.021", 5e-4)
    add("pos_vs_shuffled_final_theta_diff", A,
        "pairwise_final_theta_abs_differences.positive_vs_shuffled_random_final_theta_abs_diff",
        R, "0.794", 5e-4)
    add("n_distinct_state_final_thetas", A,
        "same_action_different_initial_state.n_distinct_final_thetas",
        R, "4 个互不相同", 0)

    S = "split_proposal.json"
    add("split_train", S, "split_counts.train", R2, "**14**", 0)
    add("split_validation", S, "split_counts.validation", R2, "**3**", 0)
    add("split_test", S, "split_counts.test", R2, "**3**", 0)
    add("windows_train", S,
        "window_isolation_demo.per_split.train.n_windows", R2, "**546**", 0)
    add("windows_validation", S,
        "window_isolation_demo.per_split.validation.n_windows", R2, "**117**", 0)
    add("windows_total", S, "window_isolation_demo.n_windows",
        R2, "780", 0)

    # git_integrity.json -> branch integrity report (current-HEAD section)
    R0 = "P1_A_BRANCH_INTEGRITY_REPORT.md"
    G = "git_integrity.json"
    add("p2_02_evidence_file_count", G, "p2_02_evidence_file_count",
        R0, "**78**", 0)
    add("ahead_commits_current_head", G, "ahead", R0, "**11**", 0)
    add("behind_commits_current_head", G, "behind", R0, "**0**", 0)
    add("changed_files_current_head", G, "changed_files_vs_main",
        R0, "**36**", 0)
    return C


# --------------------------------------------------------------------------- #
# Core verification (injectable for negative tests)
# --------------------------------------------------------------------------- #
def verify(report_texts: Dict[str, str],
           evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Run all cases. report_texts maps report filename -> full markdown text."""
    rows = []
    for case in _build_cases():
        label, lit = case["label"], case["report_literal"]
        report = case["target_report"]
        text = report_texts.get(report)
        row: Dict[str, Any] = {
            "label": label,
            "source": case["source"],
            "jsonpath": case["jsonpath"],
            "target_report": report,
            "report_literal": lit,
            "tolerance": case["tolerance"],
            "scale": case["scale"],
            "anchor": case["anchor"],
        }
        try:
            src = resolve_path(evidence[case["source"]], case["jsonpath"])
            source_value = as_float(src)
            row["source_value"] = source_value

            # (a) literal MUST appear in the SPECIFIC target report AND on the
            #     line identified by the anchor (not just anywhere in the file).
            lines = [ln for ln in (text or "").splitlines() if lit in ln]
            anchored = [ln for ln in lines if case["anchor"] in ln]
            n_lines = len(lines)
            present = len(anchored) >= 1
            row["literal_present_in_target_report"] = present
            row["occurrences_in_target_report"] = n_lines
            row["occurrences_on_anchor_line"] = len(anchored)

            # (b) an explicit numeric comparison ALWAYS runs (no default-True)
            toks = _NUMBER_RE.findall(lit)
            nums = [parse_literal(t) for t in toks]
            parsed = nums[-1] if nums else None
            row["parsed_report_value"] = parsed
            expected = source_value * case["scale"]
            if parsed is None:
                numeric_ok = False
            else:
                numeric_ok = abs(parsed - expected) <= case["tolerance"]
            row["numeric_match"] = numeric_ok
            row["numeric_delta"] = (None if parsed is None
                                    else abs(parsed - expected))
            row["passed"] = bool(present and numeric_ok)
        except (KeyError, TypeError, ValueError, IndexError, UnboundLocalError) as exc:
            row["source_value"] = None
            row["parsed_report_value"] = None
            row["literal_present_in_target_report"] = False
            row["numeric_match"] = False
            row["numeric_delta"] = None
            row["error"] = repr(exc)
            row["passed"] = False
        rows.append(row)

    n_pass = sum(1 for r in rows if r["passed"])
    verdict = "PASS" if n_pass == len(rows) else "FAIL"
    return {
        "verifier_version": "v2",
        "verdict": verdict,
        "n_pass": n_pass,
        "n_total": len(rows),
        "strictness": {
            "all_cases_numerically_compared": True,
            "default_numeric_match_true": False,
            "per_report_binding": True,
            "per_line_anchor_binding": True,
            "git_counts_from": "analysis/git_integrity.json (real git commands)",
            "historical_v1_28_28": "preserved as historical artifact; NOT proof "
                                   "of full numeric traceability (9/28 skipped "
                                   "numeric comparison, 78 was hard-coded, "
                                   "reports were concatenated)",
        },
        "rows": rows,
    }


def load_inputs() -> tuple[Dict[str, str], Dict[str, Any]]:
    report_texts = {n: (DOC / n).read_text(encoding="utf-8") for n in REPORTS}
    evidence = {n: json.loads((ANALYSIS / n).read_text(encoding="utf-8"))
                for n in EVIDENCE_FILES}
    return report_texts, evidence


def main() -> int:
    report_texts, evidence = load_inputs()
    result = verify(report_texts, evidence)
    out = ANALYSIS / "number_traceability_v2.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"number traceability v2: {result['verdict']} "
          f"({result['n_pass']}/{result['n_total']})")
    for r in result["rows"]:
        if not r["passed"]:
            print(f"  FAIL {r['label']}: "
                  f"src={r.get('source_value')} "
                  f"literal={r['report_literal']!r} in "
                  f"{r['target_report']} "
                  f"present={r.get('literal_present_in_target_report')} "
                  f"numeric={r.get('numeric_match')} "
                  f"delta={r.get('numeric_delta')} err={r.get('error')}")
    print(f"report: {out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
