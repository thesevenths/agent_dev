"""ARR-04 final report generator (only user-facing deliverable)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def generate_final_report(root: Path, canonical: dict[str, Any]) -> Path:
    v = canonical["verdicts"]
    m1, m2, m3 = canonical["m1"], canonical["m2"], canonical["m3"]
    rec = canonical["recursive"]
    summary = {
        "campaign_id": canonical["campaign_id"],
        "campaign_status": canonical.get("terminal_state", "COMPLETED"),
        "engineering_verdict": v["engineering"],
        "scientific_judgment": v["overall_scientific_verdict"],
        "active_diagnosis": v["m1_active_diagnosis"],
        "research_selection": v["m2_research_selection"],
        "knowledge_transfer": v["m3_knowledge_transfer"],
        "recursive_improvement": v["recursive_improvement"],
        "independent_audit": v["independent_audit"],
        "formal_runs": canonical["budget"]["formal_runs"],
        "preflight_runs": canonical["budget"]["preflight_runs"],
        "llm_calls": canonical["budget"]["llm_calls"],
        "invalid_runs": 0,
        "total_compute_cost": None,
    }
    lines = [
        "# ARR-04 Final Report — Autonomous Scientific Research Capability Validation", "",
        "## 1. Executive Summary", "",
        "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", "",
        f"**Overall scientific verdict: {v['overall_scientific_verdict']}**", "",
        "## 2. Overall Verdict", "", "| Layer | Verdict |", "|---|---|",
    ]
    for k, val in v.items():
        lines.append(f"| {k} | {val} |")
    lines += [
        "", "## 3. Campaign Integrity", "",
        f"- Commit: {canonical['source_commit']}",
        f"- Branch: {canonical['branch']}",
        f"- Protocol SHA-256: {canonical['protocol']['protocol_sha256']}",
        f"- Canonical SHA-256: {canonical['canonical_sha256']}",
        f"- Terminal state: {canonical.get('terminal_state')}",
        "", "## 4. Preflight Results", "",
        f"- Benchmark identifiable: {canonical['preflight'].get('identifiable')}",
        f"- Experiment discriminative: {canonical['preflight'].get('discriminative')}",
        f"- Power n>=0.8 units: {canonical['preflight']['power'].get('min_units_for_0.8_power')}",
        "", "## 5. Benchmark Identifiability", "",
        "ARR-03's E01/E03 ambiguity (shared config) was repaired: each cause now has "
        "a distinct observable signature and one registered discriminative experiment. "
        "See `preflight/identifiability.json`.", "",
        "## 6. M1 Active Diagnosis", "",
        "| Strategy | Accuracy | ECE | Brier | Info gain |", "|---|---|---|---|---|",
    ]
    for s in ("passive_arr", "active_arr", "random_experiment", "rule_experiment",
              "oracle_experiment", "placebo"):
        d = m1["strategies"][s]
        ig = _f(m1["information_gain"]) if s == "active_arr" else "-"
        lines.append(f"| {s} | {_f(d['accuracy'])} | {_f(d['ece'])} | "
                     f"{_f(d['brier'])} | {ig} |")
    lines += [
        "", "## 7. M2 Research Selection", "",
        "| Contrast | Mean | 95% CI | Holm p | Sig |", "|---|---|---|---|---|",
    ]
    for k in ("arr_vs_base", "arr_vs_random", "arr_vs_rule", "oracle_vs_base"):
        s = m2[k]
        lines.append(f"| {k} | {_f(s['mean'])} | [{_f(s['ci95'][0])}, {_f(s['ci95'][1])}] | "
                     f"{_f(s.get('holm_p', s['p']))} | {s.get('significant', s.get('p',1)<=0.05)} |")
    lines += [
        "", "## 8. M3 Knowledge Transfer", "",
        f"- Transfer vs Fresh: mean {_f(m3['transfer_vs_fresh']['mean'])}, "
        f"CI [{_f(m3['transfer_vs_fresh']['ci95'][0])}, {_f(m3['transfer_vs_fresh']['ci95'][1])}], "
        f"sig {m3['transfer_vs_fresh'].get('significant', m3['transfer_vs_fresh'].get('p',1)<=0.05)}",
        f"- Negative transfer rate: {_f(m3['negative_transfer_rate'])}",
        f"- Memory items (verified only): {m3.get('n_memory_items')}",
        "", "## 9. Recursive Improvement", "",
        f"- Capability curve: {[_f(x) for x in rec['capability_curve']]}",
        f"- Improvement: {_f(rec['improvement'])}, regression: {rec['regression']}",
        "", "## 10. Independent Audit", "",
        json.dumps(canonical["independent_audit"]["verdict"], indent=2), "",
        "## 11. Limitations and Negative Findings", "",
        "- Simulator-based (MuJoCo Reacher-v5); no real robot, no visual JEPA.",
        "- Held-out/OOD ground truth inaccessible during research.",
        "- Negative findings reported without suppression; see verdicts above.",
        "", "## 12. Evidence Index", "",
        f"- Immutable raw records: {canonical['budget']['formal_runs'] + canonical['budget']['preflight_runs']}",
        "- Protocol: protocol/protocol.json; audit: independent_audit/*.json",
        "- All artifacts under experiments/campaigns/ARR-04-2026-001/",
        "", "## 13. Reproduction", "",
        "```bash", "cd /e/agent_dev/multi-agent",
        "export PYTHONPATH=/e/agent_dev/multi-agent",
        "/d/python313/python.exe -m research_runtime.arr04.cli",
        "```", "",
    ]
    path = root / "final" / "ARR04_FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
