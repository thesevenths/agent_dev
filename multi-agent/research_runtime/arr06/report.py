"""ARR-06 report generator: all values from canonical records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def generate(root: Path, c: dict[str, Any]) -> dict[str, Path]:
    root = Path(root)
    v = c["verdicts"]
    (root / "final" / "ARR06_CAPABILITY_CURVE.json").write_text(json.dumps(
        {"rq4": c["rq4"].get("capability_curve"), "verdicts": v}, indent=2,
        sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    (root / "final" / "ARR06_EVIDENCE_INDEX.json").write_text(json.dumps(
        {"canonical_sha256": c["canonical_sha256"],
         "n_records": c["reproducibility"]["n_records"],
         "protocol_hash": c["protocol"]["protocol_sha256"]},
        indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    (root / "final" / "ARR06_STATISTICAL_REPORT.json").write_text(json.dumps(
        {"rq1": {k: c["rq1"].get(k) for k in ("accuracy", "ece",
                 "mean_information_gain", "n_cases", "n_units")},
         "rq2": {k: c["rq2"].get(k) for k in ("arr_vs_base", "arr_vs_random",
                 "arr_vs_rule", "oracle_vs_base", "n_units")},
         "rq3": {k: c["rq3"].get(k) for k in ("raw_vs_fresh", "gated_vs_fresh",
                 "gated_vs_raw", "negative_transfer_rate", "n_units")},
         "rq5": {k: c["rq5"].get(k) for k in ("arr_vs_random", "arr_vs_rule",
                 "falsifiability", "n_tasks")},
         "alpha": c["protocol"]["alpha"]},
        indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    (root / "final" / "ARR06_REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(
        {"levels": c["reproducibility"], "source_commit": c["source_commit"],
         "protocol_sha256": c["protocol"]["protocol_sha256"],
         "replay": c["independent_audit"]["audits"]["replay"],
         "root_cause_of_arr05_mismatch": c["source_audit"]["root_cause_identified"],
         "commands": ["cd /e/agent_dev/multi-agent",
                      "export PYTHONPATH=/e/agent_dev/multi-agent",
                      "/d/python313/python.exe -m pytest research_runtime/arr06/tests -q",
                      "/d/python313/python.exe -m research_runtime.arr06.cli"]},
        indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    lines = [
        "# ARR-06 Final Report — Research Evidence, Verified Memory & Recursive Methodology Validation",
        "", "## Executive Summary", "",
        f"Overall: **{v['overall']}** · Certification: **{v['certification_level']}** · "
        f"Terminal: {c['terminal_state']}", "",
        "| Layer | Verdict |", "|---|---|",
    ]
    for k, val in v.items():
        lines.append(f"| {k} | {val} |")
    lines += ["", "## Source & Determinism", "",
        f"- ARR-05 replay issue: {c['source_audit']['arr05_reported_replay']['n_match']}"
        f"/{c['source_audit']['arr05_reported_replay']['n_sampled']} matched",
        f"- Root cause identified: {c['source_audit']['root_cause_identified']}",
        f"- ARR-06 deterministic conditions: {len(c['hardening']['determinism']['conditions'])}, "
        f"all deterministic: {all(x['deterministic'] for x in c['hardening']['determinism']['conditions'])}", "",
        "## Reproducibility Levels (R0-R3)", "",
        json.dumps(c["reproducibility"], indent=2), "",
        "## RQ1 Belief Update & Calibration", "",
        f"- Cases: {c['rq1'].get('n_cases')}; accuracy {_f(c['rq1'].get('accuracy',0))}; "
        f"ECE {_f(c['rq1'].get('ece',1))}; IG {_f(c['rq1'].get('mean_information_gain',0))}",
        f"- LLM confidence used: {c['rq1'].get('llm_confidence_used')} (frozen Bayesian rule)", "",
        "## RQ2 Information-Gain Research Selection", "",
        f"- Units: {c['rq2'].get('n_units')}; ARR vs Random "
        f"{_f(c['rq2'].get('arr_vs_random',{}).get('mean',0))}, "
        f"sig {c['rq2'].get('arr_vs_random',{}).get('significant')}", "",
        "## RQ3 Verified Memory & Applicability Transfer", "",
        f"- Gated vs Fresh {_f(c['rq3'].get('gated_vs_fresh',{}).get('mean',0))}; "
        f"negative transfer {_f(c['rq3'].get('negative_transfer_rate',1))}", "",
        "## RQ4 Recursive Methodology", "",
        f"- Promotions: {c['rq4'].get('actual_promotions')}; "
        f"curve {c['rq4'].get('capability_curve')}", "",
        "## RQ5 Research Question Discovery", "",
        f"- ARR vs Random {_f(c['rq5'].get('arr_vs_random',{}).get('mean',0))}; "
        f"falsifiability {_f(c['rq5'].get('falsifiability',0))}; "
        f"novelty {c['rq5'].get('novelty')}", "",
        "## Independent Audit", "", json.dumps(c["independent_audit"]["verdict"], indent=2), "",
        "## Budget", "", json.dumps(c["budget"], indent=2), "",
        f"- Canonical SHA-256: {c['canonical_sha256']}", "",
    ]
    p = root / "final" / "ARR06_FINAL_REPORT.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return {"report": p}


__all__ = ["generate"]
