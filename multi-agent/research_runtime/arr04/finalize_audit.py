"""Re-run INDEPENDENT audit + verdicts + report from persisted evidence.

No engine runs, no LLM calls, no modification to raw/immutable data. Used when
the audit logic itself was corrected after a campaign reached terminal state.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import audit, core
from .report import generate_final_report
from .statistics import verdicts


def finalize(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    canonical = json.loads((root / "final" / "ARR04_FINAL_REPORT.json").read_text(
        encoding="utf-8"))
    proto = json.loads((root / "protocol" / "protocol.json").read_text(encoding="utf-8"))
    # BudgetController counters reset on resume; reconstruct TRUE totals from
    # immutable run records so provenance in the report is accurate.
    raw_records = list((root / "raw_data" / "immutable").glob("*.json"))
    preflight_n = sum(1 for p in raw_records
                      if p.stem.startswith(("ident__", "valid__", "pilot__")))
    canonical["budget"]["formal_runs"] = len(raw_records) - preflight_n
    canonical["budget"]["preflight_runs"] = preflight_n
    llm_files = list(root.rglob("agent_llm_interaction.json"))
    canonical["budget"]["llm_calls"] = len(llm_files)
    audit_result = audit.run_independent_audit(root, canonical["verdicts"])
    final_v = verdicts(
        engineering=canonical["engineering"], integrity=canonical["integrity"],
        m1=canonical["m1"], m2=canonical["m2"], m3=canonical["m3"],
        recursive=canonical["recursive"], audit=audit_result["verdict"],
        protocol=proto)
    canonical["independent_audit"] = audit_result
    canonical["verdicts"] = final_v
    canonical["verdict"] = final_v["overall_scientific_verdict"]
    canonical.pop("canonical_sha256", None)
    canonical["canonical_sha256"] = core.sha256_json(canonical)
    (root / "final" / "ARR04_FINAL_REPORT.json").write_text(json.dumps(
        canonical, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8")
    (root / "final" / "ARR04_VERDICT.json").write_text(json.dumps(
        final_v, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    generate_final_report(root, canonical)
    return canonical


__all__ = ["finalize"]
