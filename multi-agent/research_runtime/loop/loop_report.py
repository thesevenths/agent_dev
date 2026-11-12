"""ARR 1.1 / G3 -- aggregate, auditable report for a finished loop.

Everything in the report is derived from the recorded :class:`LoopState`
(checkpoint) and the version history -- no metric is invented here. A reader can
see, per round, what was accepted and baked into the inherited baseline.
"""
from __future__ import annotations

from typing import Any

from .loop_state import LoopState


def build_loop_report(state: LoopState) -> dict[str, Any]:
    n_rounds = len(state.version_history)
    return {
        "schema_version": "arr-1.1-loop-report-v1",
        "loop_id": state.loop_id,
        "domain": state.domain,
        "status": state.status,
        "rounds_requested": state.max_rounds,
        "rounds_completed": n_rounds,
        "final_baseline_config_path": state.baseline_config_path,
        "final_accepted_domain_overrides": dict(state.accepted_domain_overrides),
        "n_domain_accepted_total": state.n_domain_accepted,
        "n_harness_accepted_total": state.n_harness_accepted,
        "versions": [v.to_dict() for v in state.version_history],
        "note": ("a version is the accumulated, accepted scientific state; it is "
                 "NOT an automatic promotion of a model -- confirmation remains a "
                 "separate human-gated phase"),
    }


def summarise_text(report: dict[str, Any]) -> str:
    lines = [
        f"ARR 1.1 research loop {report['loop_id']} -- domain={report['domain']}",
        f"  status={report['status']} rounds={report['rounds_completed']}/"
        f"{report['rounds_requested']}",
        f"  final baseline: {report['final_baseline_config_path']}",
        f"  accumulated domain overrides: "
        f"{report['final_accepted_domain_overrides'] or '{}'}",
        f"  accepted: domain={report['n_domain_accepted_total']} "
        f"harness={report['n_harness_accepted_total']}",
    ]
    for v in report["versions"]:
        lines.append(
            f"  v{v['round_no']}: {v['accepted_domain_overrides'] or '{}'} "
            f"harness={v['accepted_harness_targets'] or []} -> "
            f"{v['baseline_config_path']}")
    return "\n".join(lines)
