"""ARR-07 repaired applicability gate.

ARR-06 bug: when the gate rejected memory, rq3 degraded to EXP_PLACEBO, which
made gated transfer strictly worse than fresh. Correct behaviour: ABSTAIN —
report that no verified knowledge applies, do not run a dummy experiment.
"""
from __future__ import annotations

from typing import Any


def repaired_applicability_gate(items: list[Any], task: Any) -> dict[str, Any]:
    """Return should_transfer + explicit abstain_reason; never silently PLACEBO."""
    candidates = []
    for it in items:
        cond = it.applicability_conditions or {}
        match = cond.get("mechanism") == task.cause and cond.get("action_conditioned") is True
        verified = getattr(it, "verification_status", "") == "VERIFIED"
        evidence = bool(getattr(it, "supporting_evidence", [])) and \
            float(getattr(it, "confidence", 0)) >= 0.6
        if match and verified and evidence:
            candidates.append(it)
    if not candidates:
        return {"should_transfer": False, "selected": [],
                "abstain_reason": "no VERIFIED memory with matching experimental "
                                  "conditions; refusing to transfer (no placebo run)",
                "transfer_risk": 1.0}
    return {"should_transfer": True, "selected": candidates,
            "abstain_reason": "", "transfer_risk": 0.0}


__all__ = ["repaired_applicability_gate"]
