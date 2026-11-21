"""M5 Hardening P1-B -- deterministic Scientific Judgment dimensions.

The existing Critic checks Evidence Validity and Falsifiability (structure).
This module adds TWO more dimensions without ever fabricating a literature
claim or letting an LLM assert novelty:

  * Scientific Novelty -- measured ONLY as overlap with objectives already
    present in the snapshot. With no real literature index, the result is
    explicitly ``literature_check=not_performed``; it can detect duplication,
    but it CANNOT claim a problem is globally novel.
  * Research Value -- a bounded, deterministic score from evidence coverage,
    measurable success criteria, an actionable intervention and a sane budget.

This is a JUDGMENT AID, not an ACCEPT/REJECT verdict. It never changes a
scientific conclusion and never lowers a gate.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def scientific_judgment(candidate: Any, snapshot: Any) -> dict[str, Any]:
    desc = str(getattr(candidate, "description", "") or "")
    toks = _tokens(desc)
    objectives = list(getattr(snapshot, "objectives", None) or [])
    overlaps: list[dict[str, Any]] = []
    for obj in objectives:
        other = _tokens(str(obj.get("description", "")))
        if not toks or not other:
            continue
        j = len(toks & other) / max(1, len(toks | other))
        if j >= 0.6:
            overlaps.append({"objective_id": obj.get("objective_id", ""),
                             "token_jaccard": round(j, 3)})

    evidence_refs = list(getattr(candidate, "evidence_refs", None) or [])
    sc = getattr(candidate, "success_criteria", None) or {}
    inter = getattr(candidate, "proposed_intervention", None) or {}
    budget = getattr(candidate, "budget", None) or {}

    novelty = {
        "duplication_overlap": overlaps,
        "novelty_status": ("possible_duplicate" if overlaps else "no_in_snapshot_duplicate"),
        # Honest boundary: without a real literature index we cannot claim
        # global novelty. Never let an LLM self-certify novelty.
        "literature_check": "not_performed",
        "claim": "cannot_assert_global_novelty_without_literature_index",
    }
    value_signals = {
        "n_evidence_refs": len(evidence_refs),
        "has_measurable_criterion": bool(sc.get("measurable") and sc.get("criterion")),
        "has_actionable_intervention": bool(
            isinstance(inter, dict) and any(
                k for k in inter if not str(k).startswith("_"))),
        "budget_bounded": bool(budget) and all(
            isinstance(v, int) and v > 0 for v in budget.values()
            if isinstance(v, int)),
    }
    value_score = sum(1 for v in (
        value_signals["n_evidence_refs"] > 0,
        value_signals["has_measurable_criterion"],
        value_signals["has_actionable_intervention"],
        value_signals["budget_bounded"]))
    research_value = {
        "signals": value_signals,
        "score": value_score,
        "max_score": 4,
        "interpretation": ("well_grounded" if value_score >= 3
                           else "thinly_grounded"),
    }
    return {"novelty": novelty, "research_value": research_value,
            "verdict_emitted": False}


__all__ = ["scientific_judgment"]
