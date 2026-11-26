"""ARR-07: give the LLM the TWO decision rights previously held by harness.

① LLM proposes the SciNet++ STRUCTURE / HYPERPARAMETERS (not a human menu).
③ LLM decides finish vs continue from measured before/after data.

Guardrails (so autonomy is executable and auditable, not free-form fantasy):
- proposal must be JSON with a WHITELIST of real config keys + type/range checks;
  invalid proposals are REJECTED and logged, never silently repaired.
- finish decision must QUOTE the measured numbers it relies on; recorded as
  evidence with the raw record hashes, auditable afterward.
- the harness does NOT override a valid LLM decision (that is the whole point).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class Proposal:
    reasoning: str
    overrides: dict[str, Any]
    raw_response: str
    valid: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"reasoning": self.reasoning, "overrides": self.overrides,
                "valid": self.valid, "rejection_reason": self.rejection_reason,
                "raw_response": self.raw_response[:2000]}


@dataclass
class FinishDecision:
    finish: bool
    reasoning: str
    quoted_metrics: dict[str, Any]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {"finish": self.finish, "reasoning": self.reasoning,
                "quoted_metrics": self.quoted_metrics,
                "raw_response": self.raw_response[:2000]}


def _parse_json(text: str) -> dict[str, Any] | None:
    t = text.strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.splitlines()
                      if not l.strip().startswith("```")).strip()
    if t.startswith("json\n"):
        t = t[5:].strip()
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except (ValueError, TypeError):
        return None


def validate_proposal(overrides: Any, allowed: dict[str, Any]) -> tuple[bool, str, dict]:
    """Check LLM-proposed overrides against real config keys and ranges."""
    if not isinstance(overrides, dict) or not overrides:
        return False, "overrides must be a non-empty dict", {}
    clean = {}
    for k, v in overrides.items():
        if k not in allowed:
            return False, f"key {k!r} not in allowed SciNet++ config keys", {}
        rule = allowed[k]
        if rule is None:  # any type accepted (e.g. hidden layer list)
            clean[k] = v
        elif isinstance(rule, tuple):  # numeric range
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, f"{k} must be a number", {}
            if not (rule[0] <= float(v) <= rule[1]):
                return False, f"{k}={v} outside [{rule[0]}, {rule[1]}]", {}
            clean[k] = v
        elif rule == "list_int":
            if (not isinstance(v, list) or not v or
                    any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                        for x in v) or len(v) > 4 or max(v) > 256):
                return False, f"{k} must be a short list of positive ints (<=256 each)", {}
            clean[k] = v
        else:
            return False, f"no validation rule for {k}", {}
    return True, "", clean


class LLMResearcher:
    """LLM with two explicit decision rights, via a caller-supplied generator."""

    def __init__(self, generate: Callable[[str], str], out_dir: str | Path):
        self._gen = generate
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0

    def _call(self, payload: dict) -> str:
        self.n_calls += 1
        prompt = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            raw = self._gen(prompt)
        except Exception:
            raw = ""
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False, default=str)
        (self.out / f"autonomy_{self.n_calls:02d}.txt").write_text(
            prompt + "\n---RESPONSE---\n" + raw, encoding="utf-8")
        return raw

    # ⓪ NEW: LLM decides WHICH single-variable screening sweep to run first.
    # This is how the agent DISCOVERS the key variable itself, instead of being
    # fed a human-computed prior_evidence.json.
    def plan_screening(self, *, objective: str, allowed: dict[str, Any],
                       baseline_metrics: dict[str, float],
                       max_knobs: int = 3, values_per_knob: int = 2) -> dict[str, Any]:
        payload = {
            "role": "autonomous_researcher",
            "task": ("Before proposing full experiments, PLAN a one-factor-at-a-time "
                     "screening sweep to DISCOVER which config knob most affects "
                     "the objective. You choose the knobs and the values; the "
                     "harness will run each single change on the real engine."),
            "objective": objective,
            "baseline_measured_metrics": baseline_metrics,
            "allowed_knobs_and_ranges": allowed,
            "budget": {"max_knobs": max_knobs, "values_per_knob": values_per_knob,
                       "rule": "only ONE knob changed per screening run"},
            "hint": ("Pick knobs you are UNSURE about; each must have a value "
                     "different from its baseline in the config. Include at least "
                     "one optimization knob (e.g. learning rate). CRITICAL: "
                     "explore BOTH directions and ACROSS ORDERS OF MAGNITUDE — "
                     "e.g. for a baseline lr=0.001, test 0.0001 AND 0.01, not "
                     "only tiny values or the baseline itself. A sweep with no "
                     "value materially different from baseline is useless."),
            "output_format": ("JSON: {\"reasoning\": str, \"sweep\": "
                              "[{\"key\": str, \"values\": [v1, v2]}]}"),
        }
        raw = self._call(payload)
        d = _parse_json(raw)
        if not d:
            return {"valid": False, "reason": "screening plan not valid JSON",
                    "raw": raw[:500], "sweep": []}
        sweep = d.get("sweep") or []
        if not isinstance(sweep, list) or not sweep:
            return {"valid": False, "reason": "empty sweep",
                    "raw": raw[:500], "sweep": []}
        clean = []
        for item in sweep[:max_knobs]:
            if not isinstance(item, dict):
                continue
            key, vals = item.get("key"), item.get("values")
            if key not in allowed or not isinstance(vals, list):
                continue
            ok, _, norm = validate_proposal({key: vals[0]}, allowed) if vals else (False, "", {})
            if not ok or len(vals) > values_per_knob:
                continue
            clean.append({"key": key, "values": vals})
        return {"valid": bool(clean), "reasoning": str(d.get("reasoning", ""))[:800],
                "sweep": clean, "raw": raw[:1000]}

    # ① LLM proposes structure / parameters
    def propose_experiment(self, *, objective: str, allowed: dict[str, Any],
                           baseline_metrics: dict[str, float],
                           history: list[dict],
                           search_hint: dict[str, Any] | None = None,
                           contract: dict[str, Any] | None = None,
                           max_changes: int = 2,
                           prior_evidence: dict[str, Any] | None = None) -> Proposal:
        payload = {
            "role": "autonomous_researcher",
            "task": "Propose ONE experiment by choosing SciNet++ model structure "
                    "and/or hyperparameters. You decide the keys and values yourself.",
            "objective": objective,
            "frozen_quantified_goal": contract or {},
            "baseline_measured_metrics": baseline_metrics,
            "previous_rounds": history,
            "search_memory_hint": search_hint or {"hint": "no trials yet"},
            "verified_prior_evidence": prior_evidence or {"note": "none supplied"},
            "prior_evidence_rule": ("These are MEASURED results from earlier real "
                                    "experiments. Treat them as strong evidence: "
                                    "if a prior config improved the objective, "
                                    "test it (or a nearby value) before inventing "
                                    "new directions."),
            "change_budget": (f"Change AT MOST {max_changes} config keys per round. "
                              "One-factor-at-a-time is preferred so the measured "
                              "effect can be attributed. Changing 7 keys at once "
                              "previously lowered cosine while MSE improved."),
            "why_constraints_exist": ("Structural limits are enforced in code because "
                                      "a 64-sample dataset cannot fit a huge model; "
                                      "see rejected attempts in previous_rounds."),
            "allowed_config_keys_and_ranges": allowed,
            "rules": ["choose values you believe will improve the objective",
                      "only use keys from allowed_config_keys_and_ranges",
                      "explain your hypothesis from the measured metrics"],
            "output_format": "JSON: {\"reasoning\": str, \"overrides\": {key: value}}",
        }
        raw = self._call(payload)
        d = _parse_json(raw)
        if not d:
            return Proposal("", {}, raw, False, "LLM response not valid JSON")
        reasoning = str(d.get("reasoning", ""))[:1200]
        ok, why, clean = validate_proposal(d.get("overrides"), allowed)
        if ok and len(clean) > max_changes:
            return Proposal(reasoning, clean, raw, False,
                            f"proposal changes {len(clean)} keys, max {max_changes} "
                            "per round (one-factor-at-a-time)")
        return Proposal(reasoning, clean, raw, ok, why)

    # ③ LLM decides finish vs continue
    def decide_finish(self, *, objective: str, target: float,
                      baseline: dict[str, float], latest: dict[str, float],
                      improvement: float | None, iteration: int,
                      max_iterations: int,
                      contract_verdict: dict[str, Any] | None = None) -> FinishDecision:
        payload = {
            "role": "autonomous_researcher",
            "task": "Decide whether the research objective is met (finish) or "
                    "another experiment is needed (continue). You are the judge.",
            "objective": objective,
            "target_improvement": target,
            "frozen_contract_evaluation": contract_verdict or {},
            "baseline_metrics": baseline,
            "latest_metrics": latest,
            "computed_improvement": improvement,
            "iteration": iteration, "max_iterations": max_iterations,
            "rules": ["quote the measured numbers you rely on",
                      "frozen_contract_evaluation is computed by the harness from "
                      "measured data (ALL goals must meet their threshold); you may "
                      "disagree only with evidence, not ignore it",
                      "finish only if the objective is genuinely met",
                      "budget exhaustion is recorded separately - it is NOT your verdict"],
            "output_format": "JSON: {\"finish\": bool, \"reasoning\": str, "
                             "\"quoted_metrics\": {key: value}}",
        }
        raw = self._call(payload)
        d = _parse_json(raw)
        if not d:
            return FinishDecision(False, "parse failed; continue", {}, raw)
        # The LLM's judgement is recorded VERBATIM. Budget exhaustion is a
        # SEPARATE field in the caller's result; it must not silently rewrite
        # what the LLM decided (that would destroy the decision-right test).
        finish = bool(d.get("finish", False))
        quoted = d.get("quoted_metrics") or {}
        return FinishDecision(finish, str(d.get("reasoning", ""))[:1200],
                              quoted if isinstance(quoted, dict) else {}, raw)


__all__ = ["Proposal", "FinishDecision", "LLMResearcher", "validate_proposal"]
