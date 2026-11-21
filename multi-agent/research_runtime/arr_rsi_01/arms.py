"""ARR-RSI-01 P0.1 -- controlled arms with FULL intervention traceability.

Every Candidate records diagnosis id, evidence refs, hypothesis, expected
effect, falsification condition, before-config, after-config, applied config
and a config-diff hash. An empty intervention is INERT. An intervention that
was not actually applied is NOT_APPLIED and can never count as improvement.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from ..arr_runtime import BASE_HARNESS, ResearchHarness
from ..harness_rsi.harness import assert_single_method_change, derive_candidate
from .diagnosis import DiagnosisOutput, INSUFFICIENT
from .tasks import ALLOWED_CONFIG_SCHEMA, sha256_json

ARMS = ("base", "candidate", "placebo", "shuffled")


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    harness: ResearchHarness
    intervention_policy: dict[str, dict[str, Any]]
    diagnosis_id: str
    diagnosis_evidence_refs: tuple[str, ...]
    intervention: dict[str, Any] | None
    source_task_id: str
    rationale: str
    component_changed: str = ""
    parent_harness_sha256: str = ""
    intervention_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["harness_sha256"] = self.harness.fingerprint()
        d["arm_sha256"] = sha256_json(d)
        return d

    def budget_signature(self, *, seeds: list[int], epochs: int = 1,
                         timeout_s: float = 900.0) -> dict[str, Any]:
        return {"campaign_rounds": self.harness.campaign_rounds,
                "max_candidates": self.harness.max_candidates,
                "paired_seeds": list(seeds), "epochs": epochs,
                "max_total_seconds": timeout_s,
                "interventions_per_round": 1}


def _dotted(config: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _payload(iv: dict[str, Any] | None) -> dict[str, Any]:
    if not iv:
        return {}
    key = iv.get("key")
    if key:
        return {str(key): iv.get("value")}
    return {str(k): v for k, v in iv.items() if k not in ("role", "direction",
                                                          "mechanism")}


def _record(*, arm: str, diagnosis: DiagnosisOutput | None, intervention,
            hypothesis: str, expected: dict[str, Any], falsification: str,
            baseline_config: dict[str, Any], status: str) -> dict[str, Any]:
    payload = _payload(intervention)
    before = {}
    after = {}
    for key, value in payload.items():
        before[key] = _dotted(baseline_config, key, None)
        after[key] = value
    diff = {"before": before, "after": after}
    record = {
        "record_id": "iv_" + sha256_json(
            {"arm": arm, "payload": payload, "status": status})[:16],
        "arm": arm,
        "diagnosis_id": getattr(diagnosis, "diagnosis_id", "") if diagnosis else "",
        "diagnosis_evidence_refs": list(getattr(diagnosis, "evidence_refs", [])
                                        if diagnosis else []),
        "intervention_hypothesis": hypothesis,
        "expected_effect": expected,
        "falsification_condition": falsification,
        "before_config": before,
        "after_config": after,
        "config_diff": diff,
        "config_diff_hash": sha256_json(diff),
        "actual_applied_config": {},
        "actual_applied_config_path": "",
        "status": status,
    }
    if not payload:
        record["status"] = "INERT"
    return record


def _base_arm() -> ArmSpec:
    harness = ResearchHarness(
        planner_strategy="fixed_first_knob", hypothesis_prompt="terse_v0",
        experience_top_k=0, max_candidates=1, campaign_rounds=1,
        require_falsification=False,
        notes="ARR-RSI-01-P0.1 Base: frozen method, no diagnosis injection.")
    return ArmSpec(
        arm="base", harness=harness, intervention_policy={},
        diagnosis_id="", diagnosis_evidence_refs=(), intervention=None,
        source_task_id="", rationale="Frozen H0; first allowed knob only.",
        intervention_record=_record(
            arm="base", diagnosis=None, intervention=None,
            hypothesis="No diagnosis-driven hypothesis (frozen Baseline).",
            expected={}, falsification="", baseline_config={}, status="INERT"))


def _harness(notes: str) -> ResearchHarness:
    return ResearchHarness(
        planner_strategy="diagnosis_targeted", hypothesis_prompt="evidence_grounded_v1",
        experience_top_k=0, max_candidates=1, campaign_rounds=1,
        require_falsification=True, notes=notes)


def _bundle_from_harness(harness: ResearchHarness):
    from ..harness_rsi.harness import ComponentSpec, HarnessBundle
    components = {}
    for name in ("objective_generator", "hypothesis_critic",
                 "experiment_designer", "failure_diagnoser"):
        components[name] = ComponentSpec(
            name=name, version="arr_base_v1", method=f"arr::{name}",
            config=harness.to_dict(),
            provenance={"source": "arr_runtime.BASE_HARNESS"})
    return HarnessBundle(
        harness_id=f"arr_harness_{harness.fingerprint()[:12]}", parent_sha="",
        components=components,
        workflow={"rounds": harness.campaign_rounds, "approval_required": True},
        prompt_policy={"version": harness.hypothesis_prompt},
        tool_policy={"allowed": ["research_api"]},
        model_config={"temperature": 0, "model_role": "research_assistant"},
        environment={"runtime": "arr-1.0"}, provenance={"source": "arr_runtime"})


def _placebo_choice(*, candidate_key: str, seed: str,
                    baseline_config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic inert knob from the generic schema (no benchmark label)."""
    keys = [s["key"] for s in ALLOWED_CONFIG_SCHEMA if s["key"] != candidate_key]
    digest = hashlib.sha256(f"{seed}|placebo".encode()).digest()
    key = keys[int.from_bytes(digest[:4], "big") % len(keys)]
    schema = next(s for s in ALLOWED_CONFIG_SCHEMA if s["key"] == key)
    base = _dotted(baseline_config, key, None)
    if isinstance(base, bool) or base is None:
        value = 0.0001 if schema["type"] == "float" else 1
    elif isinstance(base, list) and base and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in base):
        value = [max(1, int(round(v * 1.01))) if isinstance(v, int)
                 else round(float(v) * 1.01, 8) for v in base]
    elif isinstance(base, (int, float)):
        number = float(base)
        # Stay strictly inside the declared schema range: an out-of-range
        # inert value would be dropped by the config gate -> NOT_APPLIED.
        factor = 0.99 if schema["range"] == "(0, 1)" and number >= 0.9 else 1.01
        if schema["range"] == "(0, 1]" and number * factor >= 1:
            factor = 0.9
        value = round(number * factor, 8) if schema["type"] == "float" \
            else max(1, int(round(number * factor)))
    else:
        value = 0.0001
    return {"key": key, "value": value, "direction": "inert",
            "mechanism": "matched-cost control; no mechanistic relevance"}


def derive_arms(*, task: dict[str, Any], diagnosis: DiagnosisOutput,
                shuffled: DiagnosisOutput | None = None,
                baseline_config: dict[str, Any] | None = None) -> dict[str, ArmSpec]:
    """Derive Base/Candidate/Placebo/Shuffled from ONE real diagnosis."""
    baseline = dict(baseline_config or {})
    base = _base_arm()
    production_base = _bundle_from_harness(BASE_HARNESS)
    candidate_h = _harness(
        f"P0.1 Candidate from diagnosis {diagnosis.diagnosis_id} "
        f"cause={diagnosis.root_cause}")
    candidate_bundle = derive_candidate(
        production_base, component="failure_diagnoser",
        version="p0_1_evidence_diagnosis_v1",
        config={"diagnosis_id": diagnosis.diagnosis_id,
                "evidence_refs": diagnosis.evidence_refs[:10],
                "expected_metric": diagnosis.expected_metric,
                "falsification_condition": diagnosis.falsification_condition},
        candidate_id=f"p01_candidate_{task.get('agent_task_id', 'unknown')}",
        source_experience_ids=[diagnosis.diagnosis_id],
        expected_improvement=str(diagnosis.expected_metric),
        falsification_conditions=[diagnosis.falsification_condition])
    change = assert_single_method_change(production_base, candidate_bundle)

    candidate_intervention = diagnosis.proposed_intervention or None
    if diagnosis.insufficient_evidence or diagnosis.root_cause == INSUFFICIENT:
        candidate_intervention = None
    candidate_record = _record(
        arm="candidate", diagnosis=diagnosis, intervention=candidate_intervention,
        hypothesis=diagnosis.improvement_hypothesis,
        expected=diagnosis.expected_metric,
        falsification=diagnosis.falsification_condition,
        baseline_config=baseline, status="PENDING_APPLICATION")
    category = diagnosis.failure_modes[0] if diagnosis.failure_modes else "evidence"
    candidate_policy = {category: _payload(candidate_intervention)} \
        if candidate_intervention else {}

    placebo_iv = _placebo_choice(
        candidate_key=str((candidate_intervention or {}).get("key", "")),
        seed=str(task.get("agent_task_id", "")), baseline_config=baseline)
    placebo_record = _record(
        arm="placebo", diagnosis=diagnosis, intervention=placebo_iv,
        hypothesis="Matched-cost inert control; no mechanistic claim.",
        expected={}, falsification="", baseline_config=baseline,
        status="PENDING_APPLICATION")
    placebo_h = _harness("P0.1 Placebo: matched cost, generic inert knob.")

    shuffled = shuffled or diagnosis
    shuffled_iv = shuffled.proposed_intervention or None
    shuffled_record = _record(
        arm="shuffled", diagnosis=shuffled, intervention=shuffled_iv,
        hypothesis="CONTROL ONLY: diagnosis not derived from this task.",
        expected={}, falsification="", baseline_config=baseline,
        status="PENDING_APPLICATION")
    shuffled_h = _harness("P0.1 Shuffled-Diagnosis attribution control.")
    shuffled_category = (shuffled.failure_modes[0]
                         if shuffled.failure_modes else "evidence")

    arms = {
        "base": base,
        "candidate": ArmSpec(
            arm="candidate", harness=candidate_h,
            intervention_policy=candidate_policy,
            diagnosis_id=diagnosis.diagnosis_id,
            diagnosis_evidence_refs=tuple(diagnosis.evidence_refs[:10]),
            intervention=candidate_intervention,
            source_task_id=str(task.get("agent_task_id", "")),
            rationale="Derived from evidence-grounded diagnosis.",
            component_changed=change["changed_component"],
            parent_harness_sha256=change["component_sha"],
            intervention_record=candidate_record),
        "placebo": ArmSpec(
            arm="placebo", harness=placebo_h,
            intervention_policy={category: _payload(placebo_iv)},
            diagnosis_id=diagnosis.diagnosis_id,
            diagnosis_evidence_refs=tuple(diagnosis.evidence_refs[:10]),
            intervention=placebo_iv,
            source_task_id=str(task.get("agent_task_id", "")),
            rationale=f"Generic inert control {placebo_iv['key']}.",
            component_changed="failure_diagnoser",
            parent_harness_sha256=change["component_sha"],
            intervention_record=placebo_record),
        "shuffled": ArmSpec(
            arm="shuffled", harness=shuffled_h,
            intervention_policy=({shuffled_category: _payload(shuffled_iv)}
                                 if shuffled_iv else {}),
            diagnosis_id=shuffled.diagnosis_id,
            diagnosis_evidence_refs=tuple(shuffled.evidence_refs[:10]),
            intervention=shuffled_iv,
            source_task_id=shuffled.task_id,
            rationale="Diagnosis from another task / synthetic shuffle.",
            component_changed="failure_diagnoser",
            parent_harness_sha256=change["component_sha"],
            intervention_record=shuffled_record),
    }
    assert_matched_budgets(arms, seeds=list(task.get("budget", {}).get(
        "paired_seeds", [42])))
    return arms


def assert_matched_budgets(arms: dict[str, ArmSpec], *, seeds: list[int],
                           epochs: int = 1,
                           timeout_s: float = 900.0) -> dict[str, Any]:
    signatures = {name: arm.budget_signature(seeds=seeds, epochs=epochs,
                                             timeout_s=timeout_s)
                  for name, arm in arms.items()}
    reference = signatures["base"]
    mismatches = {n: s for n, s in signatures.items() if s != reference}
    if mismatches:
        raise ValueError(f"unmatched arm budgets: {mismatches}")
    return {"matched": True, "signature": reference}


def finalize_intervention_record(*, record: dict[str, Any],
                                 applied_config: dict[str, Any],
                                 applied_config_path: str) -> dict[str, Any]:
    """Bind the record to the ACTUAL engine config; fail closed on mismatch."""
    out = dict(record)
    after = out.get("after_config") or {}
    actual = {}
    mismatches = []
    for key, expected in after.items():
        found = _dotted(applied_config, key, None)
        actual[key] = found
        if found != expected:
            mismatches.append({"key": key, "expected": expected, "actual": found})
    out["actual_applied_config"] = actual
    out["actual_applied_config_path"] = applied_config_path
    out["config_mismatches"] = mismatches
    out["application_verified"] = bool(after) and not mismatches
    out["status"] = ("INERT" if not after else
                     ("APPLIED" if out["application_verified"] else "NOT_APPLIED"))
    return out


__all__ = ["ARMS", "ArmSpec", "derive_arms", "assert_matched_budgets",
           "finalize_intervention_record"]
