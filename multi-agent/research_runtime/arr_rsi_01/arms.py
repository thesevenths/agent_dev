"""ARR-RSI-01 -- controlled Base / Candidate / Placebo / Shuffled arms.

Every arm uses the SAME real Research Runtime and the SAME engine campaign
machinery. Only the research-method configuration changes; model version,
runtime version, dataset, seeds, metrics and budgets are frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..arr_runtime import BASE_HARNESS, ResearchHarness
from ..harness_rsi.harness import assert_single_method_change, derive_candidate
from .diagnosis import DiagnosisOutput
from .tasks import sha256_json

ARMS = ("base", "candidate", "placebo", "shuffled")


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    harness: ResearchHarness
    intervention_policy: dict[str, dict[str, Any]]
    diagnosis_sha256: str
    intervention: dict[str, Any] | None
    source_task_id: str
    rationale: str
    component_changed: str = ""
    parent_harness_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["harness_sha256"] = self.harness.fingerprint()
        d["arm_sha256"] = sha256_json(d)
        return d

    def budget_signature(self, *, seeds: list[int], epochs: int = 1) -> dict[str, Any]:
        """Cost-relevant configuration; must be identical across arms."""
        return {
            "campaign_rounds": self.harness.campaign_rounds,
            "max_candidates": self.harness.max_candidates,
            "paired_seeds": list(seeds),
            "epochs": epochs,
            # Every arm runs exactly one real candidate job set. The Base arm
            # derives its knob from the frozen policy; controls use the
            # recorded payload. Planner strategy is a METHOD difference, not a
            # cost difference, so it is intentionally excluded here.
            "interventions_per_round": 1,
        }


def _base_arm() -> ArmSpec:
    harness = ResearchHarness(
        planner_strategy="fixed_first_knob",
        hypothesis_prompt="terse_v0",
        experience_top_k=0,
        max_candidates=1,
        campaign_rounds=1,
        require_falsification=False,
        notes="ARR-RSI-01 Base: frozen production method, no diagnosis injection.",
    )
    return ArmSpec(
        arm="base", harness=harness, intervention_policy={},
        diagnosis_sha256="", intervention=None, source_task_id="",
        rationale="Frozen H0; first allowed knob only (no diagnosis).",
    )


def _matched_harness(*, strategy: str, diagnosis: DiagnosisOutput,
                     notes: str) -> ResearchHarness:
    """Same cost shape as Base; only the method component changes."""
    return ResearchHarness(
        planner_strategy=strategy,
        hypothesis_prompt="evidence_grounded_v1",
        experience_top_k=0,
        max_candidates=1,
        campaign_rounds=1,
        require_falsification=True,
        notes=notes,
    )


def derive_arms(*, task: dict[str, Any], diagnosis: DiagnosisOutput,
                shuffled: DiagnosisOutput | None = None) -> dict[str, ArmSpec]:
    """Derive all four arms from ONE diagnosis (and its shuffled control)."""
    base = _base_arm()
    diagnosis_dict = diagnosis.to_dict()
    trace = {
        "diagnosis_sha256": diagnosis_dict["diagnosis_sha256"],
        "root_cause": diagnosis.root_cause,
        "evidence_refs": diagnosis.evidence_refs[:10],
        "task_id": task.get("task_id", ""),
    }
    def _payload(iv: dict[str, Any] | None) -> dict[str, Any]:
        """Convert a recorded intervention record to engine knob->value."""
        if not iv:
            return {}
        key = iv.get("key")
        if key:
            return {str(key): iv.get("value")}
        return {str(k): v for k, v in iv.items() if k != "role"}

    candidate_intervention = _payload(diagnosis.proposed_intervention or {})
    # An evidence-gap diagnosis may legitimately propose nothing. In that case
    # the Candidate arm is INERT and must not be claimed as improved.
    if not candidate_intervention:
        candidate_intervention = {}

    candidate_h = _matched_harness(
        strategy="diagnosis_targeted", diagnosis=diagnosis,
        notes=f"ARR-RSI-01 Candidate derived from diagnosis {trace['diagnosis_sha256'][:12]} "
              f"cause={diagnosis.root_cause}; trace={sha256_json(trace)[:12]}.",
    )
    # Enforce the single-method-change rule against the frozen production Base.
    production_base = BASE_HARNESS
    candidate_evidence = {
        "diagnosis_sha256": diagnosis_dict["diagnosis_sha256"],
        "evidence_refs": diagnosis.evidence_refs[:10],
        "expected_metric": diagnosis.expected_metric,
        "falsification_condition": diagnosis.falsification_condition,
    }
    candidate_bundle = derive_candidate(
        _bundle_from_harness(production_base),
        component="failure_diagnoser",
        version="arr_rsi_01_diagnosis_targeted_v1",
        config=candidate_evidence,
        candidate_id=f"arr_rsi_01_candidate_{task.get('task_id', 'unknown')}",
        source_experience_ids=[diagnosis_dict["diagnosis_sha256"]],
        expected_improvement=str(diagnosis.expected_metric),
        falsification_conditions=[diagnosis.falsification_condition],
    )
    _change = assert_single_method_change(_bundle_from_harness(production_base),
                                          candidate_bundle)

    category = diagnosis.failure_modes[0] if diagnosis.failure_modes else "unclassified"
    candidate_policy = ({category: candidate_intervention}
                        if candidate_intervention else {})

    # Placebo: identical cost/harness shape, but an explicitly irrelevant knob.
    placebo_pool = [iv for iv in (task.get("valid_interventions") or [])
                    if isinstance(iv, dict) and iv.get("role") == "placebo"]
    placebo_intervention = dict(placebo_pool[0]) if placebo_pool else {}
    placebo_key = str(placebo_intervention.get("key", "train.weight_decay"))
    placebo_value = placebo_intervention.get("value", 0.0001)
    if not isinstance(placebo_value, (int, float)):
        placebo_value = 0.0001
    placebo_payload = {placebo_key: placebo_value}
    placebo_h = _matched_harness(
        strategy="diagnosis_targeted", diagnosis=diagnosis,
        notes="ARR-RSI-01 Placebo: matched cost, mechanistically irrelevant "
              f"intervention {placebo_key}={placebo_value}.",
    )

    # Shuffled diagnosis: same evidence, diagnosis from another task.
    shuffled = shuffled or diagnosis
    shuffled_intervention = _payload(shuffled.proposed_intervention or {})
    shuffled_category = (shuffled.failure_modes[0]
                         if shuffled.failure_modes else "unclassified")
    shuffled_h = _matched_harness(
        strategy="diagnosis_targeted", diagnosis=shuffled,
        notes="ARR-RSI-01 Shuffled-Diagnosis attribution control.",
    )

    arms = {
        "base": base,
        "candidate": ArmSpec(
            arm="candidate", harness=candidate_h,
            intervention_policy=candidate_policy,
            diagnosis_sha256=diagnosis_dict["diagnosis_sha256"],
            intervention=candidate_intervention,
            source_task_id=str(task.get("task_id", "")),
            rationale="Directly derived from evidence-grounded diagnosis.",
            component_changed=_change["changed_component"],
            parent_harness_sha256=_change["component_sha"],
        ),
        "placebo": ArmSpec(
            arm="placebo", harness=placebo_h,
            intervention_policy={category: placebo_payload},
            diagnosis_sha256=diagnosis_dict["diagnosis_sha256"],
            intervention=placebo_payload,
            source_task_id=str(task.get("task_id", "")),
            rationale=f"Controlled irrelevant intervention {placebo_key}.",
            component_changed="failure_diagnoser",
            parent_harness_sha256=_change["component_sha"],
        ),
        "shuffled": ArmSpec(
            arm="shuffled", harness=shuffled_h,
            intervention_policy=({shuffled_category: shuffled_intervention}
                                 if shuffled_intervention else {}),
            diagnosis_sha256=shuffled.to_dict()["diagnosis_sha256"],
            intervention=shuffled_intervention or None,
            source_task_id=shuffled.task_id,
            rationale="Diagnosis permuted from another task (attribution control).",
            component_changed="failure_diagnoser",
            parent_harness_sha256=_change["component_sha"],
        ),
    }
    assert_matched_budgets(arms, seeds=list(task.get("budget", {}).get("paired_seeds", [42])))
    return arms


def _bundle_from_harness(harness: ResearchHarness):
    """Map an ARR ResearchHarness onto the M6 versioned HarnessBundle."""
    from ..harness_rsi.harness import ComponentSpec, HarnessBundle
    components = {}
    for name in ("objective_generator", "hypothesis_critic",
                 "experiment_designer", "failure_diagnoser"):
        components[name] = ComponentSpec(
            name=name, version="arr_base_v1",
            method=f"arr::{name}", config=harness.to_dict(),
            provenance={"source": "arr_runtime.BASE_HARNESS"})
    return HarnessBundle(
        harness_id=f"arr_harness_{harness.fingerprint()[:12]}",
        parent_sha="", components=components,
        workflow={"rounds": harness.campaign_rounds,
                  "approval_required": True},
        prompt_policy={"version": harness.hypothesis_prompt},
        tool_policy={"allowed": ["research_api"]},
        model_config={"temperature": 0, "model_role": "research_assistant"},
        environment={"runtime": "arr-1.0"},
        provenance={"source": "arr_runtime"},
    )


def assert_matched_budgets(arms: dict[str, ArmSpec], *, seeds: list[int],
                           epochs: int = 1) -> dict[str, Any]:
    """Fail closed if arms do not share identical matched budgets."""
    signatures = {name: arm.budget_signature(seeds=seeds, epochs=epochs)
                  for name, arm in arms.items()}
    reference = signatures["base"]
    mismatches = {name: sig for name, sig in signatures.items()
                  if sig != reference}
    if mismatches:
        raise ValueError(f"unmatched arm budgets: {mismatches}")
    return {"matched": True, "signature": reference}


__all__ = ["ARMS", "ArmSpec", "derive_arms", "assert_matched_budgets"]
