"""M6 -- versioned Research Harness bundles.

A Harness is the researcher, not the research target. It is composed of four
INDEPENDENTLY versioned method components:

  1. objective_generator   -- what is worth studying
  2. hypothesis_critic     -- is it falsifiable / evidence-grounded
  3. experiment_designer   -- is the design valid / confounder-controlled
  4. failure_diagnoser     -- why did a run fail

Prompts are a *representation* of a method, not the method itself. A bundle
also carries workflow, tool policy, model config and provenance, but for the
primary RSI experiment exactly ONE principal method component may change at a
time; model / tool / environment changes are separately declared and are NOT
allowed to masquerade as research-method improvement.

The Candidate SHA covers the COMPLETE immutable artifact (all components,
workflow, prompts, tool policy, model config, environment and provenance),
computed from the bytes on disk after freezing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .identity import freeze_json, sha256_json, verify_frozen

RESEARCH_COMPONENTS = (
    "objective_generator",
    "hypothesis_critic",
    "experiment_designer",
    "failure_diagnoser",
)
COMPONENT_METHOD = {
    "objective_generator": "evidence_refs + critic + deterministic rank",
    "hypothesis_critic": "falsifiability + evidence binding + criterion check",
    "experiment_designer": "control/arms + seed + budget + confounder checks",
    "failure_diagnoser": "failure taxonomy + root-cause evidence chain",
}
BASE_HARNESS_ID = "harness_base_m6_v1"


class HarnessError(ValueError):
    """Raised for an invalid harness bundle or illegal change."""


@dataclass(frozen=True)
class ComponentSpec:
    """One versioned research-method component."""
    name: str
    version: str
    method: str
    config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class HarnessBundle:
    """The complete immutable research harness."""
    harness_id: str
    parent_sha: str
    components: dict[str, ComponentSpec]
    workflow: dict[str, Any]
    prompt_policy: dict[str, Any]
    tool_policy: dict[str, Any]
    model_config: dict[str, Any]
    environment: dict[str, Any]
    provenance: dict[str, Any]
    candidate_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Raw immutable content. ``candidate_sha`` is NEVER part of identity."""
        d = asdict(self)
        d["candidate_sha"] = ""
        return d

    @property
    def sha256(self) -> str:
        """Content identity over the COMPLETE artifact, stable and self-free."""
        return sha256_json(self.to_dict())

    def changed_components(self, other: "HarnessBundle") -> list[str]:
        return [name for name in RESEARCH_COMPONENTS
                if self.components[name].sha256 != other.components[name].sha256]

    def non_method_changes(self, other: "HarnessBundle") -> list[str]:
        out: list[str] = []
        for field_name in ("workflow", "prompt_policy", "tool_policy",
                           "model_config", "environment"):
            if sha256_json(getattr(self, field_name)) != sha256_json(
                    getattr(other, field_name)):
                out.append(field_name)
        return out

    def component(self, name: str) -> ComponentSpec:
        if name not in self.components:
            raise HarnessError(f"unknown research component: {name}")
        return self.components[name]


def base_harness() -> HarnessBundle:
    """Freeze the CURRENT production research methods as the Base harness.

    The values mirror the live, already-shipped M5 behaviour; no engine code is
    changed here. This is the H0 that candidates must beat under paired tests.
    """
    components = {
        name: ComponentSpec(name=name, version="base_v1", method=COMPONENT_METHOD[name],
                            config={"use_historical_experience": True},
                            provenance={"source": "m5_production_default"})
        for name in RESEARCH_COMPONENTS
    }
    return HarnessBundle(
        harness_id=BASE_HARNESS_ID,
        parent_sha="",
        components=components,
        workflow={"rounds": 2, "approval_required": True,
                  "evidence_boundary": "dev_only_for_generation"},
        prompt_policy={"version": "base_v1", "strategy": "method_neutral"},
        tool_policy={"allowed": ["research_api", "local_engine",
                                 "evidence_store", "experience_retrieval"]},
        model_config={"temperature": 0, "model_role": "research_assistant"},
        environment={"runtime": "agent_dev", "python": "3.13"},
        provenance={"stage": "baseline", "note": "frozen before any candidate"},
    )


def derive_candidate(base: HarnessBundle, *, component: str, version: str,
                     config: dict[str, Any], candidate_id: str,
                     source_experience_ids: list[str] | None = None,
                     expected_improvement: str = "",
                     falsification_conditions: list[str] | None = None
                     ) -> HarnessBundle:
    """Derive a candidate changing exactly ONE principal method component."""
    if component not in RESEARCH_COMPONENTS:
        raise HarnessError(f"not a research-method component: {component}")
    new_components = dict(base.components)
    old = base.components[component]
    new_components[component] = ComponentSpec(
        name=component, version=version,
        method=old.method, config=dict(config),
        provenance={"parent_component_sha": old.sha256,
                    "source_experience_ids": list(source_experience_ids or []),
                    "expected_improvement": expected_improvement,
                    "falsification_conditions": list(falsification_conditions or [])})
    return HarnessBundle(
        harness_id=candidate_id,
        parent_sha=base.sha256,
        components=new_components,
        workflow=dict(base.workflow),
        prompt_policy=dict(base.prompt_policy),
        tool_policy=dict(base.tool_policy),
        model_config=dict(base.model_config),
        environment=dict(base.environment),
        provenance={"stage": "candidate", "changed_component": component,
                    "base_sha": base.sha256},
    )


def assert_single_method_change(base: HarnessBundle, cand: HarnessBundle
                                ) -> dict[str, Any]:
    """Enforce 'one principal research-method change at a time'."""
    changed = cand.changed_components(base)
    non_method = cand.non_method_changes(base)
    if len(changed) != 1:
        raise HarnessError(
            f"candidate must change exactly 1 method component, got {changed}")
    if non_method:
        raise HarnessError(
            f"model/tool/environment changes are not method improvements: {non_method}")
    if cand.parent_sha != base.sha256:
        raise HarnessError("candidate parent SHA does not match the frozen Base")
    return {"changed_component": changed[0], "component_sha":
            cand.components[changed[0]].sha256, "candidate_sha": cand.sha256}


def freeze_bundle(bundle: HarnessBundle, path: str | Path) -> dict[str, Any]:
    """Freeze the COMPLETE bundle to disk.

    Returns both identities explicitly:
      * ``candidate_sha`` -- canonical content SHA (the version identity used
        for parent binding, promotion, adoption and rollback);
      * ``file_sha256`` -- SHA of the exact bytes written to disk (integrity).
    """
    content_sha = bundle.sha256
    obj = bundle.to_dict()
    obj["candidate_sha"] = content_sha
    frozen = freeze_json(obj, path)
    return {"candidate_sha": content_sha, "file_sha256": frozen["sha256"],
            "path": frozen["path"], "n_bytes": frozen["n_bytes"],
            "harness_id": bundle.harness_id, "parent_sha": bundle.parent_sha}


def load_frozen_bundle(path: str | Path, *, expected_candidate_sha: str,
                       expected_file_sha: str = "") -> dict[str, Any]:
    """Re-read a bundle and verify BOTH the content identity and disk bytes."""
    import json
    p = Path(path)
    if expected_file_sha:
        check = verify_frozen(p, expected_file_sha)
        if not check["ok"]:
            return check
    if not p.is_file():
        return {"ok": False, "reasons": ["bundle_file_missing"]}
    obj = json.loads(p.read_text(encoding="utf-8"))
    # Identity is defined over the raw content with an EMPTY sha field.
    obj["candidate_sha"] = ""
    recomputed = sha256_json(obj)
    if recomputed != expected_candidate_sha:
        return {"ok": False, "reasons": ["loaded_bundle_sha_mismatch"],
                "expected": expected_candidate_sha, "recomputed": recomputed}
    return {"ok": True, "reasons": [], "bundle": obj,
            "candidate_sha": recomputed}


__all__ = [
    "HarnessError", "RESEARCH_COMPONENTS", "ComponentSpec", "HarnessBundle",
    "base_harness", "derive_candidate", "assert_single_method_change",
    "freeze_bundle", "load_frozen_bundle", "BASE_HARNESS_ID",
]
