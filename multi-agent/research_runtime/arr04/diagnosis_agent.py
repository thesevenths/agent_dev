"""ARR-04 diagnosis agent: real configured LLM, passive + active (discriminative
experiment) diagnosis with evidence sufficiency and UNKNOWN abstention."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config  # noqa: F401
from research_runtime.llm_client import llm_config, make_generate

from .benchmark import CAUSES, DIAGNOSTIC_EXPERIMENTS
from .core import assert_no_leak, sha256_json


@dataclass
class Diagnosis:
    ok: bool
    primary_cause: str
    confidence: float
    candidates: list[str] = field(default_factory=list)
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    evidence_sufficient: bool = False
    chosen_experiment: str = ""
    experiment_rationale: str = ""
    abstain: bool = False
    generator: str = "ARR.LLM"
    model: str = ""
    prompt_sha256: str = ""
    response_sha256: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DiagnosisAgent:
    """Real ARR LLM bound to an evidence-grounded diagnosis schema."""

    def __init__(self, *, out_dir: str | Path, generate=None):
        cfg = llm_config()
        if not cfg.get("api_key"):
            raise RuntimeError("configured LLM gateway has no API key")
        self._generate = generate or make_generate()
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last_prompt = ""
        self._last_response = ""

    def _record(self, prompt: str) -> str:
        self._last_prompt = prompt
        raw = self._generate(prompt)
        if not isinstance(raw, str):
            raw = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        self._last_response = raw
        (self.out_dir / "agent_llm_interaction.json").write_text(json.dumps(
            {"mode": self.out_dir.name,
             "prompt_sha256": sha256_json({"prompt": prompt}),
             "response_sha256": sha256_json({"response": raw}),
             "prompt": prompt, "response": raw},
            indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return raw

    def _call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        prompt = json.dumps(payload, ensure_ascii=False, default=str)
        assert_no_leak({"payload": payload.get("evidence")})
        try:
            raw = self._record(prompt)
        except Exception:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(ln for ln in text.splitlines()
                             if not ln.strip().startswith("```")).strip()
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _cause(x: Any) -> str:
        c = str(x or "UNKNOWN")
        return c if c in CAUSES or c == "UNKNOWN" else "UNKNOWN"

    def passive(self, *, task_id: str, evidence: dict[str, Any]) -> Diagnosis:
        payload = {
            "task": "passive_scientific_diagnosis", "task_id": task_id,
            "evidence": evidence,
            "allowed_causes": list(CAUSES) + ["UNKNOWN"],
            "experiments": [{"id": e.experiment_id, "description": e.description}
                            for e in DIAGNOSTIC_EXPERIMENTS],
            "output_schema": {
                "primary_cause": "one of allowed_causes",
                "confidence": "0..1 calibrated to evidence",
                "candidate_ranking": "list of causes most-to-least likely",
                "evidence_for": "[concrete metric/config citations]",
                "evidence_against": "[concrete citations]",
                "missing_evidence": "[what would disambiguate]",
                "evidence_sufficient": "bool; false if ambiguous",
                "chosen_experiment": "one experiment id or NONE; choose the "
                                     "experiment that best discriminates your "
                                     "top competing hypotheses",
                "experiment_rationale": "why this experiment",
                "abstain": "true if evidence insufficient -> primary_cause=UNKNOWN",
            },
            "rules": ["Judge only from evidence, never from hidden labels.",
                      "If multiple causes fit, set evidence_sufficient=false and "
                      "choose a discriminative experiment.",
                      "Return ONLY one JSON object."],
        }
        d = self._call(payload)
        if d is None:
            return Diagnosis(False, "UNKNOWN", 0.0, abstain=True,
                             generator=self.cfg["model"], model=self.cfg["model"],
                             prompt_sha256=sha256_json({"prompt": payload}),
                             response_sha256=sha256_json({"response": self._last_response}),
                             error="invalid LLM JSON")
        cause = self._cause(d.get("primary_cause"))
        abstain = bool(d.get("abstain", False)) or cause == "UNKNOWN"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        ranking = [self._cause(x) for x in (d.get("candidate_ranking") or [])]
        chosen = str(d.get("chosen_experiment") or "NONE")
        if chosen not in {e.experiment_id for e in DIAGNOSTIC_EXPERIMENTS}:
            chosen = "NONE"
        return Diagnosis(True, "UNKNOWN" if abstain else cause, conf,
                         candidates=ranking or [cause],
                         evidence_for=[str(x) for x in d.get("evidence_for") or []],
                         evidence_against=[str(x) for x in d.get("evidence_against") or []],
                         missing_evidence=[str(x) for x in d.get("missing_evidence") or []],
                         evidence_sufficient=bool(d.get("evidence_sufficient", False)),
                         chosen_experiment=chosen,
                         experiment_rationale=str(d.get("experiment_rationale") or ""),
                         abstain=abstain, generator=self.cfg["model"],
                         model=self.cfg["model"],
                         prompt_sha256=sha256_json({"prompt": payload}),
                         response_sha256=sha256_json({"response": self._last_response}))

    def updated(self, *, task_id: str, evidence: dict[str, Any],
                passive: Diagnosis, experiment_id: str,
                experiment_result: dict[str, Any]) -> Diagnosis:
        payload = {
            "task": "belief_update_after_discriminative_experiment",
            "task_id": task_id, "original_evidence": evidence,
            "passive_diagnosis": passive.to_dict(),
            "experiment_run": {"experiment_id": experiment_id,
                               "normalized_improvement": experiment_result.get(
                                   "normalized_improvement"),
                               "post_metrics": experiment_result.get("metrics", {})},
            "allowed_causes": list(CAUSES) + ["UNKNOWN"],
            "output_schema": {
                "primary_cause": "one of allowed_causes (updated belief)",
                "confidence": "0..1", "candidate_ranking": "[causes]",
                "evidence_for": "[citations including experiment result]",
                "evidence_against": "[citations]", "missing_evidence": "[strings]",
                "information_gain": "brief description of what the experiment resolved",
                "abstain": "true if still insufficient",
            },
            "rules": ["Update belief using the experiment outcome as new evidence.",
                      "A large improvement from an experiment whose registered "
                      "mechanism matches a cause is strong evidence for that cause.",
                      "Return ONLY one JSON object."],
        }
        d = self._call(payload)
        if d is None:
            return Diagnosis(False, passive.primary_cause, passive.confidence,
                             candidates=passive.candidates, abstain=True,
                             generator=self.cfg["model"], model=self.cfg["model"],
                             error="invalid LLM JSON (update)")
        cause = self._cause(d.get("primary_cause"))
        abstain = bool(d.get("abstain", False)) or cause == "UNKNOWN"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        return Diagnosis(True, "UNKNOWN" if abstain else cause, conf,
                         candidates=[self._cause(x) for x in
                                     (d.get("candidate_ranking") or [])] or [cause],
                         evidence_for=[str(x) for x in d.get("evidence_for") or []],
                         evidence_against=[str(x) for x in d.get("evidence_against") or []],
                         missing_evidence=[str(x) for x in d.get("missing_evidence") or []],
                         evidence_sufficient=True, abstain=abstain,
                         generator=self.cfg["model"], model=self.cfg["model"],
                         prompt_sha256=sha256_json({"prompt": payload}),
                         response_sha256=sha256_json({"response": self._last_response}))


__all__ = ["Diagnosis", "DiagnosisAgent"]
