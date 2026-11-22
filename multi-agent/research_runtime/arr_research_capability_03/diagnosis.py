"""ARR-03 M1 -- evidence-grounded diagnosis via the REAL configured ARR LLM.

The agent uses the same configured gateway (config.py / llm_client.make_generate)
and the same model as the rest of ARR. It is NOT a hardcoded rule. Hardcoded
Rule-Based diagnosis is a separate baseline controller in evaluation.py.

Improvements over V2 (per M1 requirements), implemented in the prompt+parser,
not as new fixed rules:
  * structured evidence interpretation (metrics + normalized config + curves)
  * explicit alternative-hypothesis and evidence-against reasoning
  * a diagnostic experiment and falsification condition
  * confidence calibration signal and UNKNOWN abstention
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config  # noqa: F401 -- loads .env
from research_runtime.llm_client import llm_config, make_generate

from .benchmark import CAUSES
from .integrity import assert_no_ground_truth, sha256_json


@dataclass
class DiagnosisOutput:
    ok: bool
    case_id: str
    primary_cause: str            # one of CAUSES or "UNKNOWN"
    confidence: float
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    diagnostic_experiment: str = ""
    expected_outcome: str = ""
    proposed_cause: str = ""      # cause whose registered intervention is chosen
    falsification_condition: str = ""
    abstain: bool = False
    generator: str = "ARR.LLM"
    model: str = ""
    base_url: str = ""
    prompt_sha256: str = ""
    response_sha256: str = ""
    rejections: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "case_id": self.case_id,
            "primary_cause": self.primary_cause, "confidence": self.confidence,
            "alternatives": self.alternatives, "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "missing_evidence": self.missing_evidence,
            "diagnostic_experiment": self.diagnostic_experiment,
            "expected_outcome": self.expected_outcome,
            "proposed_cause": self.proposed_cause,
            "falsification_condition": self.falsification_condition,
            "abstain": self.abstain, "generator": self.generator,
            "model": self.model, "base_url": self.base_url,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "rejections": self.rejections, "error": self.error,
        }


def build_agent_evidence(*, metrics: dict[str, float],
                         config_view: dict[str, Any],
                         artifacts: list[str]) -> dict[str, Any]:
    """Blind, normalized evidence view. Never contains hidden identity."""
    curves = {k: v for k, v in metrics.items()
              if "@" in k or "mse" in k or "cosine" in k or "loss" in k}
    view = {
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "model_and_data_configuration": config_view,
        "prediction_curves": curves,
        "verified_artifacts": list(artifacts),
        "instruction": ("All values are normalized; lower prediction error is "
                        "better. Infer the dominant failure mechanism from the "
                        "pattern. Configuration values are legitimate observed "
                        "evidence, not answers."),
    }
    assert_no_ground_truth(view)
    return view


class ArrDiagnosisAgent:
    """Real configured ARR LLM bound to an evidence-grounded diagnosis schema."""

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
            {"prompt_sha256": sha256_json({"prompt": prompt}),
             "response_sha256": sha256_json({"response": raw}),
             "prompt": prompt, "response": raw},
            indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return raw

    def _prompt(self, case_id: str, evidence: dict[str, Any],
                with_knowledge: str) -> str:
        return json.dumps({
            "task": "scientific_failure_diagnosis",
            "case": case_id,
            "evidence": evidence,
            "transferred_research_knowledge": with_knowledge,
            "allowed_root_causes": list(CAUSES) + ["UNKNOWN"],
            "output_schema": {
                "primary_cause": "one of allowed_root_causes",
                "confidence": "float 0..1; calibrate to evidence strength",
                "alternatives": "[{cause, why_rejected}]",
                "evidence_for": "[strings tying to specific metric/config values]",
                "evidence_against": "[strings]",
                "missing_evidence": "[strings]",
                "diagnostic_experiment": "one cheap experiment to disambiguate",
                "expected_outcome": "what that experiment should show",
                "proposed_intervention_cause": (
                    "root cause whose registered intervention you recommend"),
                "falsification_condition": "observation that would refute you",
                "abstain": "true if evidence is insufficient; then primary_cause=UNKNOWN",
            },
            "rules": [
                "Judge from evidence, not from any hidden label.",
                "You MUST cite concrete evidence values for your primary cause.",
                "List at least one alternative and why you rejected it.",
                "Set abstain=true and primary_cause=UNKNOWN if evidence is weak "
                "or contradictory; do not guess.",
                "Return ONLY a single JSON object with the output_schema keys.",
            ],
        }, ensure_ascii=False, default=str)

    @staticmethod
    def _parse(raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines()
                     if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def diagnose(self, *, case_id: str, evidence: dict[str, Any],
                 transferred_knowledge: str = "") -> DiagnosisOutput:
        prompt = self._prompt(case_id, evidence, transferred_knowledge)
        try:
            raw = self._record(prompt)
        except Exception as exc:
            return DiagnosisOutput(False, case_id, "UNKNOWN", 0.0,
                                   generator=self.cfg.get("model", "ARR.LLM"),
                                   model=self.cfg.get("model", ""),
                                   base_url=self.cfg.get("base_url", ""),
                                   prompt_sha256=sha256_json({"prompt": prompt}),
                                   error=f"{type(exc).__name__}: {exc}")
        data = self._parse(raw)
        rejects: list[str] = []
        if data is None:
            rejects.append("invalid JSON response")
            return DiagnosisOutput(
                False, case_id, "UNKNOWN", 0.0, rejections=rejects,
                generator=self.cfg.get("model", "ARR.LLM"),
                model=self.cfg.get("model", ""),
                base_url=self.cfg.get("base_url", ""),
                prompt_sha256=sha256_json({"prompt": prompt}),
                response_sha256=sha256_json({"response": self._last_response}),
                error="agent produced invalid JSON")
        cause = str(data.get("primary_cause") or "UNKNOWN")
        if cause not in CAUSES and cause != "UNKNOWN":
            rejects.append("cause outside registry -> abstain")
            cause = "UNKNOWN"
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        abstain = bool(data.get("abstain", False)) or cause == "UNKNOWN"
        if abstain:
            cause = "UNKNOWN"
        proposed = str(data.get("proposed_intervention_cause") or cause)
        if proposed not in CAUSES:
            proposed = "" if abstain else cause
        # Normalize alternatives to [{cause, why_rejected}] regardless of
        # whether the LLM emitted dicts or plain strings.
        raw_alts = data.get("alternatives") or []
        alts: list[dict[str, Any]] = []
        for alt in raw_alts:
            if isinstance(alt, dict):
                alts.append({"cause": str(alt.get("cause", "")),
                             "why_rejected": str(alt.get("why_rejected", ""))})
            else:
                alts.append({"cause": str(alt), "why_rejected": ""})
        return DiagnosisOutput(
            True, case_id, cause, conf,
            alternatives=alts,
            evidence_for=[str(x) for x in (data.get("evidence_for") or [])],
            evidence_against=[str(x) for x in (data.get("evidence_against") or [])],
            missing_evidence=[str(x) for x in (data.get("missing_evidence") or [])],
            diagnostic_experiment=str(data.get("diagnostic_experiment") or ""),
            expected_outcome=str(data.get("expected_outcome") or ""),
            proposed_cause=proposed,
            falsification_condition=str(data.get("falsification_condition") or ""),
            abstain=abstain,
            generator=self.cfg.get("model", "ARR.LLM"),
            model=self.cfg.get("model", ""),
            base_url=self.cfg.get("base_url", ""),
            prompt_sha256=sha256_json({"prompt": prompt}),
            response_sha256=sha256_json({"response": self._last_response}),
            rejections=rejects)


__all__ = ["ArrDiagnosisAgent", "DiagnosisOutput", "build_agent_evidence"]
