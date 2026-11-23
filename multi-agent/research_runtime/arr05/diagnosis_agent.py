"""ARR-05 diagnosis agent: real configured LLM; evidence-grounded diagnosis,
abstention, evidence sufficiency, and confidence suitable for temperature scaling."""
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
    cause_probabilities: dict[str, float] = field(default_factory=dict)
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
        return dict(self.__dict__)


class DiagnosisAgent:
    def __init__(self, *, out_dir: str | Path, generate=None, memory=None):
        cfg = llm_config()
        if not cfg.get("api_key"):
            raise RuntimeError("configured LLM gateway has no API key")
        self._generate = generate or make_generate()
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory

    def _call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        assert_no_leak({"evidence": payload.get("evidence")})
        prompt = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            raw = self._generate(prompt)
        except Exception:
            return None
        if not isinstance(raw, str):
            raw = json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str)
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.splitlines()
                             if not l.strip().startswith("```")).strip()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "agent_llm_interaction.json").write_text(json.dumps(
            {"prompt_sha256": sha256_json({"prompt": prompt}),
             "response_sha256": sha256_json({"response": raw}),
             "prompt": prompt, "response": raw},
            indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        try:
            d = json.loads(text)
        except (ValueError, TypeError):
            return None
        return d if isinstance(d, dict) else None

    @staticmethod
    def _cause(x: Any) -> str:
        c = str(x or "UNKNOWN")
        return c if c in CAUSES or c == "UNKNOWN" else "UNKNOWN"

    def _base_payload(self, task_id: str, evidence: dict[str, Any],
                      mode: str, schema: dict[str, Any], rules: list[str]) -> dict:
        p = {"task": mode, "task_id": task_id, "evidence": evidence,
             "allowed_causes": list(CAUSES) + ["UNKNOWN"],
             "experiments": [{"id": e.experiment_id, "description": e.description}
                             for e in DIAGNOSTIC_EXPERIMENTS],
             "output_schema": schema, "rules": rules}
        if self.memory:
            p["verified_research_memory"] = self.memory
        return p

    def passive(self, *, task_id: str, evidence: dict[str, Any]) -> Diagnosis:
        schema = {
            "primary_cause": "one of allowed_causes",
            "confidence": "0..1; your subjective probability the primary cause is correct",
            "cause_probabilities": {c: "0..1 probability" for c in CAUSES},
            "candidate_ranking": "[causes most-to-least likely]",
            "evidence_for": "[concrete metric/config citations]",
            "evidence_against": "[concrete citations]",
            "missing_evidence": "[what would disambiguate]",
            "evidence_sufficient": "bool; false if evidence is ambiguous/insufficient",
            "chosen_experiment": "experiment id that best discriminates your top "
                                 "competing hypotheses, or NONE",
            "experiment_rationale": "why this experiment",
            "abstain": "true if evidence insufficient -> primary_cause=UNKNOWN",
        }
        rules = ["Judge ONLY from evidence, never hidden labels.",
                 "If multiple causes fit, set evidence_sufficient=false and choose "
                 "a discriminative experiment; do not force a conclusion.",
                 "cause_probabilities must reflect your full belief distribution.",
                 "Return ONLY one JSON object."]
        d = self._call(self._base_payload(task_id, evidence,
                                          "passive_diagnosis", schema, rules))
        return self._parse(d, task_id, evidence, "passive")

    def updated(self, *, task_id: str, evidence: dict[str, Any],
                passive: Diagnosis, experiment_id: str,
                experiment_result: dict[str, Any]) -> Diagnosis:
        schema = {
            "primary_cause": "updated belief (one of allowed_causes)",
            "confidence": "0..1 updated subjective probability",
            "cause_probabilities": {c: "0..1 updated probability" for c in CAUSES},
            "candidate_ranking": "[causes]", "evidence_for": "[citations incl. experiment]",
            "evidence_against": "[citations]", "missing_evidence": "[strings]",
            "belief_change": "what the experiment resolved or changed",
            "abstain": "true if still insufficient",
        }
        rules = ["Update belief using the experiment outcome as NEW evidence.",
                 "A large improvement from an experiment whose mechanism matches "
                 "a cause is strong evidence for that cause.",
                 "Return ONLY one JSON object."]
        payload = self._base_payload(task_id, evidence, "belief_update", schema, rules)
        payload["passive_diagnosis"] = passive.to_dict()
        payload["experiment"] = {
            "experiment_id": experiment_id,
            "normalized_improvement": experiment_result.get("normalized_improvement"),
            "post_metrics": experiment_result.get("metrics", {})}
        d = self._call(payload)
        return self._parse(d, task_id, evidence, "updated", fallback=passive)

    def _parse(self, d: dict[str, Any] | None, task_id: str,
               evidence: dict[str, Any], mode: str,
               fallback: Diagnosis | None = None) -> Diagnosis:
        if d is None:
            return Diagnosis(False, "UNKNOWN", 0.0, abstain=True,
                             generator=self.cfg["model"], model=self.cfg["model"],
                             error=f"invalid LLM JSON ({mode})")
        cause = self._cause(d.get("primary_cause"))
        abstain = bool(d.get("abstain", False)) or cause == "UNKNOWN"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        probs = {}
        rawp = d.get("cause_probabilities") or {}
        if isinstance(rawp, dict):
            for c in CAUSES:
                try:
                    probs[c] = max(0.0, min(1.0, float(rawp.get(c, 0.0))))
                except (TypeError, ValueError):
                    probs[c] = 0.0
        if not probs:
            probs = {c: (conf if c == cause else (1 - conf) / max(1, len(CAUSES) - 1))
                     for c in CAUSES}
        chosen = str(d.get("chosen_experiment") or "NONE")
        if chosen not in {e.experiment_id for e in DIAGNOSTIC_EXPERIMENTS}:
            chosen = "NONE"
        return Diagnosis(
            True, "UNKNOWN" if abstain else cause, conf,
            candidates=[self._cause(x) for x in (d.get("candidate_ranking") or [])]
                       or [cause],
            cause_probabilities=probs,
            evidence_for=[str(x) for x in d.get("evidence_for") or []],
            evidence_against=[str(x) for x in d.get("evidence_against") or []],
            missing_evidence=[str(x) for x in d.get("missing_evidence") or []],
            evidence_sufficient=bool(d.get("evidence_sufficient", mode == "updated")),
            chosen_experiment=chosen,
            experiment_rationale=str(d.get("experiment_rationale") or ""),
            abstain=abstain, generator=self.cfg["model"], model=self.cfg["model"])


__all__ = ["Diagnosis", "DiagnosisAgent"]
