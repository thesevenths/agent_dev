"""ARR_EMB_RSI_02 -- actual ARR LLM-backed research proposer.

This module invokes the REAL ARR ``LLMProposer`` from
``research_runtime.hypothesis_manager`` with the project's configured LLM
gateway (``config.py`` / ``llm_client.make_generate``). Hardcoded diagnosis
rules live only in the Rule-Based baseline controller, never in the ARR arm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config  # noqa: F401  -- loads .env into os.environ
from research_runtime.hypothesis_manager import LLMProposer
from research_runtime.llm_client import llm_config, make_generate
from research_runtime.schemas import ExperimentResult

from .integrity import sha256_json


@dataclass
class AgentInvocation:
    ok: bool
    hypotheses: list[dict[str, Any]]
    generator: str
    model: str
    base_url: str
    allowed_keys: list[str]
    prompt_sha256: str
    raw_response_sha256: str
    rejections: list[str]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "hypotheses": self.hypotheses,
                "generator": self.generator, "model": self.model,
                "base_url": self.base_url, "allowed_keys": self.allowed_keys,
                "prompt_sha256": self.prompt_sha256,
                "raw_response_sha256": self.raw_response_sha256,
                "rejections": self.rejections, "error": self.error}


class ActualARRResearchAgent:
    """Thin binding of the real ARR LLMProposer to a blind evidence payload."""

    def __init__(self, *, allowed_keys: list[str], out_dir: str | Path,
                 generate=None):
        cfg = llm_config()
        if not cfg.get("api_key"):
            raise RuntimeError("configured LLM gateway has no API key")
        self._generate = generate or make_generate()
        self._proposer = LLMProposer(generate=self._generate,
                                     allowed_intervention_keys=allowed_keys)
        self._prompt_sha = ""
        self._raw_sha = ""
        self._raw = ""
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _recorded_generate(self, prompt: str) -> str:
        self._prompt_sha = sha256_json({"prompt": prompt})
        raw = self._generate(prompt)
        self._raw = raw if isinstance(raw, str) else json.dumps(
            raw, sort_keys=True, ensure_ascii=False, default=str)
        self._raw_sha = sha256_json({"response": self._raw})
        (self.out_dir / "agent_llm_interaction.json").write_text(
            json.dumps({"prompt_sha256": self._prompt_sha,
                        "response_sha256": self._raw_sha,
                        "prompt": prompt, "response": self._raw},
                       indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
        return raw

    def diagnose(self, *, case_id: str, metrics: dict[str, float],
                 evidence_refs: list[str], objective: str) -> AgentInvocation:
        cfg = llm_config()
        # Bind the REAL generator once (wrapped for provenance).
        if getattr(self, "_bound", False) is False:
            self._proposer.generate = self._recorded_generate
            self._bound = True
        result = ExperimentResult(
            experiment_id=f"blind_{case_id}", job_id=f"blind_{case_id}",
            status="succeeded", engine="scinet++", stage="evaluate",
            metrics={k: float(v) for k, v in metrics.items()})
        try:
            hypotheses = self._proposer.propose(
                result, [], objective_id=f"obj_{case_id}",
                objective_context={
                    "objective": objective,
                    "evidence_refs": evidence_refs,
                    "instructions": (
                        "Return ONE hypothesis. Root cause must be one of: "
                        "insufficient_model_capacity, "
                        "insufficient_data_coverage, "
                        "dynamics_distribution_shift, "
                        "optimization_instability. "
                        "intervention MUST be a direct JSON object mapping "
                        "exactly one allowed dotted config key to its value "
                        "(e.g. {\"train.lr\": 0.0005}), NOT a nested "
                        "{key,value} object. "
                        "predicted_effects MUST be a JSON list of "
                        "objects with metric and direction. "
                        "falsification_criteria MUST be a JSON list of strings. "
                        "Include confidence and evidence_refs. "
                        "Use only these allowed key types/ranges: "
                        "model.predictor_hidden=list[int] with widths 8-64; "
                        "data.train_samples=int in [4,128]; "
                        "train.lr=float in [0.0001,0.01]. "
                        "Do not propose values outside these ranges.")} )
        except Exception as exc:
            return AgentInvocation(False, [], "ARR.LLMProposer", cfg["model"],
                                   cfg["base_url"], sorted(
                                       self._proposer.allowed or ()),
                                   self._prompt_sha, self._raw_sha,
                                   [], f"{type(exc).__name__}: {exc}")
        items = [h.model_dump() for h in hypotheses]
        for item in items:
            text = " ".join(str(item.get(k) or "") for k in (
                "statement", "mechanism", "title"))
            cause = ""
            for candidate in ("insufficient_model_capacity",
                              "insufficient_data_coverage",
                              "dynamics_distribution_shift",
                              "optimization_instability"):
                if candidate in text:
                    cause = candidate
                    break
            item["arr_root_cause"] = cause
            item["arr_intervention"] = dict(item.get("intervention") or {})
        return AgentInvocation(
            bool(items), items, "ARR.LLMProposer", cfg["model"],
            cfg["base_url"], sorted(self._proposer.allowed or ()),
            self._prompt_sha, self._raw_sha,
            list(getattr(self._proposer, "rejections", []) or []),
            "" if items else "agent produced no accepted hypothesis")


__all__ = ["ActualARRResearchAgent", "AgentInvocation"]
