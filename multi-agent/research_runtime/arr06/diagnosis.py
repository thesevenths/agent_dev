"""ARR-06 RQ discovery: real LLM research agent + INDEPENDENT judge.

The judge is a separate role/call from the researcher; novelty is never
self-certified (no literature retrieval available -> UNVERIFIED).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config  # noqa: F401
from research_runtime.llm_client import llm_config, make_generate

from .core import assert_no_leak


class ResearchAgent:
    def __init__(self, *, out_dir: str | Path, generate=None):
        cfg = llm_config()
        if not cfg.get("api_key"):
            raise RuntimeError("no LLM key")
        self._gen = generate or make_generate()
        self.model = cfg["model"]
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def propose(self, *, blind: dict[str, Any]) -> str:
        assert_no_leak(blind)
        prompt = json.dumps({
            "role": "researcher", "task": "propose_research_question",
            "observations": blind,
            "requirements": ["state a falsifiable hypothesis", "name competing hypotheses",
                             "specify a discriminative experiment", "predict outcomes",
                             "state falsification condition", "do NOT claim novelty"],
            "output": "a concise structured research question as text"},
            ensure_ascii=False)
        try:
            raw = self._gen(prompt)
        except Exception:
            raw = ""
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False, default=str)
        (self.out / "researcher.txt").write_text(raw, encoding="utf-8")
        return raw[:2000]


def judge(research_question: str, *, out_dir: str | Path) -> dict[str, Any] | None:
    """Independent LLM judge — different prompt role, not the researcher."""
    prompt = json.dumps({
        "role": "independent_blind_judge",
        "task": "score_research_question",
        "research_question": research_question,
        "dimensions": ["research_value", "falsifiability", "novelty",
                       "feasibility", "expected_information_gain",
                       "experimental_discriminability"],
        "rules": ["score each 0..1", "novelty MUST be 0 because no literature "
                  "retrieval was performed (UNVERIFIED)",
                  "return JSON with scores dict and mean"],
    }, ensure_ascii=False)
    try:
        raw = make_generate()(prompt)
    except Exception:
        return None
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False, default=str)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge.json").write_text(raw, encoding="utf-8")
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines()
                         if not l.strip().startswith("```")).strip()
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return None
    return d if isinstance(d, dict) else None


__all__ = ["ResearchAgent", "judge"]
