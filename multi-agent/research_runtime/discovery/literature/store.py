"""M7 -- query expansion, literature store and coverage accounting."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .base import (Paper, ResearchQuestionInput, SearchQuery, deduplicate,
                   validate_paper)


def _freeze(obj: Any, path: str | Path) -> dict[str, Any]:
    from research_runtime.harness_rsi.identity import freeze_json as _fj
    return _fj(obj, path)


STRATEGIES = ("exact_phrase", "synonym", "technical", "related_method",
              "alternative_explanation", "contradictory")

_SYNONYMS = {
    "world model": ["latent dynamics model", "learned simulator",
                    "action-conditioned dynamics"],
    "generalization": ["out-of-distribution", "ood robustness", "transfer"],
    "prediction": ["rollout", "forecasting", "multi-step error"],
    "planning": ["control", "decision making", "model-based planning"],
}
_CONTRADICTIONS = ["fails", "does not improve", "negative result",
                   "limited generalization"]


def expand_query(q: ResearchQuestionInput, *, domain_terms: list[str] | None = None
                 ) -> list[SearchQuery]:
    """Produce MULTIPLE documented queries; each is frozen with its own id."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    texts: list[tuple[str, str]] = [("exact_phrase", f'"{q.question}"')]
    for concept in q.concepts or [q.domain]:
        for syn in _SYNONYMS.get(concept.lower(), []):
            texts.append(("synonym", f"{concept} {syn}"))
        texts.append(("technical", f"{concept} {q.domain} empirical evaluation"))
        texts.append(("related_method", f"{concept} baseline comparison ablation"))
    for claim in q.target_claims or []:
        texts.append(("alternative_explanation", f"{claim} alternative mechanism"))
        texts.append(("contradictory", f"{claim} " + _CONTRADICTIONS[1]))
    for term in domain_terms or []:
        texts.append(("technical", f"{q.domain} {term}"))
    seen: set[str] = set()
    out: list[SearchQuery] = []
    for strategy, text in texts:
        if text in seen or not text.strip():
            continue
        seen.add(text)
        qid = "q_" + hashlib.sha256(
            f"{strategy}|{text}|{q.domain}".encode()).hexdigest()[:16]
        out.append(SearchQuery(query_id=qid, text=text, strategy=strategy,
                               provider="", created_at=ts))
    return out


class LiteratureStore:
    """Append-only paper/search/coverage ledger with SHA-bound artifacts."""

    def __init__(self, out_dir: str | Path):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact = self.root / "literature_store.json"
        self.obj: dict[str, Any] = (json.loads(self.artifact.read_text(
            encoding="utf-8")) if self.artifact.is_file() else {
                "version": "m7_literature_v1", "papers": [], "queries": [],
                "search_runs": [], "dropped_duplicates": [],
                "dropped_invalid": []})

    def record_queries(self, queries: list[SearchQuery]) -> None:
        existing = {q["query_id"] for q in self.obj["queries"]}
        for q in queries:
            if q.query_id not in existing:
                self.obj["queries"].append(q.to_dict())

    def add_papers(self, papers: list[Paper]) -> dict[str, Any]:
        accepted: list[Paper] = []
        for paper in papers:
            v = validate_paper(paper)
            if v["ok"]:
                accepted.append(paper)
            else:
                self.obj["dropped_invalid"].append({
                    "paper_id": paper.paper_id, "doi": paper.doi,
                    "reasons": v["reasons"]})
        unique, dropped = deduplicate(
            self._all_papers() + accepted)
        self.obj["papers"] = [p.to_dict() for p in unique]
        self.obj["dropped_duplicates"].extend(dropped)
        return {"n_unique": len(unique), "n_dropped": len(dropped),
                "n_invalid": len(papers) - len(accepted)}

    def _all_papers(self) -> list[Paper]:
        return [Paper(**p) for p in self.obj["papers"]]

    def record_run(self, *, provider: str, query: SearchQuery,
                   n_returned: int, events: list[dict[str, Any]],
                   error: str = "") -> None:
        self.obj["search_runs"].append({
            "provider": provider, "query": query.to_dict(),
            "n_returned": n_returned, "events": events,
            "error": error, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def finalize(self) -> dict[str, Any]:
        runs = self.obj["search_runs"]
        providers = sorted({r["provider"] for r in runs})
        ok_runs = [r for r in runs if r["n_returned"] > 0 and not r["error"]]
        coverage = {
            "version": "m7_literature_coverage_v1",
            "n_queries": len(self.obj["queries"]),
            "n_runs": len(runs),
            "providers": providers,
            "success_rate": (len(ok_runs) / len(runs)) if runs else 0.0,
            "n_unique_papers": len(self.obj["papers"]),
            "n_dropped_duplicates": len(self.obj["dropped_duplicates"]),
            "full_text_accessible": sum(1 for p in self.obj["papers"]
                                        if p.get("full_text_available")),
            "year_range": [min([p["year"] for p in self.obj["papers"]
                                if p["year"] > 0], default=0),
                           max([p["year"] for p in self.obj["papers"]],
                               default=0)],
            "limitations": ["abstract-only metadata; full text not fetched",
                            "S2 rate limits may reduce provider coverage"],
            "status": ("OK" if providers and ok_runs else "INSUFFICIENT"),
        }
        frozen = _freeze({**self.obj, "coverage": coverage}, self.artifact)
        return {"coverage": coverage, "sha256": frozen["sha256"],
                "path": frozen["path"]}


__all__ = ["STRATEGIES", "expand_query", "LiteratureStore"]
