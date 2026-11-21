"""M7 -- REAL literature API providers (official APIs only, no scraping).

OpenAlex, Crossref and Semantic Scholar. Each implements timeout, retry with
exponential backoff, HTTP 429 rate-limit handling, pagination, an on-disk cache
and a request budget. Provider failures are recorded, NEVER filled with fake
papers. Unit tests inject a fake transport; the real E2E uses the real HTTP one.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .base import LiteratureError, LiteratureProvider, Paper, SearchQuery

Transport = Callable[[str, dict[str, str], float], str]


def http_transport(url: str, headers: dict[str, str], timeout: float) -> str:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


class _BaseAPIProvider(LiteratureProvider):
    def __init__(self, *, transport: Transport | None = None,
                 cache_dir: str | Path | None = None,
                 max_retries: int = 3, backoff_base: float = 1.0):
        self.transport = transport or http_transport
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.events: list[dict[str, Any]] = []

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": "ARR-M7/1.0 (mailto:research@softsafe-tech.com)",
                "Accept": "application/json"}

    def _get(self, url: str, budget: dict[str, Any]) -> dict[str, Any]:
        if budget.get("requests_used", 0) >= budget.get("max_requests", 20):
            raise LiteratureError("request budget exhausted")
        cache_key = hashlib.sha256(url.encode()).hexdigest()[:20]
        cache_file = self.cache_dir / f"{cache_key}.json" if self.cache_dir else None
        if cache_file and cache_file.is_file():
            self.events.append({"provider": self.name, "url": url,
                                "cache": "hit"})
            return json.loads(cache_file.read_text(encoding="utf-8"))
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.transport(url, self._headers(),
                                     float(budget.get("timeout_s", 20)))
                data = json.loads(raw)
                budget["requests_used"] = budget.get("requests_used", 0) + 1
                self.events.append({"provider": self.name, "url": url,
                                    "status": "ok", "attempt": attempt})
                if cache_file:
                    cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 429:
                    wait = float(exc.headers.get("Retry-After",
                                                 self.backoff_base * 2 ** attempt))
                    self.events.append({"provider": self.name, "url": url,
                                        "status": "rate_limited",
                                        "retry_after": wait})
                    time.sleep(min(wait, 8))
                elif exc.code >= 500:
                    time.sleep(min(self.backoff_base * 2 ** attempt, 8))
                else:
                    break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(min(self.backoff_base * 2 ** attempt, 8))
        self.events.append({"provider": self.name, "url": url,
                            "status": "failed",
                            "error": f"{type(last).__name__}: {last}"})
        raise LiteratureError(f"{self.name} request failed: {last}")

    def _paper(self, *, pid: str, doi: str, title: str, authors: list[str],
               year: int, abstract: str, url: str, citations: int,
               references: list[str], query_id: str) -> Paper:
        metadata = {"title": title, "authors": authors, "year": year,
                    "doi": doi, "url": url, "citations": citations}
        return Paper(
            paper_id=pid, doi=doi, title=title, authors=authors, year=year,
            abstract=abstract or "", source=self.name, url=url,
            citations=int(citations or 0), references=list(references or []),
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            query_id=query_id,
            content_sha=hashlib.sha256((abstract or "").encode()).hexdigest(),
            metadata_sha=hashlib.sha256(json.dumps(
                metadata, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            full_text_available=False)


class OpenAlexProvider(_BaseAPIProvider):
    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def search(self, query: SearchQuery, *, limit: int,
               budget: dict[str, Any]) -> list[Paper]:
        params = {"search": query.text, "per-page": str(min(limit, 50))}
        cursor = "*"
        out: list[Paper] = []
        while len(out) < limit:
            params["cursor"] = cursor
            url = self.BASE + "?" + urllib.parse.urlencode(params)
            data = self._get(url, budget)
            for w in data.get("results", []):
                authors = [a.get("author", {}).get("display_name", "")
                           for a in (w.get("authorships") or [])]
                out.append(self._paper(
                    pid=str(w.get("id", "")), doi=str(w.get("doi") or ""),
                    title=str(w.get("title") or w.get("display_name") or ""),
                    authors=[a for a in authors if a],
                    year=int(w.get("publication_year") or 0),
                    abstract=str(w.get("abstract_inverted_index") or ""),
                    url=str(w.get("doi") or w.get("id") or ""),
                    citations=int((w.get("cited_by_count") or 0)),
                    references=list(w.get("referenced_works") or []),
                    query_id=query.query_id))
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor or len(out) >= limit:
                break
        return out[:limit]


class CrossrefProvider(_BaseAPIProvider):
    name = "crossref"
    BASE = "https://api.crossref.org/works"

    def search(self, query: SearchQuery, *, limit: int,
               budget: dict[str, Any]) -> list[Paper]:
        params = {"query": query.text, "rows": str(min(limit, 50)),
                  "select": "DOI,title,author,published,abstract,URL,"
                            "is-referenced-by-count,reference"}
        url = self.BASE + "?" + urllib.parse.urlencode(params)
        data = self._get(url, budget)
        items = ((data.get("message") or {}).get("items") or [])
        out: list[Paper] = []
        for w in items:
            year = 0
            parts = ((w.get("published") or {}).get("date-parts") or [[0]])
            if parts and parts[0]:
                year = int(parts[0][0] or 0)
            authors = [" ".join(filter(None, [a.get("given"), a.get("family")]))
                       for a in (w.get("author") or [])]
            out.append(self._paper(
                pid=str(w.get("DOI") or ""), doi=str(w.get("DOI") or ""),
                title=" ".join(w.get("title") or []),
                authors=[a for a in authors if a], year=year,
                abstract=str(w.get("abstract") or ""),
                url=str(w.get("URL") or ""),
                citations=int(w.get("is-referenced-by-count") or 0),
                references=[str(r.get("DOI") or "") for r in
                            (w.get("reference") or []) if isinstance(r, dict)],
                query_id=query.query_id))
        return out[:limit]


class SemanticScholarProvider(_BaseAPIProvider):
    name = "semantic_scholar"
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: SearchQuery, *, limit: int,
               budget: dict[str, Any]) -> list[Paper]:
        params = {"query": query.text, "limit": str(min(limit, 50)),
                  "fields": "paperId,externalIds,title,authors,year,abstract,"
                            "url,citationCount,references"}
        url = self.BASE + "?" + urllib.parse.urlencode(params)
        data = self._get(url, budget)
        out: list[Paper] = []
        for w in data.get("data", []):
            ext = w.get("externalIds") or {}
            out.append(self._paper(
                pid=str(w.get("paperId") or ""), doi=str(ext.get("DOI") or ""),
                title=str(w.get("title") or ""),
                authors=[str(a.get("name") or "")
                         for a in (w.get("authors") or [])],
                year=int(w.get("year") or 0),
                abstract=str(w.get("abstract") or ""),
                url=str(w.get("url") or ""),
                citations=int(w.get("citationCount") or 0),
                references=[str(r.get("paperId") or "") for r in
                            (w.get("references") or [])],
                query_id=query.query_id))
        return out[:limit]


PROVIDERS = {"openalex": OpenAlexProvider, "crossref": CrossrefProvider,
             "semantic_scholar": SemanticScholarProvider}

__all__ = ["Transport", "http_transport", "LiteratureProvider",
           "OpenAlexProvider", "CrossrefProvider", "SemanticScholarProvider",
           "PROVIDERS"]
