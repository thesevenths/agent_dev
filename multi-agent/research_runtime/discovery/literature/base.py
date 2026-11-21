"""M7 -- literature domain model and provider contract (framework-agnostic)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any


_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def normalize_doi(value: str) -> str:
    """Accept both bare DOIs and official resolver URLs returned by APIs."""
    raw = (value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]
            break
    return raw.strip().lower()


class LiteratureError(RuntimeError):
    """Provider failure or malformed record; never fabricate metadata."""


@dataclass(frozen=True)
class ResearchQuestionInput:
    question: str
    domain: str
    concepts: list[str] = field(default_factory=list)
    target_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    text: str
    strategy: str          # exact_phrase | synonym | technical | related_method |
                           # alternative_explanation | contradictory
    provider: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Paper:
    paper_id: str
    doi: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    source: str
    url: str
    citations: int
    references: list[str]
    retrieved_at: str
    query_id: str
    content_sha: str = ""
    metadata_sha: str = ""
    full_text_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(
            json.dumps({"title": self.title, "abstract": self.abstract,
                        "authors": self.authors, "year": self.year},
                       sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LiteratureProvider:
    """Interface all real providers implement (no web scraping)."""
    name = "base"

    def search(self, query: SearchQuery, *, limit: int,
               budget: dict[str, Any]) -> list[Paper]:
        raise NotImplementedError


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def paper_identity(p: Paper) -> str:
    """DOI first, then normalized title + year + first author."""
    doi = normalize_doi(p.doi)
    if doi:
        return "doi:" + doi
    first = p.authors[0].lower().strip() if p.authors else ""
    return f"title:{normalize_title(p.title)}|{p.year}|{first}"


def validate_paper(p: Paper) -> dict[str, Any]:
    """Fail closed on forged DOI or a full-text claim backed only by metadata."""
    problems: list[str] = []
    if not p.title.strip():
        problems.append("missing_title")
    doi = normalize_doi(p.doi)
    if p.doi and not _DOI_RE.match(doi):
        problems.append("invalid_doi")
    if p.full_text_available and p.content_sha == p.content_digest:
        problems.append("full_text_claim_without_full_text")
    return {"ok": not problems, "reasons": problems}


def deduplicate(papers: list[Paper]) -> tuple[list[Paper], list[str]]:
    """Collapse the SAME paper appearing in multiple providers."""
    seen: dict[str, Paper] = {}
    dropped: list[str] = []
    for p in papers:
        key = paper_identity(p)
        if key in seen:
            dropped.append(key)
            # Prefer the record with more metadata; never count as a new paper.
            cur = seen[key]
            if (p.citations > cur.citations
                    or (len(p.abstract) > len(cur.abstract))):
                seen[key] = p
            continue
        seen[key] = p
    return list(seen.values()), dropped


__all__ = ["LiteratureError", "ResearchQuestionInput", "SearchQuery",
           "Paper", "LiteratureProvider", "normalize_title",
           "paper_identity", "deduplicate"]
