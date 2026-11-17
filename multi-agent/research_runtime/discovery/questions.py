"""M7.1 -- evidence-grounded questions and honest value assessment.

The M7 baseline scored value from field presence and used default EIG priors.
That is not scientific judgment. This module makes missing evidence explicit:
dimensions without independent support are ``NOT_ASSESSED``, and EIG requires
an explicit hypothesis space, prior and observation model.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

VALUE_DIMENSIONS = ("scientific_significance", "practical_impact", "novelty",
                    "feasibility", "falsifiability", "expected_information_gain")
NOT_ASSESSED = "NOT_ASSESSED"


@dataclass(frozen=True)
class ResearchQuestion:
    question_id: str
    title: str
    domain: str
    motivation: str
    literature_gap: str
    existing_claims: list[str] = field(default_factory=list)
    related_experiences: list[str] = field(default_factory=list)
    expected_contribution: str = ""
    falsifiability_requirements: list[str] = field(default_factory=list)
    estimated_budget: dict[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)
    supporting_evidence: list[dict[str, Any]] = field(default_factory=list)
    source_coverage: str = "UNKNOWN"
    limitations: list[str] = field(default_factory=list)
    # M7.6 P1-A: explicit multi-paper synthesis fields. All optional to keep
    # on-disk schema and constructor compatibility with M7.1/M7.5 artifacts.
    prior_art_summary: str = ""
    contradictory_evidence: list[str] = field(default_factory=list)
    unresolved_assumptions: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    applicability_conditions: list[str] = field(default_factory=list)
    n_supporting_papers: int = 0
    synthesis_method: str = "SINGLE_SOURCE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


def _gap_text(gap: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(gap, str):
        return gap, {"text": gap, "evidence_type": "unstructured_gap"}
    text = str(gap.get("text") or gap.get("gap") or "")
    return text, dict(gap)


def generate_questions(*, domain: str,
                       novelty: dict[str, Any],
                       literature_gaps: list[str | dict[str, Any]],
                       failures: list[dict[str, Any]] | None = None,
                       experiences: list[dict[str, Any] | str] | None = None,
                       max_questions: int = 6) -> list[ResearchQuestion]:
    """Generate questions grounded in explicit evidence, never title strings."""
    sources: list[dict[str, Any]] = []
    # ---- multi-paper synthesis (M7.6 P1-A) -------------------------------
    # The M7.1 generator treated each abstract as an independent gap. Here we
    # first group evidence snippets by shared salient tokens (a deterministic,
    # auditable approximation of thematic synthesis -- NOT a claim of semantic
    # understanding), then build ONE evidence-backed gap per theme that cites
    # every supporting paper and flags cross-paper disagreement explicitly.
    lit_sources: list[dict[str, Any]] = []
    for gap in literature_gaps:
        text, meta = _gap_text(gap)
        if len(text.strip()) < 12:
            continue
        lit_sources.append({"text": text, "meta": meta})
    themes: list[dict[str, Any]] = []
    for item in lit_sources:
        toks = {t for t in __import__("re").findall(
            r"[a-z][a-z0-9]{3,}", item["text"].lower())
            if t not in {"this", "that", "with", "from", "than", "have",
                         "across", "using", "study", "results", "based"}}
        best = None
        best_overlap = 0.0
        for th in themes:
            if not th["tokens"] or not toks:
                continue
            overlap = len(th["tokens"] & toks) / len(th["tokens"] | toks)
            if overlap > best_overlap:
                best, best_overlap = th, overlap
        if best is not None and best_overlap >= 0.20:
            best["items"].append(item)
            best["tokens"] |= toks
        else:
            themes.append({"tokens": set(toks), "items": [item]})
    for th in themes:
        items = th["items"]
        # Representative text is the longest snippet; evidence aggregates ALL
        # source papers, so the resulting gap is multi-paper grounded.
        representative = max(items, key=lambda x: len(x["text"]))["text"]
        meta_all = [it["meta"] for it in items]
        paper_ids = [str(m.get("paper_id", "")) for m in meta_all
                     if m.get("paper_id")]
        # Cross-paper contradiction signal: opposing outcome words on the same
        # theme are surfaced as uncertainty, never silently averaged away.
        corpus = " ".join(it["text"].lower() for it in items)
        contradicts = []
        if ("improve" in corpus and
                any(w in corpus for w in ("fail", "no effect", "does not"))):
            contradicts.append(
                "papers report opposing outcome directions on this theme")
        sources.append({
            "kind": "literature_gap", "text": representative,
            "evidence": meta_all,
            "paper_ids": paper_ids,
            "contradictory_evidence": contradicts,
            "n_papers": len(paper_ids),
            "synthesis_method": "MULTI_PAPER_THEMATIC_TOKEN_SYNTHESIS",
            "unresolved_assumptions": [
                "abstract-level evidence cannot confirm experimental details"]})
    for f in failures or []:
        text = f"{f.get('reason','')}: {f.get('detail','')}".strip(": ")
        if len(text) >= 12:
            sources.append({"kind": "experiment_failure", "text": text,
                            "evidence": [dict(f)]})
    for e in experiences or []:
        text = e.get("claim", "") if isinstance(e, dict) else str(e)
        if len(text.strip()) >= 12:
            sources.append({"kind": "verified_experience", "text": text,
                            "evidence": [e if isinstance(e, dict)
                                         else {"text": e}]})
    for verdict in novelty.get("verdicts", []):
        if verdict.get("status") in ("POSSIBLE_PRIOR_ART", "NOVELTY_UNCERTAIN"):
            sources.append({"kind": "novelty_gap",
                            "text": str(verdict.get("rationale", "")),
                            "evidence": [{"claim_id": verdict.get("claim_id")}]})

    out: list[ResearchQuestion] = []
    for i, src in enumerate(sources[:max_questions]):
        text = src["text"]
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        qid = "rq_" + hashlib.sha256(
            f"{domain}|{src['kind']}|{text}".encode()).hexdigest()[:16]
        coverage = ("ABSTRACT_ONLY" if novelty.get("source_coverage")
                    == "ABSTRACT_ONLY" else novelty.get("literature_check",
                                                        "UNKNOWN"))
        limits = ["evidence is limited to retrieved metadata/abstracts"]
        if src["kind"] == "verified_experience":
            limits.append("memory applies only under its recorded conditions")
        out.append(ResearchQuestion(
            question_id=qid,
            title=f"{domain}: {text.strip()[:100]}",
            domain=domain,
            motivation=f"Derived from real {src['kind']}: {text[:240]}",
            literature_gap=text if src["kind"] in (
                "literature_gap", "novelty_gap") else "",
            existing_claims=[v.get("claim_id", "") for v in
                             novelty.get("verdicts", [])
                             if v.get("status") == "NOT_NOVEL"],
            related_experiences=[e.get("memory_id", str(e)[:40])
                                 for e in (experiences or [])
                                 if src["kind"] == "verified_experience"],
            expected_contribution=(
                f"Resolve {src['kind']} with a falsifiable experiment"),
            falsifiability_requirements=[
                "state one observable metric and predicted direction",
                "state an executable result that would refute the mechanism"],
            estimated_budget={"max_campaign_rounds": 1, "max_jobs": 8},
            uncertainty=0.5,
            evidence_refs=[f"{src['kind']}:{digest}"]
                          + [f"paper:{pid}" for pid in src.get("paper_ids", [])],
            supporting_evidence=src["evidence"],
            source_coverage=coverage,
            limitations=limits,
            prior_art_summary=("; ".join(src.get("paper_ids", []))
                               if src.get("paper_ids") else
                               "no prior-art IDs retrievable from evidence"),
            contradictory_evidence=list(src.get("contradictory_evidence", [])),
            unresolved_assumptions=list(src.get("unresolved_assumptions", [])),
            alternative_explanations=[
                "apparent effect is an artifact of the shared experimental setup"],
            applicability_conditions=[
                "conclusions apply only to the retrieved domain and conditions"],
            n_supporting_papers=int(src.get("n_papers", 0)),
            synthesis_method=src.get("synthesis_method", "SINGLE_SOURCE")))
    return out


@dataclass(frozen=True)
class ValueAssessment:
    question_id: str
    scores: dict[str, Any]
    uncertainties: dict[str, Any]
    rationales: dict[str, str]
    evidence_refs: list[str]
    assessment_methods: dict[str, str]
    limitations: list[str]
    evaluator_version: str
    expected_information_gain: Any
    budget_efficiency: Any
    priority: Any
    decision: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log2(max(p, 1e-12)) for p in probs)


def expected_information_gain(h_prior: list[float],
                              h_posterior: list[float]) -> float:
    """Pure entropy reduction; callers must supply REAL distributions."""
    if not h_prior or len(h_prior) != len(h_posterior):
        raise ValueError("prior/posterior must be non-empty and equal length")
    sp, sq = sum(h_prior), sum(h_posterior)
    if sp <= 0 or sq <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return max(0.0, _entropy([p / sp for p in h_prior])
               - _entropy([q / sq for q in h_posterior]))


def expected_information_gain_from_model(*, hypotheses: list[str],
                                         priors: list[float],
                                         likelihoods: dict[str, list[float]]
                                         ) -> dict[str, Any]:
    """EIG(a)=E_y[KL(p(H|D,y,a)||p(H|D))] with explicit observation model."""
    if len(hypotheses) < 2 or len(priors) != len(hypotheses) or not likelihoods:
        return {"status": NOT_ASSESSED, "reason": "incomplete_probability_model"}
    if any(abs(sum(p) - 1.0) > 1e-6 for p in [priors]
           + list(likelihoods.values())):
        return {"status": NOT_ASSESSED, "reason": "probabilities_not_normalized"}
    n_out = len(next(iter(likelihoods.values())))
    if any(len(v) != n_out for v in likelihoods.values()):
        return {"status": NOT_ASSESSED, "reason": "inconsistent_outcome_space"}
    eig = 0.0
    posterior_rows: list[dict[str, Any]] = []
    for y in range(n_out):
        p_y = sum(float(priors[h]) * float(likelihoods[hypotheses[h]][y])
                  for h in range(len(hypotheses)))
        if p_y <= 0:
            continue
        posterior = [float(priors[h]) * float(likelihoods[hypotheses[h]][y]) / p_y
                     for h in range(len(hypotheses))]
        kl = sum(posterior[h] * math.log2(max(posterior[h], 1e-12)
                                          / max(float(priors[h]), 1e-12))
                 for h in range(len(hypotheses)))
        eig += p_y * kl
        posterior_rows.append({"outcome_index": y, "p_y": p_y,
                               "posterior": posterior})
    return {"status": "ASSESSED", "value": round(eig, 6),
            "hypotheses": list(hypotheses), "priors": list(priors),
            "observation_model": {k: list(v) for k, v in likelihoods.items()},
            "posteriors": posterior_rows,
            "method": "explicit_observation_model_kl"}


EVALUATOR_VERSION = "m7_1_evidence_value_v1"


def _measured_dimensional_evidence(q: ResearchQuestion, dimension: str,
                                   novelty: dict[str, Any],
                                   evidence_map: dict[str, Any]
                                   ) -> dict[str, Any]:
    ev = evidence_map.get(dimension, {})
    if dimension == "novelty":
        if novelty.get("literature_check") != "performed":
            return {"ok": False, "reason": "literature_check_not_performed"}
        statuses = [v.get("status") for v in novelty.get("verdicts", [])]
        return {"ok": bool(statuses),
                "score": 0.7 if "POSSIBLE_PRIOR_ART" in statuses else 0.2,
                "evidence": statuses}
    if dimension == "falsifiability":
        reqs = q.falsifiability_requirements
        ok = (len(reqs) >= 2
              and any(any(t in r.lower() for t in ("metric", "mse", "rate",
                                                  "error", "increase",
                                                  "decrease")) for r in reqs)
              and any(any(t in r.lower() for t in ("refut", "reject", "fail"))
                      for r in reqs))
        return {"ok": ok, "score": 0.8 if ok else None, "evidence": list(reqs)}
    if dimension == "feasibility":
        ok = bool(q.estimated_budget) and bool(ev.get("engine_capability_check"))
        return {"ok": ok, "score": 0.7 if ok else None,
                "evidence": ev.get("engine_capability_check", [])}
    if dimension in ("scientific_significance", "practical_impact"):
        evidence = [e for e in q.supporting_evidence
                    if str(e.get("evidence_type", "")).lower()
                    not in ("unstructured_gap", "")]
        ok = bool(evidence) and bool(ev.get("independent_review"))
        return {"ok": ok, "score": 0.7 if ok else None,
                "evidence": evidence + ev.get("independent_review", [])}
    return {"ok": False, "reason": "no_assessment_method"}


def assess_value(q: ResearchQuestion, *, novelty: dict[str, Any],
                 evidence_map: dict[str, Any] | None = None,
                 eig: dict[str, Any] | None = None) -> ValueAssessment:
    """Score only dimensions with independent evidence; never invent priors."""
    evidence_map = evidence_map or {}
    scores: dict[str, Any] = {}
    rationales: dict[str, str] = {}
    methods: dict[str, str] = {}
    refs = list(q.evidence_refs)
    for dimension in VALUE_DIMENSIONS[:-1]:
        result = _measured_dimensional_evidence(q, dimension, novelty,
                                                evidence_map)
        if result["ok"]:
            scores[dimension] = result["score"]
            rationales[dimension] = (f"assessed from {len(result['evidence'])} "
                                     f"independent evidence item(s)")
            methods[dimension] = "evidence_gate_v1"
            refs.extend(str(x)[:120] for x in result["evidence"][:4])
        else:
            scores[dimension] = NOT_ASSESSED
            rationales[dimension] = result.get("reason", "insufficient_evidence")
            methods[dimension] = "fail_closed"
    eig_result = eig or {"status": NOT_ASSESSED,
                         "reason": "no_explicit_observation_model"}
    if eig_result.get("status") == "ASSESSED":
        scores["expected_information_gain"] = eig_result["value"]
        rationales["expected_information_gain"] = eig_result["method"]
        methods["expected_information_gain"] = "explicit_observation_model_kl"
    else:
        scores["expected_information_gain"] = NOT_ASSESSED
        rationales["expected_information_gain"] = eig_result.get("reason", "")
        methods["expected_information_gain"] = "fail_closed"

    numeric = [v for v in scores.values() if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    complete = all(v != NOT_ASSESSED for v in scores.values())
    status = "ASSESSED" if complete else NOT_ASSESSED
    priority = round(sum(numeric) / len(numeric), 6) if numeric and complete \
        else None
    efficiency = round(priority / max(1.0, float(
        q.estimated_budget.get("max_jobs", 1))), 6) if priority is not None \
        else None
    decision = ("PROMISING" if priority is not None and priority >= 0.65 else
                "EXPLORATORY" if priority is not None and priority >= 0.4 else
                NOT_ASSESSED if priority is None else "DEPRIORITIZED")
    return ValueAssessment(
        question_id=q.question_id, scores=scores,
        uncertainties={k: (0.2 if isinstance(v, (int, float))
                           and not isinstance(v, bool) else NOT_ASSESSED)
                       for k, v in scores.items()},
        rationales=rationales, evidence_refs=refs,
        assessment_methods=methods, limitations=list(q.limitations),
        evaluator_version=EVALUATOR_VERSION,
        expected_information_gain=scores["expected_information_gain"],
        budget_efficiency=efficiency, priority=priority,
        decision=decision, status=status)


__all__ = [
    "VALUE_DIMENSIONS", "NOT_ASSESSED", "EVALUATOR_VERSION",
    "ResearchQuestion", "ValueAssessment", "generate_questions",
    "expected_information_gain", "expected_information_gain_from_model",
    "assess_value",
]
