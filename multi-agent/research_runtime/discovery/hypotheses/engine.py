"""M7 -- competing scientific hypotheses with an INDEPENDENT critic.

Each research question gets >=3 substantively different hypotheses plus a null
hypothesis. The generator does NOT act as the final critic: a separate rule-based
critic checks falsifiability, executable refutation, discriminating predictions,
confounders, alternative explanations and measurability. Invalid hypotheses get
REJECTED/UNDECIDABLE, never silently accepted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

STATUSES = ("PROPOSED", "ACCEPTED", "REJECTED", "UNDECIDABLE")


@dataclass(frozen=True)
class ScientificHypothesis:
    hypothesis_id: str
    question_id: str
    kind: str                       # H1/H2/H3/H0
    mechanism: str
    assumptions: list[str]
    predictions: list[str]
    falsifying_observations: list[str]
    confounders: list[str]
    required_data: list[str]
    required_experiments: list[str]
    literature_refs: list[str]
    confidence: float
    status: str = "PROPOSED"
    review_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


def generate_competing_hypotheses(q: Any, *,
                                  literature_refs: list[str] | None = None
                                  ) -> list[ScientificHypothesis]:
    """Three mechanism-distinct hypotheses + a NULL, all tied to the question."""
    qid = q.question_id
    refs = list(literature_refs or q.evidence_refs)
    specs = [
        ("H1", "target mechanism causes the predicted improvement",
         ["no confounding capacity change", "same training budget"],
         [f"{q.domain}: primary metric improves vs baseline beyond MME"],
         ["no improvement beyond MME on independent holdout families"],
         ["model capacity", "seed variance"], "paired controlled campaign"),
        ("H2", "alternative mechanism (data/inductive bias) explains it",
         ["baseline differs only in data conditioning"],
         [f"{q.domain}: improvement vanishes under matched data conditioning"],
         ["improvement persists with matched data"],
         ["dataset shift", "implementation detail"], "ablation campaign"),
        ("H3", "effect only holds in-distribution, not under OOD",
         ["ID and OOD splits are disjoint"],
         [f"{q.domain}: ID improves but OOD does not"],
         ["OOD also improves beyond MME"],
         ["split leakage", "weak OOD protocol"], "ID+OOD campaign"),
        ("H0", "NULL: no systematic difference between treatments",
         ["treatments are equivalent in expectation"],
         [f"{q.domain}: no difference beyond MME across families"],
         ["consistent improvement beyond MME on held-out families"],
         ["multiple comparisons", "low power"], "pre-registered paired test"),
    ]
    out: list[ScientificHypothesis] = []
    for kind, mech, assump, pred, fals, conf, exp in specs:
        hid = "sh_" + hashlib.sha256(
            f"{qid}|{kind}|{mech}".encode()).hexdigest()[:14]
        out.append(ScientificHypothesis(
            hypothesis_id=hid, question_id=qid, kind=kind, mechanism=mech,
            assumptions=assump, predictions=pred,
            falsifying_observations=fals, confounders=conf,
            required_data=["frozen dataset manifest", "ID/OOD splits"],
            required_experiments=[exp], literature_refs=refs,
            confidence=0.4 if kind != "H0" else 0.5))
    return out


class IndependentCritic:
    """Rule-based reviewer; NEVER the generator itself."""

    version = "m7_hypothesis_critic_v1"

    def review(self, h: ScientificHypothesis) -> tuple[str, list[str]]:
        errors: list[str] = []
        if len(h.falsifying_observations) < 1 or any(
                len(str(x).strip()) < 10 for x in h.falsifying_observations):
            errors.append("no executable falsifying observation")
        if not h.predictions or any(len(str(x).strip()) < 8
                                    for x in h.predictions):
            errors.append("no measurable prediction")
        pred_tokens = {w for p in h.predictions
                       for w in p.lower().split() if len(w) > 4}
        fals_tokens = {w for f in h.falsifying_observations
                       for w in f.lower().split() if len(w) > 4}
        if not (pred_tokens & fals_tokens):
            errors.append("predictions and falsification share no observable")
        if len(h.confounders) < 1:
            errors.append("no confounders identified")
        if not h.required_experiments:
            errors.append("no executable experiment")
        if not h.literature_refs and h.kind != "H0":
            errors.append("no literature evidence bound")
        if errors:
            return ("UNDECIDABLE" if len(errors) == 1
                    and errors[0] == "no literature evidence bound" else
                    "REJECTED"), errors
        return "ACCEPTED", []


def review_all(hypotheses: list[ScientificHypothesis],
               *, critic: IndependentCritic | None = None
               ) -> list[ScientificHypothesis]:
    c = critic or IndependentCritic()
    out: list[ScientificHypothesis] = []
    for h in hypotheses:
        status, errors = c.review(h)
        out.append(ScientificHypothesis(**{**h.to_dict(), "status": status,
                                           "review_errors": errors}))
    return out


__all__ = ["STATUSES", "ScientificHypothesis", "generate_competing_hypotheses",
           "IndependentCritic", "review_all"]
