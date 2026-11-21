"""M7.1 -- competing hypotheses with an adversarial independent Critic.

The Critic checks executable falsification, controlled confounders, observable
metrics, alternative explanations, counterexamples and correlation/causation
confusion. It records model identity so self-review by the same model is never
mistaken for independent expert validation.
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
    kind: str
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
    counterexamples: list[str] = field(default_factory=list)
    critique: dict[str, Any] = field(default_factory=dict)

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
    qid = q.question_id
    refs = list(literature_refs or q.evidence_refs)
    specs = [
        ("H1", "target mechanism causes the predicted improvement",
         ["no confounding capacity change", "same training budget"],
         [f"{q.domain}: state_mse decreases vs baseline beyond MME"],
         ["state_mse does not decrease beyond MME on independent holdout families"],
         ["model capacity", "seed variance"], "paired controlled campaign",
         ["improvement disappears when target mechanism is ablated"]),
        ("H2", "alternative data/inductive-bias mechanism explains it",
         ["baseline differs only in data conditioning"],
         [f"{q.domain}: state_mse improvement vanishes under matched data conditioning"],
         ["state_mse improvement persists with matched data conditioning"],
         ["dataset shift", "implementation detail"], "matched-data ablation",
         ["effect persists despite matched data conditioning"]),
        ("H3", "effect holds in-distribution but not under OOD",
         ["ID and OOD splits are disjoint"],
         [f"{q.domain}: ID state_mse decreases but OOD state_mse does not"],
         ["OOD state_mse also decreases beyond MME"],
         ["split leakage", "weak OOD protocol"], "ID+OOD paired campaign",
         ["OOD improvement removes the ID/OOD distinction"]),
        ("H0", "NULL: no systematic treatment difference",
         ["treatments equivalent in expectation"],
         [f"{q.domain}: state_mse shows no treatment difference beyond MME across families"],
         ["consistent treatment improvement beyond MME on held-out families"],
         ["multiple comparisons", "low power"], "pre-registered paired test",
         ["a pre-registered replication shows systematic improvement"]),
    ]
    out: list[ScientificHypothesis] = []
    for kind, mech, assump, pred, fals, conf, exp, counter in specs:
        hid = "sh_" + hashlib.sha256(
            f"{qid}|{kind}|{mech}".encode()).hexdigest()[:14]
        out.append(ScientificHypothesis(
            hypothesis_id=hid, question_id=qid, kind=kind, mechanism=mech,
            assumptions=assump, predictions=pred,
            falsifying_observations=fals, confounders=conf,
            required_data=["frozen dataset manifest", "ID/OOD splits"],
            required_experiments=[exp], literature_refs=refs,
            counterexamples=[counter], confidence=0.4 if kind != "H0" else 0.5))
    return out


class IndependentCritic:
    """Rule-based adversarial reviewer; generator never reviews its own output."""

    version = "m7_1_adversarial_critic_v1"

    def __init__(self, *, model_identity: str = "rule_based_critic_v1",
                 strict_evidence: bool = False):
        self.model_identity = model_identity
        self.strict_evidence = strict_evidence

    def review_detailed(self, h: ScientificHypothesis) -> dict[str, Any]:
        errors: list[str] = []
        flags: dict[str, bool] = {}
        pred_text = " ".join(h.predictions).lower()
        fals_text = " ".join(h.falsifying_observations).lower()
        obs_terms = ("mse", "error", "rate", "metric", "score", "latency",
                     "accuracy", "increase", "decrease")
        flags["observable_prediction"] = any(t in pred_text for t in obs_terms)
        if not flags["observable_prediction"]:
            errors.append("prediction_has_no_observable_metric")
        flags["executable_refutation"] = (len(h.falsifying_observations) >= 1
                                          and all(len(str(x).strip()) >= 10
                                                  for x in h.falsifying_observations))
        if not flags["executable_refutation"]:
            errors.append("no_executable_falsifying_observation")
        pred_tokens = {w for p in h.predictions
                       for w in p.lower().split() if len(w) > 4}
        fals_tokens = {w for f in h.falsifying_observations
                       for w in f.lower().split() if len(w) > 4}
        flags["shared_observable"] = bool(pred_tokens & fals_tokens)
        if not flags["shared_observable"]:
            errors.append("predictions_and_falsification_share_no_observable")
        flags["confounders_controlled"] = (len(h.confounders) >= 1
                                           and any(any(k in e.lower() for k in
                                                       ("control", "matched",
                                                        "paired", "ablation"))
                                                   for e in h.required_experiments))
        if not flags["confounders_controlled"]:
            errors.append("confounders_not_operationally_controlled")
        flags["measurable_conditions"] = bool(h.required_data)
        if not flags["measurable_conditions"]:
            errors.append("key_variables_not_measurable")
        flags["counterexample_considered"] = bool(h.counterexamples)
        if not flags["counterexample_considered"]:
            errors.append("no_strongest_counterexample_recorded")
        flags["not_circular"] = not any(k in h.mechanism.lower() for k in
                                        ("because it is", "by definition"))
        if not flags["not_circular"]:
            errors.append("circular_mechanism")
        flags["not_correlation_only"] = not (
            "correlat" in pred_text and not any(
                k in (h.mechanism + " " + " ".join(h.required_experiments)).lower()
                for k in ("interven", "control", "ablat", "random")))
        if not flags["not_correlation_only"]:
            errors.append("correlation_treated_as_cause")
        flags["literature_bound"] = bool(h.literature_refs)
        if not flags["literature_bound"] and h.kind != "H0":
            errors.append("no_literature_evidence_bound")
        if self.strict_evidence and not flags["literature_bound"]:
            errors.append("strict_evidence_requires_references")
        status = "ACCEPTED" if not errors else (
            "UNDECIDABLE" if errors == ["no_literature_evidence_bound"] else
            "REJECTED")
        return {"status": status, "errors": errors, "flags": flags,
                "critic_model": self.model_identity,
                "critic_version": self.version}

    def review(self, h: ScientificHypothesis) -> tuple[str, list[str]]:
        d = self.review_detailed(h)
        return d["status"], d["errors"]

    def review_competition(self, hypotheses: list[ScientificHypothesis]
                           ) -> dict[str, Any]:
        kinds = {h.kind for h in hypotheses}
        pred_sets = [{w for p in h.predictions for w in p.lower().split()
                      if len(w) > 4} for h in hypotheses
                     if h.kind in ("H1", "H2", "H3")]
        common = set.intersection(*pred_sets) if pred_sets else set()
        reduced = [s - common for s in pred_sets]
        distinct = all(
            len(reduced[i] & reduced[j]) / max(
                1, len(reduced[i] | reduced[j])) < 0.5
            for i in range(len(reduced))
            for j in range(i + 1, len(reduced)))
        null_present = "H0" in kinds
        ok = (len(hypotheses) >= 4 and distinct and null_present)
        return {"ok": ok, "n_hypotheses": len(hypotheses),
                "null_present": null_present,
                "predictions_distinct": distinct,
                "reason": "competition_complete" if ok else
                "competition_incomplete"}


def review_all(hypotheses: list[ScientificHypothesis],
               *, critic: IndependentCritic | None = None
               ) -> list[ScientificHypothesis]:
    c = critic or IndependentCritic()
    out: list[ScientificHypothesis] = []
    for h in hypotheses:
        detailed = c.review_detailed(h)
        out.append(ScientificHypothesis(
            **{**h.to_dict(), "status": detailed["status"],
               "review_errors": detailed["errors"],
               "critique": detailed}))
    return out


__all__ = ["STATUSES", "ScientificHypothesis",
           "generate_competing_hypotheses", "IndependentCritic", "review_all"]
