"""P1-02: the DETERMINISTIC gate every hypothesis must pass.

Passing this reviewer means a hypothesis is well formed and evidence-grounded.
It does NOT mean its mechanism is true -- only a controlled experiment can say
that. An optional ``advisor`` (typically an LLM) may attach scientific notes
(confounders, cheaper experiments, alternative explanations), but its output is
advice only: it can never turn a rejection into an approval.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional

from .schemas import Hypothesis

Advisor = Callable[[Hypothesis], "str | list[str] | None"]

# Evidence references are colon-delimited "key:value" pairs. Only these keys are
# interpreted as resolver queries; anything else is ignored so the structural
# check stays robust to ref-format evolution.
_RESOLVER_KEYS = ("experiment", "metric", "split", "dataset", "run", "artifact")
EvidenceResolver = Callable[[str], "Any | None"]


def parse_evidence_ref(ref: str) -> dict[str, str]:
    """Parse ``experiment:X:metric:Y:split:Z``-style refs into a field map.

    Returns only the recognised keys that carry a non-empty value. Crucially the
    structural check never invents a meaning for an unrecognised token.
    """
    out: dict[str, str] = {}
    parts = ref.split(":")
    i = 0
    while i < len(parts) - 1:
        key = parts[i].strip().lower()
        value = parts[i + 1]
        if key in _RESOLVER_KEYS and value:
            out[key] = value
        i += 2
    return out


class HypothesisCritic:
    """Rules-only reviewer. No engine-specific metric names are built in.

    ``allowed_metric_patterns`` is an OPTIONAL injected policy (regex strings).
    With no policy injected, any non-empty metric name is accepted -- a generic
    research runtime must not decide which metric families a domain may use.

    ``evidence_resolver`` (ARR 0.1 / R1) is an OPTIONAL callable that maps an
    experiment_id to its REAL recorded ``ExperimentResult`` (or ``None`` when the
    experiment does not exist). When supplied, the critic additionally verifies
    that every evidence reference points at a REAL experiment and that any metric
    / split / dataset_version / run_id it cites actually appears in that result.
    This rejects fabricated citations (e.g. a ``fake_metric`` on a real run)
    independently of any LLM-generated prose. With no resolver, behaviour is
    unchanged (the structural check only).
    """

    def __init__(self, *,
                 allowed_metric_patterns: Iterable[str] | None = None,
                 advisor: Advisor | None = None,
                 evidence_resolver: EvidenceResolver | None = None,
                 high_confidence: float = 0.9,
                 min_evidence_for_high_confidence: int = 3):
        self.patterns = [re.compile(p) for p in (allowed_metric_patterns or ())]
        self.advisor = advisor
        self.evidence_resolver = evidence_resolver
        self.high_confidence = high_confidence
        self.min_evidence_for_high_confidence = min_evidence_for_high_confidence
        self.advisories: list[str] = []

    def review(self, hypothesis: Hypothesis) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not hypothesis.evidence_refs:
            errors.append("missing evidence")
        if not hypothesis.falsification_criteria:
            errors.append("missing falsification criteria")
        if not hypothesis.intervention:
            errors.append("missing intervention")
        if not hypothesis.predicted_effects:
            errors.append("missing predicted effects")

        # Evidence must actually point back at the run that motivated it.
        # Schema-level min_length alone would accept a hallucinated citation.
        if hypothesis.evidence_refs and not any(
            hypothesis.source_experiment_id in ref
            for ref in hypothesis.evidence_refs
        ):
            errors.append("evidence_refs do not reference source_experiment_id")

        for prediction in hypothesis.predicted_effects:
            if not prediction.metric or not prediction.metric.strip():
                errors.append("empty metric in predicted_effects")
            elif self.patterns and not any(
                pattern.search(prediction.metric) for pattern in self.patterns
            ):
                errors.append(
                    f"metric {prediction.metric!r} is outside the injected "
                    "allowed metric families"
                )

        # R1: verify evidence references against the REAL recorded result, not
        # just that they string-match the source id. This is the line between
        # "claims a metric" and "the metric actually exists in a real run".
        if self.evidence_resolver is not None:
            self._verify_evidence_refs(hypothesis, errors)

        if (hypothesis.confidence > self.high_confidence
                and len(hypothesis.evidence_refs)
                < self.min_evidence_for_high_confidence):
            errors.append("confidence too high for limited evidence")

        self._collect_advisory(hypothesis)
        return (not errors), errors

    def _verify_evidence_refs(self, hypothesis: Hypothesis,
                              errors: list[str]) -> None:
        """R1: ground each evidence ref in a REAL recorded experiment result."""
        for ref in hypothesis.evidence_refs:
            parsed = parse_evidence_ref(ref)
            experiment_id = parsed.get("experiment")
            if not experiment_id:
                continue  # no experiment claim -> nothing to resolve
            result = self.evidence_resolver(experiment_id)
            if result is None:
                errors.append(
                    f"evidence_ref {ref!r} references unknown experiment_id "
                    f"{experiment_id!r}; the cited result does not exist")
                continue
            metric = parsed.get("metric")
            if metric is not None:
                metrics = getattr(result, "metrics", None) or {}
                if metric not in metrics:
                    errors.append(
                        f"evidence_ref {ref!r} cites metric {metric!r} that is "
                        f"not present in experiment {experiment_id!r} (available: "
                        f"{sorted(metrics)})")
            split = parsed.get("split")
            if split is not None:
                actual = (getattr(result, "evaluation_split", None)
                          or getattr(result, "split", None))
                if actual and split != actual:
                    errors.append(
                        f"evidence_ref {ref!r} cites split {split!r} but "
                        f"experiment {experiment_id!r} evaluated on {actual!r}")
            dataset = parsed.get("dataset")
            if dataset is not None:
                actual = getattr(result, "dataset_version", None)
                if actual and dataset != actual:
                    errors.append(
                        f"evidence_ref {ref!r} cites dataset_version {dataset!r} "
                        f"but experiment {experiment_id!r} used {actual!r}")
            run = parsed.get("run")
            if run is not None:
                actual = getattr(result, "run_id", None)
                if actual and run != actual:
                    errors.append(
                        f"evidence_ref {ref!r} cites run_id {run!r} but "
                        f"experiment {experiment_id!r} reported {actual!r}")

    def _collect_advisory(self, hypothesis: Hypothesis) -> None:
        if self.advisor is None:
            return
        try:
            note = self.advisor(hypothesis)
        except Exception as exc:  # advisor is best-effort, never a gate
            self.advisories.append(f"advisor unavailable: {type(exc).__name__}")
            return
        if isinstance(note, str) and note.strip():
            self.advisories.append(note.strip())
        elif isinstance(note, (list, tuple)):
            self.advisories.extend(
                str(item).strip() for item in note if str(item).strip()
            )

    def review_all(self, hypotheses: Iterable[Hypothesis]) -> list[Hypothesis]:
        approved: list[Hypothesis] = []
        self.rejections: list[dict] = []
        for hypothesis in hypotheses:
            ok, errors = self.review(hypothesis)
            if ok:
                approved.append(
                    hypothesis.model_copy(update={"status": "approved"})
                )
            else:
                self.rejections.append({
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "errors": errors,
                })
        return approved
