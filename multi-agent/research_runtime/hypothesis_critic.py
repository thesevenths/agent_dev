"""P1-02: the DETERMINISTIC gate every hypothesis must pass.

Passing this reviewer means a hypothesis is well formed and evidence-grounded.
It does NOT mean its mechanism is true -- only a controlled experiment can say
that. An optional ``advisor`` (typically an LLM) may attach scientific notes
(confounders, cheaper experiments, alternative explanations), but its output is
advice only: it can never turn a rejection into an approval.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from .schemas import Hypothesis

Advisor = Callable[[Hypothesis], "str | list[str] | None"]


class HypothesisCritic:
    """Rules-only reviewer. No engine-specific metric names are built in.

    ``allowed_metric_patterns`` is an OPTIONAL injected policy (regex strings).
    With no policy injected, any non-empty metric name is accepted -- a generic
    research runtime must not decide which metric families a domain may use.
    """

    def __init__(self, *,
                 allowed_metric_patterns: Iterable[str] | None = None,
                 advisor: Advisor | None = None,
                 high_confidence: float = 0.9,
                 min_evidence_for_high_confidence: int = 3):
        self.patterns = [re.compile(p) for p in (allowed_metric_patterns or ())]
        self.advisor = advisor
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

        if (hypothesis.confidence > self.high_confidence
                and len(hypothesis.evidence_refs)
                < self.min_evidence_for_high_confidence):
            errors.append("confidence too high for limited evidence")

        self._collect_advisory(hypothesis)
        return (not errors), errors

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
