"""P1-02: the DETERMINISTIC gate every hypothesis must pass.

Passing this reviewer means a hypothesis is well formed and evidence-grounded.
It does NOT mean its mechanism is true -- only a controlled experiment can say
that. An optional ``advisor`` (typically an LLM) may attach scientific notes
(confounders, cheaper experiments, alternative explanations), but its output is
advice only: it can never turn a rejection into an approval.

ARR 0.1 / R1 -- Evidence Integrity
-----------------------------------
``evidence_resolver`` maps an ``experiment_id`` to its REAL recorded
``ExperimentResult`` (or ``None`` when the experiment does not exist). When
supplied, every evidence reference is grounded in that real result and each
claimed field (metric / split / dataset_version / run_id / artifact) is verified
against what the engine ACTUALLY produced. Three verdicts are possible:

* ``MATCH``     -- the claim is present in the real result and agrees.
* ``MISMATCH``  -- the claim is present but disagrees. Always a rejection.
* ``UNKNOWN``   -- the reference claims a field the real result does NOT record
                   (so it cannot be verified). In ``strict`` (production) mode
                   this is a rejection: an unverifiable claim is not a PASS.

This explicitly rejects fabricated citations (e.g. a ``fake_metric`` on a real
run) independently of any LLM-generated prose. With no resolver, behaviour is
unchanged (the structural check only) -- that path is the legacy escape hatch
and must never be the production autonomous path.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
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


class EvidenceVerdict:
    """Structured outcome of verifying ONE evidence reference.

    ``MATCH`` / ``MISMATCH`` / ``UNKNOWN`` / ``UNRESOLVED`` are the four states.
    ``UNRESOLVED`` means the referenced experiment itself does not exist.
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    UNRESOLVED = "unresolved"

    def __init__(self, ref: str, *, experiment_id: str | None = None,
                 verdict: str, field: str | None = None, detail: str = ""):
        self.ref = ref
        self.experiment_id = experiment_id
        self.verdict = verdict
        self.field = field
        self.detail = detail

    @property
    def is_pass(self) -> bool:
        return self.verdict == self.MATCH

    @property
    def is_reject(self) -> bool:
        return self.verdict in (self.MISMATCH, self.UNKNOWN, self.UNRESOLVED)

    def to_dict(self) -> dict[str, str]:
        return {
            "ref": self.ref,
            "experiment_id": self.experiment_id or "",
            "verdict": self.verdict,
            "field": self.field or "",
            "detail": self.detail,
        }


@dataclass
class _ResultShim:
    """Read the fields the critic needs from a real ExperimentResult duck-type."""
    metrics: dict[str, float]
    evaluation_split: Optional[str]
    dataset_version: Optional[str]
    run_id: Optional[str]
    artifacts: list[str]
    result_path: Optional[str]


def _as_shim(result: Any) -> _ResultShim:
    return _ResultShim(
        metrics=dict(getattr(result, "metrics", None) or {}),
        evaluation_split=getattr(result, "evaluation_split", None),
        dataset_version=getattr(result, "dataset_version", None),
        run_id=getattr(result, "run_id", None),
        artifacts=list(getattr(result, "artifacts", None) or []),
        result_path=getattr(result, "result_path", None),
    )


def _hash_file(path: str) -> str | None:
    try:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            # Try resolving relative to the result's directory.
            base = _ResultBase.get()
            if base and (base / p).exists():
                p = base / p
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# Module-level hook so artifact hashing can resolve a relative path against the
# directory of the result currently being verified (set per verification).
class _ResultBase:
    _path: Optional["Path"] = None

    @classmethod
    def set(cls, path: Optional[str]) -> None:
        from pathlib import Path as _P
        cls._path = _P(path).parent if path else None

    @classmethod
    def get(cls) -> Optional["Path"]:
        return cls._path


from pathlib import Path  # noqa: E402  (kept after the shim for readability)


def verify_evidence_ref(ref: str, resolver: EvidenceResolver, *,
                        strict: bool = True) -> EvidenceVerdict:
    """Verify one evidence reference against the REAL recorded result.

    Returns an :class:`EvidenceVerdict`. ``strict`` controls whether an
    ``UNKNOWN`` verdict (a claim the real result cannot confirm) is treated as a
    rejection. Production autonomous research MUST use ``strict=True``.
    """
    parsed = parse_evidence_ref(ref)
    experiment_id = parsed.get("experiment")

    # A reference that names no experiment cannot be grounded at all.
    if not experiment_id:
        return EvidenceVerdict(ref, verdict=EvidenceVerdict.UNRESOLVED,
                               detail="reference does not name an experiment_id")

    try:
        result = resolver(experiment_id)
    except Exception:
        # A broken evidence store must never mask the failure as "verified".
        return EvidenceVerdict(
            ref, experiment_id=experiment_id,
            verdict=EvidenceVerdict.UNRESOLVED,
            detail="evidence resolver raised; experiment could not be resolved")
    if result is None:
        return EvidenceVerdict(
            ref, experiment_id=experiment_id, verdict=EvidenceVerdict.UNRESOLVED,
            detail="referenced experiment_id does not exist")

    shim = _as_shim(result)
    _ResultBase.set(shim.result_path)
    try:
        # metric: must exist in the real result's metrics.
        metric = parsed.get("metric")
        if metric is not None and metric not in shim.metrics:
            return EvidenceVerdict(
                ref, experiment_id=experiment_id, verdict=EvidenceVerdict.MISMATCH,
                field="metric",
                detail=f"metric {metric!r} not present in experiment "
                       f"{experiment_id!r} (available: {sorted(shim.metrics)})")

        # split / dataset / run: MATCH if present and equal; UNKNOWN if the real
        # result did not record that field at all (cannot confirm the claim).
        for key, actual, label in (
            ("split", shim.evaluation_split, "evaluation_split"),
            ("dataset", shim.dataset_version, "dataset_version"),
            ("run", shim.run_id, "run_id"),
        ):
            claimed = parsed.get(key)
            if claimed is None:
                continue
            if actual is None:
                v = EvidenceVerdict.UNKNOWN
                detail = (f"claim cites {key}={claimed!r} but experiment "
                          f"{experiment_id!r} recorded no {label}; unverifiable")
                if strict:
                    return EvidenceVerdict(ref, experiment_id=experiment_id,
                                           verdict=v, field=key, detail=detail)
                # Legacy / non-strict: record but do not reject.
                return EvidenceVerdict(ref, experiment_id=experiment_id,
                                       verdict=v, field=key, detail=detail)
            if claimed != actual:
                return EvidenceVerdict(
                    ref, experiment_id=experiment_id,
                    verdict=EvidenceVerdict.MISMATCH, field=key,
                    detail=f"claim cites {key}={claimed!r} but experiment "
                           f"{experiment_id!r} reported {label}={actual!r}")

        # artifact: verify existence and optional SHA-256.
        artifact = parsed.get("artifact")
        if artifact is not None:
            cited_path = artifact
            cited_sha: str | None = None
            if "@" in artifact:
                cited_path, cited_sha = artifact.rsplit("@", 1)
            if cited_path not in shim.artifacts:
                return EvidenceVerdict(
                    ref, experiment_id=experiment_id,
                    verdict=EvidenceVerdict.MISMATCH, field="artifact",
                    detail=f"artifact {cited_path!r} not listed in experiment "
                           f"{experiment_id!r} artifacts")
            actual_sha = _hash_file(cited_path)
            if actual_sha is None:
                return EvidenceVerdict(
                    ref, experiment_id=experiment_id,
                    verdict=EvidenceVerdict.MISMATCH, field="artifact",
                    detail=f"artifact {cited_path!r} is missing on disk")
            if cited_sha is not None and cited_sha != actual_sha:
                return EvidenceVerdict(
                    ref, experiment_id=experiment_id,
                    verdict=EvidenceVerdict.MISMATCH, field="artifact",
                    detail=f"artifact {cited_path!r} SHA-256 mismatch "
                           f"(cited {cited_sha!r}, actual {actual_sha!r})")

        return EvidenceVerdict(ref, experiment_id=experiment_id,
                               verdict=EvidenceVerdict.MATCH,
                               detail="all claimed fields verified")
    finally:
        _ResultBase.set(None)


class HypothesisCritic:
    """Rules-only reviewer. No engine-specific metric names are built in.

    ``allowed_metric_patterns`` is an OPTIONAL injected policy (regex strings).
    With no policy injected, any non-empty metric name is accepted -- a generic
    research runtime must not decide which metric families a domain may use.

    ``evidence_resolver`` (ARR 0.1 / R1) is an OPTIONAL callable that maps an
    experiment_id to its REAL recorded ``ExperimentResult`` (or ``None`` when the
    experiment does not exist). When supplied, every evidence reference is
    grounded in a REAL result (see :func:`verify_evidence_ref`). With no
    resolver, behaviour is unchanged (the structural check only) -- that legacy
    path must never be the production autonomous path.

    ``strict`` (ARR 0.1 / R1) controls whether an ``UNKNOWN`` verdict (a claim
    the real result cannot confirm) is treated as a rejection. The production
    autonomous research path MUST keep this ``True``.
    """

    def __init__(self, *,
                 allowed_metric_patterns: Iterable[str] | None = None,
                 advisor: Advisor | None = None,
                 evidence_resolver: EvidenceResolver | None = None,
                 strict: bool = True,
                 high_confidence: float = 0.9,
                 min_evidence_for_high_confidence: int = 3):
        self.patterns = [re.compile(p) for p in (allowed_metric_patterns or ())]
        self.advisor = advisor
        self.evidence_resolver = evidence_resolver
        self.strict = strict
        self.high_confidence = high_confidence
        self.min_evidence_for_high_confidence = min_evidence_for_high_confidence
        self.advisories: list[str] = []
        # Per-ref verdicts from the most recent review (for evidence reporting).
        self.evidence_verdicts: list[dict[str, str]] = []

    def review(self, hypothesis: Hypothesis) -> tuple[bool, list[str]]:
        errors: list[str] = []
        self.evidence_verdicts = []
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

        # R1: ground each evidence ref in a REAL recorded experiment result.
        if self.evidence_resolver is not None:
            self._verify_evidence_refs(hypothesis, errors)

        if (hypothesis.confidence > self.high_confidence
                and len(hypothesis.evidence_refs)
                < self.min_evidence_for_high_confidence):
            errors.append("confidence too high for limited evidence")

        # A truly malformed reference (non-empty but no usable structure) is
        # rejected outright -- it is neither a recognised experiment claim nor a
        # recognised failure/metric claim.
        for ref in hypothesis.evidence_refs:
            if ref.strip() and ":" not in ref:
                errors.append(f"malformed evidence reference (no key:value): {ref!r}")

        self._collect_advisory(hypothesis)
        return (not errors), errors

    def _verify_evidence_refs(self, hypothesis: Hypothesis,
                              errors: list[str]) -> None:
        """R1: ground each evidence ref in a REAL recorded experiment result."""
        for ref in hypothesis.evidence_refs:
            parsed = parse_evidence_ref(ref)
            # Skip references that are not experiment-grounded (e.g. failure:fr_x
            # citations are a separate evidence class not resolvable here).
            if not parsed.get("experiment") and "experiment" not in ref:
                continue
            verdict = verify_evidence_ref(
                ref, self.evidence_resolver, strict=self.strict)
            self.evidence_verdicts.append(verdict.to_dict())
            reject = (
                verdict.verdict in (EvidenceVerdict.MISMATCH,
                                    EvidenceVerdict.UNRESOLVED)
                or (verdict.verdict == EvidenceVerdict.UNKNOWN and self.strict)
            )
            if reject:
                if verdict.verdict == EvidenceVerdict.UNKNOWN:
                    errors.append(
                        f"evidence_ref {ref!r} is UNVERIFIABLE "
                        f"[{verdict.verdict}]: {verdict.detail}")
                else:
                    errors.append(
                        f"evidence_ref {ref!r} fails verification "
                        f"[{verdict.verdict}]: {verdict.detail}")

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
