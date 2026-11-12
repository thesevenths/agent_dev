"""ARR 1.0 / D1 -- domain-injectable RESEARCH QUALITY protocol.

Why this module exists
----------------------
The R4 bridge used ``quality = accepted / n`` as the research runtime's score.
That is a Goodhart trap: a runtime that only ever proposes trivially-accepted
hypotheses scores 1.0, while a runtime that *falsifies a wrong hypothesis* --
one of the most valuable things science can do -- scores 0.0.

This module replaces that single scalar with an explicit, auditable evaluation
protocol::

    Q = (w_E*E + w_R*R + w_G*G + w_D*D - w_C*C) / sum(w)

    E  Evidence Integrity   hashes, refs, config/dataset provenance, no prose
    R  Reproducibility      paired seeds complete, no missing runs, split hygiene
    G  Generalization       independent holdout / OOD / cross-seed consistency
    D  Discovery Value      DECISIVE evidence, including clean falsification
    C  Cost Efficiency      jobs / time / wasted trials vs the declared budget

Hard constraints (E, R floors and domain rules such as "zero safety violations"
or "zero look-ahead leakage") are NON-COMPENSABLE: a candidate cannot trade a
lower evidence bar for a higher discovery score. This module is deliberately
outside every candidate's reach -- see ``QUALITY_FROZEN_FIELDS`` and
``adaptive_runtime.harness_evolution`` which hashes these files before/after.

Everything is computed from MEASURED fields of the immutable ResearchReport,
the persisted ComparisonReports, the engine's ExperimentResults and real run
telemetry. No field is a self-declared "I am good" boolean. When evidence is
absent the dimension degrades -- it never defaults to a passing value.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from .evidence_verifier import EvidenceVerifier

QUALITY_PROTOCOL_VERSION = "arr-1.0-quality-v1"

#: Fields a Harness / Domain candidate may NEVER modify. The evaluation
#: protocol is fixed outside the system being optimised (F4).
QUALITY_FROZEN_FIELDS = frozenset({
    "weights", "hard_constraints", "evaluator", "quality_evaluator",
    "evaluation_split", "holdout_split", "primary_metric", "metric_direction",
    "gate_a", "gate_b", "evidence_integrity", "frozen_test_used", "scoring",
    "report_sha256", "min_evidence_integrity", "min_reproducibility",
})


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _fraction(checks: Iterable[bool]) -> float:
    items = list(checks)
    return sum(1.0 for c in items if c) / len(items) if items else 0.0


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


@dataclass(frozen=True)
class QualityWeights:
    """Domain-injectable weights. Declared BEFORE experiments, never tuned after."""

    evidence: float = 0.25
    reproducibility: float = 0.25
    generalization: float = 0.20
    discovery: float = 0.25
    cost: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"weight {name!r} must be finite and >= 0")
        if sum(self.as_dict().values()) <= 0:
            raise ValueError("at least one quality weight must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "evidence": self.evidence,
            "reproducibility": self.reproducibility,
            "generalization": self.generalization,
            "discovery": self.discovery,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class QualityHardConstraints:
    """Non-compensable floors. Violating any caps Q and blocks the gate."""

    min_evidence_integrity: float = 0.50
    min_reproducibility: float = 0.50
    max_q_when_violated: float = 0.35


@dataclass(frozen=True)
class MetricRule:
    """A domain's interpretation of ONE real engine metric.

    ``hard=True`` makes a breach a non-compensable violation (e.g. a robot
    safety violation, or look-ahead leakage in a backtest). Direction is
    explicit -- never guessed from a metric name.
    """

    metric: str
    direction: str = "maximize"          # maximize | minimize | equals
    bound: float = 0.0
    hard: bool = False
    label: str = ""

    def verdict(self, metrics: Mapping[str, Any]) -> tuple[str, str]:
        value = _num(metrics.get(self.metric))
        if value is None:
            return "unverified", f"{self.metric}: not measured"
        if self.direction == "minimize":
            ok = value <= self.bound
            relation = f"<= {self.bound}"
        elif self.direction == "maximize":
            ok = value >= self.bound
            relation = f">= {self.bound}"
        else:
            ok = abs(value - self.bound) < 1e-9
            relation = f"== {self.bound}"
        status = "passed" if ok else "violated"
        return status, f"{self.metric}={value:.6g} {relation}"


@dataclass
class QualityEvidence:
    """All inputs are observed facts, never self-declared quality assertions."""

    report: dict = field(default_factory=dict)
    comparison_reports: list[dict] = field(default_factory=list)
    experiment_results: list[dict] = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)
    holdout: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = {
            "report": self.report,
            "comparison_reports": self.comparison_reports,
            "experiment_results": self.experiment_results,
            "telemetry": self.telemetry,
            "holdout": self.holdout,
            "provenance": self.provenance,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass
class QualityScore:
    Q: float
    components: dict[str, float]
    weights: dict[str, float]
    checks: dict[str, bool]
    hard_violations: list[str]
    gate_eligible: bool
    evaluator_name: str
    domain: str
    protocol_version: str = QUALITY_PROTOCOL_VERSION
    evidence_fingerprint: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "Q": round(self.Q, 6), "components": self.components,
            "weights": self.weights, "checks": self.checks,
            "hard_violations": list(self.hard_violations),
            "gate_eligible": self.gate_eligible,
            "evaluator_name": self.evaluator_name, "domain": self.domain,
            "protocol_version": self.protocol_version,
            "evidence_fingerprint": self.evidence_fingerprint,
            "detail": self.detail,
        }


@runtime_checkable
class ResearchQualityEvaluator(Protocol):
    """The D1 contract: every Domain supplies its own evaluation rules."""

    name: str
    domain: str
    version: str

    def evaluate(self, evidence: QualityEvidence) -> QualityScore: ...


class DefaultResearchQualityEvaluator:
    """Generic, deterministic evaluator; domains inject weights + metric rules.

    It is intentionally strict about ABSENT evidence: an unmeasured dimension
    lowers the score instead of silently passing. A rejected-but-well-evidenced
    hypothesis scores HIGHER on Discovery than a bare acceptance.
    """

    def __init__(self, *, domain: str = "generic",
                 name: str = "default_research_quality_v1",
                 weights: QualityWeights | None = None,
                 hard: QualityHardConstraints | None = None,
                 metric_rules: tuple[MetricRule, ...] = (),
                 practical_effect_threshold: float = 0.0,
                 version: str = QUALITY_PROTOCOL_VERSION) -> None:
        self.domain = domain
        self.name = name
        self.version = version
        self.weights = weights or QualityWeights()
        self.hard = hard or QualityHardConstraints()
        self.metric_rules = tuple(metric_rules)
        self.practical_effect_threshold = practical_effect_threshold

    # ------------------------------------------------------------ dimensions
    def _evidence_integrity(self, ev: QualityEvidence) -> tuple[float, dict[str, bool]]:
        report = ev.report or {}
        hyps = list(report.get("hypotheses") or [])
        reviews = list(report.get("reviews") or [])
        plans = list(report.get("plans") or [])
        approvals = list(report.get("approvals") or [])
        campaigns = list(report.get("campaigns") or [])
        hyp_hashes = {h.get("content_sha256") for h in hyps if h.get("content_sha256")}
        checks = {
            "report_has_sha256": bool(report.get("report_sha256")),
            "evidence_refs_present": bool(report.get("evidence_refs")),
            "hypotheses_hashed": bool(hyps) and all(h.get("content_sha256") for h in hyps),
            "reviews_bound_to_hypotheses": bool(reviews) and all(
                r.get("hypothesis_sha256") in hyp_hashes for r in reviews),
            "plans_hashed": bool(plans) and all(p.get("plan_sha256") for p in plans),
            "approvals_human_and_hashed": bool(approvals) and all(
                a.get("resource_sha256") and a.get("source") == "human"
                for a in approvals),
            "campaigns_completed": bool(campaigns) and all(
                c.get("status") in ("completed", "succeeded") for c in campaigns),
            "experience_recorded": bool(report.get("experience_refs")),
        }
        comps = ev.comparison_reports
        if comps:
            checks["comparisons_have_config_hashes"] = all(
                c.get("baseline_config_sha256") and c.get("candidate_config_sha256")
                for c in comps)
            checks["comparisons_have_dataset_version"] = all(
                bool(c.get("dataset_version")) for c in comps)
            checks["comparisons_have_split_hashes"] = all(
                bool(c.get("dataset_split_hashes")) for c in comps)
        else:
            # No persisted comparison evidence => unverifiable, NOT a pass.
            checks["comparisons_have_config_hashes"] = False
            checks["comparisons_have_dataset_version"] = False
            checks["comparisons_have_split_hashes"] = False
        results = ev.experiment_results
        if results:
            checks["engine_results_have_real_metrics"] = all(
                isinstance(r.get("metrics"), dict) and bool(r.get("metrics"))
                for r in results)
        else:
            checks["engine_results_have_real_metrics"] = False
        # D3/P0-2: recompute hashes from ACTUAL content instead of trusting the
        # mere presence of a hash field. Only checks that actually recomputed
        # something are folded in, so legacy hash-only reports stay green while
        # every new artifact is upgraded to real verification (and tampering is
        # caught when raw content is present but the hash does not match).
        if ev.report:
            vr = EvidenceVerifier().verify_report(ev.report,
                                                 only_verifiable=True)
            for c in vr.checks:
                checks[f"V.{c.name}"] = bool(c.passed)
        return _fraction(checks.values()), checks

    def _reproducibility(self, ev: QualityEvidence) -> tuple[float, dict[str, bool]]:
        report = ev.report or {}
        comps = ev.comparison_reports
        tele = ev.telemetry or {}
        checks: dict[str, bool] = {
            "frozen_test_not_used_for_selection": report.get("frozen_test_used") is False,
            "no_phantom_promotion": report.get("promotion") == "not_requested",
        }
        if comps:
            checks["paired_seeds_complete"] = all(
                (c.get("n_pairs", 0) >= 2)
                and (not c.get("missing_pairs"))
                and (c.get("n_planned", 0) == 0
                     or c.get("n_pairs", 0) == c.get("n_planned"))
                for c in comps)
            checks["selection_split_not_reserved"] = all(
                c.get("evaluation_split") not in (None, "", "test")
                for c in comps)
            checks["config_hashes_verified"] = all(
                c.get("baseline_config_sha256") and c.get("candidate_config_sha256")
                and c.get("baseline_config_sha256") != c.get("candidate_config_sha256")
                for c in comps)
        else:
            checks["paired_seeds_complete"] = False
            checks["selection_split_not_reserved"] = False
            checks["config_hashes_verified"] = False
        if tele:
            checks["no_missing_runs"] = int(tele.get("missing_runs", 1)) == 0
            checks["no_timeout_runs"] = int(tele.get("timeout_runs", 1)) == 0
            checks["all_trials_succeeded"] = (
                int(tele.get("succeeded_trials", 0)) > 0
                and int(tele.get("succeeded_trials", 0))
                == int(tele.get("total_trials", -1)))
        else:
            checks["no_missing_runs"] = False
            checks["no_timeout_runs"] = False
            checks["all_trials_succeeded"] = False
        return _fraction(checks.values()), checks

    def _generalization(self, ev: QualityEvidence) -> tuple[float, dict[str, bool]]:
        comps = ev.comparison_reports
        holdout = ev.holdout or {}
        checks: dict[str, bool] = {}
        # Cross-seed sign consistency of the measured paired deltas.
        sign_consistent = False
        if comps:
            signs: list[float] = []
            for c in comps:
                for obs in (c.get("paired_observations") or []):
                    d = _num(obs.get("delta"))
                    if d is not None and abs(d) > 1e-12:
                        signs.append(1.0 if d > 0 else -1.0)
            if signs:
                dominant = 1.0 if sum(signs) >= 0 else -1.0
                sign_consistent = sum(1 for s in signs if s == dominant) / len(signs) >= 0.60
        checks["cross_seed_sign_consistency"] = sign_consistent
        if holdout:
            deltas = [_num(d) for d in (holdout.get("holdout_deltas") or [])]
            deltas = [d for d in deltas if d is not None]
            checks["independent_holdout_evaluated"] = bool(deltas)
            agree = _num(holdout.get("holdout_direction_agreement"))
            checks["holdout_direction_agreement"] = agree is not None and agree >= 0.75
            checks["holdout_not_used_for_selection"] = (
                holdout.get("holdout_used_for_decision") is False)
        else:
            checks["independent_holdout_evaluated"] = False
            checks["holdout_direction_agreement"] = False
            checks["holdout_not_used_for_selection"] = False
        checks["no_guardrail_violations"] = bool(comps) and all(
            not (c.get("guardrail_violations") or []) for c in comps)
        ood = (ev.telemetry or {}).get("ood")
        checks["ood_task_confirmed"] = bool(ood) and bool(ood.get("confirmed"))
        # A research finding that does NOT survive an independent holdout is
        # weaker evidence than one that does. Measured on the RESERVED split,
        # never declared by the system being evaluated.
        h_delta = _num((ev.holdout or {}).get("mean_holdout_delta"))
        checks["holdout_confirmed_improvement"] = (
            bool((ev.holdout or {}).get("evaluated"))
            and h_delta is not None and h_delta > 0.0)
        return _fraction(checks.values()), checks

    def _discovery_value(self, ev: QualityEvidence) -> tuple[float, dict[str, bool]]:
        report = ev.report or {}
        decisions = [(c.get("decision") if isinstance(c, dict) else str(c))
                     for c in (report.get("comparisons") or report.get("decisions") or [])]
        credits: list[float] = []
        for item in (report.get("comparisons") or []):
            decision = (item.get("decision") if isinstance(item, dict) else str(item))
            delta = _num(item.get("mean_delta")) if isinstance(item, dict) else None
            if decision == "rejected":
                # Falsifying a wrong hypothesis with paired evidence is a REAL
                # scientific discovery (strongly positive, NOT zero as the old
                # accepted/n score did). Finding a working, holdout-confirmed
                # improvement is worth slightly more -- otherwise a runtime
                # that only ever proposes easy-to-refute strawmen would tie a
                # runtime that actually advances the science.
                credits.append(0.75)
            elif decision == "accepted_for_confirmation":
                credits.append(1.0 if (delta is not None
                                       and abs(delta) > self.practical_effect_threshold)
                               else 0.7)
            elif decision == "inconclusive":
                credits.append(0.15)
            else:
                credits.append(0.0)
        decision_credit = sum(credits) / len(credits) if credits else 0.0
        hyps = list(report.get("hypotheses") or [])
        refs = list(report.get("evidence_refs") or [])
        distinct_interventions = len({
            (item.get("candidate_id") if isinstance(item, dict) else str(item))
            for item in (report.get("comparisons") or [])
        })
        checks = {
            "decisive_verdicts": bool(credits) and decision_credit >= 0.70,
            "negative_evidence_retained": any(
                d in ("rejected", "inconclusive") for d in decisions),
            "hypotheses_are_evidence_grounded": bool(hyps) and len(refs) >= len(hyps),
            "competing_interventions_tested": distinct_interventions >= 2,
            "experience_written_back": bool(report.get("experience_refs")),
        }
        # A decisive, well-evidenced falsification is worth more than a bare
        # acceptance: this is the explicit anti-Goodhart term.
        score = (0.55 * decision_credit
                 + 0.15 * (1.0 if checks["negative_evidence_retained"] else 0.0)
                 + 0.10 * (1.0 if checks["hypotheses_are_evidence_grounded"] else 0.0)
                 + 0.10 * (1.0 if checks["competing_interventions_tested"] else 0.0)
                 + 0.10 * (1.0 if checks["experience_written_back"] else 0.0))
        return _clamp(score), checks

    def _cost_efficiency(self, ev: QualityEvidence) -> tuple[float, dict[str, Any]]:
        """Return a PENALTY in [0,1]: 0 = efficient, 1 = wasteful."""
        tele = ev.telemetry or {}
        if not tele:
            return 0.5, {"reason": "no cost telemetry supplied; unknown cost is not free"}
        budget = tele.get("budget") or {}
        jobs = _num(tele.get("jobs_used")) or 0.0
        max_jobs = _num(budget.get("max_jobs")) or 0.0
        elapsed = _num(tele.get("elapsed_seconds")) or 0.0
        max_seconds = _num(budget.get("max_total_seconds")) or 0.0
        total = _num(tele.get("total_trials")) or 0.0
        succeeded = _num(tele.get("succeeded_trials")) or 0.0
        parts = [
            _clamp(jobs / max_jobs) if max_jobs > 0 else 0.5,
            _clamp(elapsed / max_seconds) if max_seconds > 0 else 0.5,
            _clamp(1.0 - (succeeded / total)) if total > 0 else 1.0,
        ]
        return _clamp(sum(parts) / len(parts)), {
            "jobs_used": jobs, "max_jobs": max_jobs,
            "elapsed_seconds": elapsed, "max_total_seconds": max_seconds,
            "total_trials": total, "succeeded_trials": succeeded,
        }

    # -------------------------------------------------------------- evaluate
    def evaluate(self, evidence: QualityEvidence) -> QualityScore:
        E, e_checks = self._evidence_integrity(evidence)
        R, r_checks = self._reproducibility(evidence)
        G, g_checks = self._generalization(evidence)
        D, d_checks = self._discovery_value(evidence)
        C, cost_detail = self._cost_efficiency(evidence)

        checks: dict[str, bool] = {}
        for prefix, group in (("E.", e_checks), ("R.", r_checks),
                              ("G.", g_checks), ("D.", d_checks)):
            for key, value in group.items():
                checks[prefix + key] = bool(value)

        hard: list[str] = []
        if E < self.hard.min_evidence_integrity:
            hard.append(f"evidence_integrity {E:.3f} < "
                        f"{self.hard.min_evidence_integrity:.3f}")
        if R < self.hard.min_reproducibility:
            hard.append(f"reproducibility {R:.3f} < "
                        f"{self.hard.min_reproducibility:.3f}")

        metrics: dict[str, Any] = {}
        for result in evidence.experiment_results:
            if isinstance(result.get("metrics"), dict):
                metrics.update(result["metrics"])
        for rule in self.metric_rules:
            status, text = rule.verdict(metrics)
            checks[f"M.{rule.metric}"] = status == "passed"
            if status == "violated" and rule.hard:
                hard.append(f"{rule.label or rule.metric}: {text}")

        w = self.weights.as_dict()
        total_w = sum(w.values())
        raw = (w["evidence"] * E + w["reproducibility"] * R
               + w["generalization"] * G + w["discovery"] * D
               - w["cost"] * C) / total_w
        Q = _clamp(raw)
        if hard:
            Q = min(Q, self.hard.max_q_when_violated)
        return QualityScore(
            Q=Q,
            components={"E": round(E, 6), "R": round(R, 6), "G": round(G, 6),
                        "D": round(D, 6), "C_penalty": round(C, 6)},
            weights=w, checks=checks, hard_violations=hard,
            gate_eligible=not hard, evaluator_name=self.name,
            domain=self.domain, protocol_version=self.version,
            evidence_fingerprint=evidence.fingerprint(),
            detail={
                "raw_Q_before_hard_cap": round(raw, 6),
                "cost": cost_detail,
                "metric_rules": [r.metric for r in self.metric_rules],
                "practical_effect_threshold": self.practical_effect_threshold,
                "n_comparison_reports": len(evidence.comparison_reports),
                "n_experiment_results": len(evidence.experiment_results),
            },
        )


# --------------------------------------------------------------------------- #
# Domain-specific evaluators: SAME protocol, different scientific rules.
# --------------------------------------------------------------------------- #

SCINET_RULES = (
    MetricRule("cosine@1", "maximize", 0.40, hard=True,
               label="representation quality floor"),
    MetricRule("latent_mse@1", "minimize", 0.10, hard=False,
               label="one-step latent prediction error"),
    MetricRule("physics_consistency", "maximize", 0.0, hard=True,
               label="physical consistency must be non-negative"),
)
EMBODIED_RULES = (
    MetricRule("safety_violations", "minimize", 0.0, hard=True,
               label="simulated robot safety constraint"),
    MetricRule("task_success_rate", "maximize", 0.0, hard=True,
               label="task success must be non-negative"),
    MetricRule("state_mse@1", "minimize", 1.0, hard=False,
               label="one-step state prediction error"),
)
QUANT_RULES = (
    MetricRule("lookahead_leakage", "minimize", 0.0, hard=True,
               label="strict no-future-information constraint"),
    MetricRule("max_drawdown", "minimize", 0.60, hard=True,
               label="risk limit"),
    MetricRule("net_return_after_cost", "maximize", -0.50, hard=True,
               label="net-of-cost return floor"),
    MetricRule("oos_return", "maximize", -0.50, hard=False,
               label="out-of-sample return"),
)


def scinet_quality_evaluator() -> DefaultResearchQualityEvaluator:
    return DefaultResearchQualityEvaluator(
        domain="scinet", name="scinet_physics_quality_v1",
        weights=QualityWeights(evidence=0.25, reproducibility=0.25,
                               generalization=0.25, discovery=0.20, cost=0.05),
        metric_rules=SCINET_RULES,
    )


def embodied_quality_evaluator() -> DefaultResearchQualityEvaluator:
    return DefaultResearchQualityEvaluator(
        domain="embodied", name="embodied_world_model_quality_v1",
        weights=QualityWeights(evidence=0.25, reproducibility=0.20,
                               generalization=0.25, discovery=0.25, cost=0.05),
        metric_rules=EMBODIED_RULES,
    )


def quant_quality_evaluator() -> DefaultResearchQualityEvaluator:
    return DefaultResearchQualityEvaluator(
        domain="quant", name="quant_factor_quality_v1",
        weights=QualityWeights(evidence=0.30, reproducibility=0.25,
                               generalization=0.25, discovery=0.15, cost=0.05),
        metric_rules=QUANT_RULES,
    )


_EVALUATORS: dict[str, Callable[[], ResearchQualityEvaluator]] = {
    "scinet": scinet_quality_evaluator,
    "embodied": embodied_quality_evaluator,
    "quant": quant_quality_evaluator,
}


def register_quality_evaluator(domain: str, factory: Callable[[], ResearchQualityEvaluator]) -> None:
    """Additive extension point: a new domain supplies its own rules."""
    if not domain or not callable(factory):
        raise ValueError("domain and factory are required")
    _EVALUATORS[domain] = factory


def quality_evaluator_for(domain: str) -> ResearchQualityEvaluator:
    factory = _EVALUATORS.get(domain)
    if factory is None:
        return DefaultResearchQualityEvaluator(domain=domain or "generic")
    return factory()


def evaluate_research_quality(evidence: QualityEvidence, *,
                              domain: str = "",
                              evaluator: ResearchQualityEvaluator | None = None
                              ) -> QualityScore:
    """Convenience entry point used by the trace bridge, CLI and Harness gate."""
    ev = evaluator or quality_evaluator_for(domain)
    return ev.evaluate(evidence)
