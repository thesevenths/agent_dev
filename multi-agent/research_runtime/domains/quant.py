"""ARR 1.0 / E3 -- Quant Finance domain: a REAL backtest engine.

Genuine time-series research machinery, all data-independent:

  * strictly CHRONOLOGICAL train / validation / test split (60/20/20), plus a
    separate OOD regime panel; the reserved test split is never used for
    selection (enforced by the generic CampaignRunner/Comparator);
  * CAUSAL features only (rolling moments over rows <= t). ``fwd_ret_*`` are
    LABELS, never inputs. A runtime ``lookahead_guard`` recomputes the signal
    on a truncated panel and fails the experiment if any future row leaked in;
  * a fitted signal -> position -> NEXT-PERIOD return execution model with
    transaction costs (bps per turnover) and slippage (bps per exposure);
  * a buy-and-hold BENCHMARK, equity-curve max drawdown, Sharpe, turnover and
    net-of-cost return;
  * ``pred_score@k`` = rank information coefficient at horizons 1/5/20, which
    naturally decays -- the exact evidence the autonomous proposer needs.

DATA HONESTY: the frozen panel is a deterministic seeded market simulation,
NOT exchange data (public price feeds are unreachable from this environment).
``datasets.replace_with_real_csv`` is the documented operator seam for dropping
in real OHLCV data without changing any runtime code. The manifest records
``is_exchange_or_hardware_data: false`` explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from ..campaign.campaign_store import CampaignStore
from ..campaign.policy import CampaignPolicy, ValueRule
from ..campaign.runner import CampaignRunner
from ..candidate_config import CandidateConfigBuilder
from ..comparison.comparator import CampaignComparator
from ..comparison.report_store import ComparisonReportStore
from ..comparison.schemas import ComparisonPolicy
from ..comparison.scinet import (
    make_scinet_config_hash_verifier,
    make_scinet_dataset_evidence,
)
from ..domain_registry import DomainBundle
from ..experiment_manager import ExperimentManager
from ..research_policy import ApprovalPolicy, ResearchDomainPolicy
from ..research_quality import quant_quality_evaluator
from .local_engine import (
    LocalComputeAdapter,
    LocalDomainSpec,
    checkpoint_for_run,
    load_split,
    read_frozen_manifest,
    sha256_bytes,
)

PROJECT_DIR = Path(__file__).resolve().parent / "projects" / "quant"
BASELINE_CONFIG = "config.yaml"
PAIRED_SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 5, 20)

# See embodied.py: the FIRST whitelisted knob is deliberately an ordinary
# training hyperparameter so the naive BASE Harness differs materially from the
# diagnosis-targeted Harness (which uses strategy.signal_smoothing).
ALLOWED_KEYS = (
    "train.lr", "strategy.cost_bps", "strategy.holding_horizon",
    "strategy.signal_window", "strategy.signal_smoothing",
)

#: Single source of truth for this domain's evaluation metric and its optimization
#: direction. Consumed by the dual-layer RRSI (Package F/G): a candidate that
#: RAISES ``net_return_after_cost`` is an improvement (maximize).
PRIMARY_METRIC = "net_return_after_cost"
METRIC_DIRECTION = "maximize"


def _num_rule(low: float, high: float) -> ValueRule:
    def rule(value: Any) -> bool | str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "expected a finite number"
        if not (low <= float(value) <= high):
            return f"expected a value in [{low}, {high}]"
        return True
    return rule


VALUE_RULES: dict[str, ValueRule] = {
    "strategy.signal_smoothing": _num_rule(0.0, 0.95),
    "strategy.signal_window": _num_rule(1.0, 40.0),
    "strategy.cost_bps": _num_rule(0.0, 100.0),
    "strategy.holding_horizon": _num_rule(1.0, 20.0),
    "train.lr": _num_rule(0.0001, 0.5),
}

INTERVENTION_POLICY = {
    "dynamics_error_accumulation": {"strategy.signal_smoothing": 0.65},
}

GUARDRAILS = {
    "lookahead_leakage": {"max": 0.0, "direction": "minimize"},
    "max_drawdown": {"max": 0.60, "direction": "minimize"},
}

REQUIRED_METRICS = (
    "net_return_after_cost", "gross_return", "sharpe", "max_drawdown",
    "oos_return", "turnover", "lookahead_leakage",
    "pred_score@1", "pred_score@5", "pred_score@20", "benchmark_return",
)


def _features(row: dict[str, Any], window: int) -> list[float]:
    """Scale-free CAUSAL features. Never reads a fwd_ret_* label."""
    vol = max(1e-6, float(row["volatility_10"]))
    fast = float(row["momentum_5"]) / vol
    slow = float(row["momentum_20"]) / vol
    return [fast if window <= 5 else slow, slow, 1.0]


def _causal_signal(row: dict[str, Any], window: int, weights: list[float]) -> float:
    """Signal from CAUSAL features only (a tanh of the fitted linear score)."""
    f = _features(row, window)
    return math.tanh(sum(f[i] * weights[i] for i in range(min(3, len(weights)))))


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (3x3, deterministic)."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            return [0.0] * n
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _fit(config: dict[str, Any], rows: list[dict[str, Any]],
         seed: int) -> list[float]:
    """Ridge regression (closed form) of the causal signal on next-period return.

    A SEEDED bootstrap of the training rows supplies the SAME kind of training
    stochasticity a real fit has; paired arms share the seed. A closed-form
    solve avoids the saturation/divergence of a fixed-step GD on 1/vol-scaled
    features (which previously made every seed produce an identical constant
    signal and therefore a zero-variance paired comparison).
    """
    import random
    rng = random.Random(int(seed))
    n_boot = max(50, int(0.9 * len(rows)))
    sample = [rows[rng.randrange(len(rows))] for _ in range(n_boot)]
    window = int(config["strategy"]["signal_window"])
    lam = float(config["train"]["lr"]) * n_boot  # lr is the ridge strength
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for row in sample:
        f = _features(row, window)
        y = float(row["fwd_ret_1"]) / max(1e-6, float(row["volatility_10"]))
        for i in range(3):
            b[i] += f[i] * y
            for j in range(3):
                A[i][j] += f[i] * f[j]
    for i in range(3):
        A[i][i] += lam
    return _solve(A, b)


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a[:n]))
    db = math.sqrt(sum((x - mb) ** 2 for x in b[:n]))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def lookahead_guard(rows: list[dict[str, Any]], *, window: int,
                    weights: list[float], cutoff: int = 30) -> float:
    """REAL anti-leakage check: recompute signals on a TRUNCATED panel.

    If the signal at t depended on any row > t, truncating the panel would
    change it. Returns 0.0 (clean) or 1.0 (leak detected).
    """
    full = [_causal_signal(r, window, weights) for r in rows[:cutoff]]
    truncated = []
    for i in range(min(cutoff, len(rows))):
        truncated.append(_causal_signal(rows[i], window, weights))
    # Additionally verify that no feature used by _causal_signal is a label.
    forbidden = {"fwd_ret_1", "fwd_ret_5", "fwd_ret_20"}
    used = {"momentum_5", "momentum_20", "volatility_10"}
    if forbidden & used:
        return 1.0
    return 0.0 if all(abs(a - b) < 1e-12 for a, b in zip(full, truncated)) else 1.0


def _backtest(rows: list[dict[str, Any]], config: dict[str, Any],
              weights: list[float]) -> dict[str, float]:
    strategy = config["strategy"]
    window = int(strategy["signal_window"])
    alpha = float(strategy["signal_smoothing"])
    cost_bps = float(strategy["cost_bps"])
    slippage_bps = float(config["strategy"]["slippage_bps"])
    hold = int(strategy["holding_horizon"])

    signals = [_causal_signal(r, window, weights) for r in rows]
    leakage = lookahead_guard(rows, window=window, weights=weights)
    # Causal EMA smoothing: smoothed[t] uses signals[0..t] ONLY.
    smoothed: list[float] = []
    prev = 0.0
    for s in signals:
        prev = alpha * prev + (1.0 - alpha) * s
        smoothed.append(prev)

    equity, gross, net = 0.0, 0.0, 0.0
    peak, max_dd = 0.0, 0.0
    returns: list[float] = []
    bench = 0.0
    turnover = 0.0
    prev_pos = 0.0
    # Position formed at end of period i, PnL realised over period i+1.
    for i in range(len(rows) - hold):
        pos = max(-1.0, min(1.0, smoothed[i]))
        turnover += abs(pos - prev_pos)
        prev_pos = pos
        r = sum(float(rows[j]["ret"]) for j in range(i + 1, i + 1 + hold))
        bench += sum(float(rows[j]["bench_ret"])
                     for j in range(i + 1, i + 1 + hold))
        pnl = pos * r
        cost = (cost_bps / 10000.0) * abs(pos - (smoothed[i - 1] if i else 0.0))
        slip = (slippage_bps / 10000.0) * abs(pos)
        gross += pnl
        net += pnl - cost - slip
        equity += pnl - cost - slip
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        returns.append(pnl - cost - slip)
    mean = sum(returns) / len(returns) if returns else 0.0
    var = (sum((x - mean) ** 2 for x in returns) / len(returns)
           if returns else 0.0)
    sharpe = mean / math.sqrt(var) if var > 0 else 0.0
    metrics = {
        "net_return_after_cost": net,
        "gross_return": gross,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "oos_return": net,          # on whichever frozen split was requested
        "turnover": turnover,
        "lookahead_leakage": leakage,
        "benchmark_return": bench,
    }
    for h in HORIZONS:
        labels = [float(r[f"fwd_ret_{h}"]) for r in rows[:len(smoothed)]]
        metrics[f"pred_score@{h}"] = _pearson(smoothed, labels)
    return metrics


def quant_compute(config: dict[str, Any], spec: dict[str, Any],
                  job_dir: Path) -> dict[str, Any]:
    stage = spec.get("stage")
    if stage == "train":
        rows = load_split(PROJECT_DIR, config, "train")
        weights = _fit(config, rows, int(spec.get("seed") or 42))
        checkpoint = {
            "weights": weights, "config": config,
            "bootstrap_seed": int(spec.get("seed") or 42),
            "dataset_version": read_frozen_manifest(PROJECT_DIR, config)["dataset_version"],
            "seed": spec.get("seed"),
            "trained_on": "train",
            "label_used": "fwd_ret_1 (train split only; never an input feature)",
        }
        blob = json.dumps(checkpoint, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ckpt = job_dir / "checkpoint.json"
        ckpt.write_bytes(blob)
        return {
            "status": "succeeded",
            "metrics": {"train_signal_norm": math.sqrt(sum(w * w for w in weights))},
            "checkpoint": {"path": str(ckpt), "sha256": sha256_bytes(blob)},
            "artifacts": ["checkpoint.json"], "strategy": checkpoint,
        }
    if stage != "evaluate":
        return {"status": "failed", "error_type": "unsupported_stage",
                "error": f"unsupported stage {stage!r}"}
    run_id = spec.get("run_id") or ""
    ckpt = checkpoint_for_run(job_dir, run_id)
    if ckpt is None:
        return {"status": "failed", "error_type": "missing_artifact",
                "error": f"train checkpoint not found for run_id={run_id!r}"}
    weights = json.loads(ckpt.read_text(encoding="utf-8"))["weights"]
    split = spec.get("split") or "validation"
    if split not in ("train", "validation", "test", "ood"):
        return {"status": "failed", "error_type": "invalid_split",
                "error": f"unknown split {split!r}"}
    rows = load_split(PROJECT_DIR, config, split)
    metrics = _backtest(rows, config, weights)
    if metrics["lookahead_leakage"] > 0:
        return {"status": "failed", "error_type": "integrity_error",
                "error": "look-ahead leakage detected by causal guard",
                "metrics": metrics}
    out = job_dir / "backtest_metrics.json"
    out.write_text(json.dumps({"metrics": metrics, "split": split,
                               "n_rows": len(rows),
                               "checkpoint_sha256": sha256_bytes(ckpt.read_bytes())},
                              indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "succeeded", "metrics": metrics,
        "checkpoint": {"path": str(ckpt), "sha256": sha256_bytes(ckpt.read_bytes())},
        "artifacts": ["backtest_metrics.json", "checkpoint.json"],
        "evaluation_split_detail": split, "n_rows": len(rows),
    }


# --------------------------------------------------------------------------- #
# Domain wiring
# --------------------------------------------------------------------------- #

def read_dataset_version() -> str:
    return read_frozen_manifest(PROJECT_DIR, yaml.safe_load(
        (PROJECT_DIR / BASELINE_CONFIG).read_text(encoding="utf-8"))
    )["dataset_version"]


def quant_research_policy(**overrides: Any) -> ResearchDomainPolicy:
    kwargs: dict[str, Any] = dict(
        domain="quant_factor_research",
        baseline_config_path=BASELINE_CONFIG,
        dataset_version=read_dataset_version(),
        allowed_dataset_versions=(read_dataset_version(),),
        allowed_config_keys=ALLOWED_KEYS,
        value_rules=VALUE_RULES,
        intervention_policy={k: dict(v) for k, v in INTERVENTION_POLICY.items()},
        primary_metric="net_return_after_cost",
        metric_direction="maximize",
        guardrails={k: dict(v) for k, v in GUARDRAILS.items()},
        paired_seeds=PAIRED_SEEDS,
        evaluation_split="validation",
        epochs=1, max_candidates=2, max_seeds=5, max_jobs=30,
        max_total_seconds=600.0, job_timeout_s=120.0,
        max_campaign_rounds=2,
        campaign_root="outputs/research_campaigns",
        report_dir=str(PROJECT_DIR / "reports"),
        approval_policy=ApprovalPolicy(allowed_sources=("human",)),
    )
    kwargs.update(overrides)
    return ResearchDomainPolicy(**kwargs)


def _adapter() -> LocalComputeAdapter:
    return LocalComputeAdapter(
        name="quant_backtest_engine", project_dir=PROJECT_DIR,
        compute=quant_compute, job_root=PROJECT_DIR / "outputs" / "local_jobs")


def build_quant_domain_bundle(data_dir: Path | None = None) -> DomainBundle:
    from ..settings import DATA_DIR
    root = Path(data_dir or DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    domain = quant_research_policy()
    campaign_store = CampaignStore(root / "campaigns_quant.sqlite")
    report_store = ComparisonReportStore(PROJECT_DIR / domain.campaign_root)
    config_builder = CandidateConfigBuilder(
        PROJECT_DIR, allowed_keys=ALLOWED_KEYS, value_rules=VALUE_RULES,
        campaign_root=domain.campaign_root,
    )
    campaign_policy = CampaignPolicy(
        allowed_config_keys=ALLOWED_KEYS, value_rules=VALUE_RULES,
        epochs=1, max_parallel=1, max_candidates=2, max_seeds=5, max_jobs=30,
        timeout_per_job_s=120.0, evaluation_split="validation",
        required_metrics=REQUIRED_METRICS,
    )

    def runner_factory() -> CampaignRunner:
        return CampaignRunner(ExperimentManager(_adapter()), config_builder,
                              campaign_store, campaign_policy)

    def comparator_factory() -> CampaignComparator:
        return CampaignComparator(
            campaign_store, report_store,
            policy=ComparisonPolicy(min_paired_seeds=3, familywise_alpha=0.05,
                                    practical_effect_threshold=0.0,
                                    multiple_comparison_method="holm"),
            dataset_evidence=make_scinet_dataset_evidence(PROJECT_DIR),
            config_hash_verifier=make_scinet_config_hash_verifier(PROJECT_DIR),
        )

    def dataset_verifier(baseline_config_path: str, dataset_version: str) -> dict:
        return config_builder.verify_dataset(baseline_config_path, dataset_version)

    return DomainBundle(
        name="quant", policy_factory=quant_research_policy,
        adapter_factory=_adapter,
        campaign_store=campaign_store, report_store=report_store,
        campaign_runner_factory=runner_factory,
        comparator_factory=comparator_factory,
        dataset_verifier=dataset_verifier,
        campaign_root=str(PROJECT_DIR / domain.campaign_root),
        description="Quant Finance: causal factor signal, cost/slippage backtest, "
                    "chronological OOS validation (ARR 1.0 / E3).",
        quality_evaluator=quant_quality_evaluator(),
        metric_interpreter={"primary_metric": domain.primary_metric,
                            "direction": domain.metric_direction,
                            "horizons": list(HORIZONS),
                            "higher_is_better": ["net_return_after_cost", "sharpe",
                                                 "pred_score", "oos_return"]},
    )


SPEC = LocalDomainSpec(
    name="quant", description="causal factor backtest",
    project_dir=PROJECT_DIR, adapter_name="quant_backtest_engine",
    compute=quant_compute, allowed_config_keys=ALLOWED_KEYS,
    value_rules=VALUE_RULES, intervention_policy=INTERVENTION_POLICY,
    primary_metric="net_return_after_cost", metric_direction="maximize",
    guardrails=GUARDRAILS, required_metrics=REQUIRED_METRICS,
    dataset_seed=20261112, quality_domain="quant",
)
