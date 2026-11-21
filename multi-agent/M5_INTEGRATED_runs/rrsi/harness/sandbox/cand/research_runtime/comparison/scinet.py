"""DOMAIN WIRING for SciNet++ -- the ONLY comparison module allowed to name
engine-specific fields, dataset layouts or metric names.

Every other module in ``comparison/`` is a generic research runtime: it knows
about paired observations, deltas, intervals and guardrail RULE FORMS, but not
about a world model, cosine similarity or a frozen pendulum dataset. A
robotics or finance domain supplies a different wiring module without touching
the comparator.

Thresholds here were fixed BEFORE running the P1-05 acceptance experiment and
were NOT tuned to obtain an ``accepted_for_confirmation`` verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..campaign.campaign_store import CampaignStore
from ..candidate_config import sha256_file
from ..settings import DATA_DIR, DEFAULT_SCINET_DIR
from .comparator import CampaignComparator
from .report_store import ComparisonReportStore
from .schemas import ComparisonPolicy


def scinet_comparison_policy(**overrides: Any) -> ComparisonPolicy:
    """Decision policy for the frozen pendulum smoke experiments.

    ``min_paired_seeds=5`` is the P1-05 engineering gate (fewer seeds still get
    a full statistical report, but never an acceptance decision).
    ``practical_effect_threshold=0.0`` is deliberate: no domain-supplied
    minimum detectable effect exists yet, so the honest requirement is a
    statistically significant improvement. A real effect size should come
    from a power analysis, not from looking at the results first.
    """
    kwargs: dict[str, Any] = dict(
        min_paired_seeds=5,
        familywise_alpha=0.05,
        practical_effect_threshold=0.0,
        practical_effect_mode="absolute",
        multiple_comparison_method="holm",
    )
    kwargs.update(overrides)
    return ComparisonPolicy(**kwargs)


# Absolute floor from the P0/P1-04 measured baseline cosine@1 (~0.54), with
# headroom; relative tolerance guards against a candidate that improves the
# primary metric while degrading representation quality. Both are declared in
# the ExperimentPlan by its author -- the Comparator never invents a threshold.
SCINET_DEFAULT_GUARDRAILS: dict[str, dict[str, Any]] = {
    "cosine@1": {
        "min": 0.40,
        "max_relative_regression_pct": 5.0,
        "direction": "maximize",
    },
}


def make_scinet_dataset_evidence(project_dir: str | Path | None = None):
    """Read the frozen manifest's per-split SHA-256 digests from DISK.

    The comparison report records real split hashes, not just a shared
    ``dataset_version`` string. A tampered split file is detected here.
    """
    root = Path(project_dir or DEFAULT_SCINET_DIR)

    def provider(record: Any) -> dict[str, str]:
        base = (record.run_spec or {}).get("baseline_config_path")
        if not base:
            raise ValueError("campaign run_spec has no baseline_config_path")
        config_path = Path(base)
        if not config_path.is_absolute():
            config_path = root / config_path
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        frozen_dir = (config.get("data") or {}).get("frozen_dir")
        if not frozen_dir:
            raise ValueError("baseline config has no data.frozen_dir")
        frozen_root = (root / frozen_dir).resolve()
        manifest_path = frozen_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_version") != record.dataset_version:
            raise ValueError("frozen manifest dataset_version != campaign")
        out: dict[str, str] = {"manifest.json": sha256_file(manifest_path)}
        for name, info in (manifest.get("splits") or {}).items():
            split_path = frozen_root / str(info["file"])
            digest = sha256_file(split_path)
            if digest != info.get("sha256"):
                raise ValueError(f"frozen split {name!r} failed its SHA-256 check")
            out[str(name)] = digest
        return out

    return provider


def make_scinet_config_hash_verifier(project_dir: str | Path | None = None):
    """Recompute each trial's config.yaml hash on disk before comparing."""
    root = Path(project_dir or DEFAULT_SCINET_DIR)

    def verifier(trial: Any) -> str | None:
        relpath = trial.config_relpath or trial.config_path
        path = Path(relpath)
        if not path.is_absolute():
            path = root / path
        return sha256_file(path) if path.is_file() else None

    return verifier


def build_scinet_comparator(*, campaign_store: CampaignStore | None = None,
                            report_store: ComparisonReportStore | None = None,
                            project_dir: str | Path | None = None,
                            policy: ComparisonPolicy | None = None,
                            **overrides: Any) -> CampaignComparator:
    """Convenience factory for real comparison runs. Tests inject their own."""
    root = Path(project_dir or DEFAULT_SCINET_DIR)
    store = campaign_store or CampaignStore(DATA_DIR / "campaigns.sqlite")
    reports = report_store or ComparisonReportStore(
        root / "outputs" / "research_campaigns")
    kwargs: dict[str, Any] = dict(
        policy=policy or scinet_comparison_policy(),
        dataset_evidence=make_scinet_dataset_evidence(root),
        config_hash_verifier=make_scinet_config_hash_verifier(root),
    )
    kwargs.update(overrides)
    return CampaignComparator(store, reports, **kwargs)
