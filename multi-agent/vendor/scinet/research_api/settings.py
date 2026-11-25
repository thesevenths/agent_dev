"""Path / config resolution shared by the research_api modules.

Kept deliberately free of PyTorch imports: the CLI layer must also be callable
from a lightweight Python environment. Only the worker subprocesses (which
execute ``train.py`` / ``evaluate.py``) need the full SciNet++ environment.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = PROJECT_DIR / "outputs" / "agent_jobs"
DEFAULT_DB_PATH = JOBS_DIR / "jobs.sqlite"

# Kept here (not imported from data.dataset) because this module must stay
# importable in a lightweight interpreter without torch. A regression test
# asserts this tuple stays in sync with data.dataset.SPLITS.
DATA_SPLITS = ("train", "validation", "adapt", "test")


def resolve_db_path(db: str | None = None) -> Path:
    path = Path(db) if db else DEFAULT_DB_PATH
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def job_log_path(job_id: str) -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR / f"{job_id}.log"


def resolve_config_path(config: str) -> Path:
    """Config paths are relative to the SciNet++ project root."""
    path = Path(config)
    return path if path.is_absolute() else PROJECT_DIR / path


def load_config_meta(config: str) -> dict:
    """Read only the fields needed for path resolution (no model imports)."""
    path = resolve_config_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return {
        "path": str(path),
        "experiment": str(cfg.get("experiment", "default")),
        "output_dir": str(cfg.get("output_dir", "outputs")),
        "seed": int(cfg.get("seed", 42)),
        "data": cfg.get("data", {}),
    }


def run_dir_for(config_meta: dict, run_id: str) -> Path:
    """Mirror ``utils.config.ensure_output_dir``: <output_dir>/<exp>/<run_id>."""
    base = Path(config_meta["output_dir"])
    if not base.is_absolute():
        base = PROJECT_DIR / base
    return base / config_meta["experiment"] / run_id
