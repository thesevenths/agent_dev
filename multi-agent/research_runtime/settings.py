"""Environment-driven wiring for the external scientific engine.

The Agent Runtime (Python 3.13 + LangGraph, no PyTorch) and the model engine
(conda scinet_gpu, torch+cuda) are DIFFERENT interpreters. Never import the
engine's modules here; only record how to launch its CLI.

Override with environment variables::

    SCINET_DIR        path to the SciNet++ checkout
    SCINET_PYTHON     python executable that has torch / scipy / PyYAML
    SCINET_CONFIG     default config relative to SCINET_DIR
    RESEARCH_DATA_DIR where evidence / model registry live (default: here)
"""
from __future__ import annotations

import os
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent

DEFAULT_SCINET_DIR = Path(os.getenv("SCINET_DIR", r"E:\LLM\SciNet++"))
DEFAULT_SCINET_PYTHON = Path(
    os.getenv("SCINET_PYTHON", r"D:\conda_envs\scinet_gpu\python.exe")
)
DEFAULT_CONFIG = os.getenv("SCINET_CONFIG", "configs/pendulum_smoke.yaml")
DATA_DIR = Path(os.getenv("RESEARCH_DATA_DIR", str(RUNTIME_DIR / "_data")))

CLI_TIMEOUT_S = float(os.getenv("RESEARCH_CLI_TIMEOUT_S", "60"))
WAIT_TIMEOUT_S = float(os.getenv("RESEARCH_WAIT_TIMEOUT_S", "3600"))


class ConfigurationError(RuntimeError):
    pass


def resolve_engine(*, project_dir: str | Path | None = None,
                   python_executable: str | Path | None = None,
                   strict: bool = True) -> tuple[Path, Path]:
    """Return (project_dir, python_executable), checking them when ``strict``."""
    project = Path(project_dir) if project_dir else DEFAULT_SCINET_DIR
    python = Path(python_executable) if python_executable else DEFAULT_SCINET_PYTHON
    if strict:
        if not project.is_dir():
            raise ConfigurationError(
                f"SCINET_DIR not found: {project}. Set SCINET_DIR to the "
                "SciNet++ checkout, or pass project_dir=..."
            )
        if not (project / "research_api" / "cli.py").is_file():
            raise ConfigurationError(
                f"{project} has no research_api/cli.py -- the engine side of "
                "the P0 integration has not been installed."
            )
        if not python.is_file():
            raise ConfigurationError(
                f"SCINET_PYTHON not found: {python}. Set SCINET_PYTHON to the "
                "interpreter with torch installed."
            )
    return project, python


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
