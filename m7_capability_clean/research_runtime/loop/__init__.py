"""ARR 1.1 / Package G -- autonomous research-loop orchestration.

One command runs many rounds of the dual-layer RRSI (Package F); the next round
inherits whatever the previous round ACCEPTED (domain config baked into a new
baseline, harness patches applied via the real patch primitive), and a crashed
process resumes from the last completed round because a checkpoint is persisted
after every round. See ``controller.run_loop`` and ``cli loop``.
"""
from __future__ import annotations

from .controller import run_loop
from .loop_report import build_loop_report, summarise_text
from .loop_state import LoopState, VersionRecord, load, new_state, save
from .recovery import load_or_init, mark_completed, mark_failed
from .research_cycle import run_research_cycle
from .improvement_cycle import derive_next_version
from .version_inheritance import bake_baseline, default_harness_inherit

__all__ = [
    "run_loop",
    "run_research_cycle",
    "derive_next_version",
    "build_loop_report",
    "summarise_text",
    "bake_baseline",
    "default_harness_inherit",
    "load_or_init",
    "mark_completed",
    "mark_failed",
    "LoopState",
    "VersionRecord",
    "load",
    "new_state",
    "save",
]
