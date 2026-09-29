"""Per-run structured logging for the multi-agent supervisor/graph.

Each user run (identified by the thread's ``memory_key``) gets its own
timestamped log file under ``multi-agent/log/`` so the important supervisor /
agent events survive process restarts and can be inspected independently of the
``langgraph dev`` console.

Usage:
    from runlog import start_run, ensure_run, set_current, log_event

    start_run(memory_key)                 # begin a new run -> fresh timestamped file
    ensure_run(memory_key)                # create file if this thread has none yet
    set_current(memory_key)               # remember current thread (for tools without state)
    log_event("some important event")     # append to the current run's file
"""
import logging
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# keyed by memory_key (thread id); each value is a logging.Logger writing to one
# timestamped file. A fresh file is created per run so re-submits don't mix.
_loggers: dict[str, logging.Logger] = {}
_current_key: str | None = None


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name or "default")


def _create(key: str) -> logging.Logger:
    """Create a NEW timestamped file logger for ``key`` and return it (cached)."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = LOG_DIR / f"{_safe(key)}_{ts}.log"
    lg = logging.getLogger(f"runlog.{key}.{ts}")
    lg.setLevel(logging.INFO)
    lg.propagate = False  # never double-emit to the console root logger
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(fh)
    _loggers[key] = lg
    lg.info(f"=== run log started: {path} ===")
    return lg


def start_run(memory_key) -> str:
    """Begin a new run: create a fresh timestamped file for this thread."""
    global _current_key
    key = memory_key or "default"
    _current_key = key
    lg = _create(key)
    return str(lg.handlers[0].baseFilename)


def ensure_run(memory_key):
    """Ensure a log file exists for this thread (create one if not)."""
    global _current_key
    key = memory_key or "default"
    _current_key = key
    if key not in _loggers:
        _create(key)


def set_current(memory_key):
    """Remember which thread is currently executing (so tools without state can log)."""
    global _current_key
    _current_key = memory_key or "default"


def log_event(msg: str, memory_key=None):
    """Append ``msg`` to the current run's file (falls back to a lazy file if needed)."""
    key = memory_key or _current_key or "default"
    lg = _loggers.get(key) or _create(key)
    try:
        lg.info(msg)
    except Exception:
        pass


def summary_path(memory_key) -> str:
    """Per-thread summary file (key-points of completed steps). One file per thread, overwritten as it evolves."""
    key = memory_key or "default"
    return str(LOG_DIR / f"{_safe(key)}_summary.md")


def write_summary(memory_key, text: str):
    """Overwrite the per-thread summary file (key-points of completed steps)."""
    try:
        with open(summary_path(memory_key), "w", encoding="utf-8") as f:
            f.write(str(text))
    except Exception:
        pass
