"""Per-run structured logging for the multi-agent supervisor/graph.

Each user submission (one ``graph.stream`` invocation, identified by the
thread's ``memory_key``) gets its OWN uniquely-named log file under
``multi-agent/log/`` so the important supervisor / agent events survive
process restarts and can be inspected independently.

Key design point — loggers are keyed by a **unique run_id** (timestamp + 8-char
random suffix), NOT by ``memory_key``. This prevents two problems that bit us
in practice:
  1. Re-submitting on the SAME thread (e.g. the LangGraph Studio "default"
     thread) without restarting ``langgraph dev`` used to overwrite / interleave
     the previous run's log file.
  2. Concurrent runs sharing one ``memory_key`` collided on a single logger.

A fresh ``run_id`` is minted every time ``start_run`` is called (driven by the
graph's ``run_start`` entry node), so each submission is unambiguously separated
even when threads are reused.

Usage:
    from runlog import start_run, ensure_run, set_current, log_event, run_file, get_run_id

    start_run(memory_key)      # begin a new run -> fresh uniquely-named file; returns run_id
    ensure_run(memory_key)     # create a file lazily if this thread has none yet
    set_current(memory_key)    # remember current thread (for tools without state)
    log_event("some important event")   # append to the CURRENT run's file
"""
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Keyed by run_id (unique per submission). A fresh run_id is minted on every
# start_run, so re-submits / concurrent runs never share a logger.
_loggers: dict[str, logging.Logger] = {}
# memory_key (thread id) -> latest run_id for that thread (within this process).
_run_ids: dict[str, str] = {}
# The run_id currently being executed (most recent start_run / ensure_run).
_current_run_id: str | None = None
# Fallback thread id for tools that log without an explicit memory_key.
_current_key: str | None = None
# run_id -> 本次 run 的启动时刻（本进程内）。供 agents 的产物幂等守卫区分
# "本次 run 落盘的产物"与 tmp/ 里历史 run 的同名 step 产物（mtime 比较）。
_run_started: dict[str, datetime] = {}


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name or "default")


def _new_run_id(key: str) -> str:
    """Mint a unique run id: <timestamp>_<8-char-random>, suffixed to the file name."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _create_for(key: str) -> str:
    """Create a NEW uniquely-named file logger for ``key`` and return its run_id."""
    global _current_run_id
    rid = _new_run_id(key)
    path = LOG_DIR / f"{_safe(key)}_{rid}.log"
    lg = logging.getLogger(f"runlog.{rid}")
    lg.setLevel(logging.INFO)
    lg.propagate = False  # never double-emit to the console root logger
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg.addHandler(fh)
    _loggers[rid] = lg
    _run_ids[key] = rid
    _current_run_id = rid
    _run_started[rid] = datetime.now()
    lg.info(f"=== run log started: {path} ===")
    return rid


def start_run(memory_key) -> str:
    """Begin a NEW run: mint a fresh unique run_id + file for this thread. Returns run_id."""
    global _current_key, _current_run_id
    key = memory_key or "default"
    _current_key = key
    return _create_for(key)


def ensure_run(memory_key):
    """Ensure a log file exists for this thread (create one only if none exists yet)."""
    global _current_key, _current_run_id
    key = memory_key or "default"
    _current_key = key
    if key not in _run_ids:
        _create_for(key)
    else:
        # Reuse the existing run_id for this thread (do NOT mint a new file per step).
        _current_run_id = _run_ids[key]


def set_current(memory_key):
    """Remember which thread is currently executing (so tools without state can log)."""
    global _current_key
    _current_key = memory_key or "default"


def get_run_id() -> str | None:
    """The run_id of the currently executing run (None before any start_run)."""
    return _current_run_id


def run_started_at(run_id: str | None = None) -> datetime | None:
    """本次 run（或指定 run_id）的启动时刻；进程重启后无记录时返回 None。"""
    rid = run_id or _current_run_id
    return _run_started.get(rid) if rid else None


def run_file(run_id: str | None = None) -> str | None:
    """Absolute path of the log file for ``run_id`` (or the current run if omitted)."""
    rid = run_id or _current_run_id
    if not rid:
        return None
    lg = _loggers.get(rid)
    if lg and lg.handlers:
        return str(lg.handlers[0].baseFilename)
    return None


def log_event(msg: str, memory_key=None):
    """Append ``msg`` to the CURRENT run's file (falls back to a lazy file if needed)."""
    # Prefer the active run_id; fall back to this thread's latest run_id; else lazily create.
    rid = _current_run_id or _run_ids.get(memory_key or _current_key)
    lg = _loggers.get(rid) if rid else None
    if lg is None:
        key = memory_key or _current_key or "default"
        rid = _create_for(key)
        lg = _loggers[rid]
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
