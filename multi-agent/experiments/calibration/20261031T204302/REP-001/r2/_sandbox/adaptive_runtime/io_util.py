"""文件与序列化工具：原子写、JSONL、hash、时间。

全部为 stdlib 实现，避免给 adaptive_runtime 引入额外依赖。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------- 时间

def now_iso() -> str:
    """本地时间 ISO 字符串（秒级），用于人读的 meta 字段。"""
    return datetime.now().isoformat(timespec="seconds")


def now_stamp() -> str:
    """紧凑时间戳，用于目录名：20261022T180500。"""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- hash

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return sha256_text(text)[:n]


# ---------------------------------------------------------------- JSON

def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: str | Path, obj: Any) -> Path:
    """原子写：先写同目录临时文件再 os.replace，避免半截 JSON。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def read_text(path: str | Path, default: str = "") -> str:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return default


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


# ---------------------------------------------------------------- 文本

_SLUG_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def slug(text: str, fallback: str = "x") -> str:
    s = _SLUG_RE.sub("_", (text or "").strip())
    s = s.strip("._")
    return s[:60] or fallback


def safe_relpath(path: str | Path, root: str | Path) -> str:
    """把绝对路径转成相对 root 的 posix 路径（失败则返回原串）。"""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)
