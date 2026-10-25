"""Quarantine：被拒/失败的候选留档，保证「可回滚 + 可复核」。

为什么不能只删沙箱
------------------
沙箱删掉就是回滚（不留半成品污染下一次运行），但**证据不能一起删**：
事后要回答「这个候选为什么被拒」「同一个 patch 是不是又提了一遍」，只能靠归档。
所以这里的语义是：沙箱删除 = 回滚；``quarantine/<...>.json`` = 归档。

索引只追加，不覆盖：同一次候选可能先失败、再被判负，两条都要留着。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .io_util import append_jsonl, now_iso, now_stamp, read_jsonl, write_json
from .settings import PATHS


def _dir() -> Path:
    d = PATHS.quarantine
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_path() -> Path:
    return _dir() / "index.jsonl"


def record(candidate: dict, reason: str, stage: str,
           probe: dict | None = None, round_dir: str | Path | None = None,
           failures: list[dict] | None = None) -> dict:
    """归档一个被拒候选。返回归档记录。"""
    cid = str(candidate.get("candidate_id") or "unknown")
    payload = {
        "quarantined_at": now_iso(),
        "candidate_id": cid,
        "target": candidate.get("target"),
        "level": candidate.get("level"),
        "old": candidate.get("old"),
        "new": candidate.get("new"),
        "patch_hash": candidate.get("patch_hash"),
        "base_commit": candidate.get("base_commit", ""),
        "hypothesis": candidate.get("hypothesis", ""),
        "hyp_fingerprint": candidate.get("hyp_fingerprint", ""),
        "reason": reason,
        "stage": stage,
        "probe": probe or {},
        "round_dir": str(round_dir) if round_dir else "",
        "failures": list(failures or []),
        "note": "沙箱已删除（回滚）；本文件仅作证据归档，不可被执行",
    }
    name = f"{now_stamp()}-{cid}.json"
    write_json(_dir() / name, payload)
    append_jsonl(index_path(), [{
        "quarantined_at": payload["quarantined_at"], "candidate_id": cid,
        "target": payload["target"], "patch_hash": payload["patch_hash"],
        "reason": reason, "stage": stage, "file": name,
    }])
    return payload


def entries() -> list[dict]:
    return read_jsonl(index_path())


def find_by_patch(patch_hash: str) -> list[dict]:
    """同一个改动是否被隔离过（用于回答「这个假设是不是又重提了」）。"""
    return [e for e in entries() if e.get("patch_hash") == patch_hash]


def summary() -> dict:
    rows = entries()
    by_reason: dict[str, int] = {}
    for r in rows:
        by_reason[r.get("reason", "?")] = by_reason.get(r.get("reason", "?"), 0) + 1
    return {"total": len(rows), "by_reason": by_reason,
            "dir": str(_dir()), "built_at": datetime.now().isoformat(timespec="seconds")}
