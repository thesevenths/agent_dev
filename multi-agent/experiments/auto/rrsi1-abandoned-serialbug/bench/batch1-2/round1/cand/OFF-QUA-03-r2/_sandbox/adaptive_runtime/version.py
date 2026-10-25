"""四级版本模型：Base / Task / Incumbent / Candidate（V3 §4）。

为什么必须分四级
----------------
V2 里 ``Current Harness`` 和 ``Base Harness`` 同时出现却没定义关系，于是「比较基线是谁」
成了一个隐性变量：一旦拿「上一轮候选」当基线，Q 的微小正噪声会被逐轮累积成漂移，
而任何一次偶然的负噪声又会被后续轮次当成新常态。本模块把关系钉死：

- **Base**：``git tag base/vN`` + 冻结的 env 快照。变化必须过**证据门**：要么人工
  评审（``promote(require_operator=True)``），要么机器判据（``autopromote.evaluate`` +
  ``require_operator=False``）。后者不拆门，只把「是不是人」换成「证据够不够」。
- **Task**：``Base + 本轮检索到的 ≤3 条经验``，只读组装，落 ``task_harness.json``，单轮即弃。
- **Incumbent**：本轮已被接受的候选，可多个串行比较，轮末丢弃。
- **Candidate**：一次具体改动，由 proposer 产出。

于是三条性质同时成立：不漂移（base 只在证据门下变）、改进能落地（不是只产建议）、
可归因（每个候选都对着同一个 base 比较）。

实现上只读文件，不引入状态机：``base_state.json`` 存当前版本，``base_history.jsonl``
只追加历史，promote 需要显式 ``operator`` 与证据，避免「系统自己给自己升级」。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import frozen
from .io_util import append_jsonl, now_iso, read_json, read_jsonl, write_json
from .settings import PATHS

#: Task Harness 允许携带的经验条数上限（V3 §4：≤3 条）
MAX_TASK_LESSONS = 3
#: 建议每 N 轮做一次人工 promote 评审
PROMOTE_EVERY_ROUNDS = 10


def _state_path() -> Path:
    return PATHS.experiments / "base_state.json"


def _history_path() -> Path:
    return PATHS.experiments / "base_history.jsonl"


def git_head(root: Path | None = None) -> str:
    """取外层 git 仓库的 HEAD。仓库不可用/未安装 git 时返回空串（不视为错误）。"""
    try:
        p = subprocess.run(["git", "-C", str(root or PATHS.root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return (p.stdout or "").strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class BaseVersion:
    version: int
    base_commit: str = ""
    env: dict = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    created_by: str = "system"
    note: str = ""
    evidence: list = field(default_factory=list)

    @property
    def tag(self) -> str:
        return f"base/v{self.version}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tag"] = self.tag
        return d


def env_snapshot() -> dict:
    """冻结口径的 env 快照：与 Frozen Core 用同一套解析，避免两份真源。"""
    try:
        keys = list(frozen.FROZEN_ENV)
        resolved = frozen.resolved_env(keys)
    except Exception:
        return {}
    return {k: str(resolved.get(k, "")) for k in sorted(resolved)}


def current() -> BaseVersion | None:
    data = read_json(_state_path(), default=None)
    if not data:
        return None
    return BaseVersion(version=int(data.get("version") or 0),
                       base_commit=data.get("base_commit", ""),
                       env=dict(data.get("env") or {}),
                       created_at=data.get("created_at", now_iso()),
                       created_by=data.get("created_by", "system"),
                       note=data.get("note", ""),
                       evidence=list(data.get("evidence") or []))


def ensure_initial(created_by: str = "system") -> BaseVersion:
    """首次使用时把当前状态固化为 base/v1。幂等。"""
    cur = current()
    if cur is not None:
        return cur
    base = BaseVersion(version=1, base_commit=git_head(), env=env_snapshot(),
                       created_by=created_by, note="初始基线（首次运行自动建立）")
    write_json(_state_path(), base.to_dict())
    append_jsonl(_history_path(), [dict(base.to_dict(), event="init")])
    return base


def promote(evidence: list[dict], operator: str, note: str = "",
            require_operator: bool = True) -> BaseVersion:
    """promote 门：把已跨任务验证的机制合并进 Base，版本号 +1。

    ``require_operator=True``（默认）时 ``operator`` 必填且不能是系统账户 ——
    这条约束是「不漂移」的保证。无人值守跑法走 ``require_operator=False``，
    但**不是删掉门**：调用方必须先过 ``autopromote.evaluate``，那里的证据要求比
    人工评审更硬（标定必须实测通过 + holdout 配对不退 + 可自动回滚）。
    把「谁能升基线」从「是不是人」换成「证据够不够」，才是正则化而不是失控。
    """
    op = (operator or "").strip()
    if require_operator and (not op or op.lower() in
                             ("system", "auto", "adaptive_runtime", "robot")):
        raise ValueError("promote 必须由人工执行：请显式提供 operator（不能是 system/auto）")
    cur = ensure_initial()
    new = BaseVersion(version=cur.version + 1, base_commit=git_head(),
                      env=env_snapshot(), created_by=op,
                      note=note, evidence=list(evidence or []))
    write_json(_state_path(), new.to_dict())
    append_jsonl(_history_path(), [dict(new.to_dict(), event="promote",
                                        previous=cur.to_dict())])
    return new


def history() -> list[dict]:
    return read_jsonl(_history_path())


def refresh_snapshot(ver: BaseVersion | None = None,
                     base_commit: str | None = None) -> BaseVersion | None:
    """重新采集基线的 env 快照（**落地之后**必须调一次）。

    为什么需要：``promote`` 在落地之前执行（这样落地记录才能写上 ``base/vN`` 标签），
    于是 ``env_snapshot()`` 采到的是改动**前**的值。不重采就会出现「base/v2 的快照里
    还写着旧值」—— 那正好丢掉了「这一版基线到底是什么配置」这个最关键的归属信息，
    后续任何回放与对比都会对着错的 env 比。

    ``base_commit`` 可传入「git HEAD + 落地内容指纹」的复合归属：自动落地默认不
    commit（提交必须人工授权），若仍只记 git HEAD，base/v1 与 base/v2 的 commit 会相同。
    """
    cur = ver or current()
    if cur is None:
        return None
    fresh = BaseVersion(version=cur.version, base_commit=base_commit or cur.base_commit or git_head(),
                        env=env_snapshot(), created_at=cur.created_at,
                        created_by=cur.created_by, note=cur.note, evidence=cur.evidence)
    write_json(_state_path(), fresh.to_dict())
    append_jsonl(_history_path(), [dict(fresh.to_dict(), event="re-snapshot")])
    return fresh


def restore(prev: "BaseVersion", note: str = "", operator: str = "auto:adaptive") -> BaseVersion:
    """把 Base 退回指定版本（自动回滚用），并在历史里追加一条 demote 事件。

    为什么不直接删历史：历史只追加。回退也是事实，必须留下「曾经升上去又退回来」的记录，
    否则下次看台账会以为那个机制从没上过线。
    """
    cur = current()
    write_json(_state_path(), prev.to_dict())
    append_jsonl(_history_path(), [dict(prev.to_dict(), event="demote",
                                        reverted_from=(cur.to_dict() if cur else None),
                                        operator=operator, note=note)])
    return prev


def tag_hint() -> str:
    """打 tag 的命令提示。不自动执行 —— 打 tag 属于人工发版动作。"""
    cur = current() or ensure_initial()
    return f"git tag -a {cur.tag} -m 'adaptive runtime base v{cur.version}'"


# ---------------------------------------------------------------- Incumbent

def incarnate(candidates: list[dict]) -> list[dict]:
    """本轮被接受的候选（Incumbent）。轮末丢弃，只作为报告里的比较链。"""
    return [{"candidate_id": c.get("candidate_id"), "target": c.get("target"),
             "old": c.get("old"), "new": c.get("new"),
             "decision": c.get("decision"), "reason": c.get("reason")}
            for c in candidates]


def promote_due(rounds_since_promote: int | None = None) -> bool:
    """是否到了「建议人工评审 promote」的时点（默认每 10 轮）。"""
    if rounds_since_promote is None:
        n = 0
        for row in history():
            n = 0 if row.get("event") == "promote" else n + 1
        rounds_since_promote = n
    return rounds_since_promote >= PROMOTE_EVERY_ROUNDS


def summary() -> dict:
    cur = current()
    return {"current": cur.to_dict() if cur else None,
            "history_events": len(history()),
            "promote_due": promote_due(),
            "tag_hint": tag_hint() if cur else "",
            "built_at": datetime.now().isoformat(timespec="seconds")}
