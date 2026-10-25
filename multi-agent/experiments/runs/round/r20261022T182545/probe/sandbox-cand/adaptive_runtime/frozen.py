"""Frozen Core：符号级冻结 + 文件 hash + 解析后 env hash。

为什么不能按目录冻结
--------------------
本仓库真正的评分逻辑不在任何独立目录里：
  - ``agents.py`` 的 ``_rule_gate`` / ``_critic_gate`` / ``_action_contract_gate`` /
    ``_is_deliverable_step`` 决定「这一步算不算完成」「什么算交付物」；
  - ``AGENT_CRITIC_MIN_LEN`` 等 8 个 env 决定判定阈值；
  - ``planutil.members`` 决定可路由的成员集合（拓扑本身是评分口径的一部分）。
按目录冻结只会保护一个新建的空目录，而上面这些全部落在「可编辑区」。

两道断言缺一不可
----------------
1. 文件侧：对**符号源码段**做 hash（AST 取段），这样同一文件里无关的改动不会误报；
2. 环境侧：对**解析后的 env 值**做 hash。只 hash 文件不够 —— 改 ``.env`` 不动代码
   就能把 ``AGENT_CRITIC_MIN_LEN`` 从 4 改成 1，通过率立刻上升。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_util import now_iso, read_json, sha256_text, write_json
from .settings import PATHS, ROOT

#: 文件 -> 不得被候选修改的符号
FROZEN_SYMBOL_FILES: dict[str, tuple[str, ...]] = {
    "agents.py": (
        "_rule_gate", "_critic_gate", "_action_contract_gate", "_is_deliverable_step",
        "_DELIVERABLE_VERBS", "_PATH_EXTS",
        "_CRITIC_MIN_LEN", "_CRITIC_RETRIES", "_CRITIC_LLM",
        "_ACTION_PROBES_ENV", "_DEFAULT_ACTION_PROBES",
        "_TRUNCATED_SALVAGE", "_SALVAGE_EXTS", "_STEP_FAILURES_MAX",
    ),
    "planutil.py": ("members", "options"),
    "runlog.py": ("log_event", "timing_summary"),
    "graph.py": ("build_graph_with_memory",),
}

#: 不得被候选修改的环境变量（评分口径）
FROZEN_ENV: tuple[str, ...] = (
    "AGENT_CRITIC_MIN_LEN", "AGENT_ACTION_PROBES", "AGENT_CRITIC_RETRIES",
    "AGENT_STEP_FAILURES_MAX", "AGENT_TRUNCATED_SALVAGE", "AGENT_SALVAGE_EXTS",
    "AGENT_FINAL_ADJUDICATE", "AGENT_CRITIC_LLM",
)


# ---------------------------------------------------------------- env 解析

def parse_env_file(path: str | Path) -> dict[str, str]:
    """解析 .env（不做变量展开；值里允许 # 注释）。"""
    p = Path(path)
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # 去掉行尾注释：只有「空白 + #」才算注释，避免误伤含 # 的值
        for i in range(1, len(val)):
            if val[i] == "#" and val[i - 1] in " \t":
                val = val[:i].rstrip()
                break
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def resolved_env(keys: tuple[str, ...] = FROZEN_ENV, root: Path | None = None) -> dict[str, str]:
    """解析后的冻结 env：进程环境优先，其次 .env；缺省记为 <unset>。"""
    file_env = parse_env_file((root or ROOT) / ".env")
    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            v = file_env.get(k)
        out[k] = "<unset>" if v is None else v
    return out


def hash_env(env: dict[str, str]) -> str:
    return sha256_text("\n".join(f"{k}={env[k]}" for k in sorted(env)))


# ---------------------------------------------------------------- 符号提取

def _target_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for t in node.targets:
            names.extend(_names_of(t))
    elif isinstance(node, ast.AnnAssign):
        names.extend(_names_of(node.target))
    return names


def _names_of(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            out.extend(_names_of(e))
        return out
    return []


def symbol_source(text: str, name: str) -> str | None:
    """返回模块级符号的源码段（函数/类/赋值语句）。找不到返回 None。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                seg = ast.get_source_segment(text, node)
                return seg
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if name in _target_names(node):
                return ast.get_source_segment(text, node)
    return None


def symbol_hashes(root: str | Path | None = None) -> dict[str, dict[str, str | None]]:
    root = Path(root) if root else ROOT
    out: dict[str, dict[str, str | None]] = {}
    for fname, symbols in FROZEN_SYMBOL_FILES.items():
        p = root / fname
        if not p.exists():
            out[fname] = {s: None for s in symbols}
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        per: dict[str, str | None] = {}
        for s in symbols:
            seg = symbol_source(text, s)
            per[s] = sha256_text(seg) if seg is not None else None
        out[fname] = per
    return out


# ---------------------------------------------------------------- 清单

@dataclass
class Violation:
    kind: str          # symbol_changed | symbol_missing | env_changed | manifest_missing
    target: str
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target,
                "expected": self.expected, "actual": self.actual}

    def __str__(self) -> str:
        return f"{self.kind}:{self.target}"


def build_manifest(root: str | Path | None = None) -> dict:
    root = Path(root) if root else ROOT
    env = resolved_env(root=root)
    return {
        "created_at": now_iso(),
        "root": str(root),
        "symbols": symbol_hashes(root),
        "env": env,
        "env_hash": hash_env(env),
    }


def freeze(path: str | Path | None = None, root: str | Path | None = None) -> dict:
    """写入冻结清单。**必须由人工执行**（probe 不会自动调用它）。"""
    manifest = build_manifest(root)
    write_json(path or PATHS.frozen_manifest, manifest)
    return manifest


def verify(path: str | Path | None = None, root: str | Path | None = None) -> list[Violation]:
    """校验当前代码/env 是否仍与冻结清单一致。空列表 = 未越界。"""
    p = Path(path or PATHS.frozen_manifest)
    manifest = read_json(p, default=None)
    if not manifest:
        return [Violation("manifest_missing", str(p))]

    out: list[Violation] = []
    expected = manifest.get("symbols", {}) or {}
    actual = symbol_hashes(root)
    for fname, per_exp in expected.items():
        per_act = actual.get(fname, {})
        for sym, h_exp in per_exp.items():
            h_act = per_act.get(sym)
            if h_exp is None and h_act is None:
                continue                     # 冻结时就缺失：只记录，不判违规
            if h_exp is None or h_act is None:
                out.append(Violation("symbol_missing", f"{fname}:{sym}", h_exp, h_act))
            elif h_exp != h_act:
                out.append(Violation("symbol_changed", f"{fname}:{sym}", h_exp, h_act))

    env_exp = manifest.get("env", {}) or {}
    env_act = resolved_env(tuple(env_exp.keys()) or FROZEN_ENV, root=root)
    for k, v_exp in env_exp.items():
        v_act = env_act.get(k)
        if v_exp != v_act:
            out.append(Violation("env_changed", k, v_exp, v_act))
    if manifest.get("env_hash") and hash_env(env_act) != manifest.get("env_hash"):
        # 逐键 diff 已经报过就不重复；只有整体 hash 不符且逐键无差异时补一条
        if not any(v.kind == "env_changed" for v in out):
            out.append(Violation("env_changed", "<aggregate>",
                                 manifest.get("env_hash"), hash_env(env_act)))
    return out


def is_intact(path: str | Path | None = None, root: str | Path | None = None) -> bool:
    return not verify(path, root)


def manifest_summary(path: str | Path | None = None) -> dict:
    m = read_json(path or PATHS.frozen_manifest, default={}) or {}
    syms = m.get("symbols", {}) or {}
    total = sum(len(v) for v in syms.values())
    missing = sum(1 for v in syms.values() for h in v.values() if h is None)
    return {"paths": len(syms), "symbols": total, "missing": missing,
            "env_keys": len(m.get("env", {}) or {}),
            "created_at": m.get("created_at")}
