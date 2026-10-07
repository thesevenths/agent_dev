"""longterm 类型结构化 + 召回按 type 过滤 的确定性单测（无需网络/LLM）。

- AGENT_MEMORY_EMBED_API_KEY 留空 -> _embed 返回 None -> recall 走关键词回落，可离线验证。
- 临时 SQLite 文件，不污染 memory/longterm.db。
"""
import os
import tempfile
import importlib

# 必须在 import longterm 之前设好环境变量（模块顶层读取）
os.environ["AGENT_LONGTERM_MEMORY"] = "1"
os.environ["AGENT_MEMORY_EMBED_API_KEY"] = ""          # 强制关键词回落
os.environ["AGENT_MEMORY_EMBED_QUERY_PREFIX"] = ""     # 保持旧行为
os.environ.pop("DASHSCOPE_API_KEY", None)
os.environ.pop("AGENT_MEMORY_RECALL_TYPES", None)      # 默认不过滤（零回归）
fd, db = tempfile.mkstemp(suffix=".db")
os.close(fd)
os.environ["AGENT_MEMORY_DB"] = db

import longterm  # noqa: E402


def ins(text, mtype, tags):
    _, act = longterm.remember(text, mtype=mtype, tags=tags)
    assert act in ("inserted", "dup"), f"插入失败: {act}"


# === 写入 4 类 + 1 个脏 type ===
ins("User is a backend security engineer working on Java SAST scanner", "profile", ["role", "security"])
ins("Prefers conclusion-first reports with tables and source-code evidence", "preference", ["report", "format"])
ins("Chose tree-sitter over joern for call-graph extraction to lower FP", "decision", ["architecture"])
ins("Nacos version under audit is 3.2.3", "fact", ["nacos", "version"])
ins("Some one-off ephemeral note that should fall back to fact", "bogus_type", ["x"])

# --- 类型枚举校验：脏值回落 fact，不入库 junk type ---
st = longterm.stats()
assert "bogus_type" not in st["by_type"], f"脏 type 不应入库: {st['by_type']}"
assert st["by_type"].get("fact", 0) >= 2, f"fact 应包含回落的脏值条目: {st['by_type']}"
print("  ok: 脏 type 回落 fact（by_type =", st["by_type"], "）")

# === 召回按 type 过滤（关键词回落模式）===
Q = "nacos report format"   # 同时命中 fact(nacos) 与 preference(report/format)

allm = longterm.recall(Q)
assert "Nacos" in allm and "conclusion-first" in allm, f"默认应召回两类: {allm!r}"
print("  ok: 默认（不过滤）召回 fact + preference 两类")

only_fact = longterm.recall(Q, types=["fact"])
assert "Nacos" in only_fact and "conclusion-first" not in only_fact, f"types=[fact] 应只含 fact: {only_fact!r}"
print("  ok: types=['fact'] 仅召回 fact")

only_pref = longterm.recall(Q, types=["preference"])
assert "conclusion-first" in only_pref and "Nacos" not in only_pref, f"types=['preference'] 应只含 preference: {only_pref!r}"
print("  ok: types=['preference'] 仅召回 preference")

# === 全局 env 过滤 + 与参数交集 ===
os.environ["AGENT_MEMORY_RECALL_TYPES"] = "profile"
importlib.reload(longterm)   # 重读环境变量，_RECALL_TYPES 变为 {profile}

gq = "who is the user role security"   # 仅命中 profile
prof_only = longterm.recall(gq)
assert "security engineer" in prof_only, f"全局 profile 应召回 profile: {prof_only!r}"
print("  ok: 全局 AGENT_MEMORY_RECALL_TYPES=profile 仅召回 profile")

empty = longterm.recall(gq, types=["decision"])   # 全局 profile ∩ 参数 decision = 空
assert empty == "", f"全局 profile ∩ decision 应为空: {empty!r}"
print("  ok: 全局 profile ∩ 参数 decision 交集为空 -> 返回空（修复前的 bug）")

print("\n全部断言通过 ✅  — longterm 类型结构化落列 + 召回按 type 过滤（含边界：交集为空返回空）。")
