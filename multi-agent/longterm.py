"""跨会话长期记忆：向量语义召回 + SQLite 存储（memory/longterm.db）。

与既有机制的边界（各司其职、不重叠）：
- log/<thread>_summary.md：单线程「本轮要点摘要」，write-only，不跨会话召回；
- memory/checkpoints.sqlite：LangGraph 每步 state 快照（原始、按 thread 续跑）；
- memory/longterm.db（本模块）：跨会话、去重、可**语义召回**的精炼长期记忆
  （用户画像/偏好、关键决策及理由、稳定事实）。

闭环（对应 OpenClaw「养龙虾 / 越用越懂你」的实质）：
  ① 抽取：每轮 supervisor FINISH 用 1 次 LLM 从本轮产出蒸馏「值得跨会话记住的」→ remember()；
  ② 存储：embedding + 元数据入 SQLite，近重复自动合并（不堆积）；
  ③ 召回：每轮图入口 run_start 按当前 query 向量召回 top-k → 写进 state.recalled_memory；
  ④ 注入：supervisor 首轮规划 + 每个子 agent 节点，把 recalled_memory 作为「用户背景」注入 context。

向量：OpenAI 兼容 /embeddings（默认 DashScope text-embedding-v4, dim=1024）。本地 vLLM 只有 chat
模型、无 embedding（已探针确认），故默认走 DashScope；AGENT_MEMORY_EMBED_* 可改。embedding 不可达
或未配置时，recall 自动回落「关键词+标签」词面匹配，绝不 hard-break 运行。
存储：SQLite 单表（含 type 列 + 索引，类型枚举 profile/preference/decision/fact）；embedding 以
float32 BLOB 存，召回时 numpy 暴力余弦（个人助手量级几百~几千条，全表扫描毫秒级，无需额外向量库依赖）。
召回支持按 type 过滤：函数参数 types= 或全局 AGENT_MEMORY_RECALL_TYPES（逗号分隔）限定只召回某些类型，
二者取交集，留空则召回全部（零回归）。设 AGENT_LONGTERM_MEMORY=0 可整体关闭（零回归）。
"""
import os
import re
import sqlite3
import hashlib
import logging
import threading
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

from runlog import log_event

load_dotenv()
logger = logging.getLogger(__name__)

# === 开关与配置（全部 env 可调）===
_ENABLED = os.environ.get("AGENT_LONGTERM_MEMORY", "1").strip().lower() not in ("0", "false", "no", "off")
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.environ.get("AGENT_MEMORY_DB") or os.path.join(_MODULE_DIR, "memory", "longterm.db")
_TOPK = int(os.environ.get("AGENT_MEMORY_TOPK", "5"))
_BUDGET = int(os.environ.get("AGENT_MEMORY_BUDGET_CHARS", "1600"))
_MIN_SCORE = float(os.environ.get("AGENT_MEMORY_MIN_SCORE", "0.25"))     # 向量余弦下限，低于视为不相关
_NEAR_DUP = float(os.environ.get("AGENT_MEMORY_DEDUP_COS", "0.95"))      # 近重复阈值，≥则合并而非新增
# 长期记忆类型枚举（与 _EXTRACT_PROMPT 对齐）。落库时非枚举值回落 fact，召回时可按类型过滤。
_MEM_TYPES = ("profile", "preference", "decision", "fact")
# 全局召回类型过滤（逗号分隔，如 "preference,profile"）。留空=不过滤（召回全部类型，零回归）。
_RECALL_TYPES = {t.strip().lower() for t in
                 (os.environ.get("AGENT_MEMORY_RECALL_TYPES") or "").split(",") if t.strip()}
# embedding 端点（OpenAI 兼容）：默认 DashScope（本地 vLLM 无 embedding 模型）
_EMBED_BASE = (os.environ.get("AGENT_MEMORY_EMBED_BASE_URL")
               or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
_EMBED_KEY = (os.environ.get("AGENT_MEMORY_EMBED_API_KEY")
              or os.environ.get("DASHSCOPE_API_KEY") or "")
_EMBED_MODEL = os.environ.get("AGENT_MEMORY_EMBED_MODEL") or "text-embedding-v4"
_EMBED_DIM = int(os.environ.get("AGENT_MEMORY_EMBED_DIM", "1024"))
_EMBED_TIMEOUT = float(os.environ.get("AGENT_MEMORY_EMBED_TIMEOUT", "15"))
# 非对称检索的 query 侧检索指令（留空 = 完全保持旧行为，逐字节不变）。
# 为什么需要：Qwen3-Embedding 这类非对称检索模型，query 与 passage 的编码方式不同。
# query 侧不加官方指令时，两侧向量不在同一"语义刻度"上 —— 实测 33 条记忆库上噪声最高 0.69，
# 与相关条目 0.66~0.83 交错，任何全局阈值都切不干净；加上指令后噪声顶降到 0.545、
# 相关底 0.611，才出现干净切点（配 AGENT_MEMORY_MIN_SCORE=0.58）。
# 前缀**只作用于 query 侧**，passage 编码不变 → 与库内已存向量兼容，无需 re-embed。
_EMBED_QUERY_PREFIX = (os.environ.get("AGENT_MEMORY_EMBED_QUERY_PREFIX") or "").strip()

_lock = threading.RLock()   # 可重入：公开方法持锁，内部 helper 复用同一连接不再单独加锁
_conn = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id           TEXT PRIMARY KEY,
  type         TEXT,
  text         TEXT,
  tags         TEXT,
  importance   REAL DEFAULT 0.5,
  created_ts   TEXT,
  updated_ts   TEXT,
  last_used_ts TEXT,
  use_count    INTEGER DEFAULT 0,
  dim          INTEGER,
  embedding    BLOB,
  text_hash    TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(text_hash);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
"""


def _connect() -> sqlite3.Connection:
    """惰性建连接（长生命周期，check_same_thread=False：LangGraph 可能跨线程访问）+ 幂等建表。"""
    global _conn
    if _conn is None:
        d = os.path.dirname(_DB_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()
        logger.info(f"[longterm] store → {_DB_PATH} (embed={_EMBED_MODEL}@{_EMBED_BASE})")
    return _conn


# === 向量工具 ===
# embedding 失败呼叫器。用 list 做计数器，不在函数里给模块级变量赋值（避免漏写 global 那类坑）。
# 为什么必须外放：原实现只有 logger.warning —— 那只进进程控制台（langgraph dev 的终端），
# 滚动即失、跑完就没了。实测踩过：一轮 run 里 remember() 写进 2 条记忆，embedding 全为 NULL、
# dim=0 —— 这些条目**永远无法被语义召回**，而 log/<thread>_run.log 里一个字都没有，完全静默。
_EMBED_FAILS: list = []
_EMBED_FAIL_WARN_MAX = 5


def _note_embed_failure(why: str) -> None:
    """记录一次 embedding 失败：控制台 logger 一条 + log/<thread>_run.log 一条（可事后回溯）。"""
    if len(_EMBED_FAILS) >= _EMBED_FAIL_WARN_MAX:
        return
    _EMBED_FAILS.append(why)
    n = len(_EMBED_FAILS)
    msg = f"[longterm] embedding 失败 #{n}（{why}）endpoint={_EMBED_BASE}"
    if n == 1:
        msg += " —— 本次 remember() 写进去的记忆只有文本没有向量，将永远无法被语义召回；召回也退化为关键词匹配"
    logger.warning(msg)
    try:
        log_event(msg)
    except Exception:
        pass


def _embed(texts, role=None):
    """批量取 embedding（1 次 HTTP，input 为列表）。返回 list[list[float]]；未配置/失败返回 None（调用方回落）。

    role="query" 且配了 AGENT_MEMORY_EMBED_QUERY_PREFIX 时，给每个待编码文本加检索指令前缀
    （见文件上方 _EMBED_QUERY_PREFIX 的注释）。role 为 None 时行为与旧版完全一致。
    """
    if not texts:
        return None
    if not _EMBED_KEY:      # 配置缺失：以前这里静默 return，是最难查的一种"记忆变傻"
        _note_embed_failure("未配置 AGENT_MEMORY_EMBED_API_KEY（或 DASHSCOPE_API_KEY）")
        return None
    import httpx
    payload_texts = ([f"{_EMBED_QUERY_PREFIX}\nQuery: {t}" for t in texts]
                     if (role == "query" and _EMBED_QUERY_PREFIX) else list(texts))
    payload = {"model": _EMBED_MODEL, "input": payload_texts, "encoding_format": "float"}
    if _EMBED_DIM:
        payload["dimensions"] = _EMBED_DIM   # DashScope v3/v4 支持指定维度
    try:
        r = httpx.post(_EMBED_BASE + "/embeddings",
                       headers={"Authorization": f"Bearer {_EMBED_KEY}"},
                       json=payload, timeout=_EMBED_TIMEOUT)
        if r.status_code != 200:
            _note_embed_failure(f"HTTP {r.status_code}: {r.text[:120]}")
            return None
        data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))  # 保持与输入同序
        vecs = [d.get("embedding") for d in data]
        if len(vecs) != len(texts) or any(v is None for v in vecs):
            _note_embed_failure(f"返回条数/内容异常（got {len(vecs)}，expect {len(texts)}）")
            return None
        return vecs
    except Exception as e:
        _note_embed_failure(f"{type(e).__name__}: {e}")
        return None


def _to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tokenize(text: str) -> set:
    """轻量分词：ASCII 词 + 中文字 + 中文相邻二字组合（无分词器下尽量提高词面命中）。"""
    text = (text or "").lower()
    words = set(re.findall(r"[a-z0-9_]+", text))
    han = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = {"".join(p) for p in zip(han, han[1:])}
    return words | bigrams | set(han)


def _recency(ts: str) -> float:
    """时间新近度 0..1（~30 天指数衰减）；无时间戳按 0.5 中性处理。"""
    if not ts:
        return 0.5
    try:
        age_days = max(0.0, (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 86400.0)
    except Exception:
        return 0.5
    return float(np.exp(-age_days / 30.0))


def _final_score(relevance: float, row) -> float:
    """Generative-Agents 式融合：相关性为主，重要度/新近度小幅加权。"""
    return relevance + 0.10 * float(row["importance"] or 0.5) + 0.10 * _recency(
        row["last_used_ts"] or row["updated_ts"] or row["created_ts"])


# === 写入 ===
def remember(text, mtype="fact", tags=None, importance=0.5, memory_key=None):
    """写入一条长期记忆（自动 embedding + 去重）。返回 (id|None, action)。

    action ∈ {inserted, dup, skipped, disabled}。精确重复(hash)与近重复(余弦≥_NEAR_DUP)都只更新、不新增，
    避免同一条事实被反复写入堆积。任何异常都吞掉并返回 (None,'error')，绝不影响主流程。
    """
    if not _ENABLED:
        return None, "disabled"
    text = (text or "").strip()
    if len(text) < 4:
        return None, "skipped"
    # type 枚举校验（唯一写入入口，保证落库 type 干净可被过滤）：非枚举值回落 fact
    mtype = str(mtype or "fact").strip().lower()
    if mtype not in _MEM_TYPES:
        mtype = "fact"
    try:
        importance = min(1.0, max(0.0, float(importance)))
    except (TypeError, ValueError):
        importance = 0.5
    tags_str = ",".join(sorted({t.strip() for t in (tags or []) if t and t.strip()}))
    thash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    vec = (_embed([text]) or [None])[0]
    now = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _connect()
        with _lock:
            row = conn.execute("SELECT id, importance FROM memories WHERE text_hash=?", (thash,)).fetchone()
            if row:  # 精确重复 → 抬高重要性 + 更新时间，不新增
                conn.execute("UPDATE memories SET updated_ts=?, importance=? WHERE id=?",
                             (now, max(float(row["importance"] or 0), importance), row["id"]))
                conn.commit()
                return row["id"], "dup"
            if vec is not None:  # 近重复 → 视为同一条，只更新时间戳
                qv = np.asarray(vec, dtype=np.float32)
                for r in conn.execute("SELECT id, dim, embedding FROM memories WHERE embedding IS NOT NULL"):
                    if int(r["dim"] or 0) != qv.shape[0]:
                        continue
                    if _cosine(qv, _from_blob(r["embedding"])) >= _NEAR_DUP:
                        conn.execute("UPDATE memories SET updated_ts=? WHERE id=?", (now, r["id"]))
                        conn.commit()
                        return r["id"], "dup"
            mid = hashlib.sha1((thash + now).encode()).hexdigest()[:16]
            conn.execute(
                "INSERT INTO memories (id,type,text,tags,importance,created_ts,updated_ts,use_count,dim,embedding,text_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (mid, mtype, text, tags_str, importance, now, now, 0,
                 (len(vec) if vec else 0), (_to_blob(vec) if vec is not None else None), thash))
            conn.commit()
        return mid, "inserted"
    except Exception as e:
        logger.warning(f"[longterm] remember failed ({type(e).__name__}: {e})")
        return None, "error"


# === 召回 ===
def recall(query, k=None, budget=None, types=None, memory_key=None) -> str:
    """按 query 语义召回 top-k 条长期记忆，预算封顶，返回可直接注入 context 的 markdown 项目符号列表（无则空串）。

    向量不可达时回落关键词/标签词面匹配。命中的条目更新 last_used_ts/use_count（为新近度打分与后续淘汰留数据）。

    types: 可选，list/set/tuple，限定只召回这些 type（profile/preference/decision/fact）。
           与全局开关 AGENT_MEMORY_RECALL_TYPES 取交集；都为空则召回全部类型（零回归）。
    """
    if not _ENABLED:
        return ""
    query = (query or "").strip()
    if not query:
        return ""
    k = k or _TOPK
    budget = budget or _BUDGET
    # 类型过滤：全局 env (AGENT_MEMORY_RECALL_TYPES) ∩ 本次调用参数 types。
    # 任意一侧指定了类型即开启过滤；交集为空集合 -> 显式"无命中"，返回空（而非退化为全量）。
    filter_active = False
    allow: set = set()
    if _RECALL_TYPES:
        allow |= _RECALL_TYPES
        filter_active = True
    if types:
        tset = {str(t).strip().lower() for t in types if str(t).strip()}
        allow = tset if not filter_active else (allow & tset)
        filter_active = True
    try:
        conn = _connect()
        with _lock:
            rows = conn.execute("SELECT * FROM memories").fetchall()
            if filter_active:
                rows = [r for r in rows if (r["type"] or "").lower() in allow]
            if not rows:
                return ""
            scored, mode = [], "keyword"
            vecs = _embed([query], role="query")
            if vecs is not None:
                qv = np.asarray(vecs[0], dtype=np.float32)
                for r in rows:
                    if r["embedding"] is None or int(r["dim"] or 0) != qv.shape[0]:
                        continue
                    c = _cosine(qv, _from_blob(r["embedding"]))
                    if c >= _MIN_SCORE:
                        scored.append((c, r))
                mode = "vector"
            if not scored:  # 向量失败或无命中 → 关键词兜底
                mode = "keyword"
                qt = _tokenize(query)
                for r in rows:
                    hay = _tokenize((r["text"] or "") + " " + (r["tags"] or ""))
                    overlap = len(qt & hay)
                    if overlap > 0:
                        scored.append((overlap / (len(qt) ** 0.5), r))
            ranked = sorted(scored, key=lambda x: _final_score(x[0], x[1]), reverse=True)[:k]
            lines, used, picked_ids = [], 0, []
            for rel, r in ranked:
                tag = f" [{r['type']}]" if r["type"] else ""
                line = f"- ({rel:.2f}){tag} {r['text']}"
                if used + len(line) > budget:
                    break
                lines.append(line)
                used += len(line) + 1
                picked_ids.append(r["id"])
            if picked_ids:  # 记录被召回的使用情况
                now = datetime.now().isoformat(timespec="seconds")
                conn.executemany("UPDATE memories SET last_used_ts=?, use_count=use_count+1 WHERE id=?",
                                 [(now, i) for i in picked_ids])
                conn.commit()
        block = "\n".join(lines)
        if block and memory_key:
            _ty = f" types={sorted(allow)}" if allow else ""
            log_event(f"[longterm] recalled {len(picked_ids)} memory(ies) via {mode}{_ty} for query "
                      f"'{query[:60]}':\n{block}", memory_key)
        return block
    except Exception as e:
        logger.warning(f"[longterm] recall failed ({type(e).__name__}: {e})")
        return ""


# === FINISH 抽取（写入端闭环）===
_EXTRACT_PROMPT = (
    "You distill DURABLE, CROSS-SESSION memories from a just-finished multi-agent run, so the "
    "assistant can serve this user better NEXT time (\"the more you use it, the better it knows you\").\n\n"
    "Record ONLY things that stay true across future conversations:\n"
    "- profile: who the user is (role, domain, language preference).\n"
    "- preference: how they like things done (format, tone, tools, constraints, do/don't).\n"
    "- decision: a key decision made and WHY (reusable rationale).\n"
    "- fact: a stable fact worth reusing (an account, a recurring target, a standing rule).\n\n"
    "Do NOT record ephemeral task data: this run's numbers, file paths, URLs, timestamps, or one-off "
    "results — those already live in tmp/ and log/. Skip anything obvious or already implied.\n"
    "Prefer FEWER, HIGH-VALUE items (0-5). If nothing durable was learned, return an empty list.\n\n"
    "Reply with EXACTLY one JSON object, no prose:\n"
    '{"memories": [{"type": "profile|preference|decision|fact", '
    '"text": "<one concise sentence, self-contained>", "tags": ["<keyword>", ...], '
    '"importance": <0.0-1.0>]}'
)


def extract_and_remember_from_run(state, memory_key=None) -> int:
    """FINISH 时调用：用 1 次 LLM 从本轮 goal+checklist+最终产出蒸馏跨会话记忆并入库。返回新增条数。

    只记跨会话稳定信息（见 _EXTRACT_PROMPT）。任何异常都吞掉（记忆写入失败绝不影响主流程）。
    """
    if not _ENABLED:
        return 0
    try:
        from llm import supervisor_llm
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        from planutil import _extract_json_obj, _goal_text

        goal = state.get("plan_goal") or _goal_text(state) or ""
        summary = (state.get("plan_summary") or "")[:3000]
        final = ""
        for m in reversed(list(state.get("messages") or [])):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                final = m.content if isinstance(m.content, str) else str(m.content)
                break
        final = (final or "")[:3000]
        if not (goal.strip() or summary.strip() or final.strip()):
            return 0
        user = (
            f"User's request this run:\n{goal.strip()[:1500]}\n\n"
            f"Key-points checklist of completed steps:\n{summary or '(none)'}\n\n"
            f"Final answer:\n{final or '(none)'}\n\n"
            "Now output the JSON of durable memories."
        )
        ai = supervisor_llm.invoke([SystemMessage(content=_EXTRACT_PROMPT), HumanMessage(content=user)])
        raw = (ai.content if isinstance(ai, AIMessage) else str(ai)) or ""
        obj = _extract_json_obj(raw)
        items = obj.get("memories") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            return 0
        added = 0
        for it in items[:8]:
            if not isinstance(it, dict):
                continue
            txt = str(it.get("text") or "").strip()
            if len(txt) < 4:
                continue
            tags = it.get("tags") or []
            if isinstance(tags, str):
                tags = [t for t in re.split(r"[,，]", tags) if t.strip()]
            # type 枚举校验集中在 remember() 唯一写入入口，这里只透传（非枚举值由 remember 回落 fact）
            mtype = str(it.get("type") or "fact").strip().lower()
            _, action = remember(txt, mtype=mtype, tags=tags,
                                 importance=it.get("importance", 0.5), memory_key=memory_key)
            if action == "inserted":
                added += 1
        if added and memory_key:
            log_event(f"[longterm] FINISH extracted {added} new durable memory(ies) → {_DB_PATH}", memory_key)
        return added
    except Exception as e:
        logger.warning(f"[longterm] extract failed ({type(e).__name__}: {e}); skipping")
        return 0


def stats() -> dict:
    """记忆库概览（供调试/可观测）：总条数、按类型计数、有 embedding 的条数。"""
    try:
        conn = _connect()
        with _lock:
            total = conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
            by_type = {r["type"] or "?": r["c"] for r in
                       conn.execute("SELECT type, COUNT(*) c FROM memories GROUP BY type")}
            embedded = conn.execute("SELECT COUNT(*) c FROM memories WHERE embedding IS NOT NULL").fetchone()["c"]
        return {"total": total, "by_type": by_type, "embedded": embedded, "db": _DB_PATH, "enabled": _ENABLED}
    except Exception as e:
        return {"error": str(e), "db": _DB_PATH, "enabled": _ENABLED}
