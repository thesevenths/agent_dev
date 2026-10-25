"""Token 旁路观测：不修改 llm.py，也不改变任何请求参数。

为什么用 monkeypatch
--------------------
`llm.py` 在 import 时就创建了 7 个 ChatOpenAI 实例，且没有传递 callbacks。
要在「不改生产文件」的前提下拿到 usage，只能挂在**类方法**上：
    BaseChatOpenAI._generate / _agenerate / _stream / _astream
方法查找发生在调用时，因此 import 之后打补丁同样生效。

为什么同时挂四个方法
--------------------
LangGraph 的 stream_mode="messages" 会让模型走 `_stream` 分支，普通 invoke 走 `_generate`。
两者互斥（不会同一次调用触发两个），所以不会重复计数。

usage 缺失时
------------
部分 OpenAI 兼容网关（含自建 vLLM 的某些版本）在流式响应里不返回 usage。
此时按字符估算并置 ``estimated=true``；调用方必须上报 estimated 占比（V3 §6.2）。
"""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .io_util import now_iso

#: 当前运行上下文：{"run_id","task_id","arm","round_id"}
_CTX: contextvars.ContextVar[dict | None] = contextvars.ContextVar("adaptive_token_ctx", default=None)

_LOCK = threading.Lock()
_ROWS: list[dict] = []
_ORIG: dict[str, tuple[Any, str, Any]] = {}
_PATCHED = False
_SINK: Path | None = None
_MEMORY_ONLY = False

#: 估算口径：CJK 字符约 1.6 char/token，ASCII 约 4 char/token
_CJK_RATIO = 1.6
_ASCII_RATIO = 4.0


# ---------------------------------------------------------------- 上下文

def bind(**kw) -> contextvars.Token:
    """绑定运行上下文，返回可用于 reset 的 token。"""
    cur = dict(_CTX.get() or {})
    cur.update({k: v for k, v in kw.items() if v is not None})
    return _CTX.set(cur)


def reset(token: contextvars.Token) -> None:
    _CTX.reset(token)


class context:
    """with token_probe.context(run_id=..., arm=...)：自动 reset。"""

    def __init__(self, **kw):
        self.kw = kw
        self.tok = None

    def __enter__(self):
        self.tok = bind(**self.kw)
        return self

    def __exit__(self, *exc):
        if self.tok is not None:
            reset(self.tok)
        return False


def current_context() -> dict:
    return dict(_CTX.get() or {})


# ---------------------------------------------------------------- 安装

def _sink_path() -> Path | None:
    global _SINK
    if _MEMORY_ONLY:
        return None
    if _SINK is None:
        raw = os.environ.get("ADAPTIVE_TOKEN_SINK", "").strip()
        if raw:
            _SINK = Path(raw)
        else:
            base = Path(__file__).resolve().parent.parent / "experiments" / "usage"
            _SINK = base / f"tokens-{os.getpid()}.jsonl"
    return _SINK


def _pick_targets() -> list[tuple[Any, str, Any]]:
    """每个方法只挑一个「最近的实现类」来打补丁。

    为什么不能两个都打：某些版本里 ``ChatOpenAI._stream`` 会调用
    ``super()._stream``，两处都包就会**同一次调用记两行**（已实测复现）。
    """
    from langchain_openai import ChatOpenAI
    from langchain_openai.chat_models.base import BaseChatOpenAI

    out: list[tuple[Any, str, Any]] = []
    for name in ("_generate", "_agenerate", "_stream", "_astream"):
        for cls in (ChatOpenAI, BaseChatOpenAI):
            if name in vars(cls):
                out.append((cls, name, vars(cls)[name]))
                break
    return out


def install(sink: str | Path | None = None, memory_only: bool = False) -> bool:
    """安装埋点。返回 True 表示本次真正安装，False 表示此前已安装/环境不支持。"""
    global _PATCHED, _SINK, _MEMORY_ONLY
    if sink is not None:
        _SINK = Path(sink)
    if memory_only:
        _MEMORY_ONLY = True
    if _PATCHED:
        return False
    try:
        targets = _pick_targets()
    except Exception:
        return False
    if not targets:
        return False

    wrappers = {"_generate": _wrap_generate, "_agenerate": _wrap_agenerate,
                "_stream": _wrap_stream, "_astream": _wrap_astream}
    for cls, name, orig in targets:
        key = f"{cls.__module__}.{cls.__name__}.{name}"
        _ORIG[key] = (cls, name, orig)
        setattr(cls, name, wrappers[name](orig))

    _PATCHED = True
    return True


def uninstall() -> bool:
    global _PATCHED
    if not _PATCHED:
        return False
    for _key, (cls, name, orig) in list(_ORIG.items()):
        setattr(cls, name, orig)
    _ORIG.clear()
    _PATCHED = False
    return True


def is_installed() -> bool:
    return _PATCHED


# ---------------------------------------------------------------- 提取

def _norm_usage(usage: Any) -> dict | None:
    """把各种 usage 形状归一化为 {in,out,total,cached}。"""
    if usage is None:
        return None
    if isinstance(usage, dict):
        d = usage
    else:
        d = getattr(usage, "__dict__", {}) or {}
    inp = d.get("input_tokens", d.get("prompt_tokens"))
    out = d.get("output_tokens", d.get("completion_tokens"))
    total = d.get("total_tokens")
    if inp is None and out is None and total is None:
        return None
    details = d.get("input_token_details") or d.get("prompt_tokens_details") or {}
    cached = None
    if isinstance(details, dict):
        cached = details.get("cached_tokens", details.get("cache_read"))
    elif details is not None:
        cached = getattr(details, "cached_tokens", None)
    return {
        "in": int(inp or 0),
        "out": int(out or 0),
        "total": int(total or ((inp or 0) + (out or 0))),
        "cached": int(cached or 0),
    }


def _usage_from_result(result: Any) -> dict | None:
    # 1) ChatResult.generations[0].message.usage_metadata
    gens = getattr(result, "generations", None)
    if gens:
        gen0 = gens[0]
        msg = getattr(gen0, "message", None)
        u = _norm_usage(getattr(msg, "usage_metadata", None))
        if u:
            return u
        # 2) ChatResult.llm_output["token_usage"]
        llm_out = getattr(result, "llm_output", None) or {}
        if isinstance(llm_out, dict):
            u = _norm_usage(llm_out.get("token_usage") or llm_out.get("usage"))
            if u:
                return u
        # 3) generation_info
        info = getattr(gen0, "generation_info", None) or {}
        if isinstance(info, dict):
            u = _norm_usage(info.get("usage") or info.get("token_usage"))
            if u:
                return u
    return None


def _estimate(messages: Any, output_text: str) -> dict:
    def est(text: str) -> int:
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        rest = len(text) - cjk
        return int(cjk / _CJK_RATIO + rest / _ASCII_RATIO) + 1

    in_text = ""
    if isinstance(messages, (list, tuple)):
        for m in messages:
            c = getattr(m, "content", m)
            in_text += str(c) + "\n"
    else:
        in_text = str(messages)
    i, o = est(in_text), est(output_text)
    return {"in": i, "out": o, "total": i + o, "cached": 0}


def _agent_of(run_manager: Any) -> str:
    meta = getattr(run_manager, "metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("langgraph_node", "agent", "agent_name"):
            v = meta.get(key)
            if isinstance(v, str) and v:
                return v
    return "unknown"


def _record(model: str, usage: dict, t0: float, agent: str, kind: str,
            estimated: bool, error: str | None = None) -> None:
    ctx = current_context()
    row = {
        "ts": now_iso(),
        "run_id": ctx.get("run_id"),
        "task_id": ctx.get("task_id"),
        "round_id": ctx.get("round_id"),
        "arm": ctx.get("arm"),
        "agent": agent,
        "model": model,
        "call": kind,
        "in": usage["in"],
        "out": usage["out"],
        "total": usage["total"],
        "cached": usage["cached"],
        "estimated": bool(estimated),
        "latency_ms": int((time.time() - t0) * 1000),
        "pid": os.getpid(),
    }
    if error:
        row["error"] = error
    with _LOCK:
        _ROWS.append(row)
        path = _sink_path()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                pass


def _model_name(self) -> str:
    for attr in ("model_name", "model"):
        v = getattr(self, attr, None)
        if isinstance(v, str) and v:
            return v
    return "unknown"


# ---------------------------------------------------------------- wrapper

def _wrap_generate(orig):
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        try:
            result = orig(self, *args, **kwargs)
        except Exception as e:
            _record(_model_name(self), {"in": 0, "out": 0, "total": 0, "cached": 0},
                    t0, _agent_of(kwargs.get("run_manager")), "generate", False,
                    error=f"{type(e).__name__}: {e}")
            raise
        usage = _usage_from_result(result)
        est = usage is None
        if usage is None:
            out_text = ""
            gens = getattr(result, "generations", None) or []
            if gens:
                out_text = str(getattr(getattr(gens[0], "message", None), "content", ""))
            usage = _estimate(args[0] if args else kwargs.get("messages"), out_text)
        _record(_model_name(self), usage, t0, _agent_of(kwargs.get("run_manager")),
                "generate", est)
        return result
    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    return wrapper


def _wrap_agenerate(orig):
    async def wrapper(self, *args, **kwargs):
        t0 = time.time()
        try:
            result = await orig(self, *args, **kwargs)
        except Exception as e:
            _record(_model_name(self), {"in": 0, "out": 0, "total": 0, "cached": 0},
                    t0, _agent_of(kwargs.get("run_manager")), "agenerate", False,
                    error=f"{type(e).__name__}: {e}")
            raise
        usage = _usage_from_result(result)
        est = usage is None
        if usage is None:
            out_text = ""
            gens = getattr(result, "generations", None) or []
            if gens:
                out_text = str(getattr(getattr(gens[0], "message", None), "content", ""))
            usage = _estimate(args[0] if args else kwargs.get("messages"), out_text)
        _record(_model_name(self), usage, t0, _agent_of(kwargs.get("run_manager")),
                "agenerate", est)
        return result
    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    return wrapper


def _wrap_stream(orig):
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        agent = _agent_of(kwargs.get("run_manager"))
        model = _model_name(self)
        seen: dict | None = None
        text_parts: list[str] = []
        error: str | None = None
        try:
            for chunk in orig(self, *args, **kwargs):
                msg = getattr(chunk, "message", None)
                u = _norm_usage(getattr(msg, "usage_metadata", None))
                if u:
                    seen = u
                text_parts.append(str(getattr(chunk, "content", "") or ""))
                yield chunk
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            usage = seen
            est = usage is None
            if usage is None:
                usage = _estimate(args[0] if args else kwargs.get("messages"),
                                  "".join(text_parts))
            _record(model, usage, t0, agent, "stream", est, error=error)
    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    return wrapper


def _wrap_astream(orig):
    async def wrapper(self, *args, **kwargs):
        t0 = time.time()
        agent = _agent_of(kwargs.get("run_manager"))
        model = _model_name(self)
        seen: dict | None = None
        text_parts: list[str] = []
        error: str | None = None
        try:
            async for chunk in orig(self, *args, **kwargs):
                msg = getattr(chunk, "message", None)
                u = _norm_usage(getattr(msg, "usage_metadata", None))
                if u:
                    seen = u
                text_parts.append(str(getattr(chunk, "content", "") or ""))
                yield chunk
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            usage = seen
            est = usage is None
            if usage is None:
                usage = _estimate(args[0] if args else kwargs.get("messages"),
                                  "".join(text_parts))
            _record(model, usage, t0, agent, "astream", est, error=error)
    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    return wrapper


# ---------------------------------------------------------------- 汇总

def snapshot() -> list[dict]:
    with _LOCK:
        return list(_ROWS)


def clear() -> None:
    with _LOCK:
        _ROWS.clear()


def totals(rows: Iterable[dict] | None = None) -> dict:
    rows = list(rows if rows is not None else snapshot())
    total_in = sum(int(r.get("in") or 0) for r in rows)
    total_out = sum(int(r.get("out") or 0) for r in rows)
    est = sum(1 for r in rows if r.get("estimated"))
    calls = len(rows)
    return {
        "calls": calls,
        "in": total_in,
        "out": total_out,
        "total": total_in + total_out,
        "estimated_calls": est,
        "estimated_ratio": round(est / calls, 4) if calls else 0.0,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def load_rows(path: str | Path) -> list[dict]:
    from .io_util import read_jsonl
    return read_jsonl(path)


def merge_dir(path: str | Path) -> dict:
    """把目录下所有 tokens-*.jsonl 汇总（一个实验轮可能跨多个子进程）。"""
    p = Path(path)
    rows: list[dict] = []
    if p.is_dir():
        for f in sorted(p.glob("tokens-*.jsonl")):
            rows.extend(load_rows(f))
    elif p.exists():
        rows.extend(load_rows(p))
    return totals(rows)
