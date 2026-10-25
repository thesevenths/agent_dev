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

流式 usage 为什么需要显式开启
------------------------------
`llm.py` 创建的 ``ChatOpenAI`` 没有设 ``stream_usage``，因此
``BaseChatOpenAI._should_stream_usage()`` 返回 ``False``，请求体里不会带
``stream_options={"include_usage": True}``，网关（vLLM）就不会在最后一个
空 choices chunk 里回 usage —— 结果流式调用 100% 退化成字符估算。

所以这里额外补一个 ``_should_stream_usage`` 补丁：只影响「响应里是否多带一个
usage 字段」，**不改变生成内容**，且两臂（base / cand）都装同一探针，故对比较是对称的、
惰性的。若后端不支持该字段（400 且报文提到 stream_options），自动关掉注入并用同一请求
重试一次，退回估算——绝不因为「量不到 token」而让整个 run 失败。

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

#: 是否强行让流式请求带上 stream_options.include_usage（见模块 docstring）。
#: 环境变量 ADAPTIVE_TOKEN_STREAM_USAGE=0 可关闭；运行期遇到后端 400 也会自动关闭。
_INJECT_USAGE = True
_INJECT_OFF_REASON = ""


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


def _env_allows_inject() -> bool:
    raw = os.environ.get("ADAPTIVE_TOKEN_STREAM_USAGE", "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _disable_inject(reason: str) -> None:
    """后端不接受 stream_options 时关掉注入，退回估算而不是让 run 失败。"""
    global _INJECT_USAGE, _INJECT_OFF_REASON
    if _INJECT_USAGE:
        _INJECT_USAGE = False
        _INJECT_OFF_REASON = reason


def usage_injection() -> dict:
    return {"enabled": bool(_INJECT_USAGE), "reason": _INJECT_OFF_REASON}


def _wrap_should_stream_usage(orig):
    """让 ``_should_stream_usage`` 返回 True，从而请求带上 include_usage。

    尊重调用方的显式意图：若它自己传了 ``stream_usage`` 或 ``stream_options``，
    一律交回原实现判断。
    """

    def wrapper(self, stream_usage=None, **kwargs):
        if _INJECT_USAGE and stream_usage is None and not isinstance(
                kwargs.get("stream_options"), dict):
            return True
        return orig(self, stream_usage, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    return wrapper


def _patch_stream_usage() -> bool:
    from langchain_openai.chat_models.base import BaseChatOpenAI

    if "_should_stream_usage" not in vars(BaseChatOpenAI):
        return False
    key = f"{BaseChatOpenAI.__module__}.{BaseChatOpenAI.__name__}._should_stream_usage"
    if key in _ORIG:
        return True
    orig = vars(BaseChatOpenAI)["_should_stream_usage"]
    _ORIG[key] = (BaseChatOpenAI, "_should_stream_usage", orig)
    BaseChatOpenAI._should_stream_usage = _wrap_should_stream_usage(orig)
    return True


def install(sink: str | Path | None = None, memory_only: bool = False) -> bool:
    """安装埋点。返回 True 表示本次真正安装，False 表示此前已安装/环境不支持。"""
    global _PATCHED, _SINK, _MEMORY_ONLY, _INJECT_USAGE, _INJECT_OFF_REASON
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

    # 流式 usage 注入（可降级，失败不影响主流程）
    if _env_allows_inject():
        _INJECT_USAGE = True
        try:
            _patch_stream_usage()
        except Exception:
            pass
    else:
        _INJECT_USAGE = False
        _INJECT_OFF_REASON = "disabled_by_env"

    _PATCHED = True
    return True


def uninstall() -> bool:
    global _PATCHED, _INJECT_USAGE, _INJECT_OFF_REASON
    if not _PATCHED:
        return False
    for _key, (cls, name, orig) in list(_ORIG.items()):
        setattr(cls, name, orig)
    _ORIG.clear()
    _PATCHED = False
    _INJECT_USAGE = True
    _INJECT_OFF_REASON = ""
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


def _is_stream_options_rejection(exc: BaseException) -> bool:
    """只有「明确抱怨 stream_options/include_usage」的 400 才降级。

    不能对所有异常都降级：那会把真正的网络/鉴权错误吞掉，变成一次错误的成功记录。
    """
    msg = str(exc).lower()
    return "stream_options" in msg or "include_usage" in msg


def _stream_steps(orig, self, args, kwargs, st: dict):
    """跑一次流式迭代，把 usage 与文本累积进 ``st``。

    单独抽出来是为了「关掉注入后原样重试一次」时能复用同一段逻辑。
    """
    for chunk in orig(self, *args, **kwargs):
        msg = getattr(chunk, "message", None)
        u = _norm_usage(getattr(msg, "usage_metadata", None))
        if u:
            st["usage"] = u
        st["text"].append(str(getattr(chunk, "content", "") or ""))
        yield chunk


def _finalize_stream(st: dict, args, kwargs, model, t0, agent, kind, error):
    usage = st["usage"]
    est = usage is None
    if usage is None:
        usage = _estimate(args[0] if args else kwargs.get("messages"), "".join(st["text"]))
    _record(model, usage, t0, agent, kind, est, error=error)


def _wrap_stream(orig):
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        agent = _agent_of(kwargs.get("run_manager"))
        model = _model_name(self)
        st: dict = {"usage": None, "text": []}
        error: str | None = None
        try:
            try:
                yield from _stream_steps(orig, self, args, kwargs, st)
            except Exception as e:
                if not (_INJECT_USAGE and _is_stream_options_rejection(e)):
                    raise
                # 后端不支持 stream_options：关注入并原样重试一次
                _disable_inject(f"{type(e).__name__}: {e}")
                st["usage"] = None
                st["text"] = []
                yield from _stream_steps(orig, self, args, kwargs, st)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            _finalize_stream(st, args, kwargs, model, t0, agent, "stream", error)
    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    return wrapper


def _wrap_astream(orig):
    async def wrapper(self, *args, **kwargs):
        t0 = time.time()
        agent = _agent_of(kwargs.get("run_manager"))
        model = _model_name(self)
        st: dict = {"usage": None, "text": []}
        error: str | None = None

        async def once():
            async for chunk in orig(self, *args, **kwargs):
                msg = getattr(chunk, "message", None)
                u = _norm_usage(getattr(msg, "usage_metadata", None))
                if u:
                    st["usage"] = u
                st["text"].append(str(getattr(chunk, "content", "") or ""))
                yield chunk

        try:
            try:
                async for chunk in once():
                    yield chunk
            except Exception as e:
                if not (_INJECT_USAGE and _is_stream_options_rejection(e)):
                    raise
                _disable_inject(f"{type(e).__name__}: {e}")
                st["usage"] = None
                st["text"] = []
                async for chunk in once():
                    yield chunk
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            _finalize_stream(st, args, kwargs, model, t0, agent, "astream", error)
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
        # 便于报告里解释「为什么 estimated 很高」：注入被关掉时后端不回流式 usage
        "stream_usage_injection": bool(_INJECT_USAGE),
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
