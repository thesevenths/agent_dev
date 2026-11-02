#!/usr/bin/env python
"""轻量版本地 embedding 服务（OpenAI 兼容 /v1/embeddings）—— 不依赖 sentence-transformers。

为什么有这份
------------
`embed_server.py` 走 sentence-transformers，而它会拖 scikit-remodel → scikit-learn。
scikit-learn >= 1.4 改成 **meson** 构建，build requires = meson-python + Cython + numpy；
在老系统（CentOS 7 / gcc 4.8.5 / glibc 2.17）上 numpy>=2.3 只有 manylinux_2_28 轮子，
pip 匹配不上就退回 sdist 源码编译 → gcc 4.8.5 报 "Compiler cython cannot compile programs"。

本文件把 sentence-transformers 完全去掉：直接用 transformers + torch，自己按
`modules.json` / `1_Pooling/config.json` 做 pooling。依赖只剩
torch / transformers / fastapi / uvicorn / pydantic / numpy —— **全部有 manylinux2014 轮子，
不需要任何源码编译**，因此在 CentOS 7 这类老机器上可直接装。

接口与 embed_server.py **完全一致**（同一套端点、同一套 OpenAI 响应形状），
所以客户端（longterm.py）配置不用改：
    AGENT_MEMORY_EMBED_BASE_URL=http://<host>:8100/v1
    AGENT_MEMORY_EMBED_API_KEY=dummy        # 不能留空！空则 longterm._embed 直接返回 None 走关键词兜底
    AGENT_MEMORY_EMBED_MODEL=Qwen3-Embedding-0.6B
    AGENT_MEMORY_EMBED_DIM=1024

新增/差异点（相对 embed_server.py）
----------------------------------
- pooling 从 1_Pooling/config.json 的 pooling_mode 推断：0=CLS 1=MEAN 2=MAX 3=MEAN_SQRT_LEN
  缺省为 MEAN（Qwen3-Embedding 官方用的是 mean pooling）。
- 启动时若发现 sentence_transformers 已装，仍优先用它（功能等价，行为最稳）；
  没装就走本文件的手写实现。**二者输出应一致**（都在末尾做 L2 归一化）。
- 其余：单 worker + 一把锁串行 encode、显式 L2 归一化、启动 warmup、
  MRL 降维（--dim）、query 侧检索指令前缀（--query-prefix 且 input 项标 "role":"query"）。

用法
----
    python embed_server_light.py --selftest                 # 只自检，不起服务
    python embed_server_light.py --host 0.0.0.0 --port 8100 # 起服务
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any

# === 顶层导入（必须！）===
# torch / numpy 一律在模块顶层 import，不要在函数内部 import。
# 曾踩过的坑：`_pool_token_embs()` 里写 `torch.clamp(...)`，而 torch 只在 `_encode()`
# 里局部 import —— 跨函数就变成 NameError: name 'torch' is not defined。
# 顶层导入还有个好处：torch 没装时直接 ImportError fail-fast，而不是等到第一次 encode 才炸。
import numpy as np
import torch

# === 请求体模型（必须在模块顶层！）===
# 坑（已踩，2026-10-02）：本文件有 `from __future__ import annotations`，注解会变成
# **字符串**；FastAPI 在解析 "EmbedReq" 时只查**模块 globals**。若把 EmbedReq 定义在
# `build_app()` 里面（局部作用域），这个名字解析不到，FastAPI 就会把 payload 从
# request body 降级成 **query 参数**，于是客户端 POST /v1/embeddings 会收到：
#     422 {"detail":[{"type":"missing","loc":["query","payload"],"msg":"Field required"}]}
# 请求体明明发了，却报「payload 字段缺失」，且 longterm._embed 只会 warn 一行后静默
# 回落到关键词匹配 —— 极难排查。所以：EmbedReq 永久留在模块顶层。
# 对照：同文件的 `/embed` 用的是 `payload: dict`（builtin，模块级可直接解析）→ 正常。
try:
    from pydantic import BaseModel as _PydBase, Field as _PydField

    class EmbedReq(_PydBase):
        model: str | None = None
        input: Any = _PydField(default=None)
        encoding_format: str | None = None
        dimensions: int | None = None
except Exception:  # pragma: no cover - 只影响 --selftest（不需要 HTTP 层）
    EmbedReq = None  # type: ignore[assignment]

# === 配置（env 优先，命令行可覆盖）===
_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "/root/model/Qwen3-Embedding-0.6B")
_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "Qwen3-Embedding-0.6B")
_HOST = os.environ.get("EMBED_HOST", "0.0.0.0")
_PORT = int(os.environ.get("EMBED_PORT", "8100"))
_DEVICE = os.environ.get("EMBED_DEVICE", "auto")          # auto | cpu | cuda
_DIM = int(os.environ.get("EMBED_DIM", "0")) or None      # 0/None = 用模型原生维度
_NORMALIZE = os.environ.get("EMBED_NORMALIZE", "1").strip().lower() not in ("0", "false", "no", "off")
_BATCH = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
_MAX_SEQ = int(os.environ.get("EMBED_MAX_SEQ_LEN", "512"))   # 0 = 用 tokenizer 默认
_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "").strip() or None

logging.basicConfig(
    level=os.environ.get("EMBED_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [embed_light] %(message)s",
)
logger = logging.getLogger("embed_light")

_lock = threading.Lock()
# torch 相关的全局状态
_tk = None          # tokenizer
_mdl = None         # model
_dev = None         # device
_native_dim = 0
_backend = ""       # "sentence_transformers" | "transformers"
_pool_mode = 1      # 0=CLS 1=MEAN 2=MAX 3=MEAN_SQRT_LEN


# === pooling 配置解析 ===
def _read_pooling_mode(model_path: str) -> int:
    """推断 pooling 方式：优先直读 <model_path>/1_Pooling/config.json 的 pooling_mode。

    兼容 modules.json 的两种形态（各仓库写法不一，这里都吃掉）：
      - dict: {"modules":[{"type":"1_Pooling","model_name_or_path":...}, ...]}
      - list: ["1_Pooling", "0_<model>"]  或  [{"type":"1_Pooling", ...}, ...]
    都拿不到就按默认 MEAN（Qwen3-Embedding 官方即 mean pooling）。
    """
    cfg_dir = os.path.join(model_path, "1_Pooling")
    cfg_path0 = os.path.join(cfg_dir, "config.json")
    if os.path.exists(cfg_path0):
        try:
            with open(cfg_path0, "r", encoding="utf-8") as f:
                return int(json.load(f).get("pooling_mode", 1))
        except Exception as e:
            logger.warning(f"1_Pooling/config.json 解析失败({e})，继续查 modules.json")

    pool_dir = None
    mods_path = os.path.join(model_path, "modules.json")
    if os.path.exists(mods_path):
        try:
            with open(mods_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            mods = raw.get("modules", []) if isinstance(raw, dict) else raw
            pool_dir = None
            for m in mods:
                if isinstance(m, str):
                    if m.split("/")[-1].startswith("1_Pooling"):
                        pool_dir = m
                        break
                elif isinstance(m, dict):
                    if str(m.get("type", "")).startswith("1_Pooling"):
                        pool_dir = m.get("model_name_or_path") or m.get("folder_name")
                        break
        except Exception as e:
            logger.warning(f"modules.json 解析失败({e})，按默认 MEAN 处理")
    pool_dir = pool_dir or "1_Pooling"
    cfg_path = os.path.join(model_path, pool_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                mode = int(json.load(f).get("pooling_mode", 1))
            return mode
        except Exception as e:
            logger.warning(f"pooling config 解析失败({e})，按默认 MEAN 处理")
    return 1


# === 模型加载 ===
def _resolve_device() -> str:
    if _DEVICE != "auto":
        return _DEVICE
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model():
    """惰性加载 + warmup。失败直接抛——不做静默吞异常（服务起不来就该让人看见）。"""
    # 注意：本函数里若给某个模块级变量赋值，就必须把它列进 global，
    # 否则该变量在函数内被当成局部变量，前面一读就 UnboundLocalError。
    global _tk, _mdl, _dev, _native_dim, _backend, _pool_mode, _MAX_SEQ

    with _lock:
        if _mdl is not None:
            return

        _pool_mode = _read_pooling_mode(_MODEL_PATH)
        _dev = _resolve_device()

        # 优先：已装 sentence-transformers 则复用（行为最稳）
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            st = SentenceTransformer(_MODEL_PATH, device=_dev,
                                     model_kwargs={"dtype": torch.float32 if _dev == "cpu" else torch.bfloat16})
            st.eval()
            _mdl, _backend = st, "sentence_transformers"
            _native_dim = int(st.get_sentence_embedding_dimension())
            logger.info(f"backend=sentence_transformers ready → {_MODEL_PATH}")
        except Exception as e:
            logger.info(f"走 transformers 手写后端（sentence-transformers 不可用：{type(e).__name__}）")

        if _backend != "sentence_transformers":
            from transformers import AutoModel, AutoTokenizer
            dtype = torch.float32 if _dev == "cpu" else torch.bfloat16
            _tk = AutoTokenizer.from_pretrained(_MODEL_PATH)
            _mdl = AutoModel.from_pretrained(_MODEL_PATH, dtype=dtype).to(_dev)
            _mdl.eval()
            _backend = "transformers"
            cfg_dim = getattr(getattr(_mdl, "config", None), "hidden_size", 0) or 0
            _native_dim = int(cfg_dim)
            # 尊重模型自带的 max position embedding，但截断到 2048：
            # Qwen3 tokenizer 的 model_max_length 是 40960，放开会让每条记忆都 encode 几万 token（CPU 上极慢）。
            if not _MAX_SEQ:
                tm = getattr(_tk, "model_max_length", None)
                if isinstance(tm, int) and tm > 0 and tm < 100000:
                    _MAX_SEQ = min(tm, 2048)
                    logger.info(f"max_seq 自动取 min(model_max_length={tm}, 2048) = {_MAX_SEQ}")
            logger.info(f"backend=transformers ready → {_MODEL_PATH} (pooling_mode={_pool_mode}, device={_dev}, max_seq={_MAX_SEQ})")

        if _native_dim <= 0:
            raise RuntimeError(f"无法推断向量维度（hidden_size={_native_dim}），请检查 {_MODEL_PATH}/config.json")

    # warmup：把加载/首次算子耗时从真实请求里挪走
    try:
        _encode(["warmup"])
        logger.info(f"warmup ok (native_dim={_native_dim}, backend={_backend})")
    except Exception as e:
        logger.warning(f"warmup skipped ({type(e).__name__}: {e})")
    return _mdl


# === 编码核心 ===
def _pool_token_embs(last_hidden_state, attention_mask):
    """按 _pool_mode 做 pooling。last_hidden_state: (B, T, H)。"""
    mode = _pool_mode
    if mode == 0:      # CLS
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    if mode == 2:      # MAX
        masked = last_hidden_state.masked_fill(mask.bool() == 0, -1e4)
        return masked.max(dim=1).values
    summed = (last_hidden_state * mask).sum(dim=1)
    # 用张量自带 .clamp() 而不是 torch.clamp()：
    # 1) 少一个全局符号依赖（跨函数就不会再冒出 NameError）；
    # 2) 自动跟随 mask 的设备，不用管 _dev 是 cpu 还是 cuda。
    seqlen = mask.sum(dim=1).clamp(min=1e-9)
    if mode == 3:      # MEAN_SQRT_LEN
        denom = seqlen.pow(0.5)
    else:              # MEAN
        denom = seqlen
    return summed / denom


def _encode(texts: list[str]):
    """串行编码（锁保护）。返回 numpy (N, dim) float32。"""
    load_model()
    with _lock:
        t0 = time.perf_counter()
        if _backend == "sentence_transformers":
            vecs = _mdl.encode(list(texts), batch_size=_BATCH, normalize_embeddings=False,
                               convert_to_numpy=True, show_progress_bar=False)
        else:
            out = []
            for i in range(0, len(texts), _BATCH):
                chunk = texts[i:i + _BATCH]
                enc = _tk(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=_MAX_SEQ if _MAX_SEQ else None)
                enc = {k: (v.to(_dev) if hasattr(v, "to") else v) for k, v in enc.items()}
                with torch.no_grad():
                    hid = _mdl(**enc).last_hidden_state
                out.append(_pool_token_embs(hid, enc["attention_mask"]).float().cpu().numpy())
            vecs = np.concatenate(out, axis=0) if len(out) > 1 else out[0]

        vecs = np.asarray(vecs, dtype=np.float32)
        cost = (time.perf_counter() - t0) * 1000.0
    logger.info(f"encode {len(texts)} text(s) in {cost:.1f}ms shape={vecs.shape}")
    return vecs


def _apply_dim(vecs):
    """按 _DIM 截断（MRL 合法截断：取前 N 维）。"""
    if not _DIM or _DIM >= _native_dim:
        return vecs
    return vecs[:, :_DIM]


# === 请求解析 ===
def _parse_input(payload: dict) -> list[tuple[str, str | None]]:
    """兼容 input 为 str / [str] / [{"text":..,"role":..}]。返回 [(text, role)]。"""
    raw = payload.get("input")
    if raw is None:
        raw = payload.get("inputs")
    if raw is None:
        raise ValueError("missing field: input")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("input must be a non-empty string or list")

    out: list[tuple[str, str | None]] = []
    for it in raw:
        if isinstance(it, str):
            out.append((it, None))
        elif isinstance(it, dict):
            txt = it.get("text")
            if txt is None:
                txt = it.get("input")
            if txt is None:
                raise ValueError("each input item needs a 'text' field")
            out.append((str(txt), (it.get("role") or it.get("_role") or None)))
        else:
            out.append((str(it), None))
    return out


def _resolve_query_prefix(cli_value: str | None) -> str | None:
    """解析 query 检索指令前缀：CLI 优先，其次 env EMBED_QUERY_PREFIX。

    坑（已踩）：`--query-prefix` 的 argparse 默认值是空串，若直接
    `_QUERY_PREFIX = args.query_prefix.strip() or None`，就会把 import 时从
    EMBED_QUERY_PREFIX 读到的值**静默抹掉** —— 用户以为 env 配上了，实际没生效。
    """
    return ((cli_value or _QUERY_PREFIX or "").strip()) or None


def _decorate(role: str | None, text: str) -> str:
    """query 侧加 Qwen3-Embedding 官方检索指令前缀（仅当配了 --query-prefix 且标了 role=query）。"""
    return f"{_QUERY_PREFIX}\nQuery: {text}" if (role == "query" and _QUERY_PREFIX) else text


def _finalize(vecs, want_dim: int | None = None):
    """统一做：MRL 截断 → L2 归一化 → tolist。want_dim 为 None 时用 _DIM。"""
    vecs = _apply_dim(vecs)
    target = want_dim or _DIM or int(vecs.shape[-1])
    if 0 < target < vecs.shape[-1]:
        vecs = vecs[:, :target]
    if _NORMALIZE:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        vecs = vecs / np.where(norms == 0.0, 1.0, norms)
    return vecs


# === HTTP 层（fastapi 为可选：--selftest 时不需要）===
def _assert_embeddings_route_binds_body(app) -> None:
    """启动期守门：确认 POST /v1/embeddings 的 payload 真的绑在 request body 上。

    背景（已踩坑，2026-10-02）：本文件有 `from __future__ import annotations`，注解是字符串；
    FastAPI 只在**模块 globals** 里解析 "EmbedReq"。如果请求体模型被定义在函数局部作用域，
    解析失败后 FastAPI 会把 payload 降级成 **query 参数**，客户端于是收到：
        422 {"detail":[{"type":"missing","loc":["query","payload"],"msg":"Field required"}]}
    请求体明明发了却说 payload 缺失；而 longterm._embed 只会 warn 一行然后静默回落关键词召回，
    表现为"记忆检索变傻但不报错" —— 排查成本极高。这里把它变成启动时的一声巨响。

    只查路由 dependant（不依赖 openapi() 生成，后者在签名坏掉时自身会抛异常）。
    """
    routes = [r for r in getattr(app, "routes", [])
              if getattr(r, "path", None) == "/v1/embeddings"
              and "POST" in (getattr(r, "methods", None) or ())]
    if not routes:
        logger.warning("路由自检跳过：未找到 POST /v1/embeddings")
        return
    route = routes[0]
    qnames = [getattr(p, "name", "?") for p in
              (getattr(getattr(route, "dependant", None), "query_params", None) or [])]
    if "payload" in qnames or getattr(route, "body_field", None) is None:
        raise RuntimeError(
            "POST /v1/embeddings 的 payload 未被绑定为 request body"
            f"（query_params={qnames}）—— 客户端会收到 422 loc=['query','payload']。"
            "请确认请求体模型（EmbedReq）定义在**模块顶层**，而不是 build_app() 内部。")
    logger.info("路由自检通过：POST /v1/embeddings 的 payload 已绑定为 request body")


def build_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    if EmbedReq is None:
        raise RuntimeError("pydantic 不可用，无法构建 HTTP 层；请 pip install pydantic")

    app = FastAPI(title="Local Embedding Service (light)", version="1.2")

    # 注意：EmbedReq 定义在**模块顶层**（见文件头注释）。若挪到这里会因
    # `from __future__ import annotations` 解析不到名字 → payload 被当 query 参数 → 422。

    def _ok() -> bool:
        return _mdl is not None

    @app.get("/health")
    def health():
        return {"status": "ok" if _ok() else "loading", "model": _MODEL_NAME,
                "path": _MODEL_PATH, "dim": _native_dim, "device": _resolve_device(),
                "backend": _backend or "not-loaded", "normalize": _NORMALIZE,
                "pooling_mode": _pool_mode, "dim_cap": _DIM or _native_dim}

    @app.get("/info")
    def info():
        return {"model": _MODEL_NAME, "path": _MODEL_PATH, "dim": _native_dim,
                "dim_cap": _DIM or _native_dim, "device": _resolve_device(),
                "backend": _backend or "not-loaded", "pooling_mode": _pool_mode,
                "normalize": _NORMALIZE, "max_seq_length": _MAX_SEQ,
                "query_prefix": _QUERY_PREFIX or ""}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": _MODEL_NAME, "object": "model",
                                            "owned_by": "local-embed-server"}]}

    @app.post("/embed")
    def embed_simple(payload: dict):
        items = _parse_input(payload)
        texts = [_decorate(role, t) for t, role in items]
        arr = _finalize(_encode(texts)).tolist()
        return {"embeddings": arr, "dim": len(arr[0]) if arr else 0}

    @app.post("/v1/embeddings")
    def embeddings(payload: EmbedReq):
        """OpenAI 兼容。longterm.py:102 打的就是这个端点，字段多余/缺失都宽容处理。"""
        items = _parse_input(payload.model_dump())
        texts = [_decorate(role, t) for t, role in items]
        vecs = _finalize(_encode(texts), want_dim=payload.dimensions)

        raw: list = vecs.tolist()
        if str(payload.encoding_format or "float").lower() == "base64":
            import base64
            raw = [base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode() for v in raw]

        data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(raw)]
        return {"object": "list", "data": data, "model": payload.model or _MODEL_NAME,
                "usage": {"prompt_tokens": sum(len(t) for t in texts), "total_tokens": sum(len(t) for t in texts)}}

    # --- 启动期自检：/v1/embeddings 必须真的把 payload 当 request body ---
    _assert_embeddings_route_binds_body(app)

    @app.exception_handler(Exception)
    async def _any_exc(_, exc):
        logger.error(f"unhandled: {type(exc).__name__}: {exc}")
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": type(exc).__name__}})

    return app


# === 自检（不起服务，直接验证模型可用性）===
def selftest() -> int:
    print(f"[selftest] path={_MODEL_PATH}")
    print(f"[selftest] device={_resolve_device()} normalize={_NORMALIZE} dim_cap={_DIM} max_seq={_MAX_SEQ}")
    t0 = time.perf_counter()
    load_model()
    load_cost = time.perf_counter() - t0

    samples = ["用户偏好中文回复、结论先行", "我喜欢用表格对比方案", "Qwen3-Embedding-0.6B 是 SentenceTransformer 格式"]
    t1 = time.perf_counter()
    vecs = _encode(samples)
    enc_cost = (time.perf_counter() - t1) * 1000
    print(f"[selftest] backend={_backend} pooling_mode={_pool_mode}")
    print(f"[selftest] load={load_cost:.2f}s  encode {len(samples)} texts={enc_cost:.1f}ms  shape={vecs.shape}")

    sims = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sims.append(float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))))
    print(f"[selftest] pairwise cosine={[round(s, 4) for s in sims]}")
    assert vecs.shape[0] == len(samples) and vecs.ndim == 2, f"bad shape {vecs.shape}"
    assert vecs.shape[1] == _native_dim, f"dim mismatch {vecs.shape[1]} != {_native_dim}"
    print(f"[selftest] PASS (dim={vecs.shape[1]})")
    return 0


def main() -> int:
    global _MODEL_PATH, _DEVICE, _DIM, _BATCH, _MAX_SEQ, _QUERY_PREFIX, _NORMALIZE
    ap = argparse.ArgumentParser(description="Local OpenAI-compatible embedding server (no sentence-transformers)")
    ap.add_argument("--model-path", default=_MODEL_PATH)
    ap.add_argument("--host", default=_HOST)
    ap.add_argument("--port", type=int, default=_PORT)
    ap.add_argument("--device", default=_DEVICE)
    ap.add_argument("--dim", type=int, default=0, help="MRL truncate dims; 0 = native")
    ap.add_argument("--batch-size", type=int, default=_BATCH)
    ap.add_argument("--max-seq-len", type=int, default=_MAX_SEQ, help="0 = auto")
    ap.add_argument("--query-prefix", default="", help='Qwen3 检索指令，如 "Instruct: Given a web search query..."')
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="只自检不起服务")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    _MODEL_PATH, _DEVICE = args.model_path, args.device
    _DIM = args.dim or None
    _BATCH, _QUERY_PREFIX = args.batch_size, _resolve_query_prefix(args.query_prefix)
    _MAX_SEQ = args.max_seq_len
    _NORMALIZE = not args.no_normalize

    if args.selftest:
        return selftest()

    import uvicorn
    load_model()
    logger.info(f"serving on http://{args.host}:{args.port}")
    uvicorn.run(build_app(), host=args.host, port=args.port, workers=args.workers,
                log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
