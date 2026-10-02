#!/usr/bin/env python
"""端到端自检：本地 embedding 服务 → longterm 长期记忆写入/召回全链路。

为什么需要：longterm.py 在 embedding 不可达时会**静默**回落关键词匹配（longterm.py:114），
所以配置错了你根本看不出来——只是"记忆召回变傻了"而已。本脚本会把这条静默路径暴露成显式 FAIL。

跑法（服务已在 flink 起好、端口从 .env 读）：
    python check_embed_service.py
    # 指定非默认端点：
    AGENT_MEMORY_EMBED_BASE_URL=http://192.168.x.x:8100/v1 python check_embed_service.py

验证项：
  1. 服务 /health 可达
  2. /embeddings 返回条数 == 请求条数、顺序与输入一致（longterm.py:108 靠 index 排序，乱序会错位）
  3. 返回维度 == AGENT_MEMORY_EMBED_DIM
  4. 长文本（>512 token 会截断）与短 query 都能编码不报错
  5. longterm 闭环，且**必须证明走的是 vector 而不是 keyword 兜底**：
     a. longterm._embed() 返回非 None（决定性判据）
     b. remember() 落库行的 embedding 非空且维度正确
     c. 用 monkeypatch 把 longterm._tokenize 置空后再 recall 仍有结果
        （关键词分支已失效，还能召回 → 只可能是向量路径；
          上一版只用「recall 有结果」判定，被字面重合的假命中骗过）
  6. 降级检查：反向确认 base_url 填错时会 FAIL（提示你别写错）

用独立临时 DB（AGENT_MEMORY_DB 先设到 tmp/ 下），不污染真实 memory/longterm.db。
"""
from __future__ import annotations

import os
import sys
import tempfile

_FAILS: list[str] = []


def _check(cond: bool, label: str, detail: str = "") -> bool:
    mark = "✔" if cond else "✖"
    print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(label)
    return cond


def main() -> int:
    # —— 必须在 import longterm 之前设好，longterm 在模块级读 env（longterm.py:38-52）——
    tmpdb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "_embed_smoke.db")
    os.environ["AGENT_MEMORY_DB"] = tmpdb
    os.environ.setdefault("AGENT_LONGTERM_MEMORY", "1")

    from dotenv import load_dotenv
    load_dotenv()  # 让 .env 里的 AGENT_MEMORY_EMBED_* 生效（与 agent 运行时的取值一致）

    embed = os.environ.get("AGENT_MEMORY_EMBED_BASE_URL", "").rstrip("/")
    key = os.environ.get("AGENT_MEMORY_EMBED_API_KEY", "")
    model = os.environ.get("AGENT_MEMORY_EMBED_MODEL", "")
    dim_exp = int(os.environ.get("AGENT_MEMORY_EMBED_DIM", "0") or 0)
    print(f"[check] base_url={embed}  model={model}  dim_req={dim_exp or 'native'}  key={'set' if key else 'EMPTY!'}")

    import httpx

    # 1) 服务可达
    print("\n[1] 服务健康")
    try:
        r = httpx.get(embed.rsplit("/v1", 1)[0] + "/health", timeout=10)
        ok = r.status_code == 200
        _check(ok, "/health 可达", f"HTTP {r.status_code} {r.text[:120]}")
    except Exception as e:
        _check(False, "/health 可达", f"{type(e).__name__}: {e}")
        print("\n服务没起起来 → 在 flink 上跑： bash start_embed_server.sh ，日志 /root/embed_server.log")
        return 1

    # 2~4) 编码契约
    print("\n[2] /embeddings 契约（longterm._embed 的调用形状）")
    samples = ["用户偏好中文回复、结论先行", "我喜欢用表格对比方案差异", "x" * 200]
    try:
        r = httpx.post(
            embed + "/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "input": samples, "encoding_format": "float",
                  **({"dimensions": dim_exp} if dim_exp else {})},
            timeout=float(os.environ.get("AGENT_MEMORY_EMBED_TIMEOUT", "30")),
        )
        if _check(r.status_code == 200, "HTTP 200", f"HTTP {r.status_code} {r.text[:150]}"):
            data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
            _check(len(data) == len(samples), "条数一致", f"got {len(data)}")
            _check([d.get("index") for d in data] == list(range(len(samples))), "index 有序")
            vecs = [d.get("embedding") for d in data]
            _check(all(isinstance(v, list) for v in vecs), "embedding 是数组")
            d0 = len(vecs[0]) if vecs[0] else 0
            _check(dim_exp == 0 or d0 == dim_exp, f"维度 == {dim_exp}", f"got {d0}")
        elif r.status_code == 422 and "query" in r.text and "payload" in r.text:
            # 这是服务端路由把 payload 当成 query 参数的经典症状（报错体：
            # loc:["query","payload"]）。请求体明明发了却说 payload 缺失。
            print("    ↳ 提示：服务端把 payload 当成 query 参数了（不是你的配置问题）。"
                  "原因通常是请求体模型被定义在 build_app() 局部作用域 + "
                  "`from __future__ import annotations`，FastAPI 解析不到注解字符串。")
    except Exception as e:
        _check(False, "编码请求", f"{type(e).__name__}: {e}")

    # 5) longterm 闭环（必须走 vector 而非 keyword 兜底）
    print("\n[3] longterm 写入 → 语义召回（必须走 vector，不能回落 keyword）")
    try:
        import sqlite3

        import numpy as np

        import longterm

        want_dim = dim_exp or 1024

        # 3a) 底层向量通道是否真的活着 —— 决定性判据。
        #     注意：不能用「recall 有没有结果」当判据！关键词兜底也能出结果，
        #     上一版就是这么被假绿的（query 里的"调用链提取"与记忆里的字面重合）。
        pv = longterm._embed(["向量通道探针"])
        if _check(pv is not None, "longterm._embed() 返回非 None（否则一切召回都在走关键词）"):
            _check(len(pv[0]) == want_dim, f"探针维度 == {want_dim}", f"got {len(pv[0])}")

        probes = [
            ("用户是做安全攻防的后端工程师，习惯结论先行 + 表格化交付", "profile", ["偏好", "角色"]),
            ("评估 joern 与 tree-sitter 两种调用链提取方案在 nacos 上的召回率", "decision", ["joern", "tree-sitter"]),
            ("GuardFox 的 sink 驱动扫描基线是 OWASP BenchmarkJava", "fact", ["sink", "benchmark"]),
        ]
        for text, t, tags in probes:
            mid, action = longterm.remember(text, mtype=t, tags=tags, importance=0.7)
            _check(action in ("inserted", "dup"), f"remember(): {text[:22]}...", f"action={action}")

        # 3b) 落库的 embedding 是否非空、维度对不对
        with sqlite3.connect(tmpdb) as c:
            n_all, n_vec = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(embedding IS NOT NULL), 0) FROM memories").fetchone()
            dims = sorted({d for (d,) in
                           c.execute("SELECT DISTINCT dim FROM memories WHERE embedding IS NOT NULL")})
        _check(n_all > 0 and n_vec == n_all, "落库行都带向量", f"{n_vec}/{n_all}")
        _check(dims == [want_dim], f"落库维度 == {want_dim}", f"got {dims}")

        # 3c) 把关键词兜底掐掉后仍能召回 → 唯一可能是走了向量路径
        query = "我用哪个工具对比过调用链提取？"
        orig_tok = getattr(longterm, "_tokenize", None)
        longterm._tokenize = lambda _s: set()   # 关键词分支再也匹配不到任何东西
        try:
            hit = longterm.recall(query, memory_key="embed_smoke")
        finally:
            if orig_tok is not None:
                longterm._tokenize = orig_tok

        _check(bool(hit), "掐掉关键词兜底后 recall 仍有结果（→ 向量路径生效）", f"{len(hit)} chars")
        _check("joern" in hit.lower() or "tree-sitter" in hit.lower(),
               "命中的是 joern/tree-sitter 那条（而非字面重合的巧合）")
        print("\n  recall 返回：\n" + ("\n".join("    " + ln for ln in hit.splitlines()) or "    (空)"))

        # 若 3c 空但 3a/3b 通过 → 多半是 MIN_SCORE 阈值不匹配新模型的余弦分布
        qv_list = longterm._embed([query]) if (not hit and pv is not None) else None
        if not hit and pv is not None and not qv_list:
            # 向量通道本身不可用（典型：服务端 /v1/embeddings 报 422）
            print("    ↳ 向量通道不可用（_embed 返回 None）→ 先修 [2] 的报错，再回来看阈值。")
        elif not hit and qv_list:
            qv = np.asarray(qv_list[0], dtype=np.float32)
            cos = []
            with sqlite3.connect(tmpdb) as c:
                for txt, dim, blob in c.execute(
                        "SELECT text, dim, embedding FROM memories WHERE embedding IS NOT NULL"):
                    if int(dim or 0) != qv.shape[0]:
                        continue
                    cos.append((longterm._cosine(qv, longterm._from_blob(blob)), txt[:34]))
            cos.sort(reverse=True)
            print("    ↳ 诊断：query 与库内各条的余弦 top3 = "
                  + ", ".join(f"{c:.3f}" for c, _ in cos[:3]))
            print(f"      AGENT_MEMORY_MIN_SCORE={os.environ.get('AGENT_MEMORY_MIN_SCORE')}"
                  f"  AGENT_MEMORY_DEDUP_COS={os.environ.get('AGENT_MEMORY_DEDUP_COS')}")
            print("      若 top1 明显高于 MIN_SCORE 却仍为空，才需要查 dedup/去重逻辑；"
                  "否则跑 check_memory_vectors.py 校准阈值。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _check(False, "longterm 闭环", f"{type(e).__name__}: {e}")

    # 6) 降级路径反证：把端点打错，确认会 FAIL（说明第 1 步不是"碰巧都是绿的"）
    #    trust_env=False：本机若挂了 HTTP 代理，127.0.0.1:1 会被代理接管并回 502，
    #    那样"错误端点被拒绝"就变成了"被代理拒绝"，反证失去意义。
    print("\n[4] 反证：填错端点应失败（确认上面检测真的在生效）")
    try:
        bad = httpx.post("http://127.0.0.1:1/embeddings", json={"input": ["x"]},
                         timeout=3, trust_env=False)
        _check(bad.status_code != 200, "错误端点未被当成成功",
               f"HTTP {bad.status_code}")
    except Exception as e:
        _check(True, "错误端点被拒绝（符合预期）", type(e).__name__)

    print("\n" + "=" * 60)
    if _FAILS:
        print(f"FAIL（{len(_FAILS)} 项）: " + "; ".join(_FAILS))
        print("排查顺序：① 服务是否起（/health）② API_KEY 是否留空 ③ 维度是否匹配 ④ 防火墙/端口")
        return 1
    print("PASS：本地 embedding 服务已就绪，agent 直接生效（无需改代码）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
