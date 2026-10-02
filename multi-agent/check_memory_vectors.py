#!/usr/bin/env python
"""长期记忆向量体检 + 阈值校准（只读校验 / 按需重编码 / 给阈值建议）。

为什么需要这个脚本
------------------
把 `longterm.py` 的 embedding 端点从 DashScope 换成本地 Qwen3-Embedding-0.6B 之后，有两个
**不会报错、只会让召回悄悄变烂**的隐患：

1. **向量空间混用（最致命）**。旧的 memory/longterm.db 里的向量是 DashScope text-embedding-v4
   算出来的，新查询向量是 Qwen3 算的。两者维度都是 1024，所以 longterm.py:234 那句
   `int(r["dim"]) != qv.shape[0]` 的维度护栏**拦不住**，余弦值变成跨模型的噪声 ——
   结果是老记忆要么永远召不回、要么召回一堆无关的。判断办法：拿库里某条的 text 用**新模型**
   重新编码，与库里已存的向量比余弦。同模型 → ≈1.00；不同模型 → 明显低于 1。
   本脚本 Phase A 就做这个判定，Phase B 用 `--reembed` 一键修（会先备份 DB）。

2. **两个阈值是按 DashScope 的余弦分布调的**。Qwen3-Embedding 的相似度整体更高、分布更压缩：
   - `AGENT_MEMORY_MIN_SCORE`（默认 0.25）太低 → 什么都算"相关"，召回一堆无关记忆污染 context；
   - `AGENT_MEMORY_DEDUP_COS`（默认 0.95）太高 → 近重复记忆不再合并，库里开始堆积。
   本脚本 Phase C 用你**真实的记忆库**算出分布，给出这两个值的建议窗口。

用法
----
    python check_memory_vectors.py                 # 只体检 + 校准，不动数据（推荐先跑）
    python check_memory_vectors.py --reembed       # 向量与当前模型不一致时，全量重编码（自动备份）
    python check_memory_vectors.py --probe "我用哪个工具对比过调用链提取？"
                                                   # 用一条真实 query 打印 top-10 命中及分数，
                                                   # 用来肉眼确认 MIN_SCORE 该切在哪里

安全约定
--------
- 默认**只读**，不写库；`--reembed` 才会写，且写之前把 DB 备份成 `<db>.bak-<时间戳>`。
- 不改任何表结构（不加列），零 schema 漂移风险。
- 依赖只有 httpx + numpy + python-dotenv（都在你现有环境里）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DB = os.environ.get("AGENT_MEMORY_DB") or os.path.join(_ROOT, "memory", "longterm.db")
_BASE = (os.environ.get("AGENT_MEMORY_EMBED_BASE_URL") or "").rstrip("/")
_KEY = os.environ.get("AGENT_MEMORY_EMBED_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
_MODEL = os.environ.get("AGENT_MEMORY_EMBED_MODEL") or "text-embedding-v4"
_DIM = int(os.environ.get("AGENT_MEMORY_EMBED_DIM", "0") or 0)
_TIMEOUT = float(os.environ.get("AGENT_MEMORY_EMBED_TIMEOUT", "30"))
_BATCH = int(os.environ.get("CALIB_BATCH", "8"))

FAILS: list[str] = []


def ok(cond: bool, label: str, detail: str = "") -> bool:
    print(f"  {'[OK]  ' if cond else '[FAIL]'} {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(label)
    return cond


# === 编码 ===
def embed(texts: list[str]) -> np.ndarray:
    """调用 OpenAI 兼容 /embeddings，返回 L2 归一化后的 (N, D) float32（归一化后点积即余弦）。"""
    payload: dict = {"model": _MODEL, "input": list(texts), "encoding_format": "float"}
    if _DIM:
        payload["dimensions"] = _DIM
    r = httpx.post(_BASE + "/embeddings",
                   headers={"Authorization": f"Bearer {_KEY}"},
                   json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
    if len(data) != len(texts):
        raise RuntimeError(f"条数不一致：请求 {len(texts)} 条，返回 {len(data)} 条")
    v = np.asarray([d["embedding"] for d in data], dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def pct(x: np.ndarray, ps=(1, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    return {p: float(np.percentile(x, p)) for p in ps}


# === 主流程 ===
def main() -> int:
    ap = argparse.ArgumentParser(description="长期记忆向量体检 + 阈值校准")
    ap.add_argument("--reembed", action="store_true", help="向量与当前模型不一致时全量重编码（先备份）")
    ap.add_argument("--probe", default="", help="用一条真实 query 打印 top-10 命中及分数")
    ap.add_argument("--sample", type=int, default=12, help="Phase A 抽样条数（默认 12）")
    args = ap.parse_args()

    print("=" * 72)
    print(f"DB      : {_DB}")
    print(f"endpoint: {_BASE}   model={_MODEL}   dim_req={_DIM or 'native'}"
          f"   key={'set' if _KEY else 'EMPTY!'}")
    print("=" * 72)

    if not _KEY:
        print("AGENT_MEMORY_EMBED_API_KEY 为空 → longterm._embed 会直接返回 None、静默回落关键词匹配。")
        print("哪怕本地服务也要显式填 dummy。")
        return 1
    if not os.path.exists(_DB):
        print(f"记忆库还不存在：{_DB}\n先正常跑一轮 agent 让它写入几条记忆，再回来校准。")
        return 0

    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, text, dim, embedding FROM memories").fetchall()
    with_emb = [r for r in rows if r["embedding"] is not None]
    print(f"\n[0] 记忆库概览：共 {len(rows)} 条，其中带向量 {len(with_emb)} 条，无向量 {len(rows) - len(with_emb)} 条")

    # ---------- Phase A：库里向量是否来自当前模型 ----------
    print("\n[A] 判定库里向量是否由当前模型生成（同模型自比应 ≈1.00）")
    sample = with_emb[: max(1, min(args.sample, len(with_emb)))] if with_emb else []
    if not sample:
        print("  没有带向量的记忆，跳过（先跑一轮 agent 写入）。")
        same_model = False
    else:
        try:
            fresh = embed([r["text"] for r in sample])
        except Exception as e:
            print(f"  编码失败：{type(e).__name__}: {e}")
            print("  排查：① 服务起了吗（curl /health）② base_url 里的 IP 对不对 ③ 防火墙 8100")
            return 1
        sims = np.array([cos(fresh[i], from_blob(sample[i]["embedding"])) for i in range(len(sample))])
        print(f"  抽样 {len(sample)} 条，自比余弦 min={sims.min():.4f}  median={np.median(sims):.4f}  max={sims.max():.4f}")
        print(f"  实测维度：库里 {int(sample[0]['dim'] or 0)}  新编码 {fresh.shape[1]}")
        same_model = bool(sims.min() >= 0.995)
        ok(same_model, "库里向量与当前模型一致",
           "同模型" if same_model else "不一致 → 跨模型混用，召回已是噪声，需要 --reembed")
        if not same_model:
            n_bad = int((sims < 0.995).sum())
            print(f"  其中 {n_bad}/{len(sample)} 条自比明显 <1 → 这些是旧模型（DashScope）留下的向量。")
            print("  注意：两个模型维度都是 1024，longterm.py 的维度护栏拦不住，不会报错、只会召回变烂。")

    # ---------- Phase B：按需全量重编码 ----------
    if args.reembed:
        if same_model:
            print("\n[B] 跳过：库里向量已是当前模型，无需重编码。")
        else:
            print(f"\n[B] 全量重编码 {len(with_emb)} 条（CPU 上 0.6B 约 0.4s/条，预计 {len(with_emb) * 0.4:.0f}s 起）")
            bak = f"{_DB}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            shutil.copy2(_DB, bak)
            ok(True, "已备份原始 DB", bak)
            done, failed = 0, 0
            for i in range(0, len(with_emb), _BATCH):
                chunk = with_emb[i:i + _BATCH]
                try:
                    vecs = embed([r["text"] for r in chunk])
                except Exception as e:
                    failed += len(chunk)
                    print(f"  [warn] 批次 {i // _BATCH} 编码失败：{type(e).__name__}: {e}")
                    continue
                for r, v in zip(chunk, vecs):
                    conn.execute("UPDATE memories SET embedding=?, dim=? WHERE id=?",
                                 (v.tobytes(), int(v.shape[0]), r["id"]))
                conn.commit()
                done += len(chunk)
                print(f"  ... {done}/{len(with_emb)}")
            ok(failed == 0, "重编码完成", f"成功 {done} 条，失败 {failed} 条")
            if failed:
                print(f"  失败的可从备份恢复：cp '{bak}' '{_DB}'")
            conn.close()
            conn = sqlite3.connect(_DB)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, text, dim, embedding FROM memories").fetchall()
            with_emb = [r for r in rows if r["embedding"] is not None]
            print("  重编码后再验一次自比（应全部 ≈1.00）：")
            if with_emb:
                s2 = embed([r["text"] for r in with_emb[: min(8, len(with_emb))]])
                ss = [round(cos(s2[i], from_blob(with_emb[i]["embedding"])), 4) for i in range(len(s2))]
                print(f"    {ss}")
                same_model = min(ss) >= 0.995
                ok(same_model, "重编码后库里向量已与当前模型一致")
                if same_model:      # 修好了就把 Phase A 的 FAIL 撤掉，否则脚本会"修好了仍报 fail"
                    FAILS[:] = [f for f in FAILS if f != "库里向量与当前模型一致"]
                    print("  → 向量空间混用已消除；Phase C 的校准结果现在基于同一模型，可信。")

    # ---------- Phase C：阈值校准 ----------
    print("\n[C] 阈值校准（基于你真实的记忆库分布）")
    if len(with_emb) < 5:
        print(f"  样本太少（{len(with_emb)} 条），先攒到 ≥10 条再校准。")
    else:
        mat = np.stack([from_blob(r["embedding"]) for r in with_emb])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.where(norms == 0.0, 1.0, norms)
        n = mat.shape[0]
        iu = np.triu_indices(n, k=1)
        sims = mat @ mat.T
        pair = sims[iu]
        p = pct(pair)
        print(f"  两两余弦（{len(pair)} 对）分布：")
        print("    " + "  ".join(f"p{k}={v:.3f}" for k, v in p.items()))
        print(f"    min={pair.min():.3f}  max={pair.max():.3f}  mean={pair.mean():.3f}")

        order = np.argsort(pair)[::-1]
        print("\n  最相似的 10 对（肉眼确认哪些是真·近重复）：")
        for idx in order[:10]:
            i, j = int(iu[0][idx]), int(iu[1][idx])
            ti = (with_emb[i]["text"] or "")[:46].replace("\n", " ")
            tj = (with_emb[j]["text"] or "")[:46].replace("\n", " ")
            print(f"    {pair[idx]:.3f}  [{ti}]  <->  [{tj}]")

        def clamp(v, lo, hi):
            v2 = min(max(v, lo), hi)
            return round(v2, 3), ("" if abs(v2 - v) < 1e-9 else f"  [原始测算 {v:.3f}，已收敛到安全区间]")

        # DEDUP_COS：在相似度最高的一段里找最大间隔，取间隔中点；间隔太小则退到 p99
        top = np.sort(pair)[::-1][:30]
        if len(top) >= 3:
            gaps = top[:-1] - top[1:]
            k = int(np.argmax(gaps))
            raw_dup = float((top[k] + top[k + 1]) / 2) if gaps[k] > 0.01 else float(np.percentile(pair, 99))
        else:
            raw_dup = float(np.percentile(pair, 99))
        dedup, dup_note = clamp(raw_dup, 0.90, 0.99)

        # MIN_SCORE：库里绝大多数对是"无关对"，所以 p50 是噪声地板、p75 是"明显更相关"的分界、
        # p90 已进入同话题簇内部。主推 p75，并给出 [p50 .. p90] 作为调参窗口。
        min_score, min_note = clamp(float(p[75]), 0.25, 0.75)
        lo_tune, _ = clamp(float(p[50]), 0.25, 0.75)
        hi_tune, _ = clamp(float(p[90]), 0.25, 0.80)

        cur_min = float(os.environ.get("AGENT_MEMORY_MIN_SCORE", "0.25"))
        cur_dup = float(os.environ.get("AGENT_MEMORY_DEDUP_COS", "0.95"))
        print("\n  建议值（写回 F:\\agent\\multi-agent\\.env）")
        print(f"    AGENT_MEMORY_MIN_SCORE  当前 {cur_min:<5} → 建议 {min_score}{min_note}")
        print(f"                            调参窗口 [{lo_tune} .. {hi_tune}]（=p50 噪声地板 .. p90 同话题簇内）")
        print(f"    AGENT_MEMORY_DEDUP_COS  当前 {cur_dup:<5} → 建议 {dedup}{dup_note}")

        print("    依据：库里绝大多数记忆对是无关的 → p50 是噪声地板（低于它基本肯定不相干），"
              "p75 是「明显更相关」的分界，\n"
              "          p90 已经进到同话题簇内部（用它当阈值会把同话题的不同事实也砍掉）。"
              "DEDUP_COS 取最高尾部的最大间隔中点。")
        if cur_min < lo_tune - 0.02:
            print(f"  ⚠ 现在的 MIN_SCORE={cur_min} 低于噪声地板：库里约 "
                  f"{100 * (pair < cur_min).mean():.0f}% 的片段对都会通过阈值，召回会夹带无关记忆。")
        if cur_dup > dedup + 0.02:
            print(f"  ⚠ 现在的 DEDUP_COS={cur_dup} 高于建议值：约 "
                  f"{100 * (pair > dedup).mean():.1f}% 的片段对落在合并区间外，近重复记忆会开始堆积。")
        print("  最后一步务必用真实 query 校准：python check_memory_vectors.py --probe \"<你的问题>\"")

    # ---------- 可选：真实 query 探查 ----------
    if args.probe and with_emb:
        print(f"\n[D] query 探查：{args.probe!r}")
        qv = np.asarray(embed([args.probe])[0], dtype=np.float32)
        scored = sorted(((cos(qv, from_blob(r["embedding"])), r) for r in with_emb),
                        key=lambda x: x[0], reverse=True)
        cur_min = float(os.environ.get("AGENT_MEMORY_MIN_SCORE", "0.25"))
        for i, (s, r) in enumerate(scored[:10], 1):
            mark = "命中" if s >= cur_min else "被 MIN_SCORE 挡住"
            print(f"    {i:2d}. {s:.3f}  {mark}  [{(r['text'] or '')[:56]}]")
        print(f"    （当前 MIN_SCORE={cur_min}；看上面分数在哪一档开始明显掉下去，阈值就切在那儿）")

    conn.close()
    print("\n" + "=" * 72)
    if FAILS:
        print("未通过：" + "; ".join(FAILS))
        if not same_model and not args.reembed and with_emb:
            print("→ 跑 `python check_memory_vectors.py --reembed` 修复向量空间混用（会先备份 DB）")
        return 1
    print("通过：记忆库向量与阈值均可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
