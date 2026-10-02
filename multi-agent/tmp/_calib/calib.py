"""用真实 embedding 服务测量余弦分布，标定 AGENT_MEMORY_MIN_SCORE。

触发原因：check_embed_service.py 里 query「我用哪个工具对比过调用链提取？」召回出 3 条，
分数 0.83 / 0.73 / 0.69 —— 但其中「GuardFox 的 sink 驱动扫描基线是 OWASP BenchmarkJava」
与 query 主题并不相同，也拿到 0.69。说明 Qwen3-Embedding 的余弦分布**整体偏高、被压缩**，
MIN_SCORE=0.25（DashScope 时代取值）形同虚设，会把大量无关记忆灌进 context。

本脚本同时验三件工程上必须成立的事：
  D1 确定性：同一文本两次编码必须完全一致（cos=1.0），否则库里向量与查询向量不可比
  D2 批一致性：encode([t]) 与 encode([t, x, y]) 里的 t 必须一致（padding 不该改变结果）
  D3 归一化：服务端返回的向量 L2 范数必须 ≈1（否则余弦排序会偏）
"""
from __future__ import annotations

import itertools
import statistics as st

import httpx
import numpy as np

BASE = "http://192.168.50.81:8100/v1"
KEY = "dummy"
MODEL = "Qwen3-Embedding-0.6B"
H = {"Authorization": f"Bearer {KEY}"}

# 同话题（A 组：都在讲 guardfox / 静态分析 / 调用图） vs 明显无关（B 组）
A = [
    "评估 joern 与 tree-sitter 两种调用链提取方案在 nacos 上的召回率",
    "GuardFox 的 sink 驱动扫描基线是 OWASP BenchmarkJava",
    "CWE-89 检测范围从 big-4 SQL sinks 扩展到任意 raw SQL string 的 DB 抽象层",
    "SourceCallGraphExtractor 用三通道回退保证库/JDK sink 的 caller 覆盖度",
]
B = [
    "红烧肉要先把五花肉焯水再炒糖色",
    "今天下午有阵雨，出门记得带伞",
    "下周去成都的机票改签到周五晚上",
    "猫砂盆该换了，顺便买点罐头",
]
QUERY = "我用哪个工具对比过调用链提取？"


def embed(texts: list[str]) -> np.ndarray:
    r = httpx.post(f"{BASE}/embeddings", headers=H,
                   json={"model": MODEL, "input": texts, "encoding_format": "float"},
                   timeout=60, trust_env=False)
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return np.asarray([d["embedding"] for d in data], dtype=np.float32)


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stats(name: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {name}: (无)")
        return
    print(f"  {name}: min={min(vals):.3f}  p25={np.percentile(vals, 25):.3f}  "
          f"中位={st.median(vals):.3f}  p75={np.percentile(vals, 75):.3f}  max={max(vals):.3f}")


print("=" * 74)
va, vb, vq = embed(A), embed(B), embed([QUERY])[0]

# --- D1/D2/D3 ---
again = embed([A[0]])[0]
mixed = embed([A[0], "猫砂盆该换了"])[0]
print(f"[D1] 确定性   同一文本两次编码 cos = {cos(va[0], again):.6f}   "
      f"{'OK' if cos(va[0], again) > 0.9999 else '异常!'}")
print(f"[D2] 批一致性 encode([t]) vs encode([t,x]) 里的 t cos = {cos(va[0], mixed):.6f}   "
      f"{'OK' if cos(va[0], mixed) > 0.9999 else '异常!'}")
norms = np.linalg.norm(np.vstack([va, vb, vq[None, :]]), axis=1)
print(f"[D3] L2 归一化 范数 min={norms.min():.4f} max={norms.max():.4f}   "
      f"{'OK' if abs(norms.max() - 1) < 1e-3 and abs(norms.min() - 1) < 1e-3 else '异常!'}")

# --- 分布 ---
intra_a = [cos(va[i], va[j]) for i, j in itertools.combinations(range(len(A)), 2)]
intra_b = [cos(vb[i], vb[j]) for i, j in itertools.combinations(range(len(B)), 2)]
cross = [cos(va[i], vb[j]) for i in range(len(A)) for j in range(len(B))]
q_a = [cos(vq, v) for v in va]
q_b = [cos(vq, v) for v in vb]

print("\n[分布] 两两余弦")
stats("A↔A（同话题）", intra_a)
stats("B↔B（无关话题）", intra_b)
stats("A↔B（跨话题，噪声底）", cross)
stats("QUERY↔A（该召回）", q_a)
stats("QUERY↔B（不该召回）", q_b)

noise_p95 = float(np.percentile(cross, 95))
noise_max = float(max(cross))
print(f"\n关键分离点：跨话题 max = {noise_max:.3f}   同话题 min = {min(intra_a):.3f}")
print(f"           query↔无关 max = {max(q_b):.3f}   query↔相关 min = {min(q_a):.3f}")

sugg = round(min(max(max(q_b) + 0.03, noise_max + 0.02), 0.85), 2)
print(f"\n建议 AGENT_MEMORY_MIN_SCORE ≈ {sugg}   "
      f"（= 取「query 与无关记忆的最高分」再上浮 0.03；当前 0.25 会把 A/B 全部召回）")
print(f"参考：DEDUP_COS 建议 ≈ {round(min(0.99, max(0.95, max(intra_a) + 0.01)), 3)}"
      f"（同话题最高分之上，避免把同话题的不同事实当重复合并掉）")
