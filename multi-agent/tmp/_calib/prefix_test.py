"""测 Qwen3-Embedding 的 query 检索指令前缀能否拉开「相关 / 噪声」的余弦间隔（多查询版）。

背景：MIN_SCORE 在 0.25~0.70 之间扫不出干净切点 —— 33 条记忆库上噪声最高 0.69，
与相关条目 0.66~0.83 交错。怀疑根因是 query 侧没有用 Qwen3 官方的非对称检索指令。

passage 一律不加前缀（与库里存量一致），只改 query 侧编码形式。
用 3 个 query 分别测，避免结论过拟合到单个 query。
指标：
  margin = min(相关分) - max(噪声分)   → >0 才存在干净阈值切点
  noise_max                            → 阈值天花板；越低，记忆库变大时越稳
"""
from __future__ import annotations

import httpx
import numpy as np

BASE = "http://192.168.50.81:8100/v1"
H = {"Authorization": "Bearer dummy"}
MODEL = "Qwen3-Embedding-0.6B"

RELEVANT = [
    ("评估 joern 与 tree-sitter 两种调用链提取方案在 nacos 上的召回率", 0),
    ("GuardFox 的 sink 驱动扫描基线是 OWASP BenchmarkJava", 1),
    ("CWE-89 检测范围从 big-4 SQL sinks 扩展到任意 raw SQL string 的 DB 抽象层", 2),
]
NOISE = [
    "红烧肉要先把五花肉焯水再炒糖色", "今天下午有阵雨，出门记得带伞",
    "下周去成都的机票改签到周五晚上", "猫砂盆该换了，顺便买点罐头",
    "会议室预约系统升级到 v3，周五晚上停机", "新版报销单需要附上电子发票 PDF",
    "咖啡机除垢剂放在茶水间第二个柜子", "季度团建定在十月中旬，地点待定",
    "打印机墨盒型号是 CF258A", "楼下便利店八点后生鲜打折",
    "周三下午三点和客户开需求评审会", "git 仓库的 master 分支改名成 main 了",
    "饮水机滤芯每三个月换一次", "出差报销标准按城市等级分三档",
    "新同事的工位在靠窗第三排", "电梯年检通知贴在公告栏",
    "年会抽奖的一等奖是扫地机器人", "快递柜取件码每天凌晨刷新",
    "食堂周三供应饺子，排队很久", "公司停车位需要提前一天在小程序上预约",
    "防火墙策略变更需要走变更评审流程", "生产环境发版窗口定在每周二凌晨",
    "监控告警阈值调整要同步给值班同学", "办公区空调温度统一设定在 26 度",
    "打印机局域网 IP 段改成 10.20 开头了", "体检报告出来了，各项指标都正常",
    "投影仪遥控器电池该换了", "绿萝浇水频率改成一周两次",
    "楼下健身房新开了一家，年卡有折扣", "周末想去看场电影，还没定看什么",
]

# (query, 应命中的 relevant 下标)
QUERIES = [
    ("我用哪个工具对比过调用链提取？", 0),
    ("扫描器的对照基线靶场是什么？", 1),
    ("SQL 注入检测覆盖了哪些数据库访问层？", 2),
]

VARIANTS = {
    "① 无前缀": "",
    "② 英文官方": "Instruct: Given a web search query, retrieve relevant passages that answer the query",
    "③ 中文通用": "Instruct: 给定一个检索查询，召回能够回答该查询的段落",
    "④ 中文安全域": "Instruct: 给定一个关于代码安全分析的问题，召回能够回答该问题的技术记录",
    "⑤ 英文通用(短)": "Instruct: Retrieve passages that answer the query",
    "⑥ 中文记忆域": "Instruct: 给定一个关于用户偏好或历史决策的问题，召回能够回答该问题的记忆条目",
}


def embed(texts: list[str]) -> np.ndarray:
    r = httpx.post(f"{BASE}/embeddings", headers=H,
                   json={"model": MODEL, "input": texts, "encoding_format": "float"},
                   timeout=180, trust_env=False)
    r.raise_for_status()
    d = sorted(r.json()["data"], key=lambda x: x["index"])
    return np.asarray([x["embedding"] for x in d], dtype=np.float32)


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


pv = embed([t for t, _ in RELEVANT] + NOISE)
pr, pn = pv[:len(RELEVANT)], pv[len(RELEVANT):]

print(f"{'query 编码方式':<18} {'噪声max':>8} {'相关min':>8} {'间隔':>7} {'命中':>6}  逐 query 间隔")
print("-" * 96)
summary = []
for name, prefix in VARIANTS.items():
    nmax_all, rmin_all, margins, hits = [], [], [], 0
    per = []
    for q, want in QUERIES:
        qv = embed([f"{prefix}\nQuery: {q}" if prefix else q])[0]
        rs = [cos(qv, v) for v in pr]
        ns = [cos(qv, v) for v in pn]
        nmax_all.append(max(ns))
        rmin_all.append(min(rs))
        m = min(rs) - max(ns)
        margins.append(m)
        per.append(f"{m:+.3f}")
        topk = sorted([(cos(qv, v), i) for i, v in enumerate(pr)] +
                      [(cos(qv, v), -1) for v in pn], reverse=True)[:5]
        if want in [i for _, i in topk]:
            hits += 1
    summary.append((name, np.mean(nmax_all), np.mean(rmin_all), np.mean(margins), hits))
    print(f"{name:<18} {np.mean(nmax_all):>8.3f} {np.mean(rmin_all):>8.3f} "
          f"{np.mean(margins):>+7.3f} {hits:>4}/3  {' '.join(per)}")

print("\n排序（先看命中数，再看平均间隔）：")
for name, nm, rm, mg, h in sorted(summary, key=lambda x: (-x[4], -x[3])):
    print(f"  {h}/3  margin={mg:+.3f}  noise_max={nm:.3f}  rel_min={rm:.3f}   {name}")
