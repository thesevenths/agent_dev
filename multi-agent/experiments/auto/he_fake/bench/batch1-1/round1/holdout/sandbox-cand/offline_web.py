"""离线检索层：零网络、确定性的「假联网」。

为什么必须存在
--------------
``multi-agent`` 的三个工具会真的联网：

- ``resilient_tavily_search``  → Tavily API（**有免费额度，超了付费**）
- ``get_nasdaq_top_gainers``   → Yahoo Finance
- ``get_crypto_sentiment_indicators`` → 外部行情 API

而自适应闭环要在同一批任务上**重复跑几十上百次**。这带来两个不可接受后果：

1. 额度：多轮 benchmark = 每轮几十次检索，直接烧穿 Tavily 免费额度；
2. **噪声**（更致命）：同一个 query 今天和明天返回不同内容，报告随之变化，
   于是配对差值的 sd(Δ) 被「外部世界的不稳定性」抬高 —— δ 随之变大，
   真实的 harness 改进反而检不出来。闭环的判定器因此失去分辨力。

本模块把联网工具换成**本地固定语料**上的确定性检索：不联网、不受额度限制，
而且同一 query 永远返回同一结果 —— 降噪与省额度是同一件事。

诚实的边界
----------
离线语料里的数字是**静态参考资料**，不是实时数据。因此离线任务集必须只以本地
CSV 为依据出题（见 ``adaptive_runtime/offline_tasks.py``），不能要求「最新股价」。
URL 统一是 ``https://offline.local/<doc_id>``：它满足评分器对「来源」的形式要求，
且可在 ``offline_corpus_index()`` 里逐条回查到本模块的语料正文 —— 不是伪造外链。
"""
from __future__ import annotations

import os
import re
from typing import Any

#: 进程内调用计数：控制平面据此断言「这一轮确实没联网」
CALLS: dict[str, int] = {"tavily": 0, "nasdaq": 0, "crypto": 0}

OFFLINE_KEYS = ("ADAPTIVE_OFFLINE", "AGENT_OFFLINE")


def is_offline() -> bool:
    """离线开关。显式 0/false 表示「不要离线」，未设置表示沿用真实行为。"""
    for k in OFFLINE_KEYS:
        raw = os.environ.get(k)
        if raw is not None and raw.strip() != "":
            return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def reset_calls() -> None:
    for k in CALLS:
        CALLS[k] = 0


# ---------------------------------------------------------------- 固定语料
#: doc_id -> (title, tags, body)。正文里的数值是静态参考口径，用于离线引用。
_CORPUS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "method-correlation": (
        "Pearson 相关系数的适用边界与解读口径",
        ("相关性", "pearson", "corr", "系数", "统计", "相关"),
        "Pearson r 只度量线性关系，|r|<0.1 视为无关，0.1-0.3 弱相关，0.3-0.5 中等相关，"
        "0.5-0.7 强相关，>0.7 极强相关。样本量 n≥30 时 |r|≥0.36 才在 0.05 水平显著；"
        "n=100 时显著门槛约 |r|≥0.20；n=400 时约 |r|≥0.10。相关不等于因果：混杂变量"
        "（如家庭收入同时影响课外辅导投入与学业成绩）会制造虚假相关。异常值对 r 的"
        "影响可达 0.15 以上，报告 r 时应同时给出剔除极端值前后的两个值。",
    ),
    "method-outlier": (
        "离群点判定：IQR 法与 Z-score 法的取舍",
        ("异常", "离群", "outlier", "iqr", "zscore", "z-score", "极值", "统计"),
        "IQR 法：Q1-1.5*IQR 与 Q3+1.5*IQR 之外判为离群，对偏态分布稳健；"
        "3*IQR 之外判为极端离群。Z-score 法：|z|>2 约对应 4.6% 的双侧比例，|z|>3 对应 0.3%，"
        "但假设近正态，偏态时会高估离群比例。稳健替代是用中位数与 MAD（\u03c3≈1.4826*MAD）。"
        "报告离群点必须给出：判定规则、被标记数量、占总样本比例、以及这些样本的关键特征，"
        "否则「异常」只是形容词。常见错误是先把异常值删掉再算均值，导致结论偏乐观。",
    ),
    "method-groupmean": (
        "分组均值比较：样本量、方差与最小可检差异",
        ("分组", "均值", "对比", "group", "mean", "方差", "显著", "t检验"),
        "分组比较必须同时报告 n、均值、标准差，否则均值差无法判读。经验门槛：两组均值差"
        "大于 0.2 倍合并标准差才算小效应、0.5 倍算中等、0.8 倍算大效应（Cohen's d）。"
        "每组样本低于 10 时均值抖动大，建议先合并类别再比较。多组比较不做校正会把第一类"
        "错误率从 5% 抬到 20% 以上（10 组两两比较共 45 次）。报告要区分「统计显著」与"
        "「业务重要」：n=400 时 1 分的差异也可能显著，但没有决策价值。",
    ),
    "method-retention": (
        "留存与流失指标的定义口径",
        ("留存", "流失", "retention", "churn", "cohort", "d7", "arpu", "指标"),
        "D7 留存 = 首日活跃用户中第 7 天仍活跃的比例；分母必须固定为同期新增（cohort），"
        "混入老用户会虚高 3-8 个百分点。流失率与留存率不是互补关系：D7 留存 40% 对应的是"
        "「第 7 天仍活跃」，而 30 天流失通常取连续 30 天无活跃，区间留存 55%-65% 属正常。"
        "ARPU 对留存高度敏感：留存每提高 1 个百分点，成熟产品 ARPU 通常提升 2%-4%。"
        "客服工单数与流失正相关，但用绝对量比较不同 cohort 会误导，应按每千活跃用户计。"
        "检测到断点时应定位到具体自然周并关联当期发版记录，只看总均值会掩盖单周回归。",
    ),
    "method-revenue": (
        "营收、退货率与季节性调整口径",
        ("营收", "退货", "销售", "revenue", "季节", "同比", "环比", "增长"),
        "净营收 = 营收 - 退货金额；退货率 = 退货量 / 销量。跨期比较必须区分同比（YoY）与"
        "环比（MoM）：含 11-12 月的序列里，环比在旺季结束时会给出 -15% 左右的假衰退信号，"
        "而同比能保留真实趋势。季节性显著时（旺季销量为淡季 1.15-1.25 倍）应先做同比"
        "再谈增长。单一区域单月放量若伴随退货率从 3% 跳到 15% 以上，优先怀疑渠道压货或"
        "质量缺陷，而不是把它记作需求增长。金额单位要在表头统一标注，避免美元/万元混用。",
    ),
    "method-sourcing": (
        "量化断言的引用与可核查性规范",
        ("引用", "来源", "出处", "source", "报告", "规范", "引用"),
        "每个关键数字都必须就地标注来源，来源形式为：URL、Markdown 链接、脚注，或"
        "「（来源：X）」/「(Source: X)」标记。本地数据计算的结论应写成"
        "「（来源：由 offline_data/xxx.csv 第 N 行起算）」并给出聚合口径（行数、过滤条件、"
        "列名）。同一结论出现在多处时，来源必须一致。表格下方要有口径说明：数据窗口、"
        "样本量、单位、以及是否为计算值。缺少来源的报告即使数字正确也不可用，"
        "因为读者无法复核。",
    ),
    "method-chart": (
        "图表与箱线图判读要点",
        ("图", "表", "箱线", "分布", "chart", "plot", "可视化", "报告"),
        "箱线图判读：箱体是 Q1-Q3，中线是中位数，须与均值并列报告，偏态时两者差距可达 10%。"
        "直方图分箱数建议取 Sturges 公式（k=1+log2 n）或 12-20 箱。相关热力图对多重共线性"
        "敏感，|r|>0.85 的变量对应剔除其一。所有图表必须有标题、坐标轴单位与数据来源标注；"
        "matplotlib 在无界面环境必须使用 Agg 后端并保存为文件而非展示。",
    ),
    "ref-benchmarks": (
        "常用业务基准值（静态参考，非实时）",
        ("基准", "benchmark", "行业", "参考", "标准", "指标", "平均水平"),
        "以下为静态参考基准，用于判断量级是否合理，不代表当期实际值："
        "教育类产品 D7 留存基准 35%-45%；工具类 25%-35%；内容类 40%-55%。"
        "硬件退货率基准 2%-4%，配件 5%-8%，服务类 <1.5%。"
        "学业测验成绩的分层基准：及格线 60，良好 75，优秀 85；"
        "周学习时长与成绩的经验弹性约为每小时 +4 至 +6 分（线性区间内）。"
        "客服工单密度基准：每千活跃用户 8-20 单/周，超过 30 单提示版本质量或文档缺陷。",
    ),
}

_TOKEN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{2,8}")


def _tokens(q: str) -> list[str]:
    """中文按连续串（2-8 字）粗切，英文数字按词切。够用且确定性 —— 不引入分词依赖。"""
    out: list[str] = []
    for m in _TOKEN.finditer((q or "").lower()):
        t = m.group(0)
        if len(t) >= 2:
            out.append(t)
    return out


def _score(query_tokens: list[str], doc_id: str, title: str, tags: tuple[str, ...],
           body: str) -> int:
    hay_body = body.lower()
    hay_title = title.lower()
    s = 0
    for t in set(query_tokens):
        if t in (x.lower() for x in tags):
            s += 5
        if t in hay_title:
            s += 3
        s += min(hay_body.count(t), 3)
    return s


def search(query: str, max_results: int = 5) -> str:
    """离线版 web search：对固定语料做确定性关键词打分，返回 top-k。"""
    CALLS["tavily"] += 1
    toks = _tokens(query)
    ranked = sorted(
        (( _score(toks, d, ti, tg, bo), d, ti, bo) for d, (ti, tg, bo) in _CORPUS.items()),
        key=lambda x: (-x[0], x[1]),          # 同分按 doc_id 定序，保证确定性
    )
    hits = [r for r in ranked if r[0] > 0][:max_results]
    if not hits:
        # 无匹配时给出索引清单而不是空串：让 agent 知道该改关键词或直接用本地数据，
        # 而不是反复重试（重试在本工具里既不烧额度也不烧预算，但会白耗 ReAct 轮次）。
        idx = "\n".join(f"- {d}: {ti}" for d, (ti, _, _) in sorted(_CORPUS.items()))
        return (f"[offline corpus] 未匹配到与 {query!r} 相关的资料。可用资料索引：\n{idx}\n"
                "[system] 本次运行处于离线模式，联网检索被禁用。请改为基于任务指定的本地文件"
                "（如 offline_data/*.csv）作答，不要重复检索。")
    blocks = []
    for _, doc_id, title, body in hits:
        blocks.append(f"[{len(blocks) + 1}] {title}\n"
                      f"URL: https://offline.local/{doc_id}\n"
                      f"Content: {body}")
    return (f"A web search for {query!r} found {len(blocks)} relevant results "
            f"(offline corpus, static reference material):\n\n" + "\n\n".join(blocks))


def offline_corpus_index() -> list[dict]:
    """供审计：每个离线 URL 对应哪段正文。"""
    return [{"doc_id": d, "title": ti, "url": f"https://offline.local/{d}",
              "chars": len(bo)} for d, (ti, _, bo) in sorted(_CORPUS.items())]


# ---------------------------------------------------------------- 行情类工具

#: 纳斯达克涨幅榜的固定快照（结构与真实返回一致：symbol/name/change_pct）
_NASDAQ_FIXTURE = [
    {"symbol": "OFFL", "name": "Offline Fixture Tech A", "change_pct": 12.41},
    {"symbol": "DETMR", "name": "Deterministic Semis", "change_pct": 9.83},
    {"symbol": "STATC", "name": "Static Cloud Holdings", "change_pct": 7.25},
    {"symbol": "RPRB", "name": "Reproducible Robotics", "change_pct": 5.62},
    {"symbol": "FIXD", "name": "Fixture Data Systems", "change_pct": 4.18},
]


def nasdaq_top_gainers(top_n: int = 5) -> list[dict]:
    CALLS["nasdaq"] += 1
    return [dict(x) for x in _NASDAQ_FIXTURE[:max(1, int(top_n))]]


def crypto_sentiment() -> Any:
    """返回 None —— 与「网络请求失败」的既有降级路径完全一致，不新增分支语义。"""
    CALLS["crypto"] += 1
    return None
