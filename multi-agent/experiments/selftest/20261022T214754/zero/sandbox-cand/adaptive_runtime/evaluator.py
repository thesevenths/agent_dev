"""评价层：L1 确定性合成分 Q = 0.5A + 0.3E + 0.2C（主目标）+ 可选 L2 rubric（次要证据）。

为什么主目标是确定性的
----------------------
研报类任务没有 ground truth，但「过程是否合规、产物是否完整、结论是否有来源」是事实。
把这些事实做成 Q，sd(Δ) 可以压到 2–4；用裸 LLM 打分则 sd≈12，需要 71 对样本才能检出
+4 分（见 stats.required_n），成本上不可行。

A（contract_rate）来自 trace：通过融合 gate 的 step / 计划 step。
E（evidence_rate）来自成品文本：带来源的「关键数字/结论」占比。
C（completeness）来自 trace 与产物：截断、兜底、计划产物缺失。

缺失分量的处理
--------------
- live 模式：产物缺失 → C=0（不是「不可用」），因为一次真实 probe 应当有产物。
- replay 模式 + artifacts_unavailable（历史日志 mtime 被拷动过）→ C/E 不参与，
  权重按剩余分量归一化，并在 score 里标记 renormalized，绝不假装算过。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .io_util import read_text, sha256_text, write_json
from .settings import CFG, Settings
from .trace_export import Trace

#: 判分口径版本。任何改变 Q 语义的改动（权重、证据口径、完整性公式）都必须让它变化，
#: 否则历史经验会被错配到新尺子上 —— 「同一 Q 却含义不同」是最隐蔽的数据污染。
#:
#: v2：加入反刷分两项（报告最小体积 + 结构系数），并把 planned 步数写进 detail。
#: 旧分数的 C 会普遍偏高，不能与新分数混用。
_RUBRIC_TAG = ("evidence:paragraph-v1|"
               "source:url+mdlink+footnote+marker|"
               "completeness:1-(trunc+salv+missing)/(planned+1)|"
               "A:contract_rate|no_report_live:E=0|"
               "antigaming:min_bytes+structure(0.5)+thin_deliverable")


def judge_version(cfg: Settings | None = None) -> str:
    cfg = cfg or CFG
    wA, wE, wC = cfg.normalized_weights()
    return sha256_text(f"{_RUBRIC_TAG}|w={wA:.4f},{wE:.4f},{wC:.4f}|"
                       f"min_points={cfg.regression_min_points}|"
                       f"min_bytes={cfg.min_report_bytes}|share={cfg.structure_share:.3f}")[:12]


# ---------------------------------------------------------------- 提取规则

_URL = re.compile(r"https?://[^\s\)\]\"'<>]+")
_MD_LINK = re.compile(r"\[[^\]]{1,60}\]\((?:https?://|\.{0,2}/)[^)]+\)")
_FOOTNOTE = re.compile(r"\[\^?\d{1,3}\]")
_SOURCE_MARKERS = ("来源", "数据来源", "资料来源", "引自", "出处", "参考：", "参考:", "Source:", "source:")
#: 一个「关键数字」：数字 + 可选单位/百分比/金额单位
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|％|个百分点|bp|倍|亿美元|亿美元|万美元|亿美元|亿美元|亿元|万元|美元|元|亿|万|"
    r"吨|桶|股|人|家|条|个|次|天|年|月|日|小时|分钟)?")

#: 计划产物类步骤的动词（与 agents._is_deliverable_step 语义一致，独立副本）
DELIVERABLE_VERBS = ("撰写", "生成", "绘制", "作图", "画图", "报告", "图表", "导出",
                     "保存", "输出", "整理成", "发送", "邮件")

_MAX_REPORT_BYTES = 400_000
_MAX_CLAIMS_PER_PARA = 8


# ---------------------------------------------------------------- 结果

@dataclass
class Score:
    run_id: str
    task_id: str = ""
    arm: str = ""
    A: float = 0.0
    E: float | None = None
    C: float | None = None
    Q: float = 0.0
    components: dict = field(default_factory=dict)
    weights_used: dict = field(default_factory=dict)
    #: 计划步数（A 的分母）。单独提出来是为了让 comparator 能检测「靠削减计划提分」。
    planned: int = 0
    flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    l2: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- L1 分量

def evidence_rate(text: str) -> tuple[float, dict]:
    """带来源的关键数字/结论占比。

    规则（确定性，不用 LLM）：
      1. 按空行切段；
      2. 每段抽出「关键数字」（含货币/百分比/数量单位），单段最多计 8 个；
      3. 段内出现 URL / markdown 链接 / 脚注 / 来源标记 → 该段全部数字算「有来源」。
    这是保守估计：宁可低估支持率，也不把「堆砌来源」当质量。
    """
    if not text.strip():
        return 0.0, {"claims": 0, "supported": 0, "paragraphs": 0}
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    claims = supported = 0
    unsupported_samples: list[str] = []
    for p in paras:
        nums = _NUMBER.findall(p)
        if not nums:
            continue
        n = min(len(nums), _MAX_CLAIMS_PER_PARA)
        claims += n
        if _has_source(p):
            supported += n
        elif len(unsupported_samples) < 3:
            unsupported_samples.append(p[:80])
    rate = supported / claims if claims else 0.0
    return rate, {"claims": claims, "supported": supported,
                  "paragraphs": len(paras), "unsupported_samples": unsupported_samples}


def _has_source(paragraph: str) -> bool:
    if _URL.search(paragraph) or _MD_LINK.search(paragraph) or _FOOTNOTE.search(paragraph):
        return True
    low = paragraph.lower()
    return any(marker.lower() in low for marker in _SOURCE_MARKERS)


def planned_artifacts(trace: Trace) -> int:
    """计划里承诺的产物数量（用动词表判断，而不是看它实际产出了什么）。"""
    n = 0
    for p in trace.plan:
        title = p.get("title", "") or ""
        if any(v in title for v in DELIVERABLE_VERBS):
            n += 1
    return n


#: 产物结构检查（确定性，任务无关）。四项各占 1/4。
#: 为什么需要它：``C`` 的分母是「计划产物数」，于是「只规划 1 步并让它通过」能拿 C=1。
#: 结构分与计划无关，用「这份东西看起来像不像一份可交付报告」来兜住底线。
_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_TABLE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S", re.MULTILINE)
_CONCLUSION_MARKERS = ("结论", "总结", "建议", "风险", "展望", "小结",
                       "conclusion", "summary", "recommendation", "outlook")


def structure_score(text: str, min_bytes: int = 1200) -> tuple[float, dict]:
    """报告结构完备度 ∈ [0,1]。刻意不看「写得好不好」，只看有没有骨架。"""
    n = len(text.encode("utf-8", errors="replace")) if text else 0
    parts = {
        "size": n >= min_bytes,
        "headings": len(_HEADING.findall(text or "")) >= 3,
        "table_or_list": bool(_TABLE.search(text or "") or _BULLET.search(text or "")),
        "conclusion": any(m in (text or "")[-4000:].lower() for m in _CONCLUSION_MARKERS),
    }
    return sum(1 for v in parts.values() if v) / 4.0, {"bytes": n, **parts}


def completeness(trace: Trace, include: bool = True,
                 report_text: str | None = None,
                 cfg: Settings | None = None) -> tuple[float | None, dict]:
    """1 − (截断段 + 兜底段 + 缺失产物) / (计划产物 + 1)，再乘结构系数。"""
    if not include:
        return None, {"available": False}
    cfg = cfg or CFG
    planned = max(1, planned_artifacts(trace))
    deliv = len(trace.deliverables())
    missing = max(0, planned - deliv)
    trunc = sum(1 for s in trace.steps.values() if s.fail_kind == "truncated")
    salv = 1 if trace.salvaged else 0
    bad = trunc + salv + missing
    c0 = max(0.0, min(1.0, 1.0 - bad / (planned + 1)))
    detail = {"planned": planned, "deliverables": deliv, "missing": missing,
              "truncated_steps": trunc, "salvaged": salv, "available": True,
              "base_c": round(c0, 4)}

    # 结构系数：没有可读报告时给最低档，而不是跳过 ——
    # 跳过等于「没产物不扣分」，正好是最容易被利用的方向。
    s, sdetail = structure_score(report_text or "", cfg.min_report_bytes)
    share = cfg.structure_share
    c = c0 * ((1.0 - share) + share * s)
    detail["structure"] = sdetail
    detail["structure_score"] = round(s, 4)
    return max(0.0, min(1.0, c)), detail


def pick_report(trace: Trace, root: str | Path,
                min_bytes: int = 0) -> Path | None:
    """挑本次 run 的主报告：.md/.txt 里体积最大的可交付产物。

    ``min_bytes`` > 0 时，小于该体积的产物**不算报告**（返回 None）。
    这不是「差不多就行」的宽松判断：一句闲聊存成 .md 也能被挑中，
    于是「什么都没做」拿到了 E=0 但 C=1 的 70 分。
    """
    cands = [a for a in trace.deliverables() if a["ext"] in (".md", ".txt")]
    if min_bytes > 0:
        cands = [a for a in cands if int(a.get("bytes") or 0) >= min_bytes]
    if not cands:
        return None
    cands.sort(key=lambda a: (a["bytes"], 0 if "报告" in a["path"] or "report" in a["path"].lower() else 1),
               reverse=True)
    p = Path(root) / cands[0]["path"]
    return p if p.exists() else None


# ---------------------------------------------------------------- 主入口

def score_trace(trace: Trace, root: str | Path | None = None,
                mode: str = "live", cfg: Settings | None = None,
                task_id: str = "", arm: str = "") -> Score:
    cfg = cfg or CFG
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    A = max(0.0, min(1.0, trace.contract_rate))
    flags: list[str] = []
    detail: dict = {}

    replay_gap = bool(trace.result.get("artifacts_unavailable")) and mode != "live"
    report = pick_report(trace, root, min_bytes=cfg.min_report_bytes) if not replay_gap else None
    report_text: str | None = None
    if report is not None:
        report_text = read_text(report)[:_MAX_REPORT_BYTES]
        E, edetail = evidence_rate(report_text)
        detail["evidence"] = edetail
        detail["report"] = str(report)
    elif replay_gap:
        E = None
        flags.append("artifacts_unavailable_replay")
        detail["evidence"] = {"available": False}
    else:
        # live 模式下「没有报告」是失败信号，不是数据缺失：必须记 0，不能重新归一化权重，
        # 否则一个没产出报告的候选会因为权重被重分配而显得没那么差。
        E = 0.0
        flags.append("no_report")
        detail["evidence"] = {"available": False, "claims": 0, "supported": 0}
        if trace.deliverables():
            # 有产物但都不够格当报告 → 单独标出来，避免与「完全没产物」混为一谈
            flags.append("thin_deliverable")

    C, cdetail = completeness(trace, include=not replay_gap,
                             report_text=report_text, cfg=cfg)
    detail["completeness"] = cdetail
    # 计划步数：gate 用它检测「靠削减计划把 A 抬上去」的候选
    detail["planned_steps"] = len(trace.plan)
    detail["total_steps"] = trace.total_steps
    if replay_gap:
        flags.append("artifacts_unavailable_replay")
    if trace.truncated:
        flags.append("truncated")
    if trace.salvaged:
        flags.append("salvaged")
    if not trace.result.get("finalized"):
        flags.append("not_finalized")
    if not any(e.get("type") == "gate" for e in trace.events):
        # 没有任何 gate 证据时 A 必然为 0；这不是「做得差」，而是「判不了」，必须显式暴露
        flags.append("no_gate_evidence")
    if trace.unknown_events > 0 and trace.total_steps == 0:
        flags.append("unparsed_trace")

    wA, wE, wC = cfg.normalized_weights()
    parts = [(wA, A)]
    if E is not None:
        parts.append((wE, E))
    if C is not None:
        parts.append((wC, C))
    total_w = sum(w for w, _ in parts) or 1.0
    Q = 100.0 * sum(w * v for w, v in parts) / total_w

    return Score(run_id=trace.run_id, task_id=task_id, arm=arm,
                 A=round(A, 4),
                 E=None if E is None else round(E, 4),
                 C=None if C is None else round(C, 4),
                 Q=round(Q, 2),
                 planned=len(trace.plan),
                 components={"A": round(A, 4),
                             "E": None if E is None else round(E, 4),
                             "C": None if C is None else round(C, 4)},
                 weights_used={"A": round(wA / total_w, 4),
                               "E": round(wE / total_w, 4) if E is not None else 0.0,
                               "C": round(wC / total_w, 4) if C is not None else 0.0},
                 flags=flags, detail=detail)


def write_score(score: Score, out_dir: str | Path) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return write_json(d / "score.json", score.to_dict())


# ---------------------------------------------------------------- L2 rubric（可选）

_L2_DIMS = ("事实边界", "结构完整", "可执行性", "一致性", "表达")


def score_l2(text_base: str, text_cand: str, cfg: Settings | None = None,
             model_name: str | None = None) -> dict:
    """双向呈现的 rubric 打分。仅当 ADAPTIVE_L2=1 时由调用方启用。

    做法：一次调用同时给 A/B 打分（避免「只评一个」带来的位置歧义），
    再交换顺序评一次；对**同一份内容**的两次得分取均值，两次差值 >1.5 记为不稳定。
    返回 {"enabled":True,"dims":{candidate 各维},"base_dims":{...},"unstable_ratio":x}。
    失败时返回 {"enabled":False,"error":...}，绝不抛异常打断主流程。
    """
    cfg = cfg or CFG
    if not (text_base or "").strip() or not (text_cand or "").strip():
        return {"enabled": False, "error": "empty_text"}
    try:
        import os
        from llm import create_llm  # 延迟导入：单测环境无 langchain 也能跑本模块
        judge_model = model_name or os.environ.get("ADAPTIVE_JUDGE_MODEL") or None
        llm = create_llm(temperature=cfg.judge_temperature, model_name=judge_model)
    except Exception as e:  # pragma: no cover - 依赖外部环境
        return {"enabled": False, "error": f"{type(e).__name__}: {e}"}

    rubric = ("你是评审。对下面两段内容的每个维度各给 1-5 分（1=最差，5=最好）："
              "事实边界、结构完整、可执行性、一致性、表达。"
              '只输出 JSON：{"A":{"事实边界":n,...},"B":{"事实边界":n,...}}，不要解释。')

    def ask(a: str, b: str) -> dict:
        prompt = f"{rubric}\n\n【A】\n{a[:12000]}\n\n【B】\n{b[:12000]}"
        try:
            msg = llm.invoke(prompt, max_tokens=240)
            content = str(getattr(msg, "content", "") or "")
            m = re.search(r"\{.*\}", content, re.S)
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}

        def clean(d: Any) -> dict:
            if not isinstance(d, dict):
                return {}
            return {k: int(d[k]) for k in _L2_DIMS
                    if isinstance(d.get(k), (int, float)) and 1 <= int(d[k]) <= 5}
        return {"A": clean(obj.get("A")), "B": clean(obj.get("B"))}

    forward = ask(text_base, text_cand)     # A=base,  B=cand
    backward = ask(text_cand, text_base)    # A=cand,  B=base
    if not forward["B"] or not backward["A"]:
        return {"enabled": False, "error": "judge_parse_failed"}

    cand_dims = {k: int(round((forward["B"].get(k, 3) + backward["A"].get(k, 3)) / 2))
                 for k in _L2_DIMS}
    base_dims = {k: int(round((forward["A"].get(k, 3) + backward["B"].get(k, 3)) / 2))
                 for k in _L2_DIMS}
    unstable = sum(1 for k in _L2_DIMS
                   if abs(forward["B"].get(k, 3) - backward["A"].get(k, 3)) > 1.5)
    return {"enabled": True, "dims": cand_dims, "base_dims": base_dims,
            "unstable_ratio": round(unstable / len(_L2_DIMS), 3),
            "raw": {"forward": forward, "backward": backward}}
