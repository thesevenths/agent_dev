"""单次 run 的执行器：真实模式（走现有 graph）或模拟模式（供自检使用）。

为什么用子进程
--------------
1. 每次 run 需要独立的 AGENT_CHECKPOINT_DB / log / tmp / documents，否则两臂互相污染；
2. 现有 graph 在 import 时创建 sqlite 连接与全局 graph 对象，同进程跑多次会共享状态；
3. 子进程崩溃（原生库、OOM）不会带走控制平面。

用法（由 probe.py 调用）：
    python -m adaptive_runtime.runner_once --query "..." --task-id CAL-001 --rep 0 \
        --arm base --out experiments/runs/CAL-001/r01/base/rep0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# 允许以 `python -m adaptive_runtime.runner_once` 在沙箱根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_runtime import token_probe  # noqa: E402
from adaptive_runtime.evaluator import score_trace, write_score  # noqa: E402
from adaptive_runtime.io_util import write_json  # noqa: E402
from adaptive_runtime.trace_export import export_from_log  # noqa: E402


# ---------------------------------------------------------------- 假跑

_FAKE_PLAN = [
    (1, "检索标的与公开数据", "crawler_agent"),
    (2, "计算关键指标", "code_agent"),
    (3, "撰写并落盘分析报告", "code_agent"),
]

#: 假报告必须**长成真实报告的形态**：体积达标、有骨架、有结论段。
#: 否则新的反刷分口径会把两份都判成「无报告」，自检就退化成
#: 「比较两个什么都没产出的 run」—— 那样的 PASS 毫无意义。
_FAKE_REPORTS = {
    "base": """# 行业分析报告：市场规模与增长

## 一、分析背景

本报告基于公开统计数据，对本行业近三年的规模变化、盈利结构与未来增速进行梳理，
供投资决策参考。数据窗口为 2023 年至 2025 年，币种统一折算为美元。

## 二、市场概况

2025 年该市场规模约为 1200 亿美元，同比增长 18%。行业集中度较上一年度提升，
前三名厂商合计份额达到 46%。新增产能主要来自头部企业的扩产项目。

结构上，服务收入占比由 31% 提升至 38%，硬件收入占比相应下降。这一变化与行业整体
从产品销售向订阅服务转型的趋势一致。

## 三、关键指标

| 指标 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- |
| 市场规模（亿美元） | 860 | 1017 | 1200 |
| 同比增速 | 12% | 18% | 18% |
| 毛利率 | 36% | 39% | 42% |
| 服务收入占比 | 24% | 31% | 38% |

毛利率提升至 42%，主要来自产品结构变化与服务收入占比上升。期间费用率基本持平，
因此毛利率的提升大部分传导到了经营利润率。

## 四、结论与风险

预计 2026 年增速回落至 12%，规模约为 1344 亿美元。主要风险包括：下游需求周期性
波动、头部厂商价格竞争加剧、以及海外市场政策不确定性。

综合判断，行业仍处于成长期后段，增长质量优于增长速度。
""",
    "good": """# 行业分析报告：市场规模与增长

## 一、分析背景

本报告基于公开统计数据，对本行业近三年的规模变化、盈利结构与未来增速进行梳理，
供投资决策参考。数据窗口为 2023 年至 2025 年，币种统一折算为美元。
市场规模口径采用第三方机构统计（来源：https://example.com/market-2025）。

## 二、市场概况

2025 年该市场规模约为 1200 亿美元，同比增长 18%（数据来源：https://example.com/market-2025）。
行业集中度较上一年度提升，前三名厂商合计份额达到 46%（引自：https://example.com/share-2025）。
新增产能主要来自头部企业的扩产项目，其中两家宣布的扩产计划合计约 80 亿美元。

结构上，服务收入占比由 31% 提升至 38%，硬件收入占比相应下降（数据来源：公司 2025 年报）。
这一变化与行业整体从产品销售向订阅服务转型的趋势一致。

## 三、关键指标

| 指标 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- |
| 市场规模（亿美元） | 860 | 1017 | 1200 |
| 同比增速 | 12% | 18% | 18% |
| 毛利率 | 36% | 39% | 42% |
| 服务收入占比 | 24% | 31% | 38% |

数据来源：https://example.com/financials-2025 ；口径与上表一致，均为年度累计值。

毛利率提升至 42%，主要来自产品结构变化与服务收入占比上升（数据来源：公司 2025 年报）。
期间费用率基本持平（引自：https://example.com/expense-2025），因此毛利率的提升
大部分传导到了经营利润率。

## 四、结论与风险

预计 2026 年增速回落至 12%，规模约为 1344 亿美元（引自：https://example.com/forecast）。
主要风险包括：下游需求周期性波动（参考：https://example.com/risk-demand）、
头部厂商价格竞争加剧（参考：https://example.com/risk-price）、
以及海外市场政策不确定性（参考：https://example.com/risk-policy）。

综合判断，行业仍处于成长期后段，增长质量优于增长速度。
""",
    "zero": None,   # 与 base 完全一致
    "worse": "",
}


def _ts(offset_s: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() + timedelta(seconds=offset_s)).strftime("%Y-%m-%d %H:%M:%S") + ",000"


def _fake_log(profile: str, goal: str) -> str:
    lines = [f"{_ts(0)} INFO === run log started: fake ==="]
    lines.append(f"{_ts(0)} INFO [supervisor] created execution plan (goal='{goal}'):")
    for no, title, agent in _FAKE_PLAN:
        lines.append(f"{no}.    [pending] {title}: {title} → {agent}")
    lines.append(f"{_ts(1)} INFO [supervisor] run log file: fake")

    pass_steps = {"base": (1, 2), "zero": (1, 2), "good": (1, 2, 3), "worse": (1,)}[profile]
    fail_steps = {"base": (3,), "zero": (3,), "good": (), "worse": (2, 3)}[profile]
    for no, title, agent in _FAKE_PLAN:
        if no in pass_steps:
            lines.append(f"{_ts(no * 10)} INFO [critic+summary] step {no}/3 {agent}: "
                         f"PASS via FUSED judge+summary (1 LLM call)")
        if no in fail_steps:
            lines.append(f"{_ts(no * 10 + 5)} INFO [step-failure] step {no} attempt #1 "
                         f"kind=recursion agent={agent}")
            lines.append("  WHY: ReAct step budget exhausted before final answer")
        lines.append(f"{_ts(no * 10 + 6)} INFO [tool][{agent}] ✔ create_file (0.10s, 120 chars)")
    lines.append(f"{_ts(60)} INFO [supervisor] FINISH")
    return "\n".join(lines) + "\n"


def _run_fake(args, out: Path) -> dict:
    profile = args.fake_profile
    if profile not in _FAKE_REPORTS:
        raise ValueError(f"未知 fake profile: {profile}")
    root = Path.cwd()
    # 1) 写一份与真实格式一致的日志，交给真正的解析器处理（顺带回归 parser）
    log_path = out / "fake_run.log"
    log_path.write_text(_fake_log(profile, args.query), encoding="utf-8")

    # 2) 落盘报告（good 有来源，base 无来源，worse 不产出）
    text = _FAKE_REPORTS[profile]
    if text is None:            # zero 与 base 完全一致
        text = _FAKE_REPORTS["base"]
    if text:
        doc = root / "documents" / f"report_{args.arm}.md"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(text, encoding="utf-8")
        os.utime(doc, None)

    trace = export_from_log(log_path, root=root,
                            run_id=f"fake-{profile}-{args.task_id}-{args.rep}")
    score = score_trace(trace, root=root, mode="live",
                        task_id=args.task_id, arm=args.arm)
    write_json(out / "trace.json", trace.to_dict())
    write_score(score, out)
    return {"ok": True, "fake": True, "profile": profile,
            "Q": score.Q, "A": score.A, "trace": str(out / "trace.json")}


# ---------------------------------------------------------------- 真实

def _run_real(args, out: Path) -> dict:
    token_probe.install(sink=out / "tokens.jsonl")
    token_probe.clear()
    token_probe.bind(run_id=f"{args.task_id}-r{args.rep}-{args.arm}",
                     task_id=args.task_id, arm=args.arm,
                     round_id=getattr(args, "round_id", None))
    from graph import invoke_with_memory  # 延迟导入：沙箱 cwd 必须已在 sys.path
    final_state = invoke_with_memory(args.query, thread_id=args.thread_id)

    import runlog
    log_file = runlog.run_file()
    if not log_file:
        raise RuntimeError("未能定位本次 run 的日志文件（runlog.run_file() 返回 None）")

    trace = export_from_log(log_file, root=Path.cwd(), run_id=Path(log_file).stem)
    score = score_trace(trace, root=Path.cwd(), mode="live",
                        task_id=args.task_id, arm=args.arm)
    usage = token_probe.totals()
    write_json(out / "trace.json", trace.to_dict())
    write_score(score, out)
    write_json(out / "usage.json", usage)
    return {"ok": True, "fake": False, "Q": score.Q, "A": score.A,
            "tokens": usage["total"], "estimated_ratio": usage["estimated_ratio"],
            "log": log_file, "finalized": bool((final_state or {}))}


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Adaptive Runtime 单次 run 执行器")
    ap.add_argument("--query", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--arm", required=True, choices=("base", "cand"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--thread-id", default=None)
    ap.add_argument("--round-id", default=None)
    ap.add_argument("--fake-profile", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    args.thread_id = args.thread_id or f"adaptive-{args.task_id}-{args.rep}-{args.arm}"

    try:
        if args.fake_profile:
            result = _run_fake(args, out)
        else:
            result = _run_real(args, out)
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-4000:]}
    write_json(out / "result.json", result)
    if not result.get("ok"):
        print(f"[runner_once] FAILED: {result.get('error')}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
