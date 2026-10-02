# -*- coding: utf-8 -*-
"""实测：sub agent 未按预期返回数据时，supervisor 到底会不会 update plan。

直接打 plan._should_replan / plan._replan_evidence（不调 LLM，纯判定层），
覆盖"返回不符预期"的 7 类典型场景，输出判定 + 依据。
用法：python tmp/verify_replan_evidence.py
"""
import os
import sys

sys.path.insert(0, r"F:\agent\multi-agent")
os.environ.setdefault("AGENT_REPLAN_NOOP_MAX", "2")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

# 本机默认 python 是 langchain 0.3.x（无 create_agent，项目实际由 langgraph dev 用 1.x 跑）。
# 只补 llm.py import 所需的符号，plan.py 本身保持真身（判定逻辑必须跑真代码）。
from unittest.mock import MagicMock  # noqa: E402
import langchain.agents as _amod  # noqa: E402
if not hasattr(_amod, "create_agent"):
    _amod.create_agent = MagicMock()

# 自动补齐本机缺失的重依赖（plan.py 判定层只用 os/re/messages，stub 不影响判定语义）
for _ in range(15):
    try:
        import plan  # noqa: E402
        break
    except ModuleNotFoundError as e:
        n = e.name or ""
        if not n:
            raise
        sys.modules[n] = MagicMock()
        if "." in n:
            parent, child = n.rsplit(".", 1)
            if isinstance(sys.modules.get(parent), MagicMock):
                setattr(sys.modules[parent], child, sys.modules[n])
    except ImportError as e:
        # langchain 0.3 少符号：给对应模块补属性后重试
        import re as _re
        m = _re.search(r"cannot import name '(\w+)' from '([\w.]+)'", str(e))
        if not m:
            raise
        sym, mod = m.group(1), m.group(2)
        __import__(mod)
        setattr(sys.modules[mod], sym, MagicMock())
else:
    raise RuntimeError("plan 导入重试次数耗尽")

# 注意：description 刻意不写文件路径 —— 否则会命中 _replan_evidence 第 2 类证据
# （"pending 步引用了不存在的文件"），把所有场景都染成 True，掩盖第 1 类证据（内容信号）的真实判定。
PLAN = [
    {"title": "获取价格数据", "description": "检索 BTC 每日收盘价 → crawler_agent", "status": "completed"},
    {"title": "获取情绪新闻", "description": "检索近期新闻与宏观事件 → crawler_agent", "status": "completed"},
    {"title": "数据分析可视化", "description": "读取上游价格数据计算技术指标并生成图表 → code_agent", "status": "pending"},
    {"title": "撰写报告", "description": "撰写 Markdown 分析报告 → code_agent", "status": "pending"},
]
# 对照用：pending 步引用了不存在的文件（第 2 类证据）
PLAN_MISSING = [dict(s) for s in PLAN]
PLAN_MISSING[2]["description"] = ("读取 F:\\agent\\multi-agent\\tmp\\nope.json 计算指标 → code_agent")


def mk_state(obs, streak=0, artifacts=None, current=2):
    return {
        "execution_plan": [dict(s) for s in PLAN],
        "current_step": current,
        "observations": obs,
        "replan_noop_streak": streak,
        "artifacts": artifacts or [],
        "plan_goal": "BTC 近期行情分析",
    }


def verdict(state, cur=2, pl=None):
    return plan._should_replan(state, pl or PLAN, cur)


CASES = []
_PROBES = []


def _mk(name, content):
    """造真实产物文件：真实链路里 sub agent 的产出路径会记进 state.artifacts，
    产物契约证据正是据此校验内容（此前场景把 artifacts 留空，等于绕过了这条证据）。"""
    p = os.path.join(r"F:\agent\multi-agent\tmp", name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    _PROBES.append(p)
    return p


P_A = _mk("_probe_a.json", "近期比特币价格在 108000 美元附近波动，昨日小幅上涨。")  # 散文冒充 JSON
P_B = _mk("_probe_b.json", "[]")                                                   # 空数组
P_C = _mk("_probe_c.json", '[{"date": "2026-09-28"}, {"date": "2026-09-29"}]')     # 缺 close 字段
P_D = _mk("_probe_d.json", '[{"date": "2026-08-01", "close": 90000}]')            # 日期范围错
P_I = _mk("_probe_i.json", "")                                                    # 空文件


def case(name, obs, expect, note, streak=0, artifacts=None, current=2, pl=None):
    st = mk_state(obs, streak, artifacts, current)
    should, why = verdict(st, current, pl)
    CASES.append((name, should, why, expect, note))


print("=" * 100)
print("场景实测：sub agent 返回后，supervisor 是否认为「计划需要改写」(current=2, 非末步, 无 failed 步)")
print("=" * 100)

# A: 返回自由文本而非约定的 JSON，且把散文写进了 .json 文件（真实链路 artifacts 会记这个路径）
case("A 自由文本代替 JSON（无失败词，文件存在）",
     [AIMessage(content="近期比特币价格在 108,000 美元附近波动，昨日小幅上涨，市场情绪偏乐观。已保存到文件。"),
      ToolMessage(content="write_file ok: F:\\agent\\multi-agent\\tmp\\btc_price.json", tool_call_id="1")],
     True, "下游要的是结构化 JSON，实际是散文 → 前提失效，应改写", artifacts=[P_A]),
# B: 空数组 / 空数据
case("B 返回空数组 []（无失败词）",
     [AIMessage(content="查询结果如下：\n[]\n共 0 条记录。"),
      ToolMessage(content="write_file ok: tmp/btc_price.json", tool_call_id="2")],
     True, "数据为空 → 下游无米下锅，应改写", artifacts=[P_B]),
# C: JSON 字段缺失（只有 date 没有收盘价）—— 结构合法，证据门不干预（非结构性硬伤）
case("C JSON 字段缺失（无 close 字段）",
     [AIMessage(content='[{"date": "2026-09-28"}, {"date": "2026-09-29"}]'),
      ToolMessage(content="saved", tool_call_id="3")],
     False, "结构合法 → 证据门不介入；由 critic 语义层（LLM 判产出是否满足本步）兜底",
     artifacts=[P_C]),
# D: 日期范围错（要 09-28~10-02，给的是 08 月）—— 同上，属语义层职责
case("D 日期范围错误（给了上上个月）",
     [AIMessage(content='[{"date": "2026-08-01", "close": 90000}] 已保存'),
      ToolMessage(content="saved", tool_call_id="4")],
     False, "结构合法 → 证据门不介入；由 critic 语义层兜底", artifacts=[P_D]),
# E: 对照 —— 明确的失败文本
case("E 对照：明确失败文本（含关键词）",
     [AIMessage(content="获取失败：API rate limit，未能取到数据。")],
     True, "对照组，本应触发"),
# F: 对照 —— 空转计数达阈值（AGENT_REPLAN_NOOP_MAX=2）
case("F 空转达阈值(2) + 场景 A 的不符预期",
     [AIMessage(content="近期比特币价格在 108,000 美元附近波动。")],
     True, "证据优先级已高于空转停用 → 不再被一票否决", streak=2, artifacts=[P_A]),
# G: 信号被挤出窗口（_replan_evidence 只看 obs[-6:]）
#    真实一步 ReAct 15 个 super-step 会产生十几条消息，上一步的错误信号必然被挤出。
obs_g = [AIMessage(content="价格数据获取成功，已保存。") for _ in range(5)]
obs_g += [AIMessage(content="警告：检索失败，rate limit，本节数据未取到。")]   # 第 6 条：真实错误
obs_g += [AIMessage(content=f"补充处理 {i}") for i in range(6)]               # 后续 6 条把错误挤出窗口
case("G 失败信号被挤出 obs[-6:] 窗口", obs_g, True,
     f"共 {len(obs_g)} 条，错误在第 6 条，窗口只看最后 6 条 → 漏判")
# H: 对照 —— pending 步引用的文件不存在（第 2 类证据，唯一能抓到"handoff 断裂"的路径）
case("H 对照：pending 步引用的文件不存在",
     [AIMessage(content="数据已保存，一切正常。")], True,
     "handoff 断裂，应改写", pl=PLAN_MISSING)
# I: 文件存在但内容为空（第 2 类证据只看存在性，不看内容 → 靠第 3 类证据）
case("I 文件存在但内容为空",
     [AIMessage(content="数据已保存到文件。")], True,
     "存在性通过、内容为空 → 由产物契约证据拦截", artifacts=[P_I])

print(f"{'场景':<38} {'判定':<6} {'应然':<4} 依据")
print("-" * 100)
bad = 0
for name, should, why, expect, note in CASES:
    mark = "OK " if should == expect else "GAP"
    if should != expect:
        bad += 1
    print(f"{name:<38} {str(should):<6} {str(expect):<4} [{mark}] {why[:52]}")
print("-" * 100)
print(f"漏判/错判场景数：{bad}/{len(CASES)}")
print("注：C/D（字段缺失 / 日期范围错）JSON 结构合法，属语义不符，"
      "按设计由 critic 语义层（AGENT_CRITIC_LLM=1 的 LLM 判定）兜底，不在廉价证据门职责内。")
for p in _PROBES:
    try:
        os.remove(p)
    except OSError:
        pass

print()
print("=" * 100)
print("窗口定量：一步产生多少条消息就会把上一步的信号挤出 obs[-6:]")
print("=" * 100)
for n in (5, 6, 7, 12):
    obs = [AIMessage(content="早期关键错误：检索失败 rate limit，数据未取到。")]
    obs += [AIMessage(content=f"后续消息 {i}") for i in range(n)]
    ev, why = plan._replan_evidence(mk_state(obs), PLAN, 2)
    print(f"  错误信号后追加 {n:>2} 条消息 → 证据命中={ev}  {'← 已漏判' if not ev else ''}")

print()
print("=" * 100)
print("判据全集：_should_replan 在什么条件下会调 LLM 改写 plan")
print("=" * 100)
probes = [
    ("current=0（首步派发）", mk_state([], current=0), 0),
    ("存在 failed 步", None, None),
    ("最后一步 current=len-1", mk_state([], current=3), 3),
    ("无证据（默认）", mk_state([AIMessage(content="一切正常，数据已保存。")]), 2),
]
p = dict(PLAN[2]); p["status"] = "failed"
plan_failed = [dict(PLAN[0]), dict(PLAN[1]), p, dict(PLAN[3])]
st = mk_state([], current=2); st["execution_plan"] = plan_failed
s, w = plan._should_replan(st, plan_failed, 2)
print(f"  {'存在 failed 步':<28} → {s}  ({w})")
for label, stx, cur in probes:
    if stx is None:
        continue
    s, w = plan._should_replan(stx, PLAN, cur)
    print(f"  {label:<28} → {s}  ({w[:70]})")

print()
print("证据信号词表（_REPLAN_EVIDENCE_SIGNALS）：")
print("  " + ", ".join(plan._REPLAN_EVIDENCE_SIGNALS))
