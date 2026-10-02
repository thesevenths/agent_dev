# -*- coding: utf-8 -*-
"""联合自检：产物契约证据 / obs 窗口 / 证据优先于空转 / 动作契约门 / 执行者对账 / 截断挽救。

对应 2026-10-02 晚间实证的三类问题：
  1) sub agent 交回空/坏数据，plan 一字不改（证据门只有关键词 + 文件存在性）；
  2) step5 "发邮件"被换成 code_agent 用 dir+read_file 应付，critic 还判 PASS；
  3) step4 产物 23:47:42 已落盘、23:48:14 才撞上限，却被判 failed 并重做。
用法：python tmp/verify_replan_and_gates.py
"""
import os
import sys
import json
import time

sys.path.insert(0, r"F:\agent\multi-agent")
TMP = r"F:\agent\multi-agent\tmp"

# 先设 env，再 import（模块级常量在 import 时读取）
os.environ["AGENT_REPLAN_OBS_WINDOW"] = "20"
os.environ["AGENT_REPLAN_ARTIFACT_CHECK"] = "1"
os.environ["AGENT_MAX_ITERATIONS"] = "15"
os.environ["AGENT_MAX_ITERATIONS_CODE"] = "28"

from unittest.mock import MagicMock  # noqa: E402
import langchain.agents as _amod  # noqa: E402
if not hasattr(_amod, "create_agent"):
    _amod.create_agent = MagicMock()

for _ in range(15):
    try:
        import plan  # noqa: E402
        import agents  # noqa: E402
        import handoff  # noqa: E402
        break
    except ModuleNotFoundError as e:
        n = e.name or ""
        if not n:
            raise
        sys.modules[n] = MagicMock()
        if "." in n:
            p, c = n.rsplit(".", 1)
            if isinstance(sys.modules.get(p), MagicMock):
                setattr(sys.modules[p], c, sys.modules[n])
    except ImportError as e:
        import re as _re
        m = _re.search(r"cannot import name '(\w+)' from '([\w.]+)'", str(e))
        if not m:
            raise
        __import__(m.group(2))
        setattr(sys.modules[m.group(2)], m.group(1), MagicMock())
else:
    raise RuntimeError("导入重试耗尽")

_FAIL = 0


def ok(cond, msg):
    global _FAIL
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _FAIL += 1


PLAN = [
    {"title": "获取价格数据", "description": "检索 BTC 日线 → crawler_agent", "status": "completed"},
    {"title": "获取情绪新闻", "description": "检索新闻 → crawler_agent", "status": "completed"},
    {"title": "数据分析可视化", "description": "读取上游数据计算指标生成图表 → code_agent", "status": "pending"},
    {"title": "撰写报告", "description": "撰写 Markdown 报告 → code_agent", "status": "pending"},
]

# --- 造四类产物文件（用后清理） ---
P_EMPTY = os.path.join(TMP, "_probe_empty.json")
P_ARR = os.path.join(TMP, "_probe_empty_arr.json")
P_BAD = os.path.join(TMP, "_probe_broken.json")
P_GOOD = os.path.join(TMP, "_probe_good.json")
open(P_EMPTY, "w").close()
open(P_ARR, "w", encoding="utf-8").write("[]")
open(P_BAD, "w", encoding="utf-8").write("{not json,,,")
open(P_GOOD, "w", encoding="utf-8").write(json.dumps([{"date": "2026-10-02", "close": 84795}]))

print("[1] 第 3 类证据：产物契约校验（_artifact_contract_breach）")
for path, label in ((P_EMPTY, "0 字节"), (P_ARR, "空数组 []"), (P_BAD, "坏 JSON")):
    br, why = plan._artifact_contract_breach({"artifacts": [path]})
    ok(br, f"{label} → 触发证据（{why[:46]}）")
br, why = plan._artifact_contract_breach({"artifacts": [P_GOOD]})
ok(not br, f"正常 JSON（1 条记录）→ 不触发（{why[:30] or '无违约'}）")
br, _ = plan._artifact_contract_breach({"artifacts": []})
ok(not br, "无产物 → 不触发（零回归）")
br, _ = plan._artifact_contract_breach({"artifacts": [os.path.join(TMP, "_nope_xyz.json")]})
ok(br, "产物记录存在但文件不在盘上 → 触发")
plan._REPLAN_ARTIFACT_CHECK = False
br, _ = plan._artifact_contract_breach({"artifacts": [P_EMPTY]})
ok(not br, "AGENT_REPLAN_ARTIFACT_CHECK=0 → 关闭（可退回）")
plan._REPLAN_ARTIFACT_CHECK = True

print("[2] obs 窗口：AGENT_REPLAN_OBS_WINDOW=20（旧为固定 6）")
from langchain_core.messages import AIMessage  # noqa: E402
for n in (6, 12, 18):
    obs = [AIMessage(content="早期关键错误：检索失败 rate limit，数据未取到。")]
    obs += [AIMessage(content=f"后续消息 {i}") for i in range(n)]
    ev, why = plan._replan_evidence({"observations": obs, "artifacts": []}, PLAN, 2)
    ok(ev, f"错误信号后追加 {n:>2} 条 → 仍命中（旧实现 ≥6 条即漏判）")
ok(plan._REPLAN_OBS_WINDOW == 20, f"窗口生效值 = {plan._REPLAN_OBS_WINDOW}")

print("[3] 证据优先级 > 空转停用（旧顺序是 noop 先判、一票否决）")
st = {"observations": [], "artifacts": [P_EMPTY], "replan_noop_streak": 2, "plan_goal": "g"}
should, why = plan._should_replan(st, PLAN, 2)
ok(should, f"streak=2 但有产物违约 → 仍再规划（{why[:52]}）")
st2 = {"observations": [], "artifacts": [], "replan_noop_streak": 2, "plan_goal": "g"}
should2, why2 = plan._should_replan(st2, PLAN, 2)
ok(not should2, f"streak=2 且无证据 → 仍停用（省 LLM 的语义保留：{why2[:30]}）")

print("[4] 动作契约门：本步要求的动作是否真的发生")
mail_step = {"title": "发送报告", "description": "将最终报告通过邮件发送给用户 → chat_agent"}
c, p_, r = agents._action_contract_gate(mail_step, "报告已生成，路径 F:\\tmp\\r.md", "")
ok(c and not p_, f"要求发邮件但无 send_email 痕迹 → 判不合格（{r[:44]}）")
c, p_, _ = agents._action_contract_gate(mail_step, "报告已生成", "已发送", "")
ok(c and p_, "证据含'已发送' → 放行")
c, p_, _ = agents._action_contract_gate(mail_step, "x", "", "send_email")
ok(c and p_, "工具名 send_email → 放行（痕迹在工具名而非返回文本）")
analysis_step = {"title": "分析", "description": "读取上游数据计算指标生成图表 → code_agent"}
c, p_, _ = agents._action_contract_gate(analysis_step, "图表已生成", "")
ok(not c, "普通分析步 → 不干预（零回归）")
saved = agents._ACTION_PROBES
agents._ACTION_PROBES = []
c, _, _ = agents._action_contract_gate(mail_step, "x", "")
agents._ACTION_PROBES = saved
ok(not c, "AGENT_ACTION_PROBES=0 → 关闭（可退回）")

print("[5] 执行者对账：派发文本提示 EXECUTOR SUBSTITUTION")
state = {"execution_plan": [
    {"title": "获取价格", "description": "抓取 → crawler_agent", "status": "completed"},
    {"title": "发送报告", "description": "将报告邮件发给用户 → chat_agent", "status": "pending"},
], "current_step": 2, "artifacts": [], "plan_goal": "g", "step_failures": []}
txt = handoff._step_assignment_text(state, "CodeAgent")
ok("EXECUTOR SUBSTITUTION" in txt, "plan 指定 chat_agent、实际 CodeAgent → 提示替换")
ok("do NOT substitute" in txt, "明确禁止用读/描述冒充执行该动作")
txt2 = handoff._step_assignment_text(state, "ChatAgent")
ok("EXECUTOR SUBSTITUTION" not in txt2, "执行者一致 → 不插入（零回归）")

print("[6] 截断挽救：本步期间落盘的产物被识别")
# 先清掉前面用过的探针文件，否则它们也落在时间窗内，会污染"只应识别到 fresh"的断言
for p in (P_EMPTY, P_ARR, P_BAD, P_GOOD):
    try:
        os.remove(p)
    except OSError:
        pass
_t0 = time.time()
fresh = os.path.join(TMP, "_probe_fresh.md")
open(fresh, "w", encoding="utf-8").write("# report\ncontent")
hits = agents._salvage_truncated_artifacts(_t0, [])
ok(fresh in hits, f"本步期间新落盘产物 → 识别到（{len(hits)} 个）")
ok(agents._salvage_truncated_artifacts(time.time() + 300, []) == [], "时间窗之外 → 不误抓上游文件")
ok(agents._salvage_truncated_artifacts(_t0, [fresh]) == [], "已在 artifacts 清单里 → 排除")
agents._TRUNCATED_SALVAGE = False
ok(agents._salvage_truncated_artifacts(time.time() - 30, []) == [], "AGENT_TRUNCATED_SALVAGE=0 → 关闭")
agents._TRUNCATED_SALVAGE = True
ok(handoff._failure_hint("truncated").find("不要重做") > 0, "truncated hint 明确要求下游不要重做")

for p in (fresh,):
    try:
        os.remove(p)
    except OSError:
        pass
print("  (探针文件已清理)")

print("\n=== RESULT:", "ALL PASS" if _FAIL == 0 else f"{_FAIL} FAILED", "===")
sys.exit(0 if _FAIL == 0 else 1)
