# -*- coding: utf-8 -*-
"""E1 终局对账 + E2 重做载体 专项自检。

E1：FINISH 前若存在 failed 步，强制一次 LLM 决策（重试 / 降级交付 / 明确告知）。
    修补 2026-10-02 step5 邮件没发却静默 FINISH 的洞。
E2：允许 execution_plan 每次 run 净增 AGENT_PLAN_GROW_MAX 步，作为「retry of step N」载体。
    修补「只减不增」导致模型无法插入重试步的洞。

运行：PYTHONIOENCODING=utf-8 python tmp/verify_e1_e2.py
"""
import os
import sys
import json

sys.path.insert(0, r"F:\agent\multi-agent")

from unittest.mock import MagicMock  # noqa: E402
import langchain.agents as _amod  # noqa: E402
if not hasattr(_amod, "create_agent"):
    _amod.create_agent = MagicMock()

for _ in range(15):
    try:
        import plan  # noqa: E402
        import supervisor as sup  # noqa: E402
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
        m = _re.search(r"cannot import name '([^']+)' from '([^']+)'", str(e))
        if not m:
            raise
        mod = sys.modules.get(m.group(2))
        if mod is not None:
            setattr(mod, m.group(1), MagicMock())
        else:
            sys.modules[m.group(2)] = MagicMock()

MK = "__e1e2__"
CAPTURED = {}
REPLY = {}

PASS = FAIL = 0


def ok(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"\n          -> {detail}" if detail else ""))


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def _fake_structured(llm, messages, schema, label="", **kw):
    CAPTURED["messages"] = messages
    CAPTURED["label"] = label
    return REPLY.get("parsed")


sup._structured_with_retry = _fake_structured
plan._structured_with_retry = _fake_structured
sup._summarize_observations = lambda s: "· 已完成：抓取价格、生成报告"
sup._extract_longterm = lambda s, k: None

plan._PLAN_GROW_MAX = 1
sup._PLAN_GROW_MAX = 1


def step(title, desc, agent, status="pending"):
    return {"title": title, "description": f"{desc} → {agent}", "status": status}


PLAN5 = [
    step("抓取价格", "抓取 BTC 日线", "crawler_agent", "completed"),
    step("抓取情绪", "抓取新闻", "crawler_agent", "completed"),
    step("分析可视化", "计算指标出图", "code_agent", "completed"),
    step("撰写报告", "写 Markdown 报告", "code_agent", "completed"),
    step("发送邮件", "把报告邮件发给用户", "chat_agent", "failed"),
]


def st_of(plan, current, failures=None, grown=0, artifacts=None):
    return {
        "messages": [], "observations": [],
        "execution_plan": [dict(s) for s in plan],
        "current_step": current,
        "artifacts": artifacts or [],
        "step_failures": failures or [],
        "plan_goal": "分析比特币行情并发邮件",
        "memory_key": MK,
        "run_started_at": 0,
        "replan_noop_streak": 0,
        "error_count": 0,
        "plan_grown": grown,
    }


# ============================================================ E2
section("E2-A：有增长配额时，模型可插入重试步")

st = st_of(PLAN5, 4, grown=0)
want = [dict(s) for s in PLAN5[:4]] + [
    step("重试分析", "重算指标", "code_agent"),
    step("重试报告", "重写报告", "code_agent"),
    step("发送邮件", "把报告邮件发给用户", "chat_agent"),
]
REPLY["parsed"] = {"next": "code_agent", "reason": "r", "execution_plan": want}
CAPTURED.clear()
res = sup.supervisor(st)
ok(len(res["execution_plan"]) == 6,
   f"原 5 步 + 模型给 7 步 → 增长配额 1 → 实际 {len(res['execution_plan'])} 步",
   detail=f"got {len(res['execution_plan'])}")
ok(res.get("plan_grown") == 1, f"plan_grown 计数写回 = {res.get('plan_grown')}")

section("E2-B：配额耗尽后恢复「只减不增」")

st = st_of(PLAN5, 4, grown=1)
REPLY["parsed"] = {"next": "code_agent", "reason": "r", "execution_plan": want}
CAPTURED.clear()
res = sup.supervisor(st)
ok(len(res["execution_plan"]) == 5,
   f"plan_grown=1（配额用尽）→ 截断回原长 {len(res['execution_plan'])} 步",
   detail=f"got {len(res['execution_plan'])}")

section("E2-C：AGENT_PLAN_GROW_MAX=0 完全恢复原语义")

plan._PLAN_GROW_MAX = 0
sup._PLAN_GROW_MAX = 0
st = st_of(PLAN5, 4, grown=0)
REPLY["parsed"] = {"next": "code_agent", "reason": "r", "execution_plan": want}
res = sup.supervisor(st)
ok(len(res["execution_plan"]) == 5, "上限设 0 → 与改动前一致的截断行为（零回归）")
plan._PLAN_GROW_MAX = 1
sup._PLAN_GROW_MAX = 1

# ============================================================ E1
section("E1-A：无 failed 步 → 不触发对账（不浪费 LLM）")

st = st_of([dict(s) for s in PLAN5[:4] if s["status"] != "failed"] + [
    dict(PLAN5[4], status="completed")], 5, grown=0)
REPLY["parsed"] = {"next": "FINISH", "reason": "ok"}
CAPTURED.clear()
res = sup.supervisor(st)
ok(CAPTURED.get("label") is None, "终态无失败 → 完全不调对账 LLM（省一次调用）")
ok(res.get("next") == "FINISH", "正常 FINISH")

section("E1-B：有 failed 步 → 强制对账，模型选择重试")

fails = [{"step": 5, "agent": "ChatAgent", "kind": "exception",
          "reason": "SMTP 连接超时，邮件未发出", "hint": "改用备用 SMTP 端口", "attempt": 3, "at": "t"}]
st = st_of(PLAN5, 5, failures=fails, grown=0)
retry_plan = [dict(s) for s in PLAN5] + [
    step("重试发邮件", "改用备用 SMTP 重发报告邮件", "chat_agent")]
REPLY["parsed"] = {"next": "chat_agent", "reason": "retry", "execution_plan": retry_plan}
CAPTURED.clear()
res = sup.supervisor(st)
ok(CAPTURED.get("label") == "final-adjudicate", "确实调用了终局对账 LLM")
ok(res.get("next") == "chat_agent", f"next={res.get('next')}（重试发邮件）")
ok(len(res["execution_plan"]) == 6, f"plan 追加到 {len(res['execution_plan'])} 步")
ok(res.get("current_step") == 5,
   f"current_step={res.get('current_step')} 指向新追加的重试步（index 5 = 第 6 步）")
ok(res.get("plan_grown") == 1, "plan_grown 累加到 1")

section("E1-C【关键】重试步能否看到原失败步的原因（否则必然再踩同一个坑）")

carried = res.get("step_failures") or []
ok(any(int(f.get("step") or 0) == 6 for f in carried),
   f"失败原因被搬运到新步号 6（handoff 前向回溯只看最近 2 步）",
   detail=f"carried steps={[f.get('step') for f in carried]}")
ok(any("retry-of-5" in str(f.get("kind")) for f in carried),
   "搬运条目的 kind 标记为 retry-of-5（可追溯来源）")
ok(any("SMTP" in str(f.get("reason")) for f in carried),
   "具体失败原因（SMTP 连接超时）随新步号保留")

section("E1-D：模型选择降级交付 → final_note 带说明，仍 FINISH")

st = st_of(PLAN5, 5, failures=fails, grown=0)
REPLY["parsed"] = {"next": "FINISH", "reason": "报告已生成但邮件未发出，需手动发送"}
CAPTURED.clear()
res = sup.supervisor(st)
ok(res.get("next") == "FINISH", "接受失败 → FINISH")
ok("邮件未发出" in str(res.get("final_note") or ""),
   f"final_note 带上向用户的说明：{res.get('final_note')!r}")
ok("邮件未发出" in str(res.get("reason") or ""), "reason 也带上说明（不再谎称全部完成）")

section("E1-E：配额耗尽 → 强制 FINISH，不给重试选项")

st = st_of(PLAN5, 5, failures=fails, grown=1)
REPLY["parsed"] = {"next": "chat_agent", "reason": "retry", "execution_plan": retry_plan}
CAPTURED.clear()
res = sup.supervisor(st)
prompt = "\n".join(str(getattr(m, "content", "")) for m in (CAPTURED.get("messages") or []))
ok("RETRY QUOTA EXHAUSTED" in prompt,
   "prompt 明确告知配额耗尽、必须 FINISH（杜绝无限追加）")
ok(res.get("next") == "FINISH", "模型仍选重试 → 降级为带说明 FINISH，不追加")

section("E1-F：对账 LLM 异常 → fail-safe 走原 FINISH")

st = st_of(PLAN5, 5, failures=fails, grown=0)


def _boom(*a, **k):
    raise RuntimeError("backend unreachable")


sup._structured_with_retry = _boom
res = sup.supervisor(st)
ok(res.get("next") == "FINISH", "LLM 不可达 → 照常 FINISH，不卡住收尾")
sup._structured_with_retry = _fake_structured

section("E1-G：AGENT_FINAL_ADJUDICATE=0 → 恢复原静默语义")

sup._FINAL_ADJUDICATE = False
st = st_of(PLAN5, 5, failures=fails, grown=0)
REPLY["parsed"] = {"next": "chat_agent", "reason": "retry", "execution_plan": retry_plan}
CAPTURED.clear()
res = sup.supervisor(st)
ok(CAPTURED.get("label") is None and res.get("next") == "FINISH",
   "开关关闭 → 完全不调对账、直接 FINISH（零回归）")
sup._FINAL_ADJUDICATE = True

# ---------------------------------------------------------------- 清理
import glob  # noqa: E402
n = 0
for p in glob.glob(os.path.join(r"F:\agent\multi-agent", "log", f"*{MK}*")):
    try:
        os.remove(p)
        n += 1
    except OSError:
        pass
print(f"\n探针日志清理：{n} 个")
print("-" * 78)
print(f"PASS={PASS}  FAIL={FAIL}")
sys.exit(1 if FAIL else 0)
