# -*- coding: utf-8 -*-
"""自检：按 agent 类型的步数预算 + 反打磨 hint + NO-OP 判定修正。

对应 2026-10-02 23:43 run 的两个实证问题：
  1) step3/step4 连续撞 recursion_limit=15 被截断（报告其实已落盘）→ 预算分类型；
  2) step5 "plan 一字未改但换了执行者"被记成 NO-OP → 空转计数误增。
用法：python tmp/verify_budget_and_noop.py
"""
import os
import sys

sys.path.insert(0, r"F:\agent\multi-agent")

from unittest.mock import MagicMock  # noqa: E402
import langchain.agents as _amod  # noqa: E402
if not hasattr(_amod, "create_agent"):
    _amod.create_agent = MagicMock()

for _ in range(15):
    try:
        import agents  # noqa: E402
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
    raise RuntimeError("agents 导入重试耗尽")

import handoff  # noqa: E402
import inspect  # noqa: E402
import supervisor as sup  # noqa: E402

_FAIL = 0


def ok(cond, msg):
    global _FAIL
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _FAIL += 1


print("[1] _agent_max_iter 按类型解析（env 驱动）")
os.environ["AGENT_MAX_ITERATIONS"] = "15"
os.environ["AGENT_MAX_ITERATIONS_CODE"] = "28"
ok(agents._agent_max_iter("CodeAgent") == 28, "CodeAgent → 28（env AGENT_MAX_ITERATIONS_CODE）")
ok(agents._agent_max_iter("code_agent") == 28, "code_agent → 28（下划线命名同样命中）")
ok(agents._agent_max_iter("crawler_agent") == 15, "crawler_agent → 15（未配置，回落全局）")
ok(agents._agent_max_iter("chat_agent") == 15, "chat_agent → 15")
ok(agents._agent_max_iter("") == 15, "空名 → 15（不炸）")
ok(agents._agent_max_iter(None) == 15, "None → 15（不炸）")
os.environ["AGENT_MAX_ITERATIONS_CODE"] = "abc"
ok(agents._agent_max_iter("CodeAgent") == 15, "非法值 'abc' → 回落 15（不炸）")
os.environ["AGENT_MAX_ITERATIONS_CODE"] = "-3"
ok(agents._agent_max_iter("CodeAgent") == 15, "负数 → 回落 15")
os.environ["AGENT_MAX_ITERATIONS_CODE"] = "28"

print("[2] 预算换算写入失败原因（可操作诊断）")
src = inspect.getsource(agents)
ok("recursion_limit={_iter_budget}" in src, "失败文案用实际生效预算（不是写死全局值）")
ok("allows only ~{_iter_budget // 2} round-trips" in src, "文案给出「约几次工具往返」换算")
ok("_iter_budget = _agent_max_iter(agent.name)" in src.split("except GraphRecursionError:")[1][:400],
   "except 分支内重算预算，避免 try 未赋值时 NameError")

print("[3] recursion hint 反打磨约束")
h = handoff._failure_hint("recursion")
ok("立即收尾" in h, "hint 含「跑通后立即收尾」")
ok("打磨" in h, "hint 点名打磨性修改是撞上限头号原因")
ok("2 个 super-step" in h, "hint 给出预算换算（1 往返 = 2 super-step）")
ok(h.count("\n") >= 6, f"hint 结构化为多条（{h.count(chr(10))+1} 行）")

print("[4] NO-OP 判定：换执行者不算空转")
src_sup = inspect.getsource(sup.supervisor)
ok("planned_agent = target_agent" in src_sup, "保存原计划的 agent")
ok("agent_changed" in src_sup and "or agent_changed" in src_sup, "changed 计入 agent 变化")
ok('_norm_agent(target_agent) != _norm_agent(planned_agent)' in src_sup,
   "比较时规范化 _agent 后缀（code vs code_agent）")
ok("changed ONLY the executor" in src_sup, "仅换执行者时单独留痕（可观测）")

print("\n=== RESULT:", "ALL PASS" if _FAIL == 0 else f"{_FAIL} FAILED", "===")
sys.exit(0 if _FAIL == 0 else 1)
