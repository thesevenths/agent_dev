# -*- coding: utf-8 -*-
"""作废台账里那条来自 **selftest 合成数据** 的 accepted 行。

`hyp_fc7353f4` / `AGENT_MAX_ITERATIONS_CODE|60->80|COMPLETENESS_FAILURE` 写于
2026-10-22T23:00:07，evidence 是 `{n:8, dQ:38.1, p:0.0, reason:VALIDATED}`。
两个证据说明它不是真实实验：

1. `dQ=38.1 / p=0.0` 与 `run_adaptive selftest` 的正控 `positive_control_good`
   一字不差（本机每次 selftest 都打印 `ACCEPT/VALIDATED mean=38.1`）；
2. `tasks` 是 `REP-001..REP-008`，而真实闭环跑的是离线任务族（id 前缀 `OFF-`）。
   合成画像（fake profiles）走的是同一条 gate 代码，但当时 `cmd_round/auto` 还没有
   「假跑不写真台账」的守卫（本会话补上的 s5），于是它把假数据记进了真台账。

为什么必须改：`accepted` 是**终局**状态，`is_duplicate` 会因此永久拦掉这个假设，
搜索空间少一个旋钮；同时 `evolution.report` 把它算成「L1 已用 1 次」，让探索/剪枝
判断建立在一次根本没发生的成功上。改成 inconclusive 后它会重新可测 —— 因为这个值
**从来没被真实测过**。

append-only：不改历史行，只追加更正（Ledger 的 latest-wins 语义会采纳后者）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_runtime.ledger import Ledger

FP = "AGENT_MAX_ITERATIONS_CODE|60->80|COMPLETENESS_FAILURE"
NOTE = ("更正：原行 status=accepted/dQ=38.1/p=0.0 与 selftest 正控完全一致，且 tasks 为 "
        "REP-001..008（真实闭环用 OFF-* 离线任务族）—— 那是假画像跑出来的合成证据，"
        "当时还没有「假跑不写真台账」的守卫。此旋钮从未被真实测量过，故改为 inconclusive "
        "并放回搜索空间。")
EV = {"n": 8, "dQ": 38.1, "p": 0.0, "reason": "void_selftest_synthetic",
      "voided_from": "accepted"}

led = Ledger()
before = led.latest(FP)
print("before:", before.get("status"), before.get("evidence"))
led.update(FP, "inconclusive", EV, notes=NOTE)
after = led.latest(FP)
print("after :", after.get("status"), after.get("evidence"))
print("is_duplicate:", led.is_duplicate(FP)[0],
      "| exhausted(不该被当成测过了):", led.exhausted(FP)[0])
