# -*- coding: utf-8 -*-
"""作废 rrsi2 round1 被**误判**成 rejected 的台账行。

背景：候选 `AGENT_REPLAN_ARTIFACT_MAX 3->5` 的实测结果是 dQ=-3.19、p=0.16（与噪声不可区分），
但 Gate B 旧版把「局部崩溃」否决排在配对检验之前，且「崩」的定义是任务级均值 < -5 分
（3 次重复的均值自带约 12 分噪声）——13 个任务里 2 个这样的小下跌就凑出
regression_rate=0.1538 > r_max=0.15，于是判成 regression。
而 regression 属于**证伪**（gate.ledger_status → rejected），这个旋钮就被永久移出了搜索空间。

已修复（gate.gate_b 第 4/4b 步顺序 + comparator.crashed_tasks 的一致性定义）：
同样的数据现在得到 within_noise → 台账 inconclusive（不再写 rejected，不再当证伪）。

append-only：不改历史行，只追加更正（Ledger 的 latest-wins 语义会采纳后者）。
本次更正不是「作废重来」，而是**把同一批数据重新标成它本来该有的结论**：
evidence 沿用实测值（n=13、dQ=-3.1903、p=0.159288），reason 改为 within_noise。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_runtime.ledger import Ledger

FP = "AGENT_REPLAN_ARTIFACT_MAX|3->5|PLANNING_FAILURE"
NOTE = ("更正：原行 reason=regression 出自 Gate B 旧排序的误判（dQ=-3.19、p=0.16 与噪声不可区分；"
        "regression_rate=0.1538 由 3 次重复的任务级均值噪声凑出）。已按「稳定崩 + 均值为正」"
        "重定义局部崩溃否决（见 comparator.crashed_tasks / gate.gate_b 4b）。"
        "同一批数据的正确结论是 within_noise：没测出好处，也没测出坏处。")
EV = {"n": 13, "dQ": -3.1903, "p": 0.159288, "round": 1, "reason": "within_noise", "stopped": ""}

led = Ledger()
before = led.latest(FP)
print("before:", before.get("status"), before.get("evidence"))
led.update(FP, "inconclusive", EV, notes=NOTE)
after = led.latest(FP)
print("after :", after.get("status"), after.get("evidence"))
print("is_duplicate:", led.is_duplicate(FP)[0])
