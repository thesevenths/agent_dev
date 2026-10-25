# -*- coding: utf-8 -*-
"""作废 2026-10-22T23:00:07 那次 `round --fake --case good` 误写进真实台账/经验库的行。

append-only：不改动历史行，只追加更正（Ledger/Store 的 latest-wins 语义会采纳后者）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_runtime.experience import Store
from adaptive_runtime.ledger import Ledger

FP = "AGENT_MAX_ITERATIONS_CODE|60->80|COMPLETENESS_FAILURE"
NOTE = ("作废：该行由 `round --fake --case good` 写入（dQ=38.1 是 fake 画像常量，"
        "不是真实测量）。cmd_round 缺 fake 守卫，已修复。台账/经验库均不得采信。")

store = Store()
ok_refute = store.refute("exp_6c31b69d", NOTE)
print("refute experience:", ok_refute)

led = Ledger()
led.update(FP, "inconclusive", {"reason": "void_fake_row"}, notes=NOTE)
print("ledger appended")

# 复查：latest-wins 后这两条都不再具有效力
print("experience latest status:",
      [r.get("status") for r in store.latest() if r.get("experience_id") == "exp_6c31b69d"])
row = led.latest(FP)
print("ledger latest status:", row.get("status"), "| is_duplicate:", led.is_duplicate(FP)[0])
print("retrieve(COMPLETENESS_FAILURE) ->", len(store.retrieve("COMPLETENESS_FAILURE|X|0")))
