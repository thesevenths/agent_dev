"""几个 (limit, n, rounds) 组合的墙钟预估：直接复用 CLI 用的同一套估算函数。

不重复实现公式，避免「脚本里算的」和「命令行报的」两个数不一致。
"""
import io
import os
import sys

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_runtime import bench_tasks as bt
from adaptive_runtime.settings import CFG, PATHS
from adaptive_runtime import benchmark as bm

CFG.offline = True
CFG.max_parallel = 4
prof = bt.profile_for(offline=True, task_profile=CFG.task_profile)
print(f"画像：{prof.name} {prof.seconds_per_run:.0f}s/run  并发={CFG.max_parallel}")

for limit, n, rounds in [(18, 3, 2), (12, 3, 3), (18, 2, 2), (14, 3, 2), (18, 3, 3), (18, 3, 4)]:
    tasks = [{"task_id": t["task_id"], "query": t["query"]}
             for t in __import__("json").load(io.open(str(PATHS.tasks / "offline_local.json"), encoding="utf-8"))["tasks"]][:limit]
    tr, ho = bm.split(tasks, 0.3)
    plan = bt.plan_cost(len(tr), n, rounds, len(ho), profile=prof)
    extra = len(ho) * CFG.holdout_reps * 2 * rounds
    runs = plan.total_runs + extra
    hours = runs * prof.seconds_per_run / 3600.0 / CFG.max_parallel
    # δ：t 检验吃的是「任务间」差值，所以自由度 = train 任务数
    from adaptive_runtime.stats import t_crit
    sd_task = 14.9922 / max(1.0, n ** 0.5)
    delta = t_crit(max(1, len(tr) - 1), CFG.alpha) * sd_task / max(1, len(tr) ** 0.5)
    print(f"limit={limit} n={n} rounds={rounds} -> train={len(tr)} holdout={len(ho)} "
          f"runs={runs} hours={hours:.2f} delta_needed={delta:.2f}")
