"""复现 probe.run_one 的子进程调用，抓出 runner_once 的真实报错。"""
import os, sys, json, subprocess, tempfile
sys.path.insert(0, ".")
from adaptive_runtime.humaneval import load_tasks

t = load_tasks(n=1)[0]
out = tempfile.mkdtemp(prefix="rrsi_repro_")
cmd = [sys.executable, "-m", "adaptive_runtime.runner_once",
       "--query", t["query"], "--task-id", t["task_id"], "--rep", "0",
       "--arm", "base", "--out", out,
       "--benchmark", "humaneval", "--test", t["test"], "--entry-point", t["entry_point"]]
env = os.environ.copy()
env.update({
    "ADAPTIVE_OFFLINE": "1",
    "ADAPTIVE_TASK_PROFILE": "research",
    "PYTHONPATH": os.path.abspath("."),
    "PYTHONIOENCODING": "utf-8",
    "AGENT_CHECKPOINT_DB": os.path.join(out, "memory", "checkpoints.sqlite"),
    "ADAPTIVE_TOKEN_SINK": os.path.join(out, "tokens.jsonl"),
})
print("CMD:", " ".join(cmd[:6]), "...")
print("PYTHONPATH:", env["PYTHONPATH"][:80])
try:
    proc = subprocess.run(cmd, cwd=os.path.abspath("."), env=env,
                          capture_output=True, text=True, timeout=300,
                          encoding="utf-8", errors="replace")
    print("RC:", proc.returncode)
    print("=== STDOUT (tail) ===")
    print((proc.stdout or "")[-2000:])
    print("=== STDERR (tail) ===")
    print((proc.stderr or "")[-2000:])
except subprocess.TimeoutExpired as e:
    print("TIMEOUT")
    print("STDOUT:", (e.stdout or "")[-1500:])
    print("STDERR:", (e.stderr or "")[-1500:])
except Exception as e:
    print("LAUNCH FAIL:", type(e).__name__, e)
