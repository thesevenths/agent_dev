# HumanEval 正路径驱动（免 Docker，证明 Capability A：引擎机械跑通
# failure → proposal → candidate → accept → promote → land → H1 → holdout）
#
# 为什么这是「正路径」而非「研究结论」：HumanEval 只证明闭环能自己换基线并落地；
# 它**不能**单独证明 RRSI research claim（需 OOD：MBPP / APEX）。本脚本只负责把 A 跑通。
#
# 退出码即判据：cmd_auto 返回 0 = 9 条硬条件全绿（RRSI 达成），1 = 未达成。
# 所以这里 0 就停，不为「多跑几轮」而继续烧墙钟。
#
# 关键环境变量（缺一不可）：
#   ADAPTIVE_MAX_EDIT_LEVEL=3  —— 必须开。否则 B1-B4 结构性候选进不了候选池，
#                                  force_dominant=PLANNING_FAILURE 拉不出候选 → 循环 SKIP。
#   ADAPTIVE_AUTO_PROMOTE=1    —— 不许有人工签字（Capability A 的「无人」由它保证）。
#   ADAPTIVE_OFFLINE=1         —— 不碰 Tavily 额度、降噪。

$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:ADAPTIVE_OFFLINE = "1"          # 不碰 Tavily 额度、降噪
$env:ADAPTIVE_TASK_PROFILE = "research"
$env:ADAPTIVE_AUTO_PROMOTE = "1"     # 不许有人工签字
$env:ADAPTIVE_MAX_EDIT_LEVEL = "3"   # L3：必须开，否则 B1-B4 结构性候选进不了候选池
$env:ADAPTIVE_COST_GOVERNED = "0"    # 自托管 LLM：token 不是稀缺资源
$env:ADAPTIVE_MAX_PARALLEL = "4"     # Windows 提交内存上限实测出来的并发

$py = "D:\venvs\multi-agent\Scripts\python.exe"
Set-Location "e:\agent_dev\multi-agent"
$log = "e:\agent_dev\multi-agent\experiments\_he.log"

# 0) 一次性拉数据（需联网；已存在则跳过）
if (-not (Test-Path "experiments/humaneval/HumanEval.jsonl")) {
    "=== fetch HumanEval（需联网） $(Get-Date -Format o) ===" | Out-File -FilePath $log -Append -Encoding utf8
    & $py experiments/fetch_humaneval.py 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
}

# 1) Frozen Core（生产代码指纹；之后改了生产代码需重跑 freeze）
& $py -m adaptive_runtime.run_adaptive freeze 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

# 2) 校准（若已有 calibration.json 可跳过；门限 sd(Δ) 来自实测，是 noise_calibrated 的硬前提）
if (-not (Test-Path "experiments/calibration/calibration.json")) {
    "=== calibrate（无标定则补一次；已有则跳过） $(Get-Date -Format o) ===" | Out-File -FilePath $log -Append -Encoding utf8
    & $py -m adaptive_runtime.run_adaptive calibrate --offline --limit 8 --k 4 --yes --publish 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
}

# 3) 正路径：auto + HumanEval。一轮一批，达成即停（退出码 0）。
for ($k = 1; $k -le 6; $k++) {
    "=== batch: --rounds $k  $(Get-Date -Format o) ===" | Out-File -FilePath $log -Append -Encoding utf8
    $autoArgs = @(
        "auto", "--benchmark", "humaneval", "--offline",
        "--limit", "18", "--n", "6",
        "--rounds", $k, "--parallel", "4", "--task-label", "he", "--yes"
    )
    if ($k -eq 1) { $autoArgs += "--no-resume" }
    & $py -u -m adaptive_runtime.run_adaptive @autoArgs 2>&1 |
        Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "=== rounds=$k exit=$code  $(Get-Date -Format o) ===" | Out-File -FilePath $log -Append -Encoding utf8
    if ($code -eq 0) {
        "=== RRSI 已达成，停止续跑 ===" | Out-File -FilePath $log -Append -Encoding utf8
        break
    }
}
"DONE" | Out-File -FilePath $log -Append -Encoding utf8

