# 无人值守闭环驱动：一轮一次进程，跑完就落 state.json
#
# 为什么要「一轮一批」而不是一次 --rounds 6：
# auto.run_auto 只在**整批**跑完后 save_state。单批 6 小时里任何一次中断（断电、
# 需要改代码、我主动停）都会让已跑的轮次没有台账 —— 轮次产物在磁盘上但 state 不存在，
# 下次只能从头再跑。一轮一批则每轮结束就写 rounds_done/curve/verifies，判据可从
# 任意一轮出具（--verify-only）。
#
# 退出码就是判据：cmd_auto 返回 0 = 9 条硬条件全绿（RRSI 达成），1 = 未达成。
# 所以这里 0 就停，不为「多跑几轮」而继续烧墙钟。

$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:ADAPTIVE_OFFLINE = "1"          # 需求 3：不碰 Tavily 免费额度
$env:ADAPTIVE_TASK_PROFILE = "research"
$env:ADAPTIVE_AUTO_PROMOTE = "1"     # 需求 1：不许有人工签字
$env:ADAPTIVE_MAX_EDIT_LEVEL = "3"   # L3：允许结构性候选。现已含 3 个结构性 env 旋钮 + 4 个 code 级候选（supervisor.B1~B4）
$env:ADAPTIVE_COST_GOVERNED = "0"    # 自托管 LLM：token 不是稀缺资源
$env:ADAPTIVE_MAX_PARALLEL = "4"     # Windows 提交内存上限实测出来的并发

$py = "D:\venvs\multi-agent\Scripts\python.exe"
Set-Location "e:\agent_dev\multi-agent"
$log = "e:\agent_dev\multi-agent\experiments\_rrsi3.log"

# 本轮为「强制全新跑」：仅第 1 批加 --no-resume 清掉旧 rrsi3 台账，之后批次续跑
# （保留一轮一批的增量落账安全性）。--n 3 -> 6 降噪：把检出门槛从 ~5.2 分压到 ~3.x，
# 让真实小改进也能被 t 检验检出；L3 已在上方 =3，3 个结构性 env 旋钮自动进入候选。
for ($k = 1; $k -le 6; $k++) {
    "=== batch: --rounds $k  $(Get-Date -Format o) ===" | Out-File -FilePath $log -Append -Encoding utf8
    $autoArgs = @(
        "auto", "--offline", "--limit", "18", "--n", "6",
        "--rounds", $k, "--parallel", "4", "--task-label", "rrsi3", "--yes"
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
