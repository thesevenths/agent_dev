# langgraph-dev.ps1 —— 一键启动 langgraph dev（带 PYTHONUTF8=1 根治 Windows GBK 编码崩溃）
#
# 用法（在 PowerShell 里，项目根目录）：
#   .\langgraph-dev.ps1              # 默认：热重载已关闭（--no-reload），跑 agent 最稳
#   .\langgraph-dev.ps1 -Watch       # 开启热重载（仅"纯改代码、不跑 agent"时用）
#   .\langgraph-dev.ps1 --port 2025  # 追加任意 langgraph dev 参数
#
# 为什么需要它：
#   langgraph dev 经 langgraph.exe 控制台启动器拉起 Python 子进程，该子进程默认继承
#   系统 ANSI 代码页（中文 Windows = GBK）。langgraph_api 自带的 openapi.json /
#   logging.json 是 UTF-8，其 open() 未指定编码 -> 'gbk' UnicodeDecodeError 启动即崩。
#   设 PYTHONUTF8=1（PEP 540）让 Python 全局默认 UTF-8，覆盖所有未写编码的 open()，
#   且写在 env 里、重装 langgraph-api 也不丢（比直接 patch site-packages 更持久）。
#
# 为什么默认关闭热重载（--no-reload）：
#   langgraph dev 默认 reload=True，uvicorn 会递归监视项目目录下所有 *.py。而 code_agent
#   的本职工作就是把生成的分析脚本 .py 写进被监视的 tmp/（create_file + 多次 str_replace 重写），
#   每次写入都会触发热重载 -> 进程重启 -> 正在跑的 run 被杀 -> CancelledError +
#   "cannot schedule new futures after interpreter shutdown"。这就是为什么偏偏每次都是 code_agent
#   报错：crawler 写 .json/.md、chat 写 .md，只有 code_agent 写 .py。跑 agent 期间从不需要热重载，
#   故默认关闭；确需边改代码边自动重载时，用 -Watch 显式开启（此时不要同时跑 agent）。
#
# 排障备忘（2026-10-03，一次真实事故）：
#   现象：控制台只停在 "Worker stats ... active=0" 心跳，http://127.0.0.1:2024 连不上、
#        浏览器不自动弹，看着像"卡住"。
#   真因：tools.py 模块顶层执行 `Base.metadata.create_all(engine)`，而 PG
#        (192.168.50.75:5432) 不可达 -> psycopg 卡到 OS TCP 超时(133s) -> import tools 阻塞
#        -> graph 加载 GraphLoadError -> "Application startup failed. Exiting."
#        -> uvicorn 从未绑定 2024，langgraph 的 _open_browser 线程死等 /ok 而静默。
#   结论：这【不是】langgraph 或浏览器的问题，是业务代码在 import 期连 DB。已修：
#        tools.py 给 engine 加 connect_args={"connect_timeout":5} + create_all 包 try/except。

param(
    # 加 -Watch 开启热重载（仅在"纯改代码、不跑 agent"时用）；默认关闭，见上方说明。
    [switch]$Watch,
    # 加 -NoBrowser：不自动弹浏览器（无桌面/远程服务器时用）。默认会自动弹，见下方说明。
    [switch]$NoBrowser,
    # 加 -Bg：把服务丢到独立窗口后台跑，当前 PowerShell 立即释放（默认前台阻塞）。
    [switch]$Bg
)

$ErrorActionPreference = "Stop"

# 切到脚本所在目录（langgraph.json / .env 在此）
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

# 关键：全局默认 UTF-8，子进程（langgraph.exe -> python） inherited
$env:PYTHONUTF8 = "1"

# 组装 langgraph dev 参数
$devArgs = @("dev")
$callerArgs = @($args)
if (-not $Watch -and ($callerArgs -notcontains "--no-reload")) {
    $devArgs += "--no-reload"
}
# 浏览器自动打开（2026-10-03 勘误）：
#   langgraph dev **默认就会开浏览器**，开关只有一个否定式 `--no-browser`（langgraph_cli/cli.py:692，
#   Click is_flag 默认 False -> 第 844 行 open_browser=not no_browser=True -> langgraph_api/cli.py:373
#   起 _open_browser 线程）。本版本【不存在】--open-browser 选项，误加会让 langgraph dev 直接报
#   "No such option '--open-browser'. Did you mean '--no-browser'?" 并退出。
#   故这里只在 -NoBrowser 时追加 --no-browser 关闭自动打开。
if ($NoBrowser -and ($callerArgs -notcontains "--no-browser")) {
    $devArgs += "--no-browser"
}
$devArgs += $callerArgs

# 优先用 venv 里的 langgraph；venv 不存在时退化为 PATH 上的
$venvLanggraph = "D:\venvs\multi-agent\Scripts\langgraph.exe"
if (-not (Test-Path -LiteralPath $venvLanggraph)) {
    $venvLanggraph = (Get-Command langgraph -ErrorAction SilentlyContinue).Source
    if (-not $venvLanggraph) {
        throw "找不到 langgraph.exe：D:\venvs\multi-agent\Scripts\langgraph.exe 不存在且 PATH 上也没有 langgraph 命令。"
    }
}

if ($Bg) {
    # 后台模式：服务独立窗口跑，日志落盘，当前 shell 立刻还给你（默认不占 PowerShell）
    if (-not (Test-Path -LiteralPath .\log)) {
        New-Item -ItemType Directory -Path .\log -Force | Out-Null
    }
    $logFile = "F:\agent\multi-agent\log\dev_stdout.log"
    Write-Host "后台启动：$(Split-Path $venvLanggraph -Leaf) $($devArgs -join ' ')" -ForegroundColor Cyan
    $inner = "`$env:PYTHONUTF8='1'; Set-Location 'F:\agent\multi-agent'; & '" + $venvLanggraph + "' " + ($devArgs -join ' ') + " *>&1 | Tee-Object -FilePath '" + $logFile + "'"
    try {
        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-Command", $inner -WindowStyle Minimized -ErrorAction Stop
    } catch {
        Write-Host "自动开窗口失败：$($_.Exception.Message)" -ForegroundColor Red
        Write-Host "请手动另开一个 PowerShell 窗口，执行：" -ForegroundColor Yellow
        Write-Host "  cd F:\agent\multi-agent; .\langgraph-dev.ps1" -ForegroundColor Yellow
        exit 1
    }
    # 轮询探测：图加载含 DB 连接，首次可能数秒，给足 45s
    $ok = $false
    for ($i = 1; $i -le 15; $i++) {
        Start-Sleep -Seconds 3
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:2024/ok" -TimeoutSec 4 -UseBasicParsing
            Write-Host "服务已就绪（约 $($i * 3)s）：HTTP $([int]$r.StatusCode)  $($r.Content)" -ForegroundColor Green
            Write-Host "本机兜底页（不依赖外网，Smith 打不开时用）： http://127.0.0.1:2024/docs" -ForegroundColor Yellow
            $ok = $true
            break
        } catch { }
    }
    if (-not $ok) {
        Write-Host "45s 内 2024 端口仍未监听 -> 启动失败。日志末尾：" -ForegroundColor Red
        if (Test-Path -LiteralPath $logFile) {
            Get-Content -LiteralPath $logFile -Tail 12 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
        }
    }
    exit 0
}

& $venvLanggraph @devArgs
