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

param(
    # 加 -Watch 开启热重载（仅在"纯改代码、不跑 agent"时用）；默认关闭，见上方说明。
    [switch]$Watch
)

$ErrorActionPreference = "Stop"

# 切到脚本所在目录（langgraph.json / .env 在此）
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

# 关键：全局默认 UTF-8，子进程（langgraph.exe -> python） inherited
$env:PYTHONUTF8 = "1"

# 组装 langgraph dev 参数：默认追加 --no-reload（除非 -Watch，或调用方已显式传入 --no-reload）
$devArgs = @("dev")
$callerArgs = @($args)
if (-not $Watch -and ($callerArgs -notcontains "--no-reload")) {
    $devArgs += "--no-reload"
}
$devArgs += $callerArgs

# 优先用 venv 里的 langgraph；venv 不存在时退化为 PATH 上的
$venvLanggraph = "D:\venvs\multi-agent\Scripts\langgraph.exe"
if (Test-Path -LiteralPath $venvLanggraph) {
    & $venvLanggraph @devArgs
} else {
    & langgraph @devArgs
}
