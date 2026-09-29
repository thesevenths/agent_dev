# langgraph-dev.ps1 —— 一键启动 langgraph dev（带 PYTHONUTF8=1 根治 Windows GBK 编码崩溃）
#
# 用法（在 PowerShell 里，项目根目录）：
#   .\langgraph-dev.ps1              # 等价于 langgraph dev
#   .\langgraph-dev.ps1 --port 2025  # 追加任意 langgraph dev 参数
#
# 为什么需要它：
#   langgraph dev 经 langgraph.exe 控制台启动器拉起 Python 子进程，该子进程默认继承
#   系统 ANSI 代码页（中文 Windows = GBK）。langgraph_api 自带的 openapi.json /
#   logging.json 是 UTF-8，其 open() 未指定编码 -> 'gbk' UnicodeDecodeError 启动即崩。
#   设 PYTHONUTF8=1（PEP 540）让 Python 全局默认 UTF-8，覆盖所有未写编码的 open()，
#   且写在 env 里、重装 langgraph-api 也不丢（比直接 patch site-packages 更持久）。

$ErrorActionPreference = "Stop"

# 切到脚本所在目录（langgraph.json / .env 在此）
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

# 关键：全局默认 UTF-8，子进程（langgraph.exe -> python） inherited
$env:PYTHONUTF8 = "1"

# 优先用 venv 里的 langgraph；venv 不存在时退化为 PATH 上的
$venvLanggraph = "D:\venvs\multi-agent\Scripts\langgraph.exe"
if (Test-Path -LiteralPath $venvLanggraph) {
    & $venvLanggraph dev @args
} else {
    & langgraph dev @args
}
