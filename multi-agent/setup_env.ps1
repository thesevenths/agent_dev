<#
.SYNOPSIS
    One-click create venv on D: and install deps (Tsinghua mirror + official PyPI fallback + cache on D:).
.DESCRIPTION
    - venv at D:\venvs\multi-agent (no C: usage)
    - pip cache redirected to D:\pip-cache
    - Primary mirror: Tsinghua (fast in China)
    - Extra index: official PyPI as fallback (some China mirrors lag behind,
      e.g. Aliyun was missing langchain-core >=1.6.4 at the time of writing)
    - deps from requirements.txt in project root
.NOTES
    Run:
      cd E:\agent_dev\multi-agent ; .\setup_env.ps1
    or:
      powershell -ExecutionPolicy Bypass -File E:\agent_dev\multi-agent\setup_env.ps1
#>

$EnvRoot     = "D:\venvs\multi-agent"
$PipCacheDir = "D:\pip-cache"
# Primary mirror: Tsinghua (fast in China)
$PipIndex    = "https://pypi.tuna.tsinghua.edu.cn/simple"
# Fallback: official PyPI (covers any mirror lag)
$PipExtra    = "https://pypi.org/simple"

$ProjectRoot  = $PSScriptRoot
$Requirements = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $Requirements)) {
    Write-Error "requirements.txt not found: $Requirements"
    exit 1
}

if (-not (Test-Path $EnvRoot)) {
    New-Item -ItemType Directory -Force -Path $EnvRoot | Out-Null
}

$VenvPython = Join-Path $EnvRoot "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/4] Creating venv at $EnvRoot ..." -ForegroundColor Cyan
    & python -m venv $EnvRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "venv create failed (conda python may miss ensurepip). Fallback: conda create --prefix $EnvRoot python=3.13 -y"
        exit 1
    }
} else {
    Write-Host "[1/4] venv already exists, skip: $EnvRoot" -ForegroundColor Yellow
}

# [PYTHONUTF8] Force Python global default UTF-8 (PEP 540) to fix the 'gbk'
# UnicodeDecodeError when langgraph_api reads openapi.json / logging.json on Windows.
# This survives reinstalls / venv recreation. Inserted right after the VIRTUAL_ENV
# assignment in the venv activate scripts so it always executes. Idempotent.
Write-Host "[+PYTHONUTF8] Injecting UTF-8 mode into venv activate scripts..." -ForegroundColor Cyan
$ActBat = Join-Path $EnvRoot "Scripts\activate.bat"
$ActPs1 = Join-Path $EnvRoot "Scripts\Activate.ps1"

if (Test-Path $ActBat) {
    $batLines = [System.Collections.ArrayList]@(Get-Content $ActBat)
    if (-not ($batLines -match "PYTHONUTF8")) {
        $idx = -1
        for ($i = 0; $i -lt $batLines.Count; $i++) {
            if ($batLines[$i].Trim() -match '^@?set\s+"?VIRTUAL_ENV=') { $idx = $i; break }
        }
        if ($idx -ge 0) {
            $ins = @("", "rem Force Python to use UTF-8 as default encoding (PEP 540). Fixes 'gbk' UnicodeDecodeError", "rem when langgraph_api reads openapi.json / logging.json on Windows. Survives reinstalls.", 'set "PYTHONUTF8=1"')
            $batLines.InsertRange($idx + 1, $ins)
            Set-Content $ActBat $batLines -Encoding ASCII
        }
    }
}
if (Test-Path $ActPs1) {
    $ps1Lines = [System.Collections.ArrayList]@(Get-Content $ActPs1)
    if (-not ($ps1Lines -match "PYTHONUTF8")) {
        $idx = -1
        for ($i = 0; $i -lt $ps1Lines.Count; $i++) {
            if ($ps1Lines[$i].Trim() -eq '$env:VIRTUAL_ENV_PROMPT = $Prompt') { $idx = $i; break }
        }
        if ($idx -ge 0) {
            $ins = @("", "# Force Python global default UTF-8 (PEP 540); fixes 'gbk' UnicodeDecodeError", "# when langgraph_api reads openapi.json / logging.json on Windows. Survives reinstalls.", '$env:PYTHONUTF8 = "1"')
            $ps1Lines.InsertRange($idx + 1, $ins)
            Set-Content $ActPs1 $ps1Lines -Encoding UTF8
        }
    }
}

Write-Host "[2/4] Activating venv and configuring pip (mirror + cache)..." -ForegroundColor Cyan
. "$EnvRoot\Scripts\activate.ps1"

& pip config set global.index-url $PipIndex
& pip config set global.extra-index-url $PipExtra
& pip config set global.cache-dir $PipCacheDir
& python -m pip install --upgrade pip

Write-Host "[3/4] Installing dependencies (may take a while)..." -ForegroundColor Cyan
& python -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency install failed, see errors above."
    exit 1
}

Write-Host "[4/4] Smoke test: import agent module..." -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    & python -c "import agent; print('agent module import OK')"
} finally {
    Pop-Location
}

Write-Host "`nDone! Activate with:" -ForegroundColor Green
Write-Host "  $EnvRoot\Scripts\activate"
Write-Host "Or just run (sets UTF-8 + uses venv python, no activation needed):" -ForegroundColor Green
Write-Host "  .\langgraph-dev.ps1"


