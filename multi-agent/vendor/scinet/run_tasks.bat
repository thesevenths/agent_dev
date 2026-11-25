@echo off
:: NOTE: keep this file ASCII-only (no Chinese, no chcp 65001).
:: cmd reads .bat by byte offset; UTF-8 Chinese + codepage switch can
:: swallow characters of the next line (e.g. "python" -> "hon").
:: To skip a finished step, just comment its line out with ::

echo.
echo ==========================================
echo [1/3] pendulum
echo ==========================================
python train.py    --config configs/pendulum.yaml              || exit /b 1
python analyze.py  --config configs/pendulum.yaml              || exit /b 1
python symbolic.py --config configs/pendulum.yaml              || exit /b 1
python evaluate.py --config configs/pendulum.yaml              || exit /b 1
python tta.py      --config configs/pendulum.yaml --noise-std 0.05 || exit /b 1

echo.
echo ==========================================
echo [2/3] newton
echo ==========================================
python train.py    --config configs/newton.yaml                || exit /b 1
python analyze.py  --config configs/newton.yaml                || exit /b 1
python symbolic.py --config configs/newton.yaml                || exit /b 1
python evaluate.py --config configs/newton.yaml                || exit /b 1
python tta.py      --config configs/newton.yaml --noise-std 0.05 || exit /b 1

echo.
echo ==========================================
echo [3/3] double_pendulum
echo ==========================================
python train.py    --config configs/double_pendulum.yaml              || exit /b 1
python analyze.py  --config configs/double_pendulum.yaml              || exit /b 1
python symbolic.py --config configs/double_pendulum.yaml              || exit /b 1
python evaluate.py --config configs/double_pendulum.yaml              || exit /b 1
python tta.py      --config configs/double_pendulum.yaml --noise-std 0.05 || exit /b 1

echo.
echo ==========================================
echo All tasks completed!
echo ==========================================
pause
