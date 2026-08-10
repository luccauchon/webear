@echo off
setlocal enabledelayedexpansion

for %%i in (-0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05) do (
    start "OERH %%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct %%i --dataset-id day --lookahead-bars 20 --plot-sample 1800 --density-target 0.05 --target-type exact --cooldown-bars 4 --output-dir models\day_persective\day_perspective_buy_exact_la20_dt05_cdb4 --disable-plot-sample"
    timeout /t 2 /nobreak >nul
)
