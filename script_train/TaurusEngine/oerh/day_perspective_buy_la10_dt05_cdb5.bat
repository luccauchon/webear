@echo off
setlocal enabledelayedexpansion

for %%i in (-0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05) do (
    start "OERH %%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct %%i --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.0125 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
    timeout /t 2 /nobreak >nul
)
