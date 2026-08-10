@echo off
setlocal enabledelayedexpansion

start "OERH" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py  --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct 0. --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.05 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
timeout /t 2 /nobreak >nul

start "OERH" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py  --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct 0.01 --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.05 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
timeout /t 2 /nobreak >nul

start "OERH" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py  --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct 0.02 --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.05 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
timeout /t 2 /nobreak >nul

start "OERH" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py  --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct 0.03 --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.05 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
timeout /t 2 /nobreak >nul

start "OERH" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py  --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct 0.04 --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.05 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
timeout /t 2 /nobreak >nul

start "OERH" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\realtime_and_backtest_hyperparameter_search_optuna.py  --n-trials 99999 --optimize-target buy --timeout 80000 --threshold-pct 0.05 --dataset-id day --lookahead-bars 10 --plot-sample 1800 --density-target 0.05 --target-type any_half_B --cooldown-bars 5 --output-dir models\day_persective\ --disable-plot-sample"
timeout /t 2 /nobreak >nul
