@echo off
setlocal enabledelayedexpansion

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 83600 --timeout 83600 --dataset-id intraday_60min --lookahead 1 --trade-direction both --delta 0. --verbose-list-trades --output-dir models\intraday_persective\60min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 83600 --timeout 83600 --dataset-id intraday_60min --lookahead 2 --trade-direction both --delta 0. --verbose-list-trades --output-dir models\intraday_persective\60min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 83600 --timeout 83600 --dataset-id intraday_60min --lookahead 4 --trade-direction both --delta 0. --verbose-list-trades --output-dir models\intraday_persective\60min"
timeout /t 2 /nobreak >nul

