@echo off
setlocal enabledelayedexpansion

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 499 --timeout 3600 --dataset-id intraday_1min --lookahead 6 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\1min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 499 --timeout 3600 --dataset-id intraday_1min --lookahead 12 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\1min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 499 --timeout 3600 --dataset-id intraday_1min --lookahead 24 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\1min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 499 --timeout 3600 --dataset-id intraday_1min --lookahead 36 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\1min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 499 --timeout 3600 --dataset-id intraday_1min --lookahead 48 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\1min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 499 --timeout 3600 --dataset-id intraday_1min --lookahead 60 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\1min"
timeout /t 2 /nobreak >nul

