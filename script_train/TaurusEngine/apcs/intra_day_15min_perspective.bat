@echo off
setlocal enabledelayedexpansion

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 1 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 2 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 3 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 4 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 5 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 6 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 7 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

start "APCS Optuna Intraday" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold 0.0125 --n-trials 32499 --timeout 43600 --dataset-id intraday_15min --lookahead 8 --trade-direction both --verbose-list-trades --output-dir models\intraday_persective\15min"
timeout /t 2 /nobreak >nul

