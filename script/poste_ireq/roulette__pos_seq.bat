@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "TIMEOUT=400000"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\PYCHARMPROJECTS\webear\src\runners"

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target pos_seq_0__f1 --timeout %TIMEOUT%"
timeout /t 1
start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target pos_seq_1__f1 --timeout %TIMEOUT%"
timeout /t 1
start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target pos_seq_2__f1 --timeout %TIMEOUT%"
timeout /t 1
start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target pos_seq_3__f1 --timeout %TIMEOUT%"
timeout /t 1


start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target neg_seq_0__f1 --timeout %TIMEOUT%"
timeout /t 1
start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target neg_seq_1__f1 --timeout %TIMEOUT%"
timeout /t 1
start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --optimize_target neg_seq_2__f1 --timeout %TIMEOUT%"
timeout /t 1

pause
