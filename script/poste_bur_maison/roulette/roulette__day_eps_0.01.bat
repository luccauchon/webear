@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "RANGE=3333"
set "DATASET_ID=day"
set "TIMEOUT=500000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\Projets\webear\src\runners"

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_0__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_1__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_2__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_3__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq_0__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq_1__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq_2__f1 --timeout %TIMEOUT% --epsilon 0.01"
timeout /t 2 /nobreak >nul

echo All processes have been dispatched.
pause
