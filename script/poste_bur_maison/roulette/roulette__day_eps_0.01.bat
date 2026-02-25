@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "RANGE=3333"
set "DATASET_ID=day"
set "EPSILON=0.01"
set "TIMEOUT=500000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\Projets\webear\src\optimizers"

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_0__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 1

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_1__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 2

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_2__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 3

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq_3__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 4

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq_0__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 5

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq_1__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 5

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq_2__f1 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 5

echo All processes have been dispatched.
pause
