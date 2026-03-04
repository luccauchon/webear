@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "RANGE=666"
set "DATASET_ID=day"
set "EPSILON=0."
set "TIMEOUT=40000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY312"
set "WORK_DIR=C:\Projets\webear\src\optimizers"

:: --- Process 1: Positive Sequence ---
start "Roulette Search" cmd /k ^
    "call %CONDA_ACTIVATE% %CONDA_ENV% && ^
    cd /D %WORK_DIR% && ^
    python roulette_hyperparameter_search_optuna.py ^
        --step_back_range %RANGE% ^
        --dataset_id %DATASET_ID% ^
        --optimize_target pos_seq__f1 ^
        --specific_wanted_class 0 1 2 ^
        --timeout %TIMEOUT% ^
        --epsilon %EPSILON%"

timeout /t 1 >nul

:: --- Process 2: Negative Sequence ---
start "Roulette Search" cmd /k ^
    "call %CONDA_ACTIVATE% %CONDA_ENV% && ^
    cd /D %WORK_DIR% && ^
    python roulette_hyperparameter_search_optuna.py ^
        --step_back_range %RANGE% ^
        --dataset_id %DATASET_ID% ^
        --optimize_target neg_seq__f1 ^
        --specific_wanted_class 0 1 2 ^
        --timeout %TIMEOUT% ^
        --epsilon %EPSILON%"

timeout /t 2 >nul

echo All processes have been dispatched.
pause