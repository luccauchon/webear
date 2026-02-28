@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "RANGE=666"
set "DATASET_ID=day"
set "EPSILON=0."
set "TIMEOUT=40000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\Projets\webear\src\optimizers"

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target pos_seq__f1 --specific_wanted_class 0 1 2 3 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 1

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --optimize_target neg_seq__f1 --specific_wanted_class 0 1 2 --timeout %TIMEOUT% --epsilon %EPSILON%"
sleep 2

echo All processes have been dispatched.
pause
