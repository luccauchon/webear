@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "RANGE=999"
set "DATASET_ID=day"
set "TIMEOUT=400000"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\PYCHARMPROJECTS\webear\src\optimizers"

start "Roulette Search %DATASET_ID%" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --specific_wanted_class 0 1 2 3 --optimize_target pos_seq__f1 --timeout %TIMEOUT%"
timeout /t 1

start "Roulette Search %DATASET_ID%" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range %RANGE% --dataset_id %DATASET_ID% --specific_wanted_class 0 1 2 --optimize_target neg_seq__f1 --timeout %TIMEOUT%"
timeout /t 1

pause
