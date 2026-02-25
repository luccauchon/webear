@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "OPTIMIZE_TARGET=pos_seq_0__f1"
set "OBJECTIVE_NAME=2026_02_20__1pct"
set "TIMEOUT=400000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\Projets\webear\src\runners"

start "Roulette Search" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python roulette_hyperparameter_search_optuna.py --step_back_range 333 --dataset_id %DATASET_ID% --optimize_target pos_seq_0__f1 --timeout %TIMEOUT%"

echo All processes have been dispatched.
pause
