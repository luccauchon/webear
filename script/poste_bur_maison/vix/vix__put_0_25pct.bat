@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "OPTIMIZE_TARGET=put"
set "OBJECTIVE_NAME=2026_02_20__0_25pct"
set "TIMEOUT=500000"
set "CONDA_ACTIVATE=C:\Users\cj3272\AppData\Local\miniconda3\Scripts\activate.bat"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\Projets\webear\src\runners"

start "VIX Search %%i" cmd /k "call "%CONDA_ACTIVATE%" && call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python VIX_hyperparameter_search_optuna.py --optimize_target %OPTIMIZE_TARGET% --objective_name %OBJECTIVE_NAME% --timeout %TIMEOUT% --look_ahead 1"
timeout /t 2 /nobreak >nul
start "VIX Search %%i" cmd /k "call "%CONDA_ACTIVATE%" && call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python VIX_hyperparameter_search_optuna.py --optimize_target %OPTIMIZE_TARGET% --objective_name %OBJECTIVE_NAME% --timeout %TIMEOUT% --look_ahead 2"
timeout /t 2 /nobreak >nul
start "VIX Search %%i" cmd /k "call "%CONDA_ACTIVATE%" && call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python VIX_hyperparameter_search_optuna.py --optimize_target %OPTIMIZE_TARGET% --objective_name %OBJECTIVE_NAME% --timeout %TIMEOUT% --look_ahead 5"
timeout /t 2 /nobreak >nul
start "VIX Search %%i" cmd /k "call "%CONDA_ACTIVATE%" && call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python VIX_hyperparameter_search_optuna.py --optimize_target %OPTIMIZE_TARGET% --objective_name %OBJECTIVE_NAME% --timeout %TIMEOUT% --look_ahead 10"
timeout /t 2 /nobreak >nul
start "VIX Search %%i" cmd /k "call "%CONDA_ACTIVATE%" && call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python VIX_hyperparameter_search_optuna.py --optimize_target %OPTIMIZE_TARGET% --objective_name %OBJECTIVE_NAME% --timeout %TIMEOUT% --look_ahead 20"
timeout /t 2 /nobreak >nul

echo.
echo All processes have been dispatched.
pause
