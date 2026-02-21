@echo off
setlocal enabledelayedexpansion

:: --- Parameters ---
set "OPTIMIZE_TARGET=call"
set "OBJECTIVE_NAME=2026_02_20__1pct"
set "TIMEOUT=400000"
set "CONDA_ENV=PY311"
set "WORK_DIR=C:\PYCHARMPROJECTS\webear\src\runners"

:: --- Loop from 1 to 20 ---
for /l %%i in (1, 1, 20) do (

    echo Launching search with look_ahead=%%i...

    REM Start a new window, call conda, change directory, and run python
    start "VIX Search %%i" cmd /k "call conda activate %CONDA_ENV% && cd /D "%WORK_DIR%" && python VIX_hyperparameter_search_optuna.py --optimize_target %OPTIMIZE_TARGET% --objective_name %OBJECTIVE_NAME% --timeout %TIMEOUT% --look_ahead %%i"

    REM Calculate sleep time: current index
    set /a SLEEP_TIME=%%i

    REM Only run timeout if SLEEP_TIME is greater than 0
    if !SLEEP_TIME! GTR 0 (
        echo Waiting for !SLEEP_TIME! seconds...
        timeout /t !SLEEP_TIME! /nobreak >nul
    )
)

echo.
echo All 20 processes have been dispatched.
pause
