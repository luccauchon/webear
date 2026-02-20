@echo off

:: Loop from 1 to 20 with a step of 1
for /l %%i in (1, 1, 20) do (

    echo Launching search with look_ahead=%%i...

    REM 'start' opens a new window
    REM 'cmd /k' keeps the window open after the script finishes so you can see results
    REM Use 'cmd /c' instead of '/k' if you want the windows to close automatically upon completion

    REM start "VIX Search - LookAhead %%i" cmd /k "call conda activate PY312_HT && cd /D C:\Projets\webear\src\runners && python VIX_hyperparameter_search_optuna.py --optimize_target put --objective_name 2026_02_20_iron_condor_1pct --timeout 400000 --look_ahead %%i"
    start "VIX Search - LookAhead %%i" cmd /k "call conda activate PY311"
)

pause