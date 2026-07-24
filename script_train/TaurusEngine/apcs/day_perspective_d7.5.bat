@echo off
setlocal enabledelayedexpansion

:: Liste des valeurs alternées : sell1 buy1 sell2 buy2... (séparées uniquement par des espaces)
set "params=1.000 1.000 1.001 0.999 1.002 0.998 1.003 0.997 1.004 0.996 1.005 0.995 1.006 0.994 1.007 0.993 1.008 0.992 1.009 0.991 1.010 0.990 1.011 0.989 1.012 0.988 1.013 0.987 1.014 0.986 1.015 0.985"

:loop
:: Récupère les deux premiers éléments de la liste
for /f "tokens=1,2" %%a in ("%params%") do (
    set "sell=%%a"
    set "buy=%%b"
    
    echo Lancement avec --sell-offset !sell! --buy-offset !buy!
    
    start "APCS Optuna Day-Day MD0.075" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\apcs && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id day --sell-offset !sell! --buy-offset !buy! --lookahead 1 --n-trials 99999 --timeout 1200 --min-density-threshold 0.075 --output-dir models\day_persective_d7.5 --verbose --verbose-list-trades"
    
    timeout /t 2 /nobreak >nul
)

:: Supprime les deux éléments traités de la chaîne de texte
for /f "tokens=1,2*" %%a in ("%params%") do (
    set "params=%%c"
)

:: Recommence tant qu'il reste des variables dans la liste
if defined params goto loop

endlocal
