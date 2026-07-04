@echo off
setlocal enabledelayedexpansion

:: Boucle externe de 1 à 20 pour lookahead-bars
for /L %%L in (1,1,20) do (

    echo ========================================================
    echo Lancement du groupe complet pour lookahead-bars = %%L
    echo ========================================================

    :: Lancement des 30 processus en parallèle
    for %%D in (0.050 0.100 0.150 0.200 0.250 0.300) do (
        set "DENS_SUF=%%D"
        set "DENS_SUF=!DENS_SUF:.=!"

        for %%P in (0.999 0.995 0.990 0.9875 0.985) do (
            set "STRK_SUF=%%P"
            set "STRK_SUF=!STRK_SUF:.=!"

            start "DGDR_OPTUNA_L%%L" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\dgdr && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id day --lookahead-bars %%L --method final_close --min-signal-density %%D --put-strike-pct %%P --call-strike-pct 1. --n-trials 99999 --timeout 3600 --train-ratio 0.90 --signal-type buy --output-dir day_perspective --optuna-storage sqlite:///day_perspective\\day_day_buy_!STRK_SUF!_dens!DENS_SUF!_lookahead%%L.db --optuna-study-name doesnotmatter"
        )
    )

    :: Attente de 3700 secondes (1 heure) avant le lookahead suivant
    echo Les 30 processus tournent. Pause de 1 heure...
    timeout /t 3700 /nobreak

    echo Le temps imparti pour lookahead-bars = %%L est ecoule.
    echo.
)

echo Tous les lookahead-bars de 1 a 20 sont termines.
pause
