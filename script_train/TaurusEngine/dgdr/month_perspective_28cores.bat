@echo off
setlocal enabledelayedexpansion

:: Boucle sur les valeurs de densité de signal
for %%D in (0.1 0.5) do (

    :: Extraction propre pour le nom du fichier DB (ex: 0.05 devient 005, 0.1 devient 01)
    set "DENS_SUF=%%D"
    set "DENS_SUF=!DENS_SUF:.=!"

    :: Boucle sur les valeurs de put-strike-pct
    for %%P in (1.015 1.01 1.005 1.001 1. 0.999 0.995 0.99 0.985 0.98 0.975 0.97 0.96 0.95) do (

        :: Extraction propre pour le nom du fichier DB (ex: 0.99 devient 099)
        set "STRK_SUF=%%P"
        set "STRK_SUF=!STRK_SUF:.=!"

        start "DGDR Optuna Month-Month" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\dgdr && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id month --lookahead-bars 1 --method final_close --min-signal-density %%D --put-strike-pct %%P --call-strike-pct 1. --n-trials 99999 --timeout 3600 --train-ratio 0.90 --signal-type buy --output-dir month_perspective --optuna-storage sqlite:///month_perspective\\month_month_buy_!STRK_SUF!_dens!DENS_SUF!.db --optuna-study-name doesnotmatter"
    )
)
