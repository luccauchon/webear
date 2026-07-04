@echo off
setlocal enabledelayedexpansion

:: Boucle sur les valeurs de densité de signal
for %%D in (0.05 0.125 0.25 0.5) do (

    :: Extraction propre pour le nom du fichier DB (ex: 0.05 devient 005, 0.1 devient 01)
    set "DENS_SUF=%%D"
    set "DENS_SUF=!DENS_SUF:.=!"

    :: Boucle sur les valeurs de put-strike-pct
    for %%P in (1.005 1.000 0.999 0.995 0.99 0.985 0.98 0.97) do (

        :: Extraction propre pour le nom du fichier DB (ex: 0.99 devient 099)
        set "STRK_SUF=%%P"
        set "STRK_SUF=!STRK_SUF:.=!"

        start "DGDR Optuna Week-Week" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\optimizers\dgdr && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id week --lookahead-bars 1 --method final_close --min-signal-density %%D --put-strike-pct %%P --call-strike-pct 1. --n-trials 99999 --timeout 3600 --train-ratio 0.90 --signal-type buy --output-dir week_perspective --optuna-storage sqlite:///week_perspective\\week_week_buy_!STRK_SUF!_dens!DENS_SUF!.db --optuna-study-name doesnotmatter"
    )
)
