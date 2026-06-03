@echo off

start "DGDR Optuna Day-Day" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\dgdr && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id day --lookahead-bars 1 --method final_close --min-signal-density 0.075  --put-strike-pct 0.995 --call-strike-pct 1. --n-trials 99999 --timeout 86400 --train-ratio 0.7 --signal-type buy --output-dir day_perspective --optuna-storage sqlite:///day_perspective\\day_day.db --optuna-study-name day_1la"
