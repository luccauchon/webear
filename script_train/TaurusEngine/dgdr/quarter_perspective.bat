@echo off

start "DGDR Optuna Quarter-Quarter" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\dgdr && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id quarter --lookahead-bars 1 --method final_close --min-signal-density 0.25  --put-strike-pct 0.94 --call-strike-pct 1. --n-trials 99999 --timeout 86400 --train-ratio 0.7 --signal-type buy --output-dir quarter_perspective --optuna-storage sqlite:///quarter_perspective\\quarter_quarter.db --optuna-study-name quarter_1la"
