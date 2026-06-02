@echo off

start "Prime RSI Optuna Day-Day" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\prime_rsi && python .\realtime_and_backtest_hyperparameter_search_optuna.py --optimize --n-trials 99999 --timeout 86400 --dataset-id day --min-signal-density 0.075 --lookahead-bars 1 --train-ratio 0.7 --optuna-db day_perspective\day_day.db --optimize-target buy_wr --method final_close --output-dir day_perspective --put-strike-pct 0.995 --call-strike-pct 1."
