@echo off

start "Optuna Quarter" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\prime_rsi && python .\realtime_and_backtest_hyperparameter_search_optuna.py --optimize --n-trials 99999 --timeout 80000 --dataset-id quarter --min-signal-density 0.30 --lookahead-bars 1 --train-ratio 0.7 --optuna-db quarter_quarter.db --optimize-target buy_wr --method final_close --put-strike-pct 0.94 --call-strike-pct 1."

start "Optuna Month" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\prime_rsi && python .\realtime_and_backtest_hyperparameter_search_optuna.py --optimize --n-trials 99999 --timeout 80000 --dataset-id month --min-signal-density 0.15 --lookahead-bars 4 --train-ratio 0.7 --optuna-db month_quarter.db --optimize-target buy_wr --method final_close --put-strike-pct 0.94 --call-strike-pct 1."

start "Optuna Week" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\prime_rsi && python .\realtime_and_backtest_hyperparameter_search_optuna.py --optimize --n-trials 99999 --timeout 80000 --dataset-id week --min-signal-density 0.075 --lookahead-bars 16 --train-ratio 0.7 --optuna-db week_quarter.db --optimize-target buy_wr --method final_close --put-strike-pct 0.94 --call-strike-pct 1."

start "Optuna Day" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\prime_rsi && python .\realtime_and_backtest_hyperparameter_search_optuna.py --optimize --n-trials 99999 --timeout 80000 --dataset-id day --min-signal-density 0.075 --lookahead-bars 80 --train-ratio 0.7 --optuna-db day_quarter.db --optimize-target buy_wr --method final_close --put-strike-pct 0.94 --call-strike-pct 1."
