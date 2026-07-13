@echo off

start "AutoTune Optuna Day-Day" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\autotune && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id day --lookahead-bars 1 --win-threshold 0.005 --min-signal-density 0.075 --train-ratio 0.9 --optimize hold_floor --n-trials 99999 --timeout 86400 --output-dir day_perspective_d75 --storage sqlite:///day_perspective_d75\\day_day.db --study-name dayla1"
