@echo off

start "AutoTune Optuna Day-Day V2" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\autotune && python .\realtime_and_backtest_hyperparameter_search_optuna.py --dataset-id day --lookahead-bars 1 --win-threshold 0.0025 --min-signal-density 0.01 --train-ratio 0.9 --optimize hold_floor --n-trials 99999 --timeout 86400 --output-dir day_perspective_v2 --storage sqlite:///day_perspective_v2\\day_day.db --study-name dayla1"
