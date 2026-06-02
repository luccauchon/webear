@echo off

start "OERH Optuna Day" cmd /K "call conda activate PY312_HT && cd ..\..\..\src\optimizers\oerh && python .\hyperparameter_search_optuna.py --sampler tpe --target-type floor --metric long_accuracy --dataset-id day --lookahead-bars 1 --min-signal-ratio 0.15 --n-trials 99999 --timeout 86400 --train-ratio 0.7 --output-dir day_perspective --storage sqlite:///day_perspective\\day_day.db --threshold-pct -0.005"
