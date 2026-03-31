@echo off
call conda activate PY312_HT
cd ..\..\..
cd src\runners
python .\VIX_hyperparameter_search_optuna.py --objective-name 2026_02_20__1pct_balanced ^
--optimize-target call ^
--timeout 200000 ^
--step-back-range 99999 ^
--look-ahead 1 ^
--storage 1day_call_vix.db ^
--study-name sn_1day_call_vix
pause