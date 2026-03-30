@echo off
call conda activate PY312_HT
cd ..\..\..
cd src\runners
python .\MMI_hyperparameter_search_optuna.py --dataset_id week --step_back_range 9999 --n_trials 33333 ^
--use_ema true ^
--return_threshold_min 0.05 --return_threshold_max 0.05 ^
--sma_period_max 60 --mmi_period_max 60 ^
--lookahead_min 1 --lookahead_max 1 ^
--mmi_trend_max_max 100 ^
--study_name 0_05__week__la1 ^
--storage sqlite:///mmi_optuna_w500p.db
pause