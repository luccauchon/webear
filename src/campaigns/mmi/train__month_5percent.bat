@echo off
call conda activate PY312_HT
D:
cd D:\PyCharmProjects\webear\src\runners
python .\MMI_hyperparameter_search_optuna.py --dataset_id month --step_back_range 9999 --n_trials 33333 --use_ema true --return_threshold_min 0.05 --return_threshold_max 0.05 --sma_period_max 20 --mmi_period_max 20 --lookahead_min 1 --lookahead_max 1 --mmi_trend_max_max 100 --study_name 0_05__month__la1
pause