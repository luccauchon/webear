@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\roulette
python .\hyperparameter_search_optuna.py ^
--step_back_range 1111 ^
--look_ahead 1 ^
--dataset_id day ^
--timeout 500000 ^
--optimize-target seq__f1 ^
--activate_sma_space_search true ^
--activate_sma_space_search true ^
--activate_ema_space_search true ^
--activate_rsi_space_search true ^
--activate_macd_space_search true ^
--activate_vwap_space_search true ^
--add_only_vwap_z_and_vwap_triggers false ^
--min_percentage_to_keep_class 2.0 ^
--epsilon 0.01 ^
--storage my_storage_roulette_4_mp ^
--study_name opt_f1 ^
--max-ema-slots 10 ^
--ema-min 2 ^
--ema-max 80 ^
--ema-step 2 ^
--max-ema-shift-slots 5 ^
--ema-shift-min 1 ^
--ema-shift-max 10 ^
--max-rsi-slots 10 ^
--rsi-min 2 ^
--rsi-max 28 ^
--max-rsi-shift-slots 5 ^
--rsi-shift-min 1 ^
--rsi-shift-max 10 ^
--base-models lgb
pause

