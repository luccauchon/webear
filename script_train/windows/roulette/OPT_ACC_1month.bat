@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\roulette
python .\hyperparameter_search_optuna.py ^
--step_back_range 3333 ^
--look_ahead 1 ^
--dataset_id month ^
--timeout 500000 ^
--optimize-target seq__precision ^
--activate_sma_space_search true ^
--activate_sma_space_search true ^
--activate_ema_space_search true ^
--activate_rsi_space_search true ^
--activate_macd_space_search true ^
--activate_vwap_space_search true ^
--add_only_vwap_z_and_vwap_triggers false ^
--min-percentage-to-keep-class 1.0 ^
--storage my_storage_roulette_1month_precision ^
--study_name opt_ ^
--verbose-debug false
pause
