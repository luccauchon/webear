@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\roulette
python .\hyperparameter_search_optuna.py ^
--step_back_range 1111 ^
--look_ahead 1 ^
--dataset_id day ^
--timeout 500000 ^
--optimize_target seq__f1 ^
--activate_sma_space_search true ^
--activate_sma_space_search true ^
--activate_ema_space_search true ^
--activate_rsi_space_search true ^
--activate_macd_space_search true ^
--activate_vwap_space_search true ^
--add_only_vwap_z_and_vwap_triggers false ^
--storage my_storage_roulette ^
--study_name opt_f1
pause
