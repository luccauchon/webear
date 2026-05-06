@echo off
call conda activate PY312_HT
cd ..\..\..\..
cd src\optimizers\quantloop
start /low /b /wait python quant_model_trainer_phase_2.py --training-mode fixed ^
--features Fed_Rate_Diff Inflation_Rate MA_Long MA_Short RSI_Lag1 VIX_Lag1 VIX_Ratio ^
--optuna-trials 999 ^
--target-type soft_higher ^
--target-percentage 0.01 ^
--look-ahead 1 ^
--n-test 24

echo "PHASE --- MONTHLY 1BAR 1% PUT CREDIT SPREAD"
pause