@echo off
call conda activate PY312_HT
cd ..\..\..\..
cd src\optimizers\quantloop
start /low /b /wait python quant_model_trainer_phase_2.py --training-mode fixed ^
--features MA_Long RSI_Lag1 VIX ^
--optuna-trials 3333 ^
--target-type soft_higher ^
--target-percentage 0.01 ^
--look-ahead 1 ^
--n-test 6

echo "WEEKLY 1BAR 1% PUT CREDIT SPREAD"
pause