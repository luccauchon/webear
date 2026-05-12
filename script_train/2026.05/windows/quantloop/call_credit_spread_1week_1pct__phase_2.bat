@echo off
call conda activate PY312_HT
cd ..\..\..\..
cd src\optimizers\quantloop
start /low /b /wait python quant_model_trainer_phase_2.py --training-mode fixed ^
--features Dist_from_ATH Fed_Rate_Diff Inflation_Rate Log_Close MA_Long MA_Short Price_to_MA RSI_Lag1 Shifted_MA_Long Shifted_Price_to_MA Spread_10Y2Y Unrate_Diff VIX_Lag1 VIX_Ratio ^
--optuna-trials 999 ^
--target-type soft_lower ^
--target-percentage 0.01 ^
--look-ahead 1 ^
--n-test 6
--scaler FunctionTransformer ^
--estimator RandomForestClassifier ^
--training-mode random
echo "WEEKLY 1BAR 1% CALL CREDIT SPREAD"
pause