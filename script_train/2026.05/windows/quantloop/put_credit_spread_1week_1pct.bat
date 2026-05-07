@echo off
call conda activate PY312_HT
cd ..\..\..\..
cd src\optimizers\quantloop
start /low /b /wait python .\quant_model_trainer.py --dataset .\combined_week_macro.data ^
--target-type soft_higher ^
--target-percentage 0.01 ^
--look-ahead 1 ^
--n-test 32 ^
--scaler FunctionTransformer ^
--estimator RandomForestClassifier ^
--training-mode random ^
--time-limit 256000
echo "WEEKLY 1BAR 1% PUT CREDIT SPREAD"
pause