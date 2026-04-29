@echo off
call conda activate PY312_HT
cd ..\..\..
cd src\optimizers\quantloop
start /low /b /wait python .\quant_model_trainer.py --dataset .\combined_month_macro.data ^
--target-type soft_higher ^
--target-percentage 0.01 ^
--look-ahead 1 ^
--n-test 24 ^
--scaler FunctionTransformer ^
--estimator RandomForestClassifier ^
--training-mode random ^
--time-limit 256000
echo "MONTHLY 1BAR 1% PUT CREDIT SPREAD"
pause