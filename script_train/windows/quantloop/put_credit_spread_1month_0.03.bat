@echo off
call conda activate PY312_HT
cd ..\..\..
cd src\optimizers\quantloop
python .\quant_model_trainer.py --dataset .\combined_monthly_macro.data ^
--target-type soft_higher ^
--target-percentage 0.03 ^
--look-ahead 1 ^
--scaler FunctionTransformer ^
--estimator RandomForestClassifier ^
--training-mode random ^
--time-limit 86400
pause