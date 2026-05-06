@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\quantloop
echo  TRAINING COMMAND:   python quant_model_trainer_phase_2.py --training-mode fixed --features Fed_Rate_Diff Inflation_Rate MA_Long MA_Short RSI_Lag1 VIX_Lag1 VIX_Ratio --optuna-trials 999
echo ==========================================================================
python .\market_and_macro_data_collector.py --output-dir D:\Finance\compiled_models\quant\2026.05.06 --filename generated__combined_month_macro.data --freq month
echo ==========================================================================
python .\quant_model_trainer.py --real-time --clip --model-path D:\Finance\compiled_models\quant\2026.05.06\best_model_RandomForestClassifier_soft_higher_20260506_004718.pkl --dataset D:\Finance\compiled_models\quant\2026.05.06\generated__combined_month_macro.data
pause

