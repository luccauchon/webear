@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\quantloop
echo  TRAINING COMMAND:   python .\quant_model_trainer.py --training-mode random --target-type in_between --target-percentage 0.04 --time-limit 70000
echo ==========================================================================
python .\market_and_macro_data_collector.py --output-dir D:\Finance\compiled_models\quant\2026.04.22 --filename generated__combined_month_macro.data --freq month
echo ==========================================================================
python .\quant_model_trainer.py --real-time --clip --model-path D:\Finance\compiled_models\quant\2026.04.22\best_model_RandomForestClassifier_in_between_20260422_064325.pkl --dataset D:\Finance\compiled_models\quant\2026.04.22\generated__combined_month_macro.data
pause

