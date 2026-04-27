@echo off
call conda activate PY312_HT
cd ..\..\..\..\src\optimizers\regime
python .\regime_switching_optimizer.py ^
--lookback-years 99999 ^
--forward-days 5 ^
--dataset-id day ^
--strike-distance 0.03 ^
--timeout 256000 ^
--spread-type call ^
--penalize-invalid-cluster ^
--confirmation-before-run ^
--storage-url sqlite:///call5days003_1.db ^
--study-name call5days003_1
pause
