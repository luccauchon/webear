@echo off
call conda activate PY312_HT
cd ..\..\..\..\src\optimizers\regime
python .\regime_switching_optimizer.py ^
--lookback-years 99999 ^
--forward-days 20 ^
--dataset-id day ^
--spread-type buy_call ^
--strike-distance 0.04 ^
--timeout 512000 ^
--penalize-invalid-cluster ^
--confirmation-before-run ^
--min-n-in-cluster 35 ^
--min-clusters 6 ^
--max-clusters 9 ^
--storage-url sqlite:///20db4pctup.db ^
--study-name luc123
pause