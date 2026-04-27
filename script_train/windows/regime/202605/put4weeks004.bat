@echo off
call conda activate PY312_HT
cd ..\..\..\..\src\optimizers\regime
python .\regime_switching_optimizer.py ^
--lookback-years 99999 ^
--forward-days 4 ^
--dataset-id week ^
--strike-distance 0.04 ^
--timeout 512000 ^
--spread-type put ^
--penalize-invalid-cluster ^
--confirmation-before-run ^
--min-n-in-cluster 35 ^
--storage-url sqlite:///put4weeks004_3.db ^
--study-name put4weeks004_3
pause
