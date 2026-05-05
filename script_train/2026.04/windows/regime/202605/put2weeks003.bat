@echo off
call conda activate PY312_HT
cd ..\..\..\..\src\optimizers\regime
python .\regime_switching_optimizer.py ^
--lookback-years 99999 ^
--forward-days 2 ^
--dataset-id week ^
--strike-distance 0.03 ^
--timeout 512000 ^
--spread-type put ^
--penalize-invalid-cluster ^
--confirmation-before-run ^
--min-n-in-cluster 35 ^
--min-clusters 4 ^
--max-clusters 7 ^
--storage-url sqlite:///put2weeks003_3.db ^
--study-name put2weeks003_3
pause
