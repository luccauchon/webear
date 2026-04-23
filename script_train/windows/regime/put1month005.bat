@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\regime
python .\regime_switching_optimizer.py ^
--lookback-years 99999 ^
--forward-days 1 ^
--dataset-id month ^
--strike-distance 0.05 ^
--timeout 256000 ^
--spread-type put ^
--penalize_invalid_cluster ^
--confirmation-before-run ^
--min-n-in-cluster 8 ^
--storage-url sqlite:///put1month005_2.db ^
--study-name put1month005_2
pause
