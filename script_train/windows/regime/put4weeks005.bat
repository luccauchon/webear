@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\regime
python .\regime_switching_optimizer.py ^
--lookback-years 99999 ^
--forward-days 4 ^
--dataset-id week ^
--strike-distance 0.05 ^
--timeout 256000 ^
--spread-type put ^
--storage-url sqlite:///put4weeks005.db ^
--study-name put4weeks005
pause
