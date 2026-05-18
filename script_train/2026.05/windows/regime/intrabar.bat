@echo off
call conda activate PY312_HT
cd ..\..\..\..\src\optimizers\regime
python .\regime_intrabar.py ^
--lookback-years 99999 ^
--dataset-id day ^
--timeout 512000 ^
--add-enhanced-features ^
--storage-url sqlite:///intraday.db ^
--study-name intraday
pause
