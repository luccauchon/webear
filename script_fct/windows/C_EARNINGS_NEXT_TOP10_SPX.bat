@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python earnings_next_spx_.py
pause