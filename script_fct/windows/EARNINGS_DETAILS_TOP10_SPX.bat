@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python earnings_details_spx.py
pause