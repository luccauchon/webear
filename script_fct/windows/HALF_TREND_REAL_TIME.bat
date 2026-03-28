@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python half_trend_real_time.py
pause