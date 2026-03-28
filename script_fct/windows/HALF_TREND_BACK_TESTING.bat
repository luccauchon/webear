@echo off
call conda activate PY311_HT
cd ..\..\src\runners
python half_trend_back_testing.py
pause