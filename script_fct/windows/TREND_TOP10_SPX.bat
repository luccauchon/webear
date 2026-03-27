@echo off
call conda activate PY311_HT
cd ..\..\src\runners
python spx_top10_aligned_trend.py
pause