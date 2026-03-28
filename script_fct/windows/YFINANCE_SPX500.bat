@echo off
call conda activate PY312_HT
cd ..\..\src\fetchers
python sp500_download.py
pause