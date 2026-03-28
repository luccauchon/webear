@echo off
call conda activate PY312_HT
cd ..\..\src\fetchers
python serialize_fyahoo.py --all-tickers
pause