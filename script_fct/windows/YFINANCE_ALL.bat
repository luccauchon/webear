@echo off
call conda activate PY311_HT
python "D:\PyCharmProjects\webear\src\fetchers\serialize_fyahoo.py" --all-tickers
pause