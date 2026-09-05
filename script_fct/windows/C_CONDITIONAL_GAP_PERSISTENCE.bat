@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python .\conditional_gap_persistence.py --dataset-id day --use-realtime-data
pause

python .\conditional_gap_persistence.py --dataset-id week
pause

python .\conditional_gap_persistence.py --dataset-id month
pause
