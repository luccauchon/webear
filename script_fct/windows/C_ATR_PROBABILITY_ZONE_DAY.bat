@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python atr_probability_zone.py --dataset-id day
:: Attend 10 secondes et affiche un compte à rebours
timeout /t 10