@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python atr_probability_zone.py
:: Attend 60 secondes et affiche un compte à rebours
timeout /t 60