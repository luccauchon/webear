@echo off
call conda activate PY312_HT
cd ..\..\src\runners
python atr_probability_zone.py --dataset-id day --use-close-for-range --production-setup
:: Attend 10 secondes et affiche un compte à rebours
timeout /t 10

python atr_probability_zone.py --dataset-id day --no-use-close-for-range --production-setup
:: Attend 10 secondes et affiche un compte à rebours
timeout /t 10