@echo off
call conda activate PY312_HT
cd ..\..\src\crusaders\roulette
python GSPC_pqseq_1la_day.py
pause

