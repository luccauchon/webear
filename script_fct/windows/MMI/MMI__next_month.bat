call conda activate PY312_HT
cd ..\..\..\src\crusaders\mmi
python mmi_next_month_at_4_0p.py  --keep_last_step=True
python mmi_next_month_at_5_0p.py  --keep_last_step=True
pause