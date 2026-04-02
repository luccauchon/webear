call conda activate PY312_HT
cd ..\..\..\src\crusaders\mmi
python mmi_next_day_at_1_0p.py  --keep_last_step=true
python mmi_next_day_at_1_25p.py --keep_last_step=true
python mmi_next_day_at_1_5p.py  --keep_last_step=true
python mmi_next_day_at_2_0p.py  --keep_last_step=true
pause