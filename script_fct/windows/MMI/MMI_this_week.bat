call conda activate PY312_HT
cd ..\..\..\src\crusaders\mmi
python mmi_next_week_at_2_0p.py  --keep_last_step=false
python mmi_next_week_at_2_25p.py --keep_last_step=false
python mmi_next_week_at_3_0p.py  --keep_last_step=false
python mmi_next_week_at_4_0p.py  --keep_last_step=false
python mmi_next_week_at_5_0p.py  --keep_last_step=false
pause