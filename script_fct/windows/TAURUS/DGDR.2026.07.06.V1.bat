@echo off
call conda activate PY312_HT
cd ..\..\..\src\optimizers\dgdr
python .\player.py --target-dir D:\Finance\compiled_models\taurus\v1\2026.07.06\optimizers\dgdr\day_perspective --hide-zero-signal
pause