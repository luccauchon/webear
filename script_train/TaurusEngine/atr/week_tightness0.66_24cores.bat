@echo off
:: 24 processes (de 0 à 23)
for /L %%i in (0,1,23) do (
    start "WEEK ATR 14" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python -c "import os; ix=%%i; tw=0.33+ix*(4.0-0.33)/23; os.system(f'python .\\atr_backtesting.py --dataset-id week --step-back-range 1040 --atr-window 14 --tightness-weight {tw:.4f} --n-trials 1500 --n-split 0.8 --use-close-for-range')" "
    timeout /t 5 /nobreak
)
