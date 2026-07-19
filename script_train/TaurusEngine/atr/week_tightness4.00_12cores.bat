@echo off
:: 12 process
for /L %%i in (7,1,18) do (
    start "WEEK ATR %%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python .\atr_backtesting.py --dataset-id week --step-back-range 1040 --atr-window %%i --tightness-weight 4.00 --n-trials 999 --n-split 0.8 --use-close-for-range"
	timeout /t 5 /nobreak
)
