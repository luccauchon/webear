@echo off
:: 24 processes
for /L %%i in (33,14,400) do (
    start "WEEK ATR 14 TW=%%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python .\atr_backtesting.py --dataset-id week --step-back-range 1040 --atr-window 14 --tightness-weight %%i --n-trials 1500 --n-split 0.8 --use-close-for-range"
	timeout /t 5 /nobreak
)
