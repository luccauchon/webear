@echo off
:: 24 processes
for /L %%i in (2,1,25) do (
    start "DAY ATR %%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python .\atr_backtesting.py --dataset-id day --step-back-range 1825 --atr-window %%i --tightness-weight 4.00 --n-trials 1500 --n-split 0.8 --use-close-for-range"
	timeout /t 5 /nobreak
)
