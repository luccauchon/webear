@echo off
:: 12 process
for /L %%i in (3,1,14) do (
    start "DAY ATR %%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python .\atr_backtesting.py --dataset-id day --step-back-range 1500 --atr-window %%i --tightness-weight 0.33 --n-trials 1500 --n-split 0.8"
	timeout /t 5 /nobreak
)
