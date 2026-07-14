@echo off
:: 24 process
for /L %%i in (2,1,25) do (
    start "ATR Optuna Day %%i" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python .\atr_backtesting.py --dataset-id day --step-back-range 3600 --atr-window %%i --tightness-weight 0.66 --n-trials 1500 --n-split 0.8"
	timeout /t 5 /nobreak
)
