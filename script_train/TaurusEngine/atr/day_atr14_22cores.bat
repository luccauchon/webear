@echo off
setlocal enabledelayedexpansion

:: Liste des valeurs de tightness-weight
set "weights=0 0.33 0.66 0.99 1.33 1.66 2 2.5 3 4 9.99"
:: 720 * 360 = 3 jours
for %%w in (%weights%) do (
    echo Lancement avec --tightness-weight %%w
    start "DAY ATR 14 WT %%w" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python atr_backtesting.py --verbose --iterations 720 --timeout 360 --runner atr_vvix_momentum --atr-window 14 --tightness-weight %%w --use-close-for-range"
    timeout /t 5 /nobreak
)

for %%w in (%weights%) do (
    echo Lancement avec --tightness-weight %%w
    start "DAY ATR 14 WT %%w" cmd /C "call conda activate PY312_HT && cd ..\..\..\src\runners && python atr_backtesting.py --verbose --iterations 720 --timeout 360 --runner atr --atr-window 14 --tightness-weight %%w --use-close-for-range"
    timeout /t 5 /nobreak
)

pause
