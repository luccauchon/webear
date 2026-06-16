@echo off
setlocal enabledelayedexpansion

:: Boucle pour préparer les 16 lancements
for /l %%i in (0, 1, 15) do (
    :: Calcul de la densité
    for /f "delims=" %%d in ('powershell -Command "[math]::Round(0.05 + (%%i * (0.3 - 0.05) / 15), 4)"') do set "DENSITY=%%d"

    echo Lancement du script avec la densite !DENSITY!

    :: Ouvre une nouvelle fenetre et lance le code
    start "Densite !DENSITY!" cmd /k "call conda activate PY312_HT && python .\discover.py --model gradient_boosting --density !DENSITY! --timeout 512000"
)

pause
