@echo off

:: 1. Gestion robuste du répertoire (indépendante du dossier d'exécution)
:: %~dp0 représente le dossier où se trouve ce fichier
cd /d "%~dp0"

:: Chargement de la configuration commune
call "%~dp0__V1__config.bat"

:: Déplacement vers le répertoire du projet
cd ..\..\..\src\crusaders\taurus
echo Using %PRIME_RSI_MONTH_TARGET_DIR%

:: 
python player.py --nb-workers 20 --prime-rsi-target-dir %PRIME_RSI_MONTH_TARGET_DIR% --dgdr-target-dir None --autotune-target-dir None --oerh-target-dir None  --verbose

pause