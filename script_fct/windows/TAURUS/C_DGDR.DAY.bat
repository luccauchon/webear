@echo off

:: 1. Gestion robuste du répertoire (indépendante du dossier d'exécution)
:: %~dp0 représente le dossier où se trouve ce fichier
cd /d "%~dp0"

:: Chargement de la configuration commune
call "%~dp0__V1__config.bat"

call conda activate PY312_HT
cd ..\..\..\src\optimizers\dgdr
echo Using %DGDR_DAY_TARGET_DIR%
python .\player.py --target-dir %DGDR_DAY_TARGET_DIR% --hide-zero-signal --verbose-table
pause