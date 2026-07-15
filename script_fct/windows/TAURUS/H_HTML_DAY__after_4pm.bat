@echo off

:: 1. Gestion robuste du répertoire (indépendante du dossier d'exécution)
:: %~dp0 représente le dossier où se trouve ce fichier
cd /d "%~dp0"

:: Chargement de la configuration commune
call "%~dp0__V1__config.bat"

:: Déplacement vers le répertoire du projet
cd ..\..\..\src\crusaders\taurus

:: 
python player.py --nb-workers 10 --prime-rsi-target-dir %PRIME_RSI_DAY_TARGET_DIR% --dgdr-target-dir %DGDR_DAY_TARGET_DIR%  --autotune-target-dir %AUTOTUNE_DAY_TARGET_DIR% --oerh-target-dir %OERH_TARGET_DIR%  --save-to %OUTPUT_FILE_PKL%

::
python generate_json_for_visualization.py --json-file %OUTPUT_FILE_DAY_JSON% --pkl-file %OUTPUT_FILE_PKL% --dataset-id day --verbose

python visualization.py --filepath %OUTPUT_FILE_DAY_JSON% --output-dir %OUTPUT_DIR_VIZ%

