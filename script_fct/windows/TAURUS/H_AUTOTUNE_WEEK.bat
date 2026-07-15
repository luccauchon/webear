@echo off

:: 1. Gestion robuste du répertoire (indépendante du dossier d'exécution)
:: %~dp0 représente le dossier où se trouve ce fichier
cd /d "%~dp0"

:: Chargement de la configuration commune
call "%~dp0__V1__config.bat"

:: Déplacement vers le répertoire du projet
cd ..\..\..\src\crusaders\taurus

:: 
python player.py --nb-workers 2 --autotune-target-dir %AUTOTUNE_WEEK_TARGET_DIR% --hide-zero-signal --clip --optimize-target buy_wr --dgdr-target-dir None --oerh-target-dir None --prime-rsi-target-dir None --save-to %OUTPUT_FILE_AUTOTUNE_WEEK_PKL%

::
python generate_json_for_visualization.py --json-file %OUTPUT_FILE_AUTOTUNE_WEEK_JSON% --pkl-file %OUTPUT_FILE_AUTOTUNE_WEEK_PKL% --dataset-id week --verbose

::
python visualization.py --filepath %OUTPUT_FILE_AUTOTUNE_WEEK_JSON% --output-dir %OUTPUT_DIR_VIZ% --file_id autotune