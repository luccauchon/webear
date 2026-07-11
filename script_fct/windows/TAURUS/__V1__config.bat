@echo off
:: __V1__config.bat

:: Activation de l'environnement
call conda activate PY312_HT

:: Définition des répertoires de base
set BASE_DIR_PKL=D:\Finance\data\taurus\V1\_pkls
set BASE_DIR_JSON=D:\Finance\data\taurus\V1\_jsons

:: Extraction de la date actuelle
set CURRENT_DATE=%DATE:~10,4%.%DATE:~4,2%.%DATE:~7,2%

:: Extraction de la date actuelle avec ajustement pour le week-end
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command ^
    "$d = [DateTime]::Today; if ($d.DayOfWeek -eq 'Saturday') { $d = $d.AddDays(-1) } elseif ($d.DayOfWeek -eq 'Sunday') { $d = $d.AddDays(-2) }; $d.ToString('yyyy.MM.dd')"`) do (
    set CURRENT_DATE=%%i
)

:: Définition des fichiers de sortie
set OUTPUT_FILE_PKL=%BASE_DIR_PKL%\taurus_v1_%CURRENT_DATE%.pkl
set OUTPUT_FILE_DAY_JSON=%BASE_DIR_JSON%\taurus_visualization_day_%CURRENT_DATE%.json

::
set OUTPUT_DIR_VIZ=D:\Finance\data\taurus\V1\_html

::
set PRIME_RSI_TARGET_DIR=D:\Finance\compiled_models\taurus\v1\2026.07.06\optimizers\prime_rsi
set PRIME_RSI_DAY_TARGET_DIR=D:\Finance\compiled_models\taurus\v1\2026.07.06\optimizers\prime_rsi\day_perspective
::
set DGDR_TARGET_DIR=D:\Finance\compiled_models\taurus\v1\2026.07.06\optimizers\dgdr\
set DGDR_DAY_TARGET_DIR=D:\Finance\compiled_models\taurus\v1\2026.07.06\optimizers\dgdr\day_perspective
::
set AUTOTUNE_TARGET_DIR=D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\autotune
::
set OERH_TARGET_DIR=D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\oerh
