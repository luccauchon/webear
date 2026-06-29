@echo off
call conda activate PY312_HT
cd ..\..\..\src\crusaders\taurus

:: Récupère la date actuelle au format YYYY, MM, DD indépendamment des paramètres régionaux
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set YEAR=%datetime:~0,4%
set MONTH=%datetime:~4,2%
set DAY=%datetime:~6,2%

:: Construit les variables de date pour le script
set DATE_UNDERSCORE=%YEAR%_%MONTH%_%DAY%
set DATE_DOT=%YEAR%.%MONTH%.%%DAY%

python generate_json_for_visualization.py --json-file taurus_visualization_day_%DATE_UNDERSCORE%.json --prime-rsi-target-dir D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\prime_rsi --autotune-target-dir D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\autotune --dgdr-target-dir D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\dgdr --dataset-id day
python generate_json_for_visualization.py --json-file taurus_visualization_week_%DATE_UNDERSCORE%.json --prime-rsi-target-dir D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\prime_rsi --autotune-target-dir D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\autotune --dgdr-target-dir D:\Finance\compiled_models\taurus\v1\2026.06.29\optimizers\dgdr --dataset-id week
python visualization.py --filepath taurus_visualization_day_%DATE_UNDERSCORE%.json
python visualization.py --filepath taurus_visualization_week_%DATE_UNDERSCORE%.json