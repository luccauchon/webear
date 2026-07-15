@echo off

:: Remplacer par le chemin vers votre installation de Conda si ce n'est pas le dossier par défaut
set CONDA_ACTIVATE_PATH=C:\Users\cj3272\miniconda3\Scripts\activate.bat
if not exist "%CONDA_ACTIVATE_PATH%" set CONDA_ACTIVATE_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat

:: Script 1 : Minutes
start "Download Minutes" cmd /k "call "%CONDA_ACTIVATE_PATH%" PY312_HT && d: && cd D:\PyCharmProjects\webear\src\fetchers && set PYTHONPATH=D:\PyCharmProjects\webear\src && python download_minutes.py"

:: Script 2 : 30 Minutes
start "Download 30 Minutes" cmd /k "call "%CONDA_ACTIVATE_PATH%" PY312_HT && d: && cd D:\PyCharmProjects\webear\src\fetchers && set PYTHONPATH=D:\PyCharmProjects\webear\src && python download_30minutes.py"

:: Script 3 : Futures
start "Download Futures" cmd /k "call "%CONDA_ACTIVATE_PATH%" PY312_HT && d: && cd D:\PyCharmProjects\webear\src\fetchers && set PYTHONPATH=D:\PyCharmProjects\webear\src && python download_futures.py"

:: Script 4 : Lancement du second fichier .bat
start "YFinance Subset HTML" cmd /k "call R_YFINANCE_SUBSET+HTML_DAY___16h05m.bat"