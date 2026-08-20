@echo off

:: Remplacer par le chemin vers votre installation de Conda si ce n'est pas le dossier par défaut
set CONDA_ACTIVATE_PATH=C:\Users\cj3272\miniconda3\Scripts\activate.bat
if not exist "%CONDA_ACTIVATE_PATH%" set CONDA_ACTIVATE_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat

:: Script 1 : Minutes  17h00
start "Download Minutes" cmd /k "call "%CONDA_ACTIVATE_PATH%" PY312_HT && d: && cd D:\PyCharmProjects\webear\src\fetchers && set PYTHONPATH=D:\PyCharmProjects\webear\src && python download_minutes.py"

:: Script 2 : 30 Minutes  17h00
start "Download 30 Minutes" cmd /k "call "%CONDA_ACTIVATE_PATH%" PY312_HT && d: && cd D:\PyCharmProjects\webear\src\fetchers && set PYTHONPATH=D:\PyCharmProjects\webear\src && python download_30minutes.py"

:: Script 3 : Futures  18h00
start "Download Futures" cmd /k "call "%CONDA_ACTIVATE_PATH%" PY312_HT && d: && cd D:\PyCharmProjects\webear\src\fetchers && set PYTHONPATH=D:\PyCharmProjects\webear\src && python download_futures.py"

:: Script 4 : Lancement du second fichier .bat
start "YFinance Subset HTML" cmd /k "call R_YFINANCE_SUBSET+HTML_DAY___16h05m.bat"

:: Script 5 : 
start "HTML Week" cmd /k "call R_HTML___Saturday.bat"

:: Script 6 : 
:: Fonctionne pas
:: start "HTML End Month" cmd /k "call R_HTML___End_of_month.bat"

:: Script 7 : ATR quoditien à 09h31 et ATR hebdomadaire le lundi à 09h39
start "ATR P ZONE" cmd /k "call R_ATR_PROBABILITY_ZONE___09h31_09h39m.bat"

:: Script 8 :
start "STREAK PROBABILITY" cmd /k "call R_STREAK_PROBABILITY___16h19_15h45.bat"
