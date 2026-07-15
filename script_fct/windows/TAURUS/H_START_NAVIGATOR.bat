@echo off

:: 1. Gestion robuste du répertoire (indépendante du dossier d'exécution)
:: %~dp0 représente le dossier où se trouve ce fichier
cd /d "%~dp0"

:: Chargement de la configuration commune
call "%~dp0__V1__config.bat"

:: Déplacement vers le répertoire du projet
cd ..\..\..\src\crusaders\taurus

:: Lancement avec le chemin absolu complet (%cd% représente le dossier actuel)
start chrome --allow-file-access-from-files "%cd%\navigator.html"
