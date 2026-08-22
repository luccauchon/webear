@echo off
title [WATCHER] BOUCLIER SPX
chcp 65001 > nul

:BOUCLE_INFINIE
:: 1. Récupère le jour (1=Lundi, 7=Dimanche) et l'heure (HH:mm) via PowerShell
:: Ajout de -NoProfile pour accélérer l'exécution et simplification de l'écriture
for /f "tokens=1,2" %%A in ('powershell -NoProfile -Command "$d=(Get-Date).DayOfWeek.value__; if($d -eq 0){$d=7}; Write-Host $d (Get-Date -Format 'HH:mm')"') do (
    set "JOUR=%%A"
    set "HEURE=%%B"
)

:: Sécurité : Si PowerShell a échoué, on saute ce tour de boucle pour éviter les erreurs
if not defined JOUR goto SUIVANT
if not defined HEURE goto SUIVANT

:: Le caractère | doit être échappé avec ^ en batch
:: echo [%DATE% %TIME%] Debug - Jour: !JOUR! ^| Heure: !HEURE!

:: 2. Vérification du Lundi
if "%JOUR%"=="1" (
    if "!HEURE!"=="08:31" (        
        start "BOUCLIER SPX" cmd /c "@echo off & call conda activate PY312_HT & cd ..\..\src\crusaders\SPX_drop & python player.py --production-setup --update-dataset"
        :: Attendre la minute suivante pour éviter les doublons
        timeout /t 60 /nobreak > nul
    )
)


:SUIVANT
:: 4. Attendre 10 secondes avant la prochaine vérification
timeout /t 10 /nobreak > nul
goto BOUCLE_INFINIE
