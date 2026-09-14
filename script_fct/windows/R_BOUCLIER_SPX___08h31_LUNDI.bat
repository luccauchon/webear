@echo off
title [WATCHER] BOUCLIER SPX
chcp 65001 > nul
setlocal enabledelayedexpansion

:BOUCLE_INFINIE
:: 1. Récupère le jour (1=Lundi, 7=Dimanche) et l'heure (HH:mm) via PowerShell
for /f "tokens=1,2" %%A in ('powershell -NoProfile -Command "$d=[int](Get-Date).DayOfWeek; if($d -eq 0){$d=7}; Write-Host $d (Get-Date -Format 'HH:mm')"') do (
    set "JOUR=%%A"
    set "HEURE=%%B"
)

:: Sécurité : Si PowerShell a échoué
if not defined JOUR goto SUIVANT
if not defined HEURE goto SUIVANT

:: Décommenter la ligne ci-dessous si vous voulez debugger dans la console
:: echo [%DATE% %TIME%] Debug - Jour: !JOUR! ^| Heure: !HEURE!

:: 2. Vérification du Lundi à 08h31
if "!JOUR!"=="1" (
    if "!HEURE!"=="08:31" (        
        start "[BOUCLIER SPX]" cmd /c "@echo off & call conda activate PY312_HT & cd ..\..\src\crusaders\SPX_drop & python player.py --production-setup --update-dataset"
        :: Attendre 61 secondes pour dépasser la minute actuelle et éviter les doublons
        timeout /t 61 /nobreak > nul
    )
)

:SUIVANT
:: 4. Attendre 10 secondes avant la prochaine vérification
timeout /t 10 /nobreak > nul
goto BOUCLE_INFINIE
