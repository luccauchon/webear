@echo off
title [WATCHER] STREAK PROBABILITY
chcp 65001 > nul
setlocal enabledelayedexpansion

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

:: 2. Vérification du Lundi au Vendredi à 16:19
if !JOUR! geq 1 if !JOUR! leq 5 (
    if "!HEURE!"=="16:19" (        
        start "STREAK PROBABILITY PZ DAY" cmd /c "@echo off & call conda activate PY312_HT & cd ..\..\src\runners & python streak_probability_informer.py --dataset-id day"
        :: Attendre la minute suivante pour éviter les doublons
        timeout /t 60 /nobreak > nul
    )
)

:: 3. Vérification du Lundi (1 = Lundi) à 09:39
if "!JOUR!"=="1" (
    if "!HEURE!"=="09:39" (                
        :: Attendre la minute suivante pour éviter les doublons
        timeout /t 60 /nobreak > nul
    )
)

:SUIVANT
:: 4. Attendre 10 secondes avant la prochaine vérification
timeout /t 10 /nobreak > nul
goto BOUCLE_INFINIE