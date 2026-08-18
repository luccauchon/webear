@echo off
title [WATCHER] ATR PROBABILITY ZONE
chcp 65001 > nul

:BOUCLE_INFINIE
:: 1. Récupère le jour (1=Lundi, 5=Vendredi) et l'heure (HH:mm) via PowerShell
for /f "tokens=1,2 delims= " %%A in ('powershell -Command "$d=[int](Get-Date).DayOfWeek; if($d -eq 0){$d=7}; $h=Get-Date -Format 'HH:mm'; Write-Output \"$d $h\""') do (
    set "JOUR=%%A"
    set "HEURE=%%B"
)

:: 2. Vérification propre et robuste
if %JOUR% geq 1 if %JOUR% leq 5 (
    if "%HEURE%"=="09:31" (        
        :: METTEZ VOTRE CODE ICI
        start "ATR PROBABILITY ZONE" cmd /c "D:\PyCharmProjects\webear\script_fct\windows\C_ATR_PROBABILITY_ZONE.bat"
        timeout /t 50 /nobreak > nul
    )
)

:: 3. Attendre 30 secondes avant la prochaine vérification
timeout /t 30 /nobreak > nul
goto BOUCLE_INFINIE
