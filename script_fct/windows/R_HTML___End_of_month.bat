@echo off
title HTML End of month
chcp 65001 > nul

:BOUCLE_INFINIE
:: 1. Vérifie si on est le dernier samedi du mois ET récupère l'heure (HH:mm)
for /f "tokens=1,2 delims= " %%A in ('powershell -Command "$now=Get-Date; $isLastSat = ($now.DayOfWeek -eq 'Saturday' -and $now.AddDays(7).Month -ne $now.Month) ? '1' : '0'; $h=$now.Format('HH:mm'); Write-Output \"$isLastSat $h\""') do (
    set "DERNIER_SAMEDI=%%A"
    set "HEURE=%%B"
)

:: 2. Vérification des conditions (1 = Vrai) et de l'heure
if "%DERNIER_SAMEDI%"=="1" (
    if "%HEURE%"=="16:05" (
        echo Lancement du script...
        
        :: METTEZ VOTRE CODE ICI        
        start "" cmd /c "D:\PyCharmProjects\webear\script_fct\windows\TAURUS\H_AUTOTUNE_MONTH.bat"
        timeout /t 65 /nobreak > nul
    )
)

:: 3. Attendre 30 secondes avant la prochaine vérification
timeout /t 30 /nobreak > nul
goto BOUCLE_INFINIE
