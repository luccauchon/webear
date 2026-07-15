@echo off
title HTML Week
chcp 65001 > nul

:BOUCLE_INFINIE
:: 1. Récupère le jour (6=Samedi) et l'heure (HH:mm) via PowerShell
for /f "tokens=1,2 delims= " %%A in ('powershell -Command "$d=[int](Get-Date).DayOfWeek; if($d -eq 0){$d=7}; $h=Get-Date -Format 'HH:mm'; Write-Output \"$d $h\""') do (
    set "JOUR=%%A"
    set "HEURE=%%B"
)

:: 2. Vérification du jour (6 = Samedi) et de l'heure
if "%JOUR%"=="6" (
    if "%HEURE%"=="16:05" (
        echo Lancement du script...
        
        :: METTEZ VOTRE CODE ICI        
		start "" cmd /c "D:\PyCharmProjects\webear\script_fct\windows\TAURUS\H_AUTOTUNE_WEEK.bat"
        timeout /t 65 /nobreak > nul
    )
)

:: 3. Attendre 30 secondes avant la prochaine vérification
timeout /t 30 /nobreak > nul
goto BOUCLE_INFINIE
