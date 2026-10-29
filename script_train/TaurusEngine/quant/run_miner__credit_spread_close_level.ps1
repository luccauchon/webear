# --- Configuration des variables globales ---
$dossierInitial = Get-Location
$scriptPath = "..\..\..\src\optimizers\quantloop"
$scriptPython1 = ".\quant_morning_alpha_miner.py"
$scriptPython2 = ".\quant_morning_beta_miner_realtime_and_backtesting.py"

# Paramètres fixes du script
$workers = 10
$featureDim = 14
$et1 = "15:59"
$outputDir = "quant\miner1__credit_spread_close_level"

# Variables d'environnement
$env:Q__DISPLAY_RATE = 1000

# --- Listes des valeurs variables ---
$listeSt2      = @("09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00","13:30", "14:00", "15:00")
$listeSeuil    = @(0.999, 1.001)
$listeEtGtOpen = @(0, 1)

# Listes pour accumuler les arguments du second script
$argsEt1 = @()
$argsEt2 = @()
$argsSt1 = @()
$argsSt2 = @()
$argsSeuil = @()

# --- Exécution du script ---
Set-Location $scriptPath

foreach ($etGtOpen in $listeEtGtOpen) {
    foreach ($st2 in $listeSt2) {
        foreach ($seuil in $listeSeuil) {
            python $scriptPython1 --target-definition last_morning_close_as_level --limit-feature-dim $featureDim --seuil-pos $seuil --et-gt-open $etGtOpen --st2 $st2 --et1 $et1 --output-dir $outputDir --nb-worker $workers

            # 2. Accumulation des valeurs pour le second script
            $argsEt1 += $et1
            $argsEt2 += "16:00"  # Valeur fixe par itération basée sur votre exemple
            $argsSt1 += "09:30"  # Valeur fixe par itération basée sur votre exemple
            $argsSt2 += $st2
            $argsSeuil += [string]$seuil # Converti en chaîne pour éviter les problèmes de locale (ex: point vs virgule)
        }
    }
}

# --- Exécution du second script avec tous les arguments en liste ---
Write-Host "Lancement de Beta Miner avec les listes d'arguments accumulees..."

python $scriptPython2 --models-dir $outputDir --et1 $argsEt1 --et2 $argsEt2 --st1 $argsSt1 --st2 $argsSt2 --seuil-pos $argsSeuil

# Retour au dossier d'origine
Set-Location $dossierInitial
