# --- Configuration des variables globales ---
$dossierInitial = Get-Location
$scriptPath = "..\..\..\src\optimizers\quantloop"
$scriptPython = ".\quant_morning_alpha_miner.py"

# Paramètres fixes du script
$workers = 12
$featureDim = 12
$et1 = "15:59"
$outputDir = "quant\miner1__credit_spread_2x"

# Variables d'environnement
$env:Q__DISPLAY_RATE = 1000

# --- Listes des valeurs variables ---
$listeSt2      = @("09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:30", "13:00","13:30", "14:00", "15:00")
$listeSeuil    = @(0.999, 0.9975, 0.9950, 0.9900, 1.001, 1.0025, 1.0050, 1.0100)
$listeEtGtOpen = @(0, 1)
# --- Exécution du script ---
Set-Location $scriptPath

foreach ($etGtOpen in $listeEtGtOpen) {
    foreach ($st2 in $listeSt2) {
        foreach ($seuil in $listeSeuil) {
            python $scriptPython --nb-worker $workers --limit-feature-dim $featureDim --seuil-pos $seuil --et-gt-open $etGtOpen --st2 $st2 --et1 $et1 --output-dir $outputDir
        }
    }
}

# Retour au dossier d'origine
Set-Location $dossierInitial
