# --- Configuration des variables globales ---
$dossierInitial = Get-Location
$scriptPath = "..\..\..\src\optimizers\quantloop"
$scriptPython = ".\quant_morning_alpha_miner.py"

# Paramètres fixes du script
$workers = 14
$featureDim = 14
$etGtOpen = 1
$et1 = "15:55"
$outputDir = "quant\miner1_put_credit_spread"

# Variables d'environnement
$env:Q__DISPLAY_RATE = 1000

# --- Listes des valeurs variables ---
$listeSt2 = @("09:45", "10:00", "11:00", "12:00", "13:00")
$listeSeuil = @(0.9999, 0.9975, 0.9950, 0.9900, 1.0001, 1.0025, 1.0050, 1.0100)

# --- Exécution du script ---
Set-Location $scriptPath

foreach ($st2 in $listeSt2) {
    foreach ($seuil in $listeSeuil) {
        # Exécution de la commande
        python $scriptPython --nb-worker $workers --limit-feature-dim $featureDim --seuil-pos $seuil --et-gt-open $etGtOpen --st2 $st2 --et1 $et1 --output-dir $outputDir
    }
}

# Retour au dossier d'origine
Set-Location $dossierInitial
