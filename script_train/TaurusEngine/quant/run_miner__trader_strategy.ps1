# --- Configuration des variables globales ---
$dossierInitial = Get-Location
$scriptPath = "..\..\..\src\optimizers\quantloop"
$scriptPython1 = ".\quant_morning_alpha_miner.py"
$scriptPython2 = ".\quant_morning_beta_miner_realtime_and_backtesting.py"

# Paramètres fixes du script
$workers = 8
$featureDim = 15

$outputDir = "quant\miner1__trader_strategy"

# Variables d'environnement
$env:Q__DISPLAY_RATE = 1000

# --- Exécution du script ---
Set-Location $scriptPath

python $scriptPython1 --style-execution trader_style --target-definition last_morning_close_as_level --limit-feature-dim $featureDim --seuil-pos 0.00050 --et-gt-open 1 --st2 "12:00" --et1 "12:01" --et2 "13:05" --output-dir $outputDir --nb-worker $workers
python $scriptPython1 --style-execution trader_style --target-definition last_morning_close_as_level --limit-feature-dim $featureDim --seuil-pos -0.00050 --et-gt-open 0 --st2 "12:00" --et1 "12:01" --et2 "13:05" --output-dir $outputDir --nb-worker $workers

# --- Exécution du second script avec tous les arguments en liste ---
Write-Host "Lancement de Beta Miner avec les listes d'arguments accumulees..."

# python $scriptPython2 --models-dir $outputDir --et1 $argsEt1 --et2 $argsEt2 --st1 $argsSt1 --st2 $argsSt2 --seuil-pos $argsSeuil

# Retour au dossier d'origine
Set-Location $dossierInitial
