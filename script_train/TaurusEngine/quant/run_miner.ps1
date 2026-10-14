$dossierInitial = Get-Location

cd ..\..\..\src\optimizers\quantloop

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9995 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9975 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9955 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9905 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9995 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9975 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9955 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9905 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9995 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9975 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9955 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --reduce-feature -1 --seuil-pos 0.9905 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1

Set-Location $dossierInitial