$dossierInitial = Get-Location

cd ..\..\..\src\optimizers\quantloop

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9999 --et-gt-open 0 --st2 09:45 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9975 --et-gt-open 0 --st2 09:45 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9950 --et-gt-open 0 --st2 09:45 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9900 --et-gt-open 0 --st2 09:45 --et1 15:55 --output-dir quant\miner1

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9999 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9975 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9950 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9900 --et-gt-open 0 --st2 10:00 --et1 15:55 --output-dir quant\miner1

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9999 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9975 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9950 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9900 --et-gt-open 0 --st2 11:00 --et1 15:55 --output-dir quant\miner1

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9999 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9975 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9950 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9900 --et-gt-open 0 --st2 12:00 --et1 15:55 --output-dir quant\miner1

$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9999 --et-gt-open 0 --st2 13:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9975 --et-gt-open 0 --st2 13:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9950 --et-gt-open 0 --st2 13:00 --et1 15:55 --output-dir quant\miner1
$env:Q__DISPLAY_RATE=10000 ; python .\quant_morning_alpha_miner.py --nb-worker 12 --limit-feature-dim 12 --seuil-pos 0.9900 --et-gt-open 0 --st2 13:00 --et1 15:55 --output-dir quant\miner1

Set-Location $dossierInitial