#!/bin/bash

# La commande exacte à exécuter (adaptée pour macOS avec des /)
COMMAND="conda activate PY312_HT && cd /Users/luccauchon/WORK/webear/src/optimizers/prime_rsi && python ./realtime_and_backtest_hyperparameter_search_optuna.py --verbose --optimize --timeout 80000 --n-trials 99999 --dataset-id day --put-strike-pct 1. --lookahead 1 --call-strike-pct 1. --verbose-optuna-progression --reduce-n 0 --optimize-target buy_wr --min-signal-density 0.0001 --method final_close --train-ratio 0.9 --output-dir models --optuna-db journal://test/buy_la1_day.db --use-pullback-buy --use-ema-cross-buy --no-use-ma-conf-buy --no-use-fib-rsi-buy --no-use-reg-bull-div --no-use-hid-bull-div --use-macd-buy --use-bb-buy --no-use-vol-buy --no-use-stoch-buy --no-use-vwap-buy"

# Boucle pour ouvrir 4 nouvelles fenêtres Terminal
for i in {1..4}
do
   osascript -e "tell application \"Terminal\" to do script \"$COMMAND\""
done


