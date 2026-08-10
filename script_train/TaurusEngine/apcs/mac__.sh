#!/bin/bash

THRESHOLDS=("0.0100" "0.0500" "0.1500" "0.2500")
# Boucle pour ouvrir 4 nouvelles fenêtres Terminal
for i in {1..4}
do
   VAL="{THRESHOLDS[i]}"
   COMMAND="conda activate PY312_HT && cd /Users/luccauchon/WORK/webear/src/optimizers/apcs && python ./realtime_and_backtest_hyperparameter_search_optuna.py --verbose --buy-offset 1. --sell-offset 1. --min-density-threshold $VAL --n-trials 83600 --timeout 86400 --dataset-id intraday_60min --lookahead 4 --trade-direction both --delta 0. --verbose-list-trades --output-dir models\intraday_persective\60min"
   osascript -e "tell application \"Terminal\" to do script \"$COMMAND\""
done
