#!/bin/bash

# 1. Récupère le 1er argument ($1). Si vide, utilise 30000 par défaut.
WEBEAR__TIMEOUT=${1:-30000}

# 2. Exporte la variable pour qu'elle soit visible par Python
export WEBEAR__TIMEOUT

echo "Demarrage avec TIMEOUT = $WEBEAR__TIMEOUT"

cd ../../../src/optimizers/autotune

python ./realtime_and_backtest_hyperparameter_search_optuna.py  --signal-type long \
    --optimize win_rate_3 \
    --timeout $WEBEAR__TIMEOUT \
    --lookahead-bars 5 \
    --win-threshold 0.04 \
    --n-trials 999999 \
    --length-dataset 99999 \
    --min-signal-density 0.06 &

PID1=$!

python ./realtime_and_backtest_hyperparameter_search_optuna.py  --signal-type long \
     --optimize win_rate_3 \
     --timeout $WEBEAR__TIMEOUT \
     --lookahead-bars 20 \
     --win-threshold 0.08 \
     --n-trials 999999 \
     --length-dataset 99999 \
     --min-signal-density 0.06 &
PID2=$!

# ⏳ Wait for both background jobs to complete
wait $PID1 $PID2

echo "✅ All processes finished!"
read  # Pause before exit