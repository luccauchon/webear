#!/bin/bash

WEBEAR__TIMEOUT=${1:-30000}
export WEBEAR__TIMEOUT

WEBEAR__BAR=${2:-5}
export WEBEAR__BAR

WEBEAR__THRESHOLD=${3:-0.}
export WEBEAR__THRESHOLD
echo "Demarrage avec TIMEOUT = $WEBEAR__TIMEOUT"
cd ../../../src/optimizers/autotune

python ./realtime_and_backtest_hyperparameter_search_optuna.py \
    --optimize hold_floor \
    --timeout "$WEBEAR__TIMEOUT" \
    --lookahead-bars "$WEBEAR__BAR" \
    --win-threshold "$WEBEAR__THRESHOLD" \
    --n-trials 999999 \
    --length-dataset 99999 \
    --output-dir calls \
    --storage sqlite:///calls.db \
    --study-name 1223 \
    --train-ratio 0.7 \
    --min-signal-density 0.05 &

    # 4. Stockage du PID
    PIDS+=($!)
done

echo "Processus lances : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont termines !"
read
