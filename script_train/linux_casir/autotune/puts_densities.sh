#!/bin/bash

WEBEAR__TIMEOUT=${1:-30000}
export WEBEAR__TIMEOUT

echo "Démarrage avec TIMEOUT = $WEBEAR__TIMEOUT"
cd ../../../src/optimizers/autotune

BARS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# Définit l'incrément ici (ex: 0.01 pour 0.03, 0.04, 0.05...)
DENSITIES=$(seq 0.03 0.01 0.2)
PIDS=()

for bar in "${BARS[@]}"; do
    case $bar in
        1|2) THRESHOLD="0.01" ;;
        3|4|5) THRESHOLD="0.02" ;;
        6|7|8|9|10) THRESHOLD="0.03" ;;
        11|12|13|14) THRESHOLD="0.04" ;;
        15|16|17|18|19|20) THRESHOLD="0.05" ;;
        *) THRESHOLD="0.04" ;;
    esac

    for density in $DENSITIES; do
        python ./realtime_and_backtest_hyperparameter_search_optuna.py \
            --signal-type long \
            --optimize win_rate_3 \
            --timeout "$WEBEAR__TIMEOUT" \
            --lookahead-bars "$bar" \
            --win-threshold "$THRESHOLD" \
            --n-trials 999999 \
            --length-dataset 99999 \
            --output-dir puts_2 \
            --min-signal-density "$density" &

        PIDS+=($!)
    done
done

echo "Processus lancés : ${PIDS[*]}"

wait "${PIDS[@]}"
echo "✅ Tous les processus sont terminés !"
read
