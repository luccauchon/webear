#!/bin/bash

WEBEAR__TIMEOUT=${1:-30000}
export WEBEAR__TIMEOUT

echo "Démarrage avec TIMEOUT = $WEBEAR__TIMEOUT"
cd ../../../src/optimizers/autotune

# 1. Définition des valeurs de lookahead
BARS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
PIDS=()

for bar in "${BARS[@]}"; do
    # 2. Ajustement dynamique du seuil selon le bar
    case $bar in
        1|2)
            THRESHOLD="0.02"
            ;;
        3|4|5)
            THRESHOLD="0.03"
            ;;
        6|7|8|9|10)
            THRESHOLD="0.04"
            ;;
        11|12|13|14)
            THRESHOLD="0.05"
            ;;
        15|16|17|18|19|20)
            THRESHOLD="0.06"
            ;;
        *)
            THRESHOLD="0.04"
            ;;
    esac
    # 3. Lancement en arrière-plan
    python ./realtime_and_backtest_hyperparameter_search_optuna.py \
        --signal-type long \
        --optimize win_rate_3 \
        --timeout "$WEBEAR__TIMEOUT" \
        --lookahead-bars "$bar" \
        --win-threshold "$THRESHOLD" \
        --n-trials 999999 \
        --length-dataset 99999 \
        --min-signal-density 0.06 &

    # 4. Stockage du PID
    PIDS+=($!)
done

echo "Processus lancés : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont terminés !"
read
