#!/bin/bash
TIMEOUT_VAL=${1:-80000}
cd ../../../src/optimizers/regime
PIDS=()

python regime_switching_optimizer.py --storage-url sqlite:///call_5_daye1.db \
    --study-name _ \
    --timeout "$TIMEOUT_VAL" \
    --spread-type call \
    --strike-distance 0.01 \
    --forward-days 5 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///call_5_daye2.db \
    --study-name _ \
    --timeout "$TIMEOUT_VAL" \
    --spread-type call \
    --strike-distance 0.015 \
    --forward-days 5 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///call_5_daye3.db \
    --study-name _ \
    --timeout "$TIMEOUT_VAL" \
    --spread-type call \
    --strike-distance 0.02 \
    --forward-days 5 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///call_5_daye4.db \
    --study-name _ \
    --timeout "$TIMEOUT_VAL" \
    --spread-type call \
    --strike-distance 0.025 \
    --forward-days 5 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///call_5_daye5.db \
    --study-name _ \
    --timeout "$TIMEOUT_VAL" \
    --spread-type call \
    --strike-distance 0.030 \
    --forward-days 5 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

echo "Processus lancés : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont terminés !"