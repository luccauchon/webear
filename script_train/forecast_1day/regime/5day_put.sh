#!/bin/bash
TIMEOUT_VAL=${1:-80000}
cd ../../../src/optimizers/regime
PIDS=()

python regime_switching_optimizer.py --storage-url sqlite:///put_5_dayd1.db \
    --study-name put_1_day \
    --timeout "$TIMEOUT_VAL" \
    --spread-type put \
    --strike-distance 0.01 \
    --forward-days 1 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///put_5_dayd2.db \
    --study-name put_1_day \
    --timeout "$TIMEOUT_VAL" \
    --spread-type put \
    --strike-distance 0.015 \
    --forward-days 1 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///put_5_dayd3.db \
    --study-name put_1_day \
    --timeout "$TIMEOUT_VAL" \
    --spread-type put \
    --strike-distance 0.02 \
    --forward-days 1 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///put_5_dayd4.db \
    --study-name put_1_day \
    --timeout "$TIMEOUT_VAL" \
    --spread-type put \
    --strike-distance 0.025 \
    --forward-days 1 \
    --penalize-invalid-cluster \
    --min-n-in-cluster 35 \
    --min-clusters 5 \
    --max-clusters 9 \
    --lookback-years 99999 &
PIDS+=($!)

python regime_switching_optimizer.py --storage-url sqlite:///put_5_dayd5.db \
    --study-name put_1_day \
    --timeout "$TIMEOUT_VAL" \
    --spread-type put \
    --strike-distance 0.030 \
    --forward-days 1 \
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