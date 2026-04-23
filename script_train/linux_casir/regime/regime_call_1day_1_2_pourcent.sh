#!/bin/bash
cd ../../../src/optimizers/regime
export WEBEAR__TIMEOUT=80000
python regime_switching_optimizer.py --storage-url sqlite:///call_1_day_1pct.db \
    --study-name call_1_day_1pct \
    --timeout $WEBEAR__TIMEOUT \
    --spread-type call \
    --strike-distance 0.01 \
    --forward-days 1 \
    --lookback-years 99999 &
PID1=$!

python regime_switching_optimizer.py --storage-url sqlite:///call_1_day_2pct.db \
    --study-name call_1_day_2pct \
    --timeout $WEBEAR__TIMEOUT \
    --spread-type call \
    --strike-distance 0.02 \
    --forward-days 1 \
    --lookback-years 99999 &
PID2=$!

# ⏳ Wait for both background jobs to complete
wait $PID1 $PID2

echo "✅ All processes finished!"
read  # Pause before exit