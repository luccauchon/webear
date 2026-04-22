#!/bin/bash
cd ../../../src/optimizers/regime
export WEBEAR__TIMEOUT=80000
python regime_switching_optimizer.py --storage-url sqlite:///put_20_days__1.db \
    --study-name put_20_days \
    --timeout $WEBEAR__TIMEOUT \
    --spread-type put \
    --strike-distance 0.03 \
    --forward-days 20 \
    --lookback-years 99999 &
PID1=$!

python regime_switching_optimizer.py --storage-url sqlite:///put_20_days__2.db \
    --study-name put_20_days \
    --timeout $WEBEAR__TIMEOUT \
    --spread-type put \
    --strike-distance 0.04 \
    --forward-days 20 \
    --lookback-years 99999 &
PID2=$!

python regime_switching_optimizer.py --storage-url sqlite:///put_20_days__3.db \
    --study-name put_20_days \
    --timeout $WEBEAR__TIMEOUT \
    --spread-type put \
    --strike-distance 0.05 \
    --forward-days 20 \
    --lookback-years 99999 &
PID3=$!

# ⏳ Wait for both background jobs to complete
wait $PID1 $PID2 $PID3

echo "✅ All processes finished!"
read  # Pause before exit