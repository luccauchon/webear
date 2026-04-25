#!/bin/bash
cd ../../../src/optimizers/regime
export WEBEAR__TIMEOUT=36000
python .\autoTune.py  --signal-type long \
    --optimize win_rate_3 \
    --timeout $WEBEAR__TIMEOUT \
    --lookahead-bars 5 \
    --win-threshold 0.04 \
    --n-trials 999999 \
    --length-dataset 99999 \
    --min-signal-density 0.06 &

PID1=$!

pyython .\autoTune.py  --signal-type long \
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