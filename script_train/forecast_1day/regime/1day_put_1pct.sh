#!/bin/bash
TIMEOUT_VAL=${1:-80000}
cd ../../../src/optimizers/regime
python regime_switching_optimizer.py --storage-url sqlite:///put_1_dayc2.db \
    --study-name put_1_day \
    --timeout "$TIMEOUT_VAL" \
    --spread-type put \
    --strike-distance 0.01 \
    --forward-days 1 \
    --lookback-years 99999