#!/bin/bash
cd ../../../src/optimizers/regime
python regime_switching_optimizer.py --storage-url sqlite:///put_1_day.db \
    --study-name put_1_day \
    --timeout 430000 \
    --spread-type put \
    --strike-distance 0.02 \
    --forward-days 1