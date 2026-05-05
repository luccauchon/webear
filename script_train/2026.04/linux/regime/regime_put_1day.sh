#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate PY312_HT
cd ../../..
cd src/optimizers/regime
python regime_switching_optimizer.py --storage-url sqlite:///put_1_day.db \
    --study-name put_1_day \
    --timeout 430000 \
    --spread-type put \
    --strike-distance 0.02 \
    --forward-days 1
read