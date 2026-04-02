#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate PY312_HT
cd ../../..
cd src/optimizers/regime
python regime_switching_optimizer.py --storage-url sqlite:///put_5_days.db \
    --study-name put_5_days \
    --timeout 430000
read