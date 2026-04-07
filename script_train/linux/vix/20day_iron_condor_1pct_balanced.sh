#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate PY312_HT
cd ../../..
cd src/runners
python VIX_hyperparameter_search_optuna.py \
    --objective-name 2026_02_20__1pct_balanced \
    --optimize-target iron_condor \
    --timeout 200000 \
    --step-back-range 99999 \
    --look-ahead 20 \
    --storage 20day_ic_vix.db \
    --study-name sn_20day_ic_vix
read