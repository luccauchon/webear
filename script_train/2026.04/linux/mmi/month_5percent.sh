#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate PY312_HT
cd ../../..
cd src/runners
python MMI_hyperparameter_search_optuna.py \
    --dataset_id month \
    --step-back-range 99999 \
    --n-trials 99999 \
    --timeout 86400 \
    --use_ema true \
    --return_threshold_min 0.05 \
    --return_threshold_max 0.05 \
    --sma_period_min 1 \
    --sma_period_max 100 \
    --mmi_period_max 1 \
    --mmi_period_max 100 \
    --mmi_trend_max_min 1 \
    --mmi_trend_max_max 100 \
    --lookahead_min 1 \
    --lookahead_max 1 \
    --study_name 0_05__month__la1 \
    --storage sqlite:///mmi_optuna_m500p.db
read