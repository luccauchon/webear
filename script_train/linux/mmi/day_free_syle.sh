#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate PY312_HT
cd ../../..
cd src/runners
python MMI_hyperparameter_search_optuna.py \
    --dataset_id day \
    --step-back-range 99999 \
    --n-trials 999999 \
    --timeout 432000 \
    --use_ema true \
    --return_threshold_min 0.02 \
    --return_threshold_max 0.04 \
    --sma_period_min 1 \
    --sma_period_max 20 \
    --mmi_period_max 1 \
    --mmi_period_max 20 \
    --mmi_trend_max_min 1 \
    --mmi_trend_max_max 20 \
    --lookahead_min 1 \
    --lookahead_max 20 \
    --study_name freestyle__day__ \
    --storage sqlite:///mmi_optuna_day_freestyle.db
read