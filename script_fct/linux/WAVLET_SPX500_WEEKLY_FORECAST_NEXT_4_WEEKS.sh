#!/bin/bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate PY312_HT
cd ../../src/runners
python wavelet_realtime.py \
  --ticker "^GSPC" \
  --col "Close" \
  --verbose True \
  --dataset_id "week" \
  --n_forecast_length 4 \
  --n_forecast_length_in_training 4 \
  --thresholds_ep "(0.0175,0.0175)" \
  --enable-plot false \
  --enable-loop true
read