#!/bin/bash

export WEBEAR__NTRIALS=99999
export WEBEAR__THRESHOLD=0.015
export WEBEAR__LOOKAHEAD=2
export WEBEAR__METRIC=long_accuracy
export WEBEAR__OUTPUT_DIR=models_put_2B_1.5pct

echo "Demarrage avec N-TRIALS=$WEBEAR__NTRIALS  WEBEAR__THRESHOLD=$WEBEAR__THRESHOLD  WEBEAR__LOOKAHEAD=$WEBEAR__LOOKAHEAD  WEBEAR__METRIC=$WEBEAR__METRIC  WEBEAR__OUTPUT_DIR=$WEBEAR__OUTPUT_DIR"
cd ../../../../src/optimizers/oerh

MIN_SIGNAL_DENSITIES=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36)
for msr in "${MIN_SIGNAL_DENSITIES[@]}"; do
  python ./hyperparameter_search_optuna.py \
      --metric long_accuracy $WEBEAR__METRIC \
      --target-type exact \
      --lookahead-bars $WEBEAR__LOOKAHEAD \
      --threshold-pct $WEBEAR__THRESHOLD \
      --n-trials $WEBEAR__NTRIALS \
      --storage none \
      --output-dir $WEBEAR__OUTPUT_DIR \
      --min-signal-ratio $msr &
    PIDS+=($!)
done

echo "Processus lances : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont termines !"
read
