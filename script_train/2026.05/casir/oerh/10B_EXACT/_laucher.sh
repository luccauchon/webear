#!/bin/bash

# Valeurs par défaut d'origine
WEBEAR__NTRIALS=44444
WEBEAR__THRESHOLD=0.020
WEBEAR__LOOKAHEAD=5
WEBEAR__METRIC="long_accuracy"

# Analyse des arguments nommés
while [[ $# -gt 0 ]]; do
  case $1 in
    --trials)
      WEBEAR__NTRIALS="$2"
      shift 2
      ;;
    --threshold)
      WEBEAR__THRESHOLD="$2"
      shift 2
      ;;
    --lookahead)
      WEBEAR__LOOKAHEAD="$2"
      shift 2
      ;;
    --metric)
      WEBEAR__METRIC="$2"
      shift 2
      ;;
    *)
      echo "❌ Option inconnue : $1"
      echo "Options valides : --trials, --threshold, --lookahead, --metric"
      exit 1
      ;;
  esac
done

# Calcul dynamique du dossier de sortie
PCT_VALUE=$(awk "BEGIN {print $WEBEAR__THRESHOLD * 100}")
WEBEAR__OUTPUT_DIR="models_put_${WEBEAR__LOOKAHEAD}B_${PCT_VALUE}pct"

echo "Démarrage avec N-TRIALS=$WEBEAR__NTRIALS  WEBEAR__THRESHOLD=$WEBEAR__THRESHOLD  WEBEAR__LOOKAHEAD=$WEBEAR__LOOKAHEAD  WEBEAR__METRIC=$WEBEAR__METRIC  WEBEAR__OUTPUT_DIR=$WEBEAR__OUTPUT_DIR"

cd ../../../../../src/optimizers/oerh || exit 1

MIN_SIGNAL_DENSITIES=(0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.0625 0.065 0.07 0.0725 0.075 0.08 0.0825 0.085 0.09 0.0925 0.095 0.10 0.1025 0.105 0.11 0.115 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20)
PIDS=()

for msr in "${MIN_SIGNAL_DENSITIES[@]}"; do
  python ./hyperparameter_search_optuna.py \
      --metric "$WEBEAR__METRIC" \
      --lookahead-bars "$WEBEAR__LOOKAHEAD" \
      --threshold-pct "$WEBEAR__THRESHOLD" \
      --n-trials "$WEBEAR__NTRIALS" \
      --storage none \
      --output-dir "$WEBEAR__OUTPUT_DIR" \
      --target-type exact \
      --timeout 80000 \
      --min-signal-ratio "$msr" &
    PIDS+=($!)
done

echo "Processus lancés : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont terminés !"
