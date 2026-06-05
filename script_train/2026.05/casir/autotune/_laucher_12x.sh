#!/bin/bash

# Valeurs par défaut d'origine
WEBEAR__NTRIALS=44444
WEBEAR__THRESHOLD=0.020
WEBEAR__LOOKAHEAD=5
WEBEAR__OPTIMIZE="finish_above"

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
      WEBEAR__OPTIMIZE="$2"
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

echo "Démarrage avec N-TRIALS=$WEBEAR__NTRIALS  WEBEAR__THRESHOLD=$WEBEAR__THRESHOLD  WEBEAR__LOOKAHEAD=$WEBEAR__LOOKAHEAD  WEBEAR__OPTIMIZE=$WEBEAR__OPTIMIZE  WEBEAR__OUTPUT_DIR=$WEBEAR__OUTPUT_DIR"

cd ../../../../src/optimizers/autotune || exit 1

MIN_SIGNAL_DENSITIES=(0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19)
PIDS=()

for msr in "${MIN_SIGNAL_DENSITIES[@]}"; do
  python ./realtime_and_backtest_hyperparameter_search_optuna.py \
      --dataset-id day \
      --lookahead-bars "$WEBEAR__LOOKAHEAD" \
      --win-threshold "$WEBEAR__THRESHOLD" \
      --min-signal-density "$msr" \
      --train-ratio 0.7 \
      --optimize "$WEBEAR__OPTIMIZE" \
      --n-trials "$WEBEAR__NTRIALS" \
      --output-dir "$WEBEAR__OUTPUT_DIR" \
      --timeout 80000 &
    PIDS+=($!)
done

echo "Processus lancés : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont terminés !"
