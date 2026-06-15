#!/bin/bash

# Valeurs par défaut d'origine
WEBEAR__NTRIALS=44444
WEBEAR__THRESHOLD=0.020
WEBEAR__LOOKAHEAD=5
WEBEAR__DATASET_ID="day"
WEBEAR__TIMEOUT=80000
WEBEAR__METRIC="long_accuracy"
WEBEAR__OPTIMIZE="floor"

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
    --timeout)
      WEBEAR__TIMEOUT="$2"
      shift 2
      ;;
    --optimize)
      WEBEAR__OPTIMIZE="$2"
      shift 2
      ;;
    --dataset-id|-d)
      # Vérification de la valeur passée
      if [[ "$2" =~ ^(day|week|month|quarter|year)$ ]]; then
        WEBEAR__DATASET_ID="$2"
        shift 2
      else
        echo "❌ Erreur : --dataset-id doit être : day, week, month, quarter ou year"
        exit 1
      fi
      ;;
    *)
      echo "❌ Option inconnue : $1"
      echo "Options valides : --trials, --dataset-id, --threshold, --lookahead, --metric"
      exit 1
      ;;
  esac
done

# Calcul dynamique du dossier de sortie
PCT_VALUE=$(awk "BEGIN {print $WEBEAR__THRESHOLD * 100}")
WEBEAR__OUTPUT_DIR="models__${WEBEAR__LOOKAHEAD}B_${WEBEAR__OPTIMIZE}_${PCT_VALUE}pct"

echo "Démarrage avec N-TRIALS=$WEBEAR__NTRIALS  WEBEAR__THRESHOLD=$WEBEAR__THRESHOLD  WEBEAR__LOOKAHEAD=$WEBEAR__LOOKAHEAD  WEBEAR__METRIC=$WEBEAR__METRIC  WEBEAR__OUTPUT_DIR=$WEBEAR__OUTPUT_DIR"

cd ../../../../src/optimizers/oerh || exit 1

MIN_SIGNAL_DENSITIES=(0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40)
PIDS=()

for msr in "${MIN_SIGNAL_DENSITIES[@]}"; do
  python ./hyperparameter_search_optuna.py \
      --dataset-id "$WEBEAR__DATASET_ID" \
      --metric "$WEBEAR__METRIC" \
      --lookahead-bars "$WEBEAR__LOOKAHEAD" \
      --threshold-pct "$WEBEAR__THRESHOLD" \
      --n-trials "$WEBEAR__NTRIALS" \
      --storage none \
      --output-dir "$WEBEAR__OUTPUT_DIR" \
      --target-type "$WEBEAR__OPTIMIZE" \
      --timeout "$WEBEAR__TIMEOUT" \
      --min-signal-ratio "$msr" &
    PIDS+=($!)
done

echo "Processus lancés : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont terminés !"
