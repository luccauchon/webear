#!/bin/bash

# Valeurs par défaut d'origine
WEBEAR__NTRIALS=44444
WEBEAR__LOOKAHEAD=5
WEBEAR__OPTIMIZE="buy_wr"
WEBEAR__DATASET_ID="day"
WEBEAR__TIMEOUT=80000
WEBEAR__PUT_STRIKE_PCT=0.99
WEBEAR__CALL_STRIKE_PCT=1.0

# Analyse des arguments nommés
while [[ $# -gt 0 ]]; do
  case $1 in
    --trials)
      WEBEAR__NTRIALS="$2"
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
    --timeout)
      WEBEAR__TIMEOUT="$2"
      shift 2
      ;;
    --put-strike-pct)
      WEBEAR__PUT_STRIKE_PCT="$2"
      shift 2
      ;;
    --call-strike-pct)
      WEBEAR__CALL_STRIKE_PCT="$2"
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
      echo "Options valides : --trials, --lookahead, --metric, --dataset-id, --timeout, --put-strike-pct, --call-strike-pct"
      exit 1
      ;;
  esac
done

# Calcul dynamique du dossier de sortie
PUT_PCT_VALUE=$(awk "BEGIN {print $WEBEAR__PUT_STRIKE_PCT * 100}")
CALL_PCT_VALUE=$(awk "BEGIN {print $WEBEAR__CALL_STRIKE_PCT * 100}")
WEBEAR__OUTPUT_DIR="models__${WEBEAR__OPTIMIZE}__${WEBEAR__LOOKAHEAD}B__put_${PUT_PCT_VALUE}pct__call_${CALL_PCT_VALUE}pct"

echo "Démarrage avec N-TRIALS=$WEBEAR__NTRIALS  WEBEAR__LOOKAHEAD=$WEBEAR__LOOKAHEAD  WEBEAR__OPTIMIZE=$WEBEAR__OPTIMIZE  DATASET-ID=$WEBEAR__DATASET_ID  TIMEOUT=$WEBEAR__TIMEOUT  PUT-STRIKE-PCT=$WEBEAR__PUT_STRIKE_PCT  CALL-STRIKE-PCT=$WEBEAR__CALL_STRIKE_PCT  WEBEAR__OUTPUT_DIR=$WEBEAR__OUTPUT_DIR"

cd ../../../../src/optimizers/prime_rsi || exit 1

MIN_SIGNAL_DENSITIES=(0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40)
PIDS=()

for msr in "${MIN_SIGNAL_DENSITIES[@]}"; do
  python ./realtime_and_backtest_hyperparameter_search_optuna.py \
      --dataset-id "$WEBEAR__DATASET_ID" \
      --lookahead-bars "$WEBEAR__LOOKAHEAD" \
      --min-signal-density "$msr" \
      --train-ratio 0.7 \
      --put-strike-pct "$WEBEAR__PUT_STRIKE_PCT" \
      --call-strike-pct "$WEBEAR__CALL_STRIKE_PCT" \
      --wr-weight 0.9 \
      --td-weight 0.1 \
      --optimize \
      --optimize-target "$WEBEAR__OPTIMIZE" \
      --n-trials "$WEBEAR__NTRIALS" \
      --output-dir "$WEBEAR__OUTPUT_DIR" \
      --optuna-db "$WEBEAR__OUTPUT_DIR\optuna_db.db" \
      --timeout "$WEBEAR__TIMEOUT" &
    PIDS+=($!)
done

echo "Processus lancés : ${PIDS[*]}"

# ⏳ Attente de tous les processus
wait "${PIDS[@]}"

echo "✅ Tous les processus sont terminés !"
