#!/bin/bash

# --- Parameters ---
OPTIMIZE_TARGET="call"
OBJECTIVE_NAME="2026_02_20__1pct"
TIMEOUT=400000
CONDA_ENV="PY311"
# IMPORTANT: Update this path to your actual Linux directory
WORK_DIR="$HOME/PYCHARMPROJECTS/webear/src/runners"

# --- Initialize Conda ---
# This ensures 'conda run' works correctly within the script
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "Error: conda command not found. Please ensure conda is installed and in PATH."
    exit 1
fi

# --- Loop from 1 to 20 ---
for i in {1..20}; do

    echo "Launching search with look_ahead=$i..."

    # --- Execution Method ---

    # OPTION 1: Run in Background (Recommended for Linux)
    # This detaches the process and saves output to a log file instead of opening 20 windows.
    (
        cd "$WORK_DIR" || exit
        conda run -n "$CONDA_ENV" python VIX_hyperparameter_search_optuna.py \
            --optimize_target "$OPTIMIZE_TARGET" \
            --objective_name "$OBJECTIVE_NAME" \
            --timeout "$TIMEOUT" \
            --look_ahead "$i"
    ) > "log_lookahead_${i}.txt" 2>&1 &

    # --- Calculate sleep time ---
    SLEEP_TIME=$i

    # --- Wait ---
    if [ "$SLEEP_TIME" -gt 0 ]; then
        echo "Waiting for $SLEEP_TIME seconds..."
        sleep "$SLEEP_TIME"
    fi
done

echo ""
echo "All 20 processes have been dispatched."
echo "Logs are being saved to log_lookahead_*.txt in the current directory."

# --- Pause (Wait for user input) ---
read -p "Press Enter to exit..."