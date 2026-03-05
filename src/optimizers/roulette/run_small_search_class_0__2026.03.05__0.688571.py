#!/usr/bin/env python3
"""
Quick optimization run with minimal search space.
"""
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import subprocess
import sys
from constants import IS_RUNNING_ON_CASIR
cmd = [
    sys.executable, "hyperparameter_search_optuna.py",
    "--ticker", "^GSPC",
    "--dataset_id", "day",
    "--look_ahead", "1",
    "--step_back_range", "999",
    "--epsilon", "0.0",
    "--verbose", "true",
    "--n_trials", "999999",
    "--n_jobs", "1",
    "--timeout", "160000",
    "--optimize_target", "pos_seq_0__f1",
    "--objective_name", "base_configuration",
    "--activate_sma_space_search", "false",
    "--activate_ema_space_search", "true",
    "--activate_rsi_space_search", "true",
    "--activate_macd_space_search", "true",
    "--activate_vwap_space_search", "true",
    "--add_only_vwap_z_and_vwap_triggers", "true",
    "--add_close_diff", "true",
    "--base_models", "xgb",
    "--min_percentage_to_keep_class", "4.",
    "--specific_wanted_class", "0", "1", "2",

    # EMA 3, Shift: 1,2
    "--max_ema_slots", "1",
    "--ema_min", "2",
    "--ema_max", "4",
    "--ema_step", "1",
    "--max_ema_shift_slots", "4",
    "--ema_shift_min", "1",
    "--ema_shift_max", "10",

    # SMA (Very Small)
    "--max_sma_slots", "1",
    "--sma_min", "20",
    "--sma_max", "100",
    "--sma_step", "20",
    "--max_sma_shift_slots", "1",
    "--sma_shift_min", "1",
    "--sma_shift_max", "2",

    # RSI 8,9,10
    "--max_rsi_slots", "5",
    "--rsi_min", "5",
    "--rsi_max", "15",
    "--max_rsi_shift_slots", "1",
    "--rsi_shift_min", "1",
    "--rsi_shift_max", "2",

    # MACD 16,40,15
    "--macd_fast_min", "11",
    "--macd_fast_max", "21",
    "--macd_slow_min", "30",
    "--macd_slow_max", "50",
    "--macd_signal_min", "10",
    "--macd_signal_max", "20",

    # VWAP (Very Small)
    "--vwap_min_window", "20",
    "--vwap_max_window", "40",

    # Sequence
    "--shift_seq_col_min", "1",
    "--shift_seq_col_max", "5",
]

subprocess.run(cmd)