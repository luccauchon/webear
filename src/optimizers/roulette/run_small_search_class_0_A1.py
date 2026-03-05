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
    "--step_back_range", "333",
    "--epsilon", "0.0",
    "--verbose", "true",
    "--n_trials", "999999",
    "--n_jobs", "1",
    "--timeout", "86400",
    "--optimize_target", "pos_seq_0__f1_penalty_others",
    "--objective_name", "base_configuration",
    "--activate_sma_space_search", "true",
    "--activate_ema_space_search", "true",
    "--activate_rsi_space_search", "true",
    "--activate_macd_space_search", "true",
    "--activate_vwap_space_search", "true",
    "--add_only_vwap_z_and_vwap_triggers", "true",
    "--add_close_diff", "true",
    "--base_models", "xgb",
    "--min_percentage_to_keep_class", "0.",
    # "--specific_wanted_class",
    # EMA (Very Small)
    "--max_ema_slots", "1",
    "--ema_min", "2",
    "--ema_max", "20",
    "--ema_step", "4",
    "--max_ema_shift_slots", "1",
    "--ema_shift_min", "1",
    "--ema_shift_max", "2",
    # SMA (Very Small)
    "--max_sma_slots", "1",
    "--sma_min", "20",
    "--sma_max", "100",
    "--sma_step", "20",
    "--max_sma_shift_slots", "1",
    "--sma_shift_min", "1",
    "--sma_shift_max", "2",
    # RSI (Very Small)
    "--max_rsi_slots", "1",
    "--rsi_min", "5",
    "--rsi_max", "10",
    "--max_rsi_shift_slots", "1",
    "--rsi_shift_min", "1",
    "--rsi_shift_max", "2",
    # MACD (Very Small)
    "--macd_fast_min", "10",
    "--macd_fast_max", "12",
    "--macd_slow_min", "22",
    "--macd_slow_max", "24",
    "--macd_signal_min", "8",
    "--macd_signal_max", "10",
    # VWAP (Very Small)
    "--vwap_min_window", "10",
    "--vwap_max_window", "12",
    # Sequence
    "--shift_seq_col_min", "1",
    "--shift_seq_col_max", "5",
]

subprocess.run(cmd)