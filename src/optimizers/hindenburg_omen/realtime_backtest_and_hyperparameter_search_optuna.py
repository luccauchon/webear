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

import os
import random
import warnings
import pickle
import math
import json
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from argparse import Namespace
import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
from utils import str2bool

# =========================================================
# PARAMETER SAVE/LOAD UTILITIES
# =========================================================
def save_best_params(params: dict, filepath: str, verbose: bool):
    """
    Save best parameters to a JSON file with numpy type handling.
    """

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj

    safe_params = convert_numpy_types(params)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True) if os.path.dirname(filepath) else None

    with open(filepath, 'w') as f:
        json.dump(safe_params, f, indent=2)
    if verbose:
        print(f"\n✅ Best parameters saved to: {filepath}")


def load_best_params(filepath: str, verbose: bool) -> dict:
    """
    Load parameters from a JSON file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parameters file not found: {filepath}")

    with open(filepath, 'r') as f:
        params = json.load(f)
    if verbose:
        print(f"✅ Parameters loaded from: {filepath}")
    return params


# =========================================================
# DATA LOADING
# =========================================================
def load_close_data(_ticker):
    from utils import get_filename_for_dataset
    filename = get_filename_for_dataset("day", older_dataset=None)
    with open(filename, "rb") as f:
        cache = pickle.load(f)
    df = cache[_ticker].copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    return df


# =========================================================
# OPTIMIZATION
# =========================================================
def run_professional_optimization(args):
    # =========================================================
    # CONFIGURATION (From Args)
    # =========================================================
    FORWARD_DAYS = args.forward_days
    THRESHOLD = args.threshold
    CLUSTER_MODE = args.cluster_mode
    TICKER = args.ticker
    RANDOM_SEED = args.seed
    assert (args.mode == "drop" and args.threshold < 0) or (args.mode == "upper" and args.threshold > 0)

    # Set seeds based on args
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

    # Min signals logic based on cluster mode
    if args.min_signals_required:
        MIN_SIGNALS_REQUIRED = args.min_signals_required
    else:
        MIN_SIGNALS_REQUIRED = 50 if CLUSTER_MODE == "crossover" else 250

    # Fixed parameters override
    fixed__cluster_threshold = args.fixed_cluster_threshold
    fixed__cluster_window = args.fixed_cluster_window

    df = load_close_data(_ticker=TICKER)
    if args.verbose:
        print(f"Data loaded: {len(df)} rows | "
              f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    close = df["Close"]

    # =========================================================
    # TARGET CALCULATION (MODE-AWARE)
    # =========================================================
    if args.mode == "drop":
        # Drop mode: look for minimum price in forward window
        future_extreme = close.shift(-1).rolling(FORWARD_DAYS, min_periods=FORWARD_DAYS).min()
        df["Target_Pct_Change"] = (future_extreme - close) / close
        # Negative threshold: e.g., -0.03 means 3% drop
        df["Is_Event"] = (df["Target_Pct_Change"] <= THRESHOLD).astype(int)
        event_direction = "drop"
    else:  # upper mode
        # Upper mode: look for maximum price in forward window
        future_extreme = close.shift(-1).rolling(FORWARD_DAYS, min_periods=FORWARD_DAYS).max()
        df["Target_Pct_Change"] = (future_extreme - close) / close
        # Positive threshold: e.g., 0.03 means 3% spike
        df["Is_Event"] = (df["Target_Pct_Change"] >= THRESHOLD).astype(int)
        event_direction = "spike"

    target = df["Is_Event"]
    valid_mask = df["Target_Pct_Change"].notna()

    total_days = valid_mask.sum()
    total_events = target.sum()
    baseline = total_events / total_days * 100
    assert 0 < args.threshold_penalty_for_low_events <= 1
    args.threshold_penalty_for_low_events = int(args.threshold_penalty_for_low_events * total_events)
    def penalty_for_having_low_number_of_events(n):
        return min(n / args.threshold_penalty_for_low_events, 1.0)
    if args.verbose:
        print(f"\nMode: {event_direction.upper()} | "
              f"Threshold: {THRESHOLD * 100:+.1f}% | "
              f"Baseline event probability: {baseline:.2f}%   ({total_events} / {total_days}) | "
              f"Penalty threshold @ {args.threshold_penalty_for_low_events}")
    improve_score_function = args.use_z_score_boost

    def objective(trial):
        # New boolean toggle via Optuna
        use_ema_stretch = trial.suggest_categorical("use_ema_stretch", [True, False])
        use_volatility_compression = trial.suggest_categorical("use_volatility_compression", [True, False])
        use_market_breadth_proxy = trial.suggest_categorical("use_market_breadth_proxy", [True, False])
        use_trend_regime_filter = trial.suggest_categorical("use_trend_regime_filter", [True, False])
        use_rsi_indicator = trial.suggest_categorical("use_rsi_indicator", [True, False])
        use_macd_indicator = trial.suggest_categorical("use_macd_indicator", [True, False])
        base_signal = trial.suggest_categorical("base_signal", ["simple_ma", "ecart_type", "slope_3days", "bull_market_global"])

        cluster_window = fixed__cluster_window if fixed__cluster_window is not None else trial.suggest_int("cluster_window", 2, 120)
        cluster_threshold = fixed__cluster_threshold if fixed__cluster_window is not None else trial.suggest_int("cluster_threshold", 2, 12)

        # ---------------- Base signal ----------------
        cond_price = None
        if base_signal == "simple_ma":
            sma_len = trial.suggest_int("sma_len", 2, 100)
            sma = ta.sma(close, length=sma_len)
            cond_price = close > sma
        elif base_signal == "ecart_type":
            # Calcule à combien d'écarts-types le prix se trouve de la moyenne
            sma_len = trial.suggest_int("sma_len", 2, 100)
            sma = ta.sma(close, length=sma_len)
            std = close.rolling(window=sma_len).std()
            z_score = (close - sma) / std
            cond_price = z_score > 2.0  # Le prix est "étiré" vers le haut
        elif base_signal == "slope_3days":
            sma_len = trial.suggest_int("sma_len", 2, 100)
            sma = ta.sma(close, length=sma_len)
            sma_slope = sma.diff(3)  # Pente sur les 3 derniers jours
            cond_price = (close > sma) & (sma_slope > 0)
        elif base_signal == "bull_market_global":
            # On ne prend des signaux d'achat que si on est en "Bull Market" global
            sma_len = trial.suggest_int("sma_len", 2, 100)
            sma = ta.sma(close, length=sma_len)
            cond_regime = ta.sma(close, 50) > ta.sma(close, 200)
            cond_price = (close > sma) & cond_regime
        assert cond_price is not None
        base_signal = cond_price

        # ---------------- Indicators ----------------
        if use_rsi_indicator:
            rsi_len = trial.suggest_int("rsi_len", 10, 20)
            rsi_thresh = trial.suggest_int("rsi_thresh", 60, 75)
            rsi_lookback = trial.suggest_int("rsi_lookback", 3, 10)
            rsi = ta.rsi(close, length=rsi_len)
            rsi_max = rsi.rolling(rsi_lookback).max()
            cond_rsi = rsi_max > rsi_thresh
            base_signal = base_signal & cond_rsi

        if use_macd_indicator:
            macd_fast = trial.suggest_int("macd_fast", 8, 15)
            macd_slow = trial.suggest_int("macd_slow", macd_fast + 5, 40)
            macd_signal = trial.suggest_int("macd_signal", 5, 15)
            macd = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd is None: return 0
            macd_line = macd.iloc[:, 0]
            macd_sig = macd.iloc[:, 1]
            cond_macd = macd_line < macd_sig
            base_signal = base_signal & cond_macd

        if use_ema_stretch:
            ema_len = trial.suggest_int("ema_len", 2, 200)
            stretch_thresh = trial.suggest_float("stretch_thresh", 0.01, 0.08)
            ema = ta.ema(close, length=ema_len)
            stretch = (close - ema) / ema
            cond_stretch = stretch > stretch_thresh
            base_signal = base_signal & cond_stretch

        if use_volatility_compression:
            bb_len = trial.suggest_int("bb_len", 2, 30)
            bb_width_thresh = trial.suggest_float("bb_width_thresh", 0.02, 0.10)
            bb = ta.bbands(close, length=bb_len)
            if bb is None:
                return 0
            bb_width = (bb.iloc[:, 2] - bb.iloc[:, 0]) / close
            cond_volatility = bb_width < bb_width_thresh
            base_signal = base_signal & cond_volatility

        if use_market_breadth_proxy:
            roc_len = trial.suggest_int("roc_len", 5, 20)
            roc_thresh = trial.suggest_float("roc_thresh", -0.02, 0.02)
            roc = ta.roc(close, length=roc_len)
            cond_roc = roc < roc_thresh
            base_signal = base_signal & cond_roc

        if use_trend_regime_filter:
            ema200 = ta.ema(close, length=200)
            bull_regime = close > ema200
            base_signal = base_signal & bull_regime

        # ---------------- Cluster logic ----------------
        omen_count = base_signal.rolling(cluster_window).sum()

        if CLUSTER_MODE == "every_day":
            cluster = omen_count >= cluster_threshold
        else:
            prev = omen_count.shift(1)
            cluster = (omen_count >= cluster_threshold) & (prev < cluster_threshold)

        signal_dates = cluster[cluster].index
        if len(signal_dates) < MIN_SIGNALS_REQUIRED: return 0

        cluster_targets = target.loc[signal_dates].dropna()
        if len(cluster_targets) < MIN_SIGNALS_REQUIRED: return 0

        win_rate = cluster_targets.mean() * 100
        n = len(cluster_targets)
        penalty = penalty_for_having_low_number_of_events(n)
        assert 0 <= penalty <= 1
        trial.set_user_attr("penalty", penalty)
        score = win_rate * penalty
        trial.report(score, step=n)
        if trial.should_prune(): raise optuna.TrialPruned()
        if improve_score_function:
            z = (win_rate / 100 - baseline / 100) / math.sqrt((baseline / 100) * (1 - baseline / 100) / n)
            score = win_rate * penalty + z * 10
        return score

    def stop_at_threshold(study, trial):
        if trial.value is not None and trial.value > 99.9:
            study.stop()

    callbacks = [stop_at_threshold]

    # Sampler and Pruner Configuration
    sampler = TPESampler(seed=RANDOM_SEED)
    if args.sampler == "cmaes":
        try:
            from optuna.samplers import CmaEsSampler
            sampler = CmaEsSampler(seed=RANDOM_SEED)
        except ImportError:
            print("Warning: CmaEsSampler not available, falling back to TPE.")

    pruner = MedianPruner(n_startup_trials=args.n_startup_trials)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    if args.verbose:
        print(f"\nStarting Optimization: {args.trials} trials or {args.timeout}s timeout...")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=True if args.verbose else False, callbacks=callbacks)

    # PLOTS
    if not args.no_plot:
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            import matplotlib.pyplot as plt
            # Save plots instead of showing (for batch runs)
            fig1 = plot_optimization_history(study)
            fig1.write_image("optimization_history.png")
            fig2 = plot_param_importances(study)
            fig2.write_image("param_importances.png")
            if args.verbose:
                print("Plots saved as optimization_history.png and param_importances.png")
        except Exception as e:
            print(f"\n⚠️ Plotting skipped: {e}")

    best_params = dict(study.best_params)
    if fixed__cluster_threshold is not None:
        best_params.update({'cluster_threshold': fixed__cluster_threshold})
    if fixed__cluster_window is not None:
        best_params.update({'cluster_window': fixed__cluster_window})
    best_params.update({'threshold': THRESHOLD, 'ticker': args.ticker, 'forward_days': FORWARD_DAYS, 'cluster_mode': CLUSTER_MODE, 'mode': args.mode})
    info = verify_best(df_data=close, cluster_mode=CLUSTER_MODE, params=best_params, target=target, valid_mask=valid_mask, baseline=baseline,
                       forward_days=FORWARD_DAYS, threshold=THRESHOLD, event_direction="drop" if args.mode == "drop" else "upper", verbose=args.verbose)
    best_params.update({'win_rate': info["win_rate"], 'baseline': info["baseline"], 'threshold_penalty_for_low_events': args.threshold_penalty_for_low_events,
                        'penalty': study.best_trial.user_attrs['penalty']})
    if args.verbose:
        print(f"Infos extra: {study.best_trial.user_attrs}")
    # =========================================================
    # SAVE BEST PARAMETERS
    # =========================================================
    if args.save_params_to:
        save_best_params(best_params, args.save_params_to, args.verbose)


# =========================================================
# VERIFICATION & REAL-TIME PREDICTION
# =========================================================
def verify_best(df_data, cluster_mode, params, target, valid_mask, baseline, forward_days, threshold, verbose, event_direction):
    # 1. Base Signal Logic
    base_signal = params["base_signal"]
    cond_price = None
    if base_signal == "simple_ma":
        sma = ta.sma(df_data, length=params["sma_len"])
        cond_price = df_data > sma
    elif base_signal == "ecart_type":
        # Calcule à combien d'écarts-types le prix se trouve de la moyenne
        sma_len = params["sma_len"]
        sma = ta.sma(df_data, length=sma_len)
        std = df_data.rolling(window=sma_len).std()
        z_score = (df_data - sma) / std
        cond_price = z_score > 2.0  # Le prix est "étiré" vers le haut
    elif base_signal == "slope_3days":
        sma = ta.sma(df_data, length=params["sma_len"])
        sma_slope = sma.diff(3)  # Pente sur les 3 derniers jours
        cond_price = (df_data > sma) & (sma_slope > 0)
    elif base_signal == "bull_market_global":
        # On ne prend des signaux d'achat que si on est en "Bull Market" global
        sma = ta.sma(df_data, length=params["sma_len"])
        cond_regime = ta.sma(df_data, 50) > ta.sma(df_data, 200)
        cond_price = (df_data > sma) & cond_regime
    assert cond_price is not None
    base_signal = cond_price

    # 2. Calculate Indicators
    if params.get("use_rsi_indicator"):
        rsi = ta.rsi(df_data, length=params["rsi_len"])
        rsi_max = rsi.rolling(params["rsi_lookback"]).max()
        cond_rsi = rsi_max > params["rsi_thresh"]
        base_signal = base_signal & cond_rsi

    if params.get("use_macd_indicator"):
        macd = ta.macd(df_data, fast=params["macd_fast"], slow=params["macd_slow"], signal=params["macd_signal"])
        cond_macd = macd.iloc[:, 0] < macd.iloc[:, 1]
        base_signal = base_signal & cond_macd

    if params.get("use_ema_stretch"):
        ema = ta.ema(df_data, length=params["ema_len"])
        stretch = (df_data - ema) / ema
        base_signal &= (stretch > params["stretch_thresh"])

    if params.get("use_volatility_compression"):
        bb = ta.bbands(df_data, length=params["bb_len"])
        bb_width = (bb.iloc[:, 2] - bb.iloc[:, 0]) / df_data
        base_signal &= (bb_width < params["bb_width_thresh"])

    if params.get("use_market_breadth_proxy"):
        roc = ta.roc(df_data, length=params["roc_len"])
        cond_roc = roc < params["roc_thresh"]
        base_signal &= cond_roc

    if params.get("use_trend_regime_filter"):
        ema200 = ta.ema(df_data, length=200)
        bull_regime = df_data > ema200
        base_signal &= bull_regime

    # 3. Cluster Logic
    omen_count = base_signal.rolling(params["cluster_window"]).sum()
    if cluster_mode == "every_day":
        cluster = omen_count >= params["cluster_threshold"]
    else:
        cluster = (omen_count >= params["cluster_threshold"]) & (omen_count.shift(1) < params["cluster_threshold"])

    # 4. Metrics Calculation (Precision, Recall, F1)
    # Align signals with known outcomes
    actual_drops = target[valid_mask]
    predicted_drops = cluster[valid_mask]

    tp = ((predicted_drops == 1) & (actual_drops == 1)).sum()
    fp = ((predicted_drops == 1) & (actual_drops == 0)).sum()
    fn = ((predicted_drops == 0) & (actual_drops == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 5. Historical Performance Output
    if verbose:
        print("\n" + "=" * 40)
        print(f"      VERIFICATION RESULTS - {event_direction.upper()} MODE")
        print("=" * 40)
        print(f"Total Signals Found: {tp + fp}")
        print(f"Precision (Win Rate): {precision * 100:.2f}%")
        print(f"Recall (Capture Rate): {recall * 100:.2f}%")
        print(f"F1 SCORE:             {f1_score:.4f}")
        print("-" * 40)
        print(f"Baseline Win Rate:   {baseline:.2f}%")
        print(f"Edge vs Baseline:    {(precision * 100) - baseline:.2f}%")

    # 4. Historical Performance
    signal_dates = cluster[cluster].index
    # We only verify dates where we actually know the outcome (Target is not NaN)
    verified_signals = target.loc[signal_dates].dropna()

    hits = verified_signals.sum()
    total = len(verified_signals)
    win_rate = (hits / total * 100) if total > 0 else 0

    # 5. Real-Time Status (The Last Row)
    last_date = cluster.index[-1]
    is_active_now = cluster.iloc[-1]  # Is the cluster active today?
    current_count = omen_count.iloc[-1]

    threshold_abs = abs(threshold)
    direction_word = "DROP" if event_direction == "drop" else "SPIKE"
    direction_verb = "drops" if event_direction == "drop" else "rises"
    _tmp_is_active_str = f"High probability SPX {direction_verb} ≥{threshold_abs * 100:.1f}% within the next {forward_days} days."
    if verbose:
        print("\n" + "=" * 40)
        print("      BEST PARAMETERS")
        print("=" * 40)
        for key, value in params.items():
            print(f"  - {key}: {value}")
        print("-" * 40)
        print(f"Total Signals Found: {total}")
        assert np.allclose(win_rate, precision*100), f"\t\t{win_rate=}  vs  {precision*100=}"
        print(f"Historical Win Rate: {win_rate:.2f}%")
        print(f"Baseline was:        {baseline:.2f}%")
        print(f"Edge vs Baseline:    {win_rate - baseline:.2f}%")
        print("\n" + "=" * 40)
        print(f"      REAL-TIME PREDICTION - {direction_word}")
        print("=" * 40)
        print(f"Latest Date:    {last_date.strftime('%Y-%m-%d')}")
        print(f"Current Count:  {current_count} / {params['cluster_threshold']}")
        print(f"SIGNAL ACTIVE:  {'YES - PREDICTING ' + direction_word if is_active_now else 'NO - NEUTRAL'}")
        if is_active_now:
            print(_tmp_is_active_str)
        print("=" * 40)

    return {'win_rate': win_rate, 'baseline': baseline, 'last_date': last_date, 'current_count': current_count, 'cluster_threshold':params['cluster_threshold'],
            'is_active_now': is_active_now, 'event_direction': event_direction, 'threshold': threshold , 'is_active_str': _tmp_is_active_str}


# =========================================================
# REAL-TIME ONLY MODE
# =========================================================
def run_realtime_only(params_file, verbose):
    """
    Run only the real-time prediction part using pre-saved best parameters.
    Skips the entire optimization process.
    """
    # Load pre-saved parameters
    best_params = load_best_params(params_file, verbose)

    # Load data
    df = load_close_data(_ticker=best_params['ticker'])
    if verbose:
        print(f"Data loaded: {len(df)} rows | "
              f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    close = df["Close"]

    # Calculate target for verification (historical performance only)
    FORWARD_DAYS = best_params['forward_days']
    THRESHOLD = best_params['threshold']

    if best_params['mode'] == "drop":
        # Drop mode: look for minimum price in forward window
        future_extreme = close.shift(-1).rolling(FORWARD_DAYS, min_periods=FORWARD_DAYS).min()
        df["Target_Pct_Change"] = (future_extreme - close) / close
        # Negative threshold: e.g., -0.03 means 3% drop
        df["Is_Event"] = (df["Target_Pct_Change"] <= THRESHOLD).astype(int)
        event_direction = "drop"
    else:  # upper mode
        # Upper mode: look for maximum price in forward window
        future_extreme = close.shift(-1).rolling(FORWARD_DAYS, min_periods=FORWARD_DAYS).max()
        df["Target_Pct_Change"] = (future_extreme - close) / close
        # Positive threshold: e.g., 0.03 means 3% spike
        df["Is_Event"] = (df["Target_Pct_Change"] >= THRESHOLD).astype(int)
        event_direction = "spike"

    target = df["Is_Event"]
    valid_mask = df["Target_Pct_Change"].notna()

    total_days = valid_mask.sum()
    total_events = target.sum()
    baseline = total_events / total_days * 100
    if verbose:
        print(f"\nMode: {event_direction.upper()} | "
              f"Threshold: {THRESHOLD * 100:+.1f}% | "
              f"Baseline event probability: {baseline:.2f}%   ({total_events} / {total_days})")

    # Run verification with loaded params
    info = verify_best(df_data=close, cluster_mode=best_params['cluster_mode'], params=best_params,
                       target=target,valid_mask=valid_mask, baseline=baseline,forward_days=FORWARD_DAYS, threshold=THRESHOLD,
                       event_direction="drop" if best_params['mode'] == "drop" else "upper",verbose = verbose,)
    return info

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Professional Stock Drop Prediction Optimization using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python script.py --ticker ^GSPC --trials 100 --timeout 3600"
    )

    # --- Data Configuration ---
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--ticker",
        type=str,
        default="^GSPC",
        help="Stock ticker symbol to analyze (e.g., ^GSPC, AAPL, BTC-USD)."
    )
    data_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (Python, Numpy, Optuna)."
    )

    # --- Strategy Parameters ---
    strat_group = parser.add_argument_group("Strategy Parameters")
    strat_group.add_argument(
        "--forward-days",
        type=int,
        default=20,
        help="Number of days into the future to check for a drop."
    )
    strat_group.add_argument(
        "--threshold",
        type=float,
        default=-0.03,
        help="Percentage threshold for event classification. "
             "For --mode drop: use negative value (e.g., -0.03 for 3%% drop). "
             "For --mode upper: use positive value (e.g., 0.03 for 3%% spike)."
    )
    # Keep backward compatibility alias
    strat_group.add_argument(
        "--drop-threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS  # Hidden, for backward compatibility
    )
    strat_group.add_argument(
        "--cluster-mode",
        type=str,
        choices=["every_day", "crossover"],
        default="every_day",
        help="Logic for signal clustering. 'every_day' checks threshold daily, 'crossover' checks when threshold is crossed."
    )
    strat_group.add_argument(
        "--min-signals-required",
        type=int,
        default=None,
        help="Minimum number of signals required to validate a trial. If None, defaults to 50 (crossover) or 250 (every_day)."
    )
    strat_group.add_argument(
        "--fixed-cluster-window",
        type=int,
        default=None,
        help="If set, fixes the rolling window size for clustering instead of optimizing it."
    )
    strat_group.add_argument(
        "--fixed-cluster-threshold",
        type=int,
        default=None,
        help="If set, fixes the signal count threshold for clustering instead of optimizing it."
    )
    strat_group.add_argument(
        "--threshold-penalty-for-low-events",
        type=float,
        default=0.5,
        help="Minimal number (pourcentage of the total number of events) of required past events to disable penalty on the score."
    )
    strat_group.add_argument(
        "--mode",
        type=str,
        choices=["drop", "upper"],
        default="drop",
        help="Prediction mode: 'drop' for downward movements (price decreases), "
             "'upper' for upward spikes (price increases). Threshold sign should match mode."
    )

    # --- Optimization Settings ---
    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument(
        "--trials",
        type=int,
        default=999,
        help="Maximum number of optimization trials."
    )
    opt_group.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Maximum time in seconds to run the optimization."
    )
    opt_group.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "cmaes"],
        default="tpe",
        help="Optuna sampler algorithm."
    )
    opt_group.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="Number of startup trials for the MedianPruner before pruning begins."
    )
    opt_group.add_argument(
        "--use-z-score-boost",
        action="store_true",
        help="Enable Z-Score statistical boost in the objective function to favor statistical significance."
    )

    # --- Output & Logging ---
    out_group = parser.add_argument_group("Output & Logging")
    out_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable generation of Optuna visualization plots."
    )
    out_group.add_argument(
        "--optuna-verbose",
        action="store_true",
        help="Enable verbose logging (currently sets Optuna to WARNING by default)."
    )
    out_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    # --- Parameter Save/Load Options ---
    param_group = parser.add_argument_group("Parameter Save/Load Options")
    param_group.add_argument(
        "--save-params-to",
        type=str,
        default=None,
        help="Path to save best parameters as JSON after optimization (e.g., 'best_params.json')."
    )
    param_group.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Path to load pre-saved parameters JSON for real-time-only mode."
    )
    param_group.add_argument(
        "--real-time-only",
        action="store_true",
        help="Skip optimization entirely; run only real-time prediction using params from --params-file."
    )

    args = parser.parse_args()

    # Handle backward compatibility for --drop-threshold
    if args.drop_threshold is not None:
        args.threshold = args.drop_threshold
        print(f"⚠️  --drop-threshold is deprecated, using --threshold={args.threshold}")

    # Validate threshold sign matches mode
    if args.mode == "drop" and args.threshold > 0:
        parser.error(f"--mode 'drop' requires negative threshold, got {args.threshold}")
    if args.mode == "upper" and args.threshold < 0:
        parser.error(f"--mode 'upper' requires positive threshold, got {args.threshold}")

    # Handle Optuna Logging Verbosity
    if args.optuna_verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # =========================================================
    # EXECUTION ROUTING
    # =========================================================
    if args.real_time_only:
        # Real-time only mode: skip optimization, use pre-saved params
        if not args.params_file:
            parser.error("--params-file is required when using --real-time-only")
        if args.verbose:
            print("🔄 Running in REAL-TIME ONLY mode (optimization skipped)")
        run_realtime_only(args.params_file, args.verbose)
    else:
        # Full optimization mode
        if args.verbose:
            print("🚀 Running full optimization...")
        run_professional_optimization(args)