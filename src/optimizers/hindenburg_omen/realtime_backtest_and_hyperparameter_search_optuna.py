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
from optuna.pruners import MedianPruner,NopPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.samplers import RandomSampler, GridSampler
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
from utils import str2bool, DATASET_AVAILABLE, format_execution_time
from sklearn.model_selection import ParameterGrid, ParameterSampler
import time

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
def load_data(_ticker, _dataset_choice):
    from utils import get_filename_for_dataset
    filename = get_filename_for_dataset(dataset_choice=_dataset_choice, older_dataset=None)
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
    assert (args.mode == "drop" and args.threshold < 0) or (args.mode == "spike" and args.threshold > 0)

    # Set seeds based on args
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

    # Min signals logic based on cluster mode
    if args.min_signals_required:
        MIN_SIGNALS_REQUIRED = args.min_signals_required
    else:
        MIN_SIGNALS_REQUIRED = 50 if CLUSTER_MODE == "crossover" else 250

    if args.verbose:
        if args.disable_ema_stretch:
            print(f"Disabling EMA strech indicator")
        if args.disable_volatility_compression:
            print(f"Disabling Volatility Compression indicator")
        if args.disable_market_breadth_proxy:
            print(f"Disabling Market Breadth Proxy indicator")
        if args.disable_trend_regime_filter:
            print(f"Disabling Trend Regime Filter indicator")
        if args.disable_rsi:
            print(f"Disabling RSI indicator")
        if args.disable_macd:
            print(f"Disabling MACD indicator")
        if args.disable_stochastic:
            print(f"Disabling Stochastic indicator")
    df = load_data(_ticker=TICKER, _dataset_choice=args.dataset_id)
    if args.verbose:
        print(f"Data loaded: {len(df)} rows | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    close = df["Close"]
    df_open, df_low, df_high  = df['Open'], df['Low'], df['High']
    use_closing_price_strategy = False
    if args.verbose:
        print(f"Using the {'close-to-close stability' if use_closing_price_strategy else 'intra-day sensitivity'} to compute the target")
    # =========================================================
    # TARGET CALCULATION (MODE-AWARE)
    # =========================================================
    if args.mode == "drop":
        # Pour chaque ligne t:
        # close[::-1] : On retourne la liste des prix (le futur devient le passé).
        # rolling(FORWARD_DAYS).min() : On trouve le prix le plus bas dans les jours précédents (qui sont en fait les jours suivants dans le temps réel).
        # shift(-1) : On décale pour s'assurer que le prix actuel (t) n'est pas inclus dans le calcul du "futur". On ne veut comparer t qu'avec [t+1,...t+N]
        # Comparaison : Si le rendement entre le prix actuel et le pire prix futur est inférieur ou égal à votre seuil (ex: -0.03), le 1 est déclenché.
        # Drop mode: look for minimum price in forward window
        if use_closing_price_strategy:
            future_extreme  = close[::-1].rolling(window=FORWARD_DAYS, min_periods=1).min()[::-1].shift(-1)
            df["Target_Pct_Change"] = (future_extreme - close) / close
            # Negative threshold: e.g., -0.03 means 3% drop
            df["Is_Event"] = (df["Target_Pct_Change"] <= THRESHOLD).astype(int)
            df.loc[df.index[-FORWARD_DAYS:], "Is_Event"] = 0
        else:
            # 1. Calcul du point le plus bas sur les N prochains jours
            # On utilise df['low'] pour capturer l'extrême intra-day
            future_low = df_low[::-1].rolling(window=FORWARD_DAYS, min_periods=1).min()[::-1].shift(-1)
            # 2. Calcul du rendement potentiel (Close actuel vs Low futur)
            df["Target_Pct_Change"] = (future_low - close) / close
            # 3. Déclenchement de l'événement
            # On utilise <= car le THRESHOLD est négatif (ex: -0.03)
            df["Is_Event"] = (future_low <= close * (1 + THRESHOLD)).astype(int)
            # 4. SÉCURITÉ : Invalider les dernières lignes
            # Si FORWARD_DAYS = 10, les 10 dernières lignes n'ont pas assez de futur pour être valides
            df.loc[df.index[-FORWARD_DAYS:], "Is_Event"] = 0

    else:  # spike mode
        # # spike mode: look for maximum price in forward window
        if use_closing_price_strategy:
            future_extreme  = close[::-1].rolling(window=FORWARD_DAYS, min_periods=1).max()[::-1].shift(-1)
            df["Target_Pct_Change"] = (future_extreme - close) / close
            # Positive threshold: e.g., 0.03 means 3% spike
            df["Is_Event"] = (df["Target_Pct_Change"] >= THRESHOLD).astype(int)
        else:
            # 1. On cherche le prix le PLUS HAUT sur les N prochains jours
            # On utilise .max() au lieu de .min()
            future_high = df_high[::-1].rolling(window=FORWARD_DAYS, min_periods=1).max()[::-1].shift(-1)
            # 2. Calcul du gain potentiel (Close actuel vs High futur)
            # THRESHOLD doit être positif (ex: 0.05 pour +5%)
            df["Target_Pct_Change"] = (future_high - close) / close
            # 3. Déclenchement de l'événement si le gain >= Seuil
            df["Is_Event"] = (future_high >= close * (1 + THRESHOLD)).astype(int)
            # 4. SÉCURITÉ : On invalide les dernières lignes (données incomplètes)
            df.loc[df.index[-FORWARD_DAYS:], "Is_Event"] = 0

    event_direction = "drop" if args.mode == "drop" else "spike"
    target = df["Is_Event"]
    valid_mask = df["Target_Pct_Change"].notna()

    total_days = valid_mask.sum()
    total_events = target.sum()
    baseline = total_events / total_days * 100
    assert 0 < args.threshold_penalty_for_low_events <= 1
    args.threshold_penalty_for_low_events = int(args.threshold_penalty_for_low_events * total_events)
    def penalty_for_having_low_number_of_events(n):
        return min(n / args.threshold_penalty_for_low_events, 1.0)**2
    if args.softer_penalty_for_low_events:
        if args.verbose:
            print(f"Switching to sigmoid-like curve for penalty of low number of events")
        def penalty_for_having_low_number_of_events(n):
            # Softer penalty: sigmoid-like curve instead of linear cap
            if n >= args.threshold_penalty_for_low_events:
                return 1.0
            return 0.5 + 0.5 * (n / args.threshold_penalty_for_low_events)  # 0.5→1.0 range
    improve_score_function = args.use_z_score_boost
    if args.verbose:
        print(f"\nMode: {event_direction.upper()} | Forward {FORWARD_DAYS} {args.dataset_id} | "
              f"Threshold: {THRESHOLD * 100:+.1f}% | "
              f"Baseline event probability: {baseline:.2f}%   ({total_events} / {total_days}) | "
              f"Penalty threshold @ {args.threshold_penalty_for_low_events}"
              f"{' | Use Z-Score boost' if improve_score_function else ''} | "
              f"Optimizing Edge")
    assert event_direction in ['drop', 'spike']
    def objective(trial):
        base_signal = None
        use_simple_ma = trial.suggest_categorical("use_simple_ma", [True, False])
        if use_simple_ma:
            base_signal = "simple_ma"
        else:
            use_slope = trial.suggest_categorical("use_slope_3days", [True, False])
            if use_slope:
                base_signal = "slope_3days"
            else:
                base_signal = "slope_3days"
        assert base_signal is not None

        use_ema_stretch = trial.suggest_categorical("use_ema_stretch", [True, False]) if not args.disable_ema_stretch else False
        use_volatility_compression = trial.suggest_categorical("use_volatility_compression", [True, False]) if not args.disable_volatility_compression else False
        use_market_breadth_proxy = trial.suggest_categorical("use_market_breadth_proxy", [True, False]) if not args.disable_market_breadth_proxy else False
        use_trend_regime_filter = trial.suggest_categorical("use_trend_regime_filter", [True, False]) if not args.disable_trend_regime_filter else False
        use_rsi_indicator = trial.suggest_categorical("use_rsi_indicator", [True, False]) if not args.disable_rsi else False
        use_macd_indicator = trial.suggest_categorical("use_macd_indicator", [True, False]) if not args.disable_macd else False
        use_stochastic_indicator = trial.suggest_categorical("use_stochastic_indicator", [True, False]) if not args.disable_stochastic else False

        p1, p2 , p3, p4 = int(str(args.cluster_window_params).split(",")[0]), int(str(args.cluster_window_params).split(",")[1]), bool(str(args.cluster_window_params).split(",")[2]), int(str(args.cluster_window_params).split(",")[3])
        cluster_window = trial.suggest_int(name="cluster_window", low=p1, high=p2, log=p3, step=p4)
        p1, p2, p3, p4 = int(str(args.cluster_threshold_params).split(",")[0]), int(str(args.cluster_threshold_params).split(",")[1]), bool(str(args.cluster_threshold_params).split(",")[2]), int(str(args.cluster_threshold_params).split(",")[3])
        cluster_threshold = trial.suggest_int("cluster_threshold", low=p1, high=p2, log=p3, step=p4)

        # ---------------- Base signal ----------------
        cond_price = None
        if base_signal in ["simple_ma", "ecart_type", "slope_3days", "bull_market_global"]:
            sma_len_min, sma_len_max, sma_len_log, sma_len_step = int(str(args.sma_len_params).split(",")[0]), int(str(args.sma_len_params).split(",")[1]), bool(str(args.sma_len_params).split(",")[2]), int(str(args.sma_len_params).split(",")[3])
            sma_len = trial.suggest_int(name="sma_len", low=sma_len_min, high=sma_len_max, log=sma_len_log, step=sma_len_step)
            sma = ta.sma(close, length=sma_len)
            if base_signal == "simple_ma":
                if event_direction == 'drop':
                    cond_price = close > sma
                else:
                    cond_price = close < sma
            elif base_signal == "ecart_type":
                # Calcule à combien d'écarts-types le prix se trouve de la moyenne
                std = close.rolling(window=sma_len).std()
                z_score = (close - sma) / std
                cond_price = z_score > 2.0  # Le prix est "étiré" vers le haut
            elif base_signal == "slope_3days":
                sma_slope = sma.diff(3)  # Pente sur les 3 derniers jours
                if event_direction == 'drop':
                    cond_price = (close > sma) & (sma_slope > 0)
                else:
                    cond_price = (close < sma) & (sma_slope < 0)
            elif base_signal == "bull_market_global":
                cond_regime = ta.sma(close, 50) > ta.sma(close, 200)
                cond_price = (close > sma) & cond_regime
        elif base_signal == "breakout":
            lookback = trial.suggest_int("breakout_lookback", 10, 50, log=False, step=1)
            resistance = close.rolling(lookback).max().shift(1)
            cond_price = close > resistance  # Price breaks above recent high
        assert cond_price is not None, f"Unknown base signal: {base_signal}"
        base_signal = cond_price

        # ---------------- Indicators ----------------
        if use_rsi_indicator:
            rsi_len = trial.suggest_int("rsi_len", 10, 20)
            rsi_thresh = trial.suggest_int("rsi_thresh", 60, 75)
            rsi_lookback = trial.suggest_int("rsi_lookback", 3, 10)
            rsi = ta.rsi(close, length=rsi_len)
            if args.mode == "spike":
                # Momentum without overextension
                cond_rsi = (rsi > 50) & (rsi < rsi_thresh)
            else:
                # Overbought for drop prediction
                cond_rsi = rsi.rolling(rsi_lookback).max() > rsi_thresh
            base_signal = base_signal & cond_rsi

        if use_macd_indicator:
            macd_fast = trial.suggest_int("macd_fast", 8, 15)
            macd_slow = trial.suggest_int("macd_slow", macd_fast + 5, 40)
            macd_signal = trial.suggest_int("macd_signal", 5, 15)
            macd = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd is None: return -10
            macd_line = macd.iloc[:, 0]
            macd_sig = macd.iloc[:, 1]
            # Mode-aware MACD
            if args.mode == "spike":
                cond_macd = macd_line > macd_sig  # Bullish crossover
            else:
                cond_macd = macd_line < macd_sig  # Bearish crossover
            base_signal = base_signal & cond_macd

        if use_ema_stretch:
            ema_len_min, ema_len_max, ema_len_log, ema_len_step = str(args.ema_stretch_params_ema_len).split(",")
            ema_len = trial.suggest_int(name="ema_len", low=int(ema_len_min), high=int(ema_len_max), log=str2bool(ema_len_log), step=int(ema_len_step))
            stretch_thresh_min, stretch_thresh_max, stretch_thresh_log, stretch_thresh_step = str(args.ema_stretch_params_stretch_treshold).split(",")
            stretch_thresh = trial.suggest_float(name="stretch_thresh", low=float(stretch_thresh_min), high=float(stretch_thresh_max), log=str2bool(stretch_thresh_log), step=float(stretch_thresh_step))
            ema = ta.ema(close, length=ema_len)
            stretch = (close - ema) / ema
            cond_stretch = stretch > stretch_thresh
            base_signal = base_signal & cond_stretch

        if use_volatility_compression:
            bb_len = trial.suggest_int("bb_len", 2, 30)
            bb_width_thresh = trial.suggest_float("bb_width_thresh", 0.02, 0.10)
            bb = ta.bbands(close, length=bb_len)
            if bb is None:
                return -10
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

        if use_stochastic_indicator:
            stoch_len = trial.suggest_int("stoch_len", 10, 20)
            stoch_k = trial.suggest_int("stoch_k", 3, 5)
            stoch_d = trial.suggest_int("stoch_d", 3, 5)
            stoch_overbought = trial.suggest_int("stoch_overbought", 70, 85)
            stoch_oversold = trial.suggest_int("stoch_oversold", 15, 30)

            stoch = ta.stoch(close=close, high=df_high, low=df_low, k=stoch_k, d=stoch_d, smooth_k=stoch_len)
            if stoch is None or stoch.empty:
                return -10

            stoch_k_line = stoch.iloc[:, 0]  # %K
            stoch_d_line = stoch.iloc[:, 1]  # %D

            if args.mode == "spike":
                # Bullish: Stochastic rising from oversold or %K > %D in neutral zone
                cond_stoch = ((stoch_k_line > stoch_oversold) & (stoch_k_line < stoch_overbought) &
                              (stoch_k_line > stoch_d_line))
            else:
                # Bearish: Stochastic falling from overbought or %K < %D in overbought zone
                cond_stoch = ((stoch_k_line < stoch_overbought) & (stoch_k_line > stoch_oversold) &
                              (stoch_k_line < stoch_d_line)) | (stoch_k_line > stoch_overbought)

            base_signal = base_signal & cond_stoch

        # ---------------- Cluster logic ----------------
        omen_count = base_signal.rolling(cluster_window).sum()
        if CLUSTER_MODE == "every_day":
            cluster = omen_count >= cluster_threshold
        else:
            prev = omen_count.shift(1)
            cluster = (omen_count >= cluster_threshold) & (prev < cluster_threshold)
        signal_dates = cluster[cluster].index
        # Only evaluate signals where the outcome is actually known
        # valid_mask is defined in the parent scope run_professional_optimization
        valid_signal_indices = signal_dates.intersection(valid_mask[valid_mask].index)
        if len(valid_signal_indices) < MIN_SIGNALS_REQUIRED: return -10
        cluster_targets = target.loc[valid_signal_indices]
        # Safety check again after filtering
        if len(cluster_targets) < MIN_SIGNALS_REQUIRED: return -10
        win_rate = cluster_targets.mean() * 100
        n = len(cluster_targets)
        penalty = penalty_for_having_low_number_of_events(n)
        assert 0 <= penalty <= 1
        trial.set_user_attr("penalty", penalty)
        z = (win_rate / 100 - baseline / 100) / math.sqrt((baseline / 100) * (1 - baseline / 100) / n)
        trial.set_user_attr("z_score", float(z))
        if improve_score_function:
            edge = (win_rate - baseline) / 100  # Edge as decimal
            statistical_confidence = min(abs(z) / 1.96, 1.0)  # Normalize to p<0.05 threshold
            score = (edge * penalty) * (1 + 0.5 * statistical_confidence)  # Cap boost at 50%
        else:
            score = (win_rate - baseline) * penalty  # Only reward edge over baseline
        trial.report(score, step=n)
        trial.set_user_attr("total_events", int(total_events))
        trial.set_user_attr("total_days", int(total_days))
        trial.set_user_attr("n", int(n))
        return score

    def stop_at_threshold(study, trial):
        if trial.value is not None and trial.value > 99.9:
            study.stop()

    callbacks = [stop_at_threshold]
    pruner = MedianPruner(n_startup_trials=25, n_warmup_steps=50)
    # Sampler and Pruner Configuration
    sampler = TPESampler(
        seed=RANDOM_SEED,
        multivariate=True,  # Model parameter dependencies
        warn_independent_sampling=False,
        group=True,  # Decompose conditional search space
        n_startup_trials=60  # More exploration before pruning
    )
    if args.sampler == "cmaes":
        try:
            from optuna.samplers import CmaEsSampler
            sampler = CmaEsSampler(seed=RANDOM_SEED)
        except ImportError:
            print("Warning: CmaEsSampler not available, falling back to TPE.")
    if args.sampler == "grid":
        search_space = {
            "use_ema_stretch": [True, False],
            "use_volatility_compression": [True, False],
            "use_stochastic_indicator": [True, False],
            "base_signal": ["simple_ma", "ecart_type", "slope_3days", "bull_market_global", "breakout"],
            "sma_len": [10, 20, 50, 100],
            "cluster_window": [20, 60, 120],
            "cluster_threshold": [2, 5, 10],
            "stoch_len": [10, 15, 20],
            "stoch_k": [3, 4, 5],
            "stoch_d": [3, 4, 5],
            "stoch_overbought": [70, 75, 80, 85],
            "stoch_oversold": [15, 20, 25, 30],
            "breakout_lookback": [10, 20, 30, 40, 50],
        }
        sampler = GridSampler(seed=RANDOM_SEED, search_space=search_space)
        pruner = None

    if args.sampler == "random":
        pruner = None

    # =========================================================
    # STORAGE & STUDY INITIALIZATION
    # =========================================================
    best_value, best_params, best_study = float(0), None, None

    # Check if Persistent Storage is enabled
    if args.storage:
        # Ensure study name is unique based on config if not provided
        if not args.study_name:
            args.study_name = f"{args.ticker}_{args.mode}_f{args.forward_days}_t{args.threshold}_s{args.seed}"

        # Normalize storage URL for SQLite
        storage_url = args.storage
        if not storage_url.startswith("sqlite:///") and not storage_url.startswith("sqlite:"):
            storage_url = f"sqlite:///{storage_url}"

        if args.verbose:
            print(f"💾 Using Persistent Storage: {storage_url}")
            print(f"📛 Study Name: {args.study_name}")

        # Create or Load Study
        study = optuna.create_study(
            storage=storage_url,
            study_name=args.study_name,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Run Optimization (Continues from last trial if exists)
        if args.verbose:
            print(f"Starting Optimization: {args.trials} trials (or {args.timeout}s timeout)...")
            existing_trials = len(study.trials)
            print(f"Found {existing_trials} existing trials in storage.")

            # Added Optimization Status Information ---
            if existing_trials > 0:
                print(f"🔄 Resuming existing study...")
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

            print(f"📊 Trial Status: {len(complete_trials)} Complete, {len(pruned_trials)} Pruned, {len(fail_trials)} Failed")

            if len(complete_trials) > 0:
                print(f"🏆 Current Best Value: {study.best_value:.4f}")
            # Optional: Uncomment below to see best params on resume
            # print(f"🏆 Current Best Params: {study.best_params}")
            else:
                print(f"⚠️ No complete trials found yet in storage (all pending/pruned).")
        else:
            print(f"🆕 Starting fresh study (no existing trials found).")
        # -----------------------------------------------

        study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                       show_progress_bar=True if args.verbose else False,
                       callbacks=callbacks, gc_after_trial=False)

        # Set best study for downstream processing
        best_study = study
        best_value = study.best_value
        best_params = study.best_params
    else:
        # =========================================================
        # IN-MEMORY MODE (Original Logic)
        # =========================================================
        if args.sampler == "random":
            local_trial = 10000
            n_studies = int(args.trials // local_trial)
            # Ensure at least 1 study if trials < local_trial
            if n_studies == 0 and args.trials > 0:
                n_studies = 1
                local_trial = args.trials
            if args.verbose:
                print(f"Using Random Sampler")
                print(f"There will be {n_studies} studies of {local_trial} passes each")
            t1 = time.time()
            for j in range(n_studies):
                t2 = time.time()
                # In memory mode, we vary seed per study chunk
                sampler_chunk = RandomSampler(seed=RANDOM_SEED + j)
                study = optuna.create_study(direction="maximize", sampler=sampler_chunk, pruner=pruner)
                study.optimize(objective, n_trials=local_trial, timeout=args.timeout,
                               show_progress_bar=True if args.verbose else False,
                               callbacks=callbacks, gc_after_trial=False)
                t3 = time.time()
                # On garde seulement le meilleur score de ce bloc
                if study.best_value > best_value:
                    best_study = study
                    best_value = study.best_value
                    best_params = study.best_params
                print(f"@iter={j + 1} Best value is {best_value}. Time remaining: {format_execution_time((t3 - t2) * (n_studies - j))}")
            print(f"Best value is {best_value}\nBest parameters are:\n{best_params}")
            study = best_study  # For the rest of the code
        else:
            # Create or Load Study
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )

            # Run Optimization (Continues from last trial if exists)
            if args.verbose:
                print(f"\nStarting Optimization: {args.trials} trials (or {args.timeout}s timeout)...")

            study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                           show_progress_bar=True if args.verbose else False,
                           callbacks=callbacks, gc_after_trial=False)

            # Set best study for downstream processing
            best_study = study
            best_value = study.best_value
            best_params = study.best_params

    # =========================================================
    # POST-OPTIMIZATION REPORTING
    # =========================================================

    if args.verbose:
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"Total des essais : {len(study.trials)}    Essais utiles (calculés) : {len(complete_trials)}    Doublons évités (élagués) : {len(pruned_trials)}")

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
    best_params.update({'threshold': THRESHOLD, 'ticker': args.ticker, 'dataset_id': args.dataset_id, 'forward_days': FORWARD_DAYS, 'cluster_mode': CLUSTER_MODE,
                        'mode': args.mode, 'use_closing_price_strategy': use_closing_price_strategy})
    info = verify_best(df_data=close, df_open=df_open, df_low=df_low, df_high=df_high, cluster_mode=CLUSTER_MODE, params=best_params, target=target, valid_mask=valid_mask, baseline=baseline,
                       forward_days=FORWARD_DAYS, threshold=THRESHOLD, mode=args.mode, verbose=args.verbose)
    best_params.update({'win_rate': info["win_rate"], 'baseline': info["baseline"], 'threshold_penalty_for_low_events': args.threshold_penalty_for_low_events,
                        'penalty': study.best_trial.user_attrs['penalty'], 'total_events': total_events, 'total_days': total_days, 'edge': info["win_rate"] - info["baseline"]})
    if args.verbose:
        print(f"Infos extra: {study.best_trial.user_attrs}")
        # How many trials had statistically significant results?
        significant = [t for t in study.trials if t.user_attrs.get('z_score', 0) > 1.96]
        # What was the Z-score of the best trial?
        print(f"Best trial Z-score: {study.best_trial.user_attrs['z_score']:.2f}   Trials with z_score>1.96: {len(significant)} / {len(study.trials)}")
        # 0      Your result = baseline           No edge
        # ±1     Result is 1 SD from baseline     ~68% of random results fall here
        # ±1.96  Result is ~2 SD from baseline    p < 0.05 (common significance threshold)
        # ±2.58  Result is ~2.6 SD from baseline  p < 0.01 (strong evidence)
        # ±3+    Result is very far from baseline p < 0.003 (very strong evidence)
    # =========================================================
    # SAVE BEST PARAMETERS
    # =========================================================
    if args.save_params_to:
        save_best_params(best_params, args.save_params_to, args.verbose)


# =========================================================
# VERIFICATION & REAL-TIME PREDICTION
# =========================================================
def verify_best(df_data, df_open, df_low, df_high, cluster_mode, params, target, valid_mask, baseline, forward_days, threshold, verbose, mode):
    event_direction = "drop" if mode == "drop" else "spike"
    # 1. Base Signal Logic
    base_signal = "simple_ma" if 'use_simple_ma' in params and params['use_simple_ma'] else ""
    base_signal = "slope_3days" if 'use_slope_3days' in params and params['use_slope_3days'] else base_signal
    cond_price = None
    if base_signal == "simple_ma":
        sma = ta.sma(df_data, length=params["sma_len"])
        if event_direction == 'drop':
            cond_price = df_data > sma
        else:
            cond_price = df_data < sma
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
        if event_direction == 'drop':
            cond_price = (df_data > sma) & (sma_slope > 0)
        else:
            cond_price = (df_data < sma) & (sma_slope < 0)
    elif base_signal == "bull_market_global":
        # On ne prend des signaux d'achat que si on est en "Bull Market" global
        sma = ta.sma(df_data, length=params["sma_len"])
        cond_regime = ta.sma(df_data, 50) > ta.sma(df_data, 200)
        cond_price = (df_data > sma) & cond_regime
    elif base_signal == "breakout":
        lookback = params["breakout_lookback"]
        resistance = df_data.rolling(lookback).max().shift(1)
        cond_price = df_data > resistance  # Price breaks above recent high
    assert cond_price is not None
    base_signal = cond_price

    # 2. Calculate Indicators
    if params.get("use_rsi_indicator"):
        rsi = ta.rsi(df_data, length=params["rsi_len"])
        if mode == "spike":
            # Momentum without overextension
            cond_rsi = (rsi > 50) & (rsi < params["rsi_thresh"])
        else:
            # Overbought for drop prediction
            cond_rsi = rsi.rolling(params["rsi_lookback"]).max() > params["rsi_thresh"]

        base_signal = base_signal & cond_rsi

    if params.get("use_macd_indicator"):
        macd = ta.macd(df_data, fast=params["macd_fast"], slow=params["macd_slow"], signal=params["macd_signal"])
        macd_line = macd.iloc[:, 0]
        macd_sig = macd.iloc[:, 1]
        # Mode-aware MACD
        if mode == "spike":
            cond_macd = macd_line > macd_sig  # Bullish crossover
        else:
            cond_macd = macd_line < macd_sig  # Bearish crossover
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

    if params.get("use_stochastic_indicator"):
        stoch = ta.stoch(close=df_data,high=df_high,low=df_low,k=params["stoch_k"], d=params["stoch_d"], smooth_k=params["stoch_len"])
        if stoch is None or stoch.empty:
            stoch_k_line = pd.Series(50, index=df_data.index)  # Fallback neutral
            stoch_d_line = pd.Series(50, index=df_data.index)
        else:
            stoch_k_line = stoch.iloc[:, 0]
            stoch_d_line = stoch.iloc[:, 1]

        if params.get("mode") == "spike":
            cond_stoch = ((stoch_k_line > params["stoch_oversold"]) &
                          (stoch_k_line < params["stoch_overbought"]) &
                          (stoch_k_line > stoch_d_line))
        else:
            cond_stoch = ((stoch_k_line < params["stoch_overbought"]) &
                          (stoch_k_line > params["stoch_oversold"]) &
                          (stoch_k_line < stoch_d_line)) | (stoch_k_line > params["stoch_overbought"])

        base_signal = base_signal & cond_stoch

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
        assert np.allclose(win_rate, precision*100, atol=0.5), f"\t\t{win_rate=}  vs  {precision*100=}"
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
            'is_active_now': is_active_now, 'event_direction': event_direction, 'threshold': threshold , 'is_active_str': _tmp_is_active_str, 'forward_days': forward_days}


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
    df = load_data(_ticker=best_params['ticker'], _dataset_choice=best_params['dataset_id'])
    if verbose:
        print(f"Data loaded: {len(df)} rows | "
              f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    close = df["Close"]
    df_open, df_low, df_high = df['Open'], df['Low'], df['High']

    # Calculate target for verification (historical performance only)
    FORWARD_DAYS = best_params['forward_days']
    THRESHOLD = best_params['threshold']
    use_closing_price_strategy = best_params['use_closing_price_strategy']
    if best_params['mode'] == "drop":
        # Pour chaque ligne t:
        # close[::-1] : On retourne la liste des prix (le futur devient le passé).
        # rolling(FORWARD_DAYS).min() : On trouve le prix le plus bas dans les jours précédents (qui sont en fait les jours suivants dans le temps réel).
        # Comparaison : Si le rendement entre le prix actuel et le pire prix futur est inférieur ou égal à votre seuil (ex: -0.03), le 1 est déclenché.
        # Drop mode: look for minimum price in forward window
        if use_closing_price_strategy:
            future_extreme = close[::-1].rolling(window=FORWARD_DAYS, min_periods=1).min()[::-1].shift(-1)
            df["Target_Pct_Change"] = (future_extreme - close) / close
            # Negative threshold: e.g., -0.03 means 3% drop
            df["Is_Event"] = (df["Target_Pct_Change"] <= THRESHOLD).astype(int)
            df.loc[df.index[-FORWARD_DAYS:], "Is_Event"] = 0
        else:
            # 1. Calcul du point le plus bas sur les N prochains jours
            # On utilise df['low'] pour capturer l'extrême intra-day
            future_low = df_low[::-1].rolling(window=FORWARD_DAYS, min_periods=1).min()[::-1].shift(-1)
            # 2. Calcul du rendement potentiel (Close actuel vs Low futur)
            df["Target_Pct_Change"] = (future_low - close) / close
            # 3. Déclenchement de l'événement
            # On utilise <= car le THRESHOLD est négatif (ex: -0.03)
            df["Is_Event"] = (future_low <= close * (1 + THRESHOLD)).astype(int)
            # 4. SÉCURITÉ : Invalider les dernières lignes
            # Si FORWARD_DAYS = 10, les 10 dernières lignes n'ont pas assez de futur pour être valides
            df.loc[df.index[-FORWARD_DAYS:], "Is_Event"] = 0
    else:  # spike mode
        # # Spike mode: look for maximum price in forward window
        if use_closing_price_strategy:
            future_extreme = close[::-1].rolling(window=FORWARD_DAYS, min_periods=1).max()[::-1].shift(-1)
            df["Target_Pct_Change"] = (future_extreme - close) / close
            # Positive threshold: e.g., 0.03 means 3% spike
            df["Is_Event"] = (df["Target_Pct_Change"] >= THRESHOLD).astype(int)
        else:
            # 1. On cherche le prix le PLUS HAUT sur les N prochains jours
            # On utilise .max() au lieu de .min()
            future_high = df_high[::-1].rolling(window=FORWARD_DAYS, min_periods=1).max()[::-1].shift(-1)
            # 2. Calcul du gain potentiel (Close actuel vs High futur)
            # THRESHOLD doit être positif (ex: 0.05 pour +5%)
            df["Target_Pct_Change"] = (future_high - close) / close
            # 3. Déclenchement de l'événement si le gain >= Seuil
            df["Is_Event"] = (future_high >= close * (1 + THRESHOLD)).astype(int)
            # 4. SÉCURITÉ : On invalide les dernières lignes (données incomplètes)
            df.loc[df.index[-FORWARD_DAYS:], "Is_Event"] = 0

    event_direction = "drop" if best_params['mode'] == "drop" else "spike"
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
    info = verify_best(df_data=close, df_open=df_open, df_low=df_low, df_high=df_high, cluster_mode=best_params['cluster_mode'], params=best_params,
                       target=target,valid_mask=valid_mask, baseline=baseline,forward_days=FORWARD_DAYS, threshold=THRESHOLD,
                       mode=best_params['mode'], verbose = verbose,)
    info.update({"total_days": total_days, "total_events": total_events})
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
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
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
             "For --mode spike: use positive value (e.g., 0.03 for 3%% spike)."
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
        "--threshold-penalty-for-low-events",
        type=float,
        default=0.95,
        help="Minimal number (pourcentage of the total number of events) of required past events to disable penalty on the score."
    )
    strat_group.add_argument(
        "--softer-penalty-for-low-events",
        action="store_true",
        help="Use a sigmoid more like function for penalty of low events."
    )
    strat_group.add_argument(
        "--mode",
        type=str,
        choices=["drop", "spike"],
        default="drop",
        help="Prediction mode: 'drop' for downward movements (price decreases), "
             "'spike' for upward spikes (price increases). Threshold sign should match mode."
    )
    strat_group.add_argument(
        "--disable-ema-stretch",
        action="store_true",
        help="Do not use the EMA stretch indicator"
    )
    strat_group.add_argument(
        "--disable-volatility-compression",
        action="store_true",
        help="Do not use the Volatility Compression indicator"
    )
    strat_group.add_argument(
        "--disable-market-breadth-proxy",
        action="store_true",
        help="Do not use the Market Breadth Proxy indicator"
    )
    strat_group.add_argument(
        "--disable-trend-regime-filter",
        action="store_true",
        help="Do not use the Trend Regime Filter indicator"
    )
    strat_group.add_argument(
        "--disable-rsi",
        action="store_true",
        help="Do not use the RSI indicator"
    )
    strat_group.add_argument(
        "--disable-macd",
        action="store_true",
        help="Do not use the MACD indicator"
    )
    strat_group.add_argument(
        "--disable-stochastic",
        action="store_true",
        help="Do not use the Stochastic indicator"
    )

    bound_group = parser.add_argument_group("Bound Parameters")
    bound_group.add_argument("--sma-len-params", type=str, default="2,100,false,1", )
    bound_group.add_argument("--ema-stretch-params-ema-len", type=str, default="2,200,false,1", )
    bound_group.add_argument("--ema-stretch-params-stretch-treshold", type=str, default="0.01,0.08,false,0.01", )
    bound_group.add_argument("--cluster-window-params", type=str, default="2,120,false,1", )
    bound_group.add_argument("--cluster-threshold-params", type=str, default="2,12,false,1", )

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
        choices=["tpe", "cmaes", "random", "grid"],
        default="tpe",
        help="Optuna sampler algorithm."
    )
    opt_group.add_argument(
        "--use-z-score-boost",
        action="store_true",
        help="Enable Z-Score statistical boost in the objective function to favor statistical significance."
    )

    # --- Storage & Resuming ---
    storage_group = parser.add_argument_group("Storage & Resuming")
    storage_group.add_argument("--storage", type=str, default=None,
                               help="Path to SQLite DB for persistent storage (e.g., 'optuna.db'). Enables resuming.")
    storage_group.add_argument("--study-name", type=str, default=None,
                               help="Unique name for the study. If using --storage, use same name to resume.")

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
    if args.mode == "spike" and args.threshold < 0:
        parser.error(f"--mode 'spike' requires positive threshold, got {args.threshold}")

    # Handle Optuna Logging Verbosity
    if args.optuna_verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.verbose:
        print(f"===== BASELINE =====")
        print(f"What it is: The natural frequency of your target event occurring in the data, expressed as a percentage."
              f"\nExample: If SPX drops ≥3% within 20 days on 150 out of 1000 valid days → baseline = 15.0%"
              f"\nInterpretation: \"If I randomly picked days (or always predicted 'drop'), I'd be right ~15% of the time.\"")
        print(f"===== WIN RATE =====")
        print(f"What it is: The percentage of your model's signals that correctly predicted an event."
              f"\nExample: Your strategy generated 80 signals; 40 of them were followed by a ≥3% drop → win_rate = 50.0%"
              f"\nInterpretation: \"When my model says 'drop coming', it's correct 50% of the time.\"")

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