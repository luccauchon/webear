#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Optimization Pipeline – Bullish Day Prediction
=============================================================

Optimizes clustering-based market regime detection to identify conditions
favorable for bullish price action (Close > Open) using Optuna for hyperparameter tuning.

Key Features:
• Regime detection via K-Means or Gaussian Mixture clustering
• Binary target: 1 if Close > Open (bullish day), 0 otherwise
• Expectancy-based filtering with risk-adjusted edge ratio
• Optuna optimization with SQLite/PostgreSQL persistence
• CLI interface for flexible configuration
• Configurable enhanced features via --add-enhanced-features flag

Author: Luc Cauchon
Date: 2026
"""

# =========================================================
# IMPORTS
# =========================================================
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import shutil

# Technical analysis
import pandas_ta as ta
from pandas_ta import macd
from utils import (
    DATASET_AVAILABLE, next_weekday, next_week, next_month,
    get_next_step, is_weekday, is_last_weekend_of_month
)

# Machine Learning
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

# Optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Statistics
from statsmodels.stats.proportion import proportion_confint

# Logging configuration
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
DEFAULT_RANDOM_SEED = 42


# =========================================================
# ALL REGIMES SUMMARY – For Real-Time Mode
# =========================================================
def print_all_regimes_summary(_stats, _n_clusters, _args):
    """
    Print a concise summary table of ALL regimes for quick comparison.
    """
    print("\n" + "═" * 80)
    print(" " * 25 + "📊 ALL REGIMES SUMMARY")
    print("═" * 80 + "\n")

    print(f"Target: Close > Open × (1 + {args.bullish_threshold * 100:.4%}) (Bullish Day)\n")

    # Table header
    print(f"{'Regime':<8} {'Samples':>10} {'Bullish %':>12} {'95% CI Lower':>14} {'95% CI Upper':>14} {'EV/$1':>10} {'Recommendation':>18}")
    print("─" * 80)

    for r in range(_n_clusters):
        if r not in _stats:
            print(f"{r:<8} {'--':>10} {'--':>12} {'--':>14} {'--':>14} {'--':>10} {'(insufficient data)':>18}")
            continue

        stats = _stats[r]
        prob_bullish = stats['prob_bullish']
        ci_low = stats['ci_95_lower']
        ci_upp = stats['ci_95_upper']
        ev = 2 * prob_bullish - 1  # EV per $1 risked: p*(+1) + (1-p)*(-1)

        # Simple recommendation logic
        rec = "✅ FAVORABLE"
        print(f"{r:<8} {stats['count']:>10,} {prob_bullish * 100:>11.2f}% {ci_low * 100:>13.2f}% {ci_upp * 100:>13.2f}% {ev:>+9.3f} {rec:>18}")

    print("─" * 80)
    print("💡 Tip: Use regime ID with custom logic to filter trades by regime profile.\n")


# =========================================================
# MODEL NAMING HELPER
# =========================================================
def generate_model_filename(_ticker, _study_name, _params, _metadata_extra=None):
    """
    Generate a unique, descriptive model filename based on experiment configuration.
    """
    import re
    from datetime import datetime

    ticker_clean = re.sub(r'[^\w\-]', '', _ticker.replace('^', ''))
    study_clean = re.sub(r'[^\w\-]', '', _study_name)

    algo = _params.get('clustering_algo', 'unknown')
    n_clusters = _params.get('n_clusters', 'NA')
    enhanced = "enh" if _metadata_extra and _metadata_extra.get('add_enhanced_features', False) else "std"

    # For Close>Open target, these are metadata only (not used in logic)
    target_type = "bullish"  # New target identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{ticker_clean}__{study_clean}__"
        f"{algo}c{n_clusters}__{enhanced}__"
        f"{target_type}__"
        f"{timestamp}.pkl"
    )
    return filename


# =========================================================
# CLUSTER CHARACTERIZATION
# =========================================================
def characterize_clusters(_features, _model, _scaler, _stats):
    """Return human-readable characteristics for each cluster"""
    X = _scaler.transform(_features.dropna())
    regimes = _model.predict(X)
    df_clustered = _features.dropna().copy()
    df_clustered['regime'] = regimes

    cluster_chars = {}
    for r in _stats.keys():
        subset = df_clustered[df_clustered['regime'] == r]
        cluster_chars[r] = {
            'n_samples': len(subset),
            'prob_bullish': _stats[r]['prob_bullish'],
            'avg_vol_10': subset['vol_10'].mean(),
            'avg_rsi': subset['rsi'].mean(),
            'avg_trend_strength': subset['trend_strength'].mean(),
            'avg_atr_ratio': subset['atr_ratio'].mean(),
            'dominant_month': subset['month'].mode().iloc[0] if not subset['month'].empty else None,
        }
    return cluster_chars


# =========================================================
# PRINT ALL CLUSTER CHARACTERISTICS
# =========================================================
def print_all_cluster_characteristics(_stats, _features, _model, _scaler, _n_clusters):
    """Print comprehensive characteristics for ALL discovered clusters"""
    print("\n" + "═" * 70)
    print(" " * 20 + "🔍 ALL REGIME CHARACTERISTICS")
    print("═" * 70 + "\n")

    X = _scaler.transform(_features.dropna())
    regimes = _model.predict(X)
    df_clustered = _features.dropna().copy()
    df_clustered['regime'] = regimes

    for r in range(_n_clusters):
        if r not in _stats:
            print(f"⚠️  Regime #{r}: Skipped (insufficient samples)\n")
            continue

        stats = _stats[r]
        cluster_data = df_clustered[df_clustered['regime'] == r]

        print(f"📦 REGIME #{r}")
        print(f"   ├─ Samples: {stats['count']}")
        print(f"   ├─ Bullish Probability: {stats['prob_bullish'] * 100:.2f}%")
        print(f"   ├─ 95% CI (Wilson): [{stats['ci_95_lower'] * 100:.2f}%, {stats['ci_95_upper'] * 100:.2f}%]")
        print(f"   ├─ Outcome Std Dev: {stats['std_outcome']:.3f}")
        print(f"   ├─ 🎯 Feature Profile:")
        print(f"   │  • Avg Volatility (10d): {cluster_data['vol_10'].mean():.4f}")
        print(f"   │  • Avg RSI: {cluster_data['rsi'].mean():.2f}")
        print(f"   │  • Avg Trend Strength: {cluster_data['trend_strength'].mean():.4f}")
        print(f"   │  • Avg ATR Ratio: {cluster_data['atr_ratio'].mean():.4f}")
        print(f"   │  • Avg Volume Ratio: {cluster_data['volume_ratio'].mean():.2f}")
        print(f"   │  • Dominant Month: {cluster_data['month'].mode().iloc[0] if not cluster_data['month'].empty else 'N/A'}")

        ev_per_dollar = 2 * stats['prob_bullish'] - 1
        recommendation = "✅ FAVORABLE" if stats['prob_bullish'] > 0.6 else "⚠️ NEUTRAL" if stats['prob_bullish'] > 0.45 else "❌ AVOID"
        print(f"   └─ 💡 Trading Implication: {recommendation} | EV per $1: ${ev_per_dollar:.3f}")
        print()


# =========================================================
# TRADE DECISION HELPER – Generalized Binary Outcome
# =========================================================
def should_trade_binary_outcome(_regime_stats, _win_payout, _loss_amount, _min_edge_ratio):
    """
    Evaluate whether a trade has positive expectancy given regime probabilities.

    This function is generalized for any binary outcome (e.g., Close > Open).

    Parameters:
    -----------
    _regime_stats : dict
        Must contain 'prob_bullish' (probability of favorable outcome)
    _win_payout : float
        Profit if outcome is favorable (e.g., +1.0 for +100% return)
    _loss_amount : float
        Loss if outcome is unfavorable (e.g., 1.0 for -100% return)
    _min_edge_ratio : float
        Minimum edge per dollar risked to approve trade

    Returns:
    --------
    dict with trade decision and metrics
    """
    prob_bullish = _regime_stats.get('prob_bullish')

    if prob_bullish is None:
        return {
            'trade': False, 'expectancy': None, 'edge_ratio': None,
            'break_even_prob': None, 'message': "❌ Missing 'prob_bullish' in regime stats"
        }

    prob_bearish = 1 - prob_bullish
    expectancy = (prob_bullish * _win_payout) - (prob_bearish * _loss_amount)
    edge_ratio = expectancy / _loss_amount if _loss_amount > 0 else 0

    # Break-even: p*win - (1-p)*loss = 0 → p = loss / (win + loss)
    denominator = _win_payout + _loss_amount
    break_even_prob = _loss_amount / denominator if denominator > 0 else 1.0

    trade = (edge_ratio >= _min_edge_ratio) and (prob_bullish > break_even_prob)

    if not trade:
        if edge_ratio < _min_edge_ratio:
            reason = f"Edge ratio {edge_ratio * 100:.2f}% < minimum {_min_edge_ratio * 100:.1f}%"
        elif prob_bullish <= break_even_prob:
            reason = f"Win rate {prob_bullish * 100:.1f}% ≤ break-even {break_even_prob * 100:.1f}%"
        else:
            reason = "Unknown filter failed"
        message = f"⛔ SKIP: {reason}"
    else:
        message = f"✅ APPROVE: Edge {edge_ratio * 100:.2f}% | EV ${expectancy:.3f}"

    return {
        'trade': trade, 'expectancy': expectancy, 'edge_ratio': edge_ratio,
        'break_even_prob': break_even_prob, 'message': message
    }


# =========================================================
# DATA LOADING
# =========================================================
def load_data(_ticker, _dataset_id):
    """Load preprocessed price data from pickle cache."""
    try:
        from utils import get_filename_for_dataset
        filename = get_filename_for_dataset(_dataset_id, older_dataset=None)
    except ImportError:
        filename = f"data/{_dataset_id}_cache.pkl"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")

    with open(filename, "rb") as f:
        cache = pickle.load(f)

    if _ticker not in cache:
        raise KeyError(f"Ticker '{_ticker}' not found. Available: {list(cache.keys())}")

    df = cache[_ticker].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index().dropna()
    if len(df) == 0:
        raise ValueError(f"No valid data for ticker {_ticker} after cleaning")
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def build_features(_df, _pct1, _pct2, _pct3, _vol1, _vol2, _vol3,
                   _rsi_length, _ema1, _ema2, _atr_period, _add_enhanced_features):
    """
    Build technical features for regime classification.

    Features include: returns, volatility, momentum, trend, volume,
    seasonality, and enhanced metrics (tail risk, Bollinger Bands, ADX, etc.)
    """
    close = _df["Close"]
    _features = pd.DataFrame(index=_df.index)

    # ===== Return Features =====
    _features["ret_5"] = close.pct_change(_pct1)
    _features["ret_10"] = close.pct_change(_pct2)
    _features["ret_20"] = close.pct_change(_pct3)

    # ===== Volatility Features =====
    returns = close.pct_change()
    _features["vol_10"] = returns.rolling(_vol1, min_periods=_vol1).std()
    _features["vol_20"] = returns.rolling(_vol2, min_periods=_vol2).std()
    _features["vol_60"] = returns.rolling(_vol3, min_periods=_vol3).std()

    # ===== Momentum & Trend =====
    _features["rsi"] = ta.rsi(close, _rsi_length)
    ema_short = ta.ema(close, _ema1)
    ema_long = ta.ema(close, _ema2)
    _features["dist_ema50"] = (close - ema_short) / ema_short
    _features["dist_ema200"] = (close - ema_long) / ema_long
    _features["trend_strength"] = (ema_short - ema_long) / ema_long

    # ===== Volatility Normalization =====
    atr = ta.atr(_df["High"], _df["Low"], _df["Close"], length=_atr_period)
    _features["atr_ratio"] = atr / close

    # ===== Volume & Seasonality =====
    _features["volume_ratio"] = _df["Volume"] / _df["Volume"].rolling(20).mean()
    _features["month"] = _df.index.month
    _features["day_of_week"] = _df.index.dayofweek
    _features["quarter_end"] = (_df.index.month % 3 == 0).astype(int)

    # ===== MACD =====
    macd_data = macd(close)
    _features["macd_histogram"] = macd_data.get("MACDh_12_26_9", pd.Series(0, index=close.index))
    _features["vol_regime_change"] = (_features["vol_10"] / _features["vol_60"].shift(1)) - 1

    # ===== ENHANCED FEATURES (Optional) =====
    if _add_enhanced_features:
        # Tail risk
        for window in [20, 60]:
            rw = returns.rolling(window, min_periods=window)
            _features[f"skew_{window}"] = rw.apply(lambda x: x.skew(), raw=False)
            _features[f"kurt_{window}"] = rw.apply(lambda x: x.kurtosis(), raw=False)

        # Volatility term structure
        _features["vol_term_structure"] = _features["vol_10"] / (_features["vol_60"] + 1e-8)
        _features["vol_expansion"] = _features["vol_10"] - _features["vol_60"].shift(1)

        # Bollinger Bands
        bb_middle = ta.sma(close, 20)
        bb_std = returns.rolling(20, min_periods=20).std()
        bb_upper = bb_middle + 2 * bb_std * close
        bb_lower = bb_middle - 2 * bb_std * close
        _features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        _features["bb_width"] = (bb_upper - bb_lower) / close
        _features["bb_squeeze"] = (_features["bb_width"] < _features["bb_width"].rolling(60).quantile(0.25)).astype(float)

        # ADX + Trend Quality
        adx_data = ta.adx(_df["High"], _df["Low"], close, length=14)
        _features["adx"] = adx_data["ADX_14"]
        _features["trend_quality"] = _features["adx"] * abs(_features["trend_strength"])

        # Drawdown & Recovery
        roll_max = close.rolling(60, min_periods=1).max()
        roll_min = close.rolling(60, min_periods=1).min()
        _features["drawdown"] = (close - roll_max) / roll_max
        _features["recovery_ratio"] = ((close - roll_min) / (roll_max - roll_min + 1e-8)).clip(0, 1)

        # Momentum Divergence
        _div = 5
        _features["rsi_bearish_div"] = ((close > close.rolling(_div).max().shift(1)) &
                                        (_features["rsi"] < _features["rsi"].rolling(_div).max().shift(1))).astype(float)
        _features["rsi_bullish_div"] = ((close < close.rolling(_div).min().shift(1)) &
                                        (_features["rsi"] > _features["rsi"].rolling(_div).min().shift(1))).astype(float)

        # Volume Confirmation
        obv = ta.obv(close, _df["Volume"])
        _features["obv_momentum"] = obv.pct_change(10)
        _features["volume_vol"] = _df["Volume"].rolling(20).std() / _df["Volume"].rolling(20).mean()
        _features["volume_spike"] = (_df["Volume"] > _df["Volume"].rolling(60).quantile(0.9)).astype(float)

        # Gap Analysis (if Open available)
        if "Open" in _df.columns:
            _features["gap_pct"] = (_df["Open"] - close.shift(1)) / close.shift(1)
            _features["gap_vol"] = _features["gap_pct"].rolling(20).std()
            _features["gap_filled"] = (((close >= _df["Open"]) & (close.shift(1) <= _df["Open"])) |
                                       ((close <= _df["Open"]) & (close.shift(1) >= _df["Open"]))).astype(float)
        else:
            for col in ["gap_pct", "gap_vol", "gap_filled"]:
                _features[col] = np.nan

        # Volatility Persistence
        abs_ret = returns.abs()

        def roll_autocorr(s, lag=1, win=20):
            def _ac(x):
                xc = x.dropna()
                return xc.autocorr(lag=lag) if len(xc) > lag + 1 else np.nan

            return s.rolling(win, min_periods=win).apply(_ac, raw=False)

        _features["vol_persistence_1"] = roll_autocorr(abs_ret, lag=1, win=20)
        _features["vol_persistence_5"] = roll_autocorr(abs_ret, lag=1, win=60)

        # Seasonal Volatility Interaction
        hv_thresh = _features["vol_10"].rolling(252, min_periods=126).quantile(0.75)
        _features["high_vol_month"] = ((_features["vol_10"] > hv_thresh) &
                                       (_features["month"].isin([9, 10, 11]))).astype(float)
        _features["q4_vol_interaction"] = ((_features["month"].isin([10, 11, 12])) &
                                           (_features["vol_10"] > _features["vol_10"].rolling(60).median())).astype(float)

        # Clip extreme values for stability
        for col in ["skew_20", "skew_60", "kurt_20", "kurt_60", "bb_position",
                    "drawdown", "recovery_ratio", "obv_momentum", "volume_vol", "gap_pct"]:
            if col in _features.columns:
                lo, hi = _features[col].quantile([0.05, 0.95])
                if pd.notna(lo) and pd.notna(hi) and lo < hi:
                    _features[col] = _features[col].clip(lo, hi)

    return _features


# =========================================================
# TARGET CONSTRUCTION – Close > Open * (1 + threshold)
# =========================================================
def build_target(_df, _bullish_threshold=0.0):
    """
    Build binary target: 1 if Close > Open * (1 + threshold), 0 otherwise.

    Parameters:
    -----------
    _df : pd.DataFrame with 'Close' and 'Open' columns
    _bullish_threshold : float, optional (default=0.0)
        Minimum fractional gain required. E.g., 0.005 = Close must exceed Open by 0.5%%

    Returns:
    --------
    pd.Series : Binary target aligned with dataframe index
    """
    close = _df["Close"]
    open_price = _df["Open"]
    threshold_multiplier = 1.0 + _bullish_threshold
    return (close > (open_price * threshold_multiplier)).astype(int)


# =========================================================
# LATEST PREDICTION HELPER
# =========================================================
def predict_latest(_features, _model, _scaler, _stats):
    """Predict regime for the most recent valid data point."""
    valid_features = _features.dropna()
    if len(valid_features) == 0:
        print("⚠️  No valid features available for prediction")
        return None

    latest_date = valid_features.index[-1]
    latest = valid_features.iloc[[-1]]
    X = _scaler.transform(latest)
    regime = _model.predict(X)[0]

    if regime not in _stats:
        print(f"⚠️  Predicted regime {regime} not found in stats")
        return None, None
    return regime, _stats[regime]


# =========================================================
# REPORT PRINTING
# =========================================================
def print_report(_regime, _stats, _latest_date=None, _latest_close_value=None):
    """Print formatted regime analysis report for bullish day prediction."""
    print("\n" + "=" * 60)
    print(" " * 15 + "BULLISH DAY REGIME ANALYSIS")
    print("=" * 60 + "\n")

    print(f"📊 Detected Regime: #{_regime}")
    if _latest_date:
        print(f"📅 Latest Date: {_latest_date.strftime('%Y-%m-%d')} | Close: ${_latest_close_value:.2f}")
    print()

    print("📋 Regime Statistics:")
    print(f"   • Historical Samples: {_stats['count']} (of {_stats['total_count']})")
    print(f"   • Bullish Probability: {_stats['prob_bullish'] * 100:.2f}%")
    print(f"   • Bearish Probability: {_stats['prob_itm'] * 100:.2f}%")
    print()
    print("🔒 95% Confidence Interval (Wilson):")
    print(f"   [{_stats['ci_95_lower'] * 100:.2f}% — {_stats['ci_95_upper'] * 100:.2f}%]")
    print()
    print("⚡ Risk Metrics:")
    print(f"   • Outcome Std Dev: {_stats['std_outcome']:.3f}")
    print("\n" + "=" * 60 + "\n")


# =========================================================
# REAL-TIME INFERENCE FUNCTION
# =========================================================
def run_real_time_inference(args, ticker, list_models, model_filename):
    """Load a saved model and run inference on the latest data point."""
    normal_verbose = not args.short_verbose
    hyper_silence = args.hypershort_verbose

    if normal_verbose and not hyper_silence:
        print("⚡ REAL-TIME INFERENCE MODE – Bullish Day Prediction")
        print("-" * 60)

    # Handle --list-models
    if list_models:
        models_dir = Path(args.output_dir)
        if not models_dir.exists():
            print("📁 No models directory found.")
            return
        pattern = f"{ticker.replace('^', '')}__*.pkl"
        models = list(models_dir.glob(pattern))
        if not models:
            print(f"🔍 No models found: {pattern}")
            return
        print(f"\n📦 Available models for {ticker}:")
        print("─" * 70)
        for m in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
            stat = m.stat()
            print(f"• {m.name}")
            print(f"  └─ Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')} | {stat.st_size / 1024:.1f} KB")
        print("─" * 70)
        return

    # Determine model path
    if model_filename:
        model_path = f"{args.output_dir}/{args.model_filename}"
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model not found: {model_path}")
            return
    else:
        latest_link = f"{args.output_dir}/{ticker.replace('^', '')}_regime_model_latest.pkl"
        if os.path.exists(latest_link):
            model_path = latest_link
        else:
            models_dir = Path(args.output_dir)
            pattern = f"{ticker.replace('^', '')}__*.pkl"
            models = list(models_dir.glob(pattern))
            if not models:
                print(f"❌ ERROR: No models for {ticker}")
                return
            model_path = str(max(models, key=lambda x: x.stat().st_mtime))

    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    _model = model_data["model"]
    _scaler = model_data["scaler"]
    _stats = model_data["stats"]
    _params = model_data["params"]
    _metadata = model_data.get("metadata", {})
    assert "bullish_threshold" in _metadata
    model_threshold = _metadata.get("bullish_threshold", 0.0)
    assert "add_enhanced_features" in _metadata
    use_enhanced_features = _metadata.get("add_enhanced_features", False)

    print(f"Target Threshold: Close > Open × (1 + {model_threshold * 100:.4%})")
    if not hyper_silence:
        print(f"✅ Model loaded ({_metadata.get('timestamp', 'N/A')})")

    # Load data
    try:
        df = load_data(ticker, _metadata['dataset_id'])
        if not hyper_silence:
            print(f"✅ Data loaded for \"{_metadata['dataset_id']}\"")
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return

    # Clip incomplete bar if needed
    if _metadata['dataset_id'] in ['day', 'week', 'month'] and args.clip:
        now = datetime.now()
        if _metadata['dataset_id'] == 'day' and now.date() == df.index[-1].date():
            df = df.iloc[:-1].copy()
        elif _metadata['dataset_id'] == 'week' and is_weekday(now):
            df = df.iloc[:-1].copy()
        elif _metadata['dataset_id'] == 'month' and not is_last_weekend_of_month(now):
            df = df.iloc[:-1].copy()

    # Build features with saved params
    features = build_features(
        _df=df,
        _pct1=_params.get('pct1', 5), _pct2=_params.get('pct2', 10), _pct3=_params.get('pct3', 20),
        _vol1=_params.get('vol1', 10), _vol2=_params.get('vol2', 20), _vol3=_params.get('vol3', 60),
        _rsi_length=_params.get('rsi_length', 14),
        _ema1=_params.get('ema1', 50), _ema2=_params.get('ema2', 200),
        _atr_period=_params.get('atr_period', 14),
        _add_enhanced_features=use_enhanced_features,
    )

    # Predict latest regime
    valid_features = features.dropna()
    if len(valid_features) == 0:
        print("❌ ERROR: No valid features")
        return

    latest_date = valid_features.index[-1]
    latest_close = df['Close'].iloc[-1]
    X_latest = _scaler.transform(valid_features.iloc[[-1]])
    regime = _model.predict(X_latest)[0]

    if regime not in _stats:
        print(f"⚠️  Regime #{regime} has no statistics")
        return

    regime_stats = _stats[regime]

    # Print report
    if normal_verbose and not hyper_silence:
        print_report(regime, regime_stats, latest_date, latest_close)

    # Show all regimes summary
    if normal_verbose and not hyper_silence:
        print_all_regimes_summary(_stats, _params['n_clusters'], args)

    # Final output
    regime_probs = {r: _stats[r]['prob_bullish'] for r in range(_params['n_clusters']) if r in _stats}
    regime_counts = {r: _stats[r]['count'] for r in range(_params['n_clusters']) if r in _stats}
    sorted_r = dict(sorted(regime_probs.items(), key=lambda x: x[1], reverse=True))

    parts = []
    for k, v in sorted_r.items():
        fmt = f"{k}:{v:.1%}:{regime_counts[k]}"
        parts.append(f"\033[1m**{fmt}**\033[0m" if k == regime else fmt)

    print(f"📊 Detected Regime: #{regime} → [{','.join(parts)}]")
    print(f"   Bullish Probability: {regime_stats['prob_bullish'] * 100:.2f}%")

    if normal_verbose and not hyper_silence:
        print("\n" + "═" * 60 + "\n✨ INFERENCE COMPLETE\n" + "═" * 60)


# =========================================================
# MAIN PIPELINE FUNCTION
# =========================================================
def entry_main(args):
    """Main execution pipeline for regime optimization."""
    # Extract arguments
    ticker = getattr(args, 'ticker', '^GSPC')
    dataset_id = getattr(args, 'dataset_id', 'day')
    study_name = getattr(args, 'study_name', 'bullish_clustering')
    storage_url = getattr(args, 'storage_url', 'sqlite:///bullish_study.db')
    max_n_trials = getattr(args, 'max_n_trials', 99999)
    timeout = getattr(args, 'timeout', 86400)
    n_jobs = getattr(args, 'n_jobs', 1)
    random_seed = getattr(args, 'random_seed', DEFAULT_RANDOM_SEED)
    min_n_in_cluster = getattr(args, 'min_n_in_cluster', 160)

    # Trade evaluation params (generalized for binary outcome)
    win_payout = getattr(args, 'credit_received', 1.0)  # Profit if Close > Open
    loss_amount = getattr(args, 'spread_width', 1.0) - getattr(args, 'credit_received', 0.0)  # Loss if not
    min_edge_ratio = getattr(args, 'min_edge_ratio', 0.04)

    np.random.seed(random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Starting Bullish Day Regime Optimization")
    print(f"   Ticker: {ticker} | Dataset: {dataset_id}")
    print(f"   Target: Close > Open | Mininum number of samples in a cluster: {min_n_in_cluster}  (yet don't understand why this is needed)")
    print(f"   Trials: {max_n_trials} | Timeout: {timeout}s")
    print(f"   Enhanced Features: {'✅ ENABLED' if args.add_enhanced_features else '❌ DISABLED (standard)'}")
    print("-" * 60)

    # Load data
    df = load_data(ticker, dataset_id)
    if args.lookback_years > 0:
        rows_per_year = {'day': 252, 'week': 52, 'month': 12}.get(dataset_id, 252)
        cutoff = args.lookback_years * rows_per_year
        print(f"📅 Using {args.lookback_years}-year lookback: ~{cutoff} rows")
        df = df.iloc[-cutoff:].copy()

    print(f"📦 Loaded {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

    total_rows = len(df)
    split_idx = int(len(df) * 0.8)
    print(f"   Train: {split_idx} | Test: {total_rows - split_idx}")

    # =========================================================
    # OPTUNA OBJECTIVE FUNCTION
    # =========================================================
    def objective(trial):
        """Maximize regime-based predictive edge for bullish days."""
        # Hyperparameter suggestions (dataset-specific ranges)
        if dataset_id == 'day':
            _n_clusters = trial.suggest_int("n_clusters", args.min_clusters, args.max_clusters)
            _pct1, _pct2, _pct3 = trial.suggest_int("pct1", 1, 5), trial.suggest_int("pct2", 5, 10), trial.suggest_int("pct3", 10, 20)
            _vol1, _vol2, _vol3 = trial.suggest_int("vol1", 1, 10), trial.suggest_int("vol2", 10, 20), trial.suggest_int("vol3", 20, 60)
            _ema1, _ema2 = trial.suggest_int("ema1", 40, 60), trial.suggest_int("ema2", 180, 220)
            _atr_period, _rsi_length = trial.suggest_int("atr_period", 2, 21), trial.suggest_int("rsi_length", 10, 20)
        elif dataset_id == 'week':
            _n_clusters = trial.suggest_int("n_clusters", args.min_clusters, args.max_clusters)
            _pct1, _pct2, _pct3 = trial.suggest_int("pct1", 1, 3), trial.suggest_int("pct2", 3, 8), trial.suggest_int("pct3", 8, 16)
            _vol1, _vol2, _vol3 = trial.suggest_int("vol1", 1, 10), trial.suggest_int("vol2", 10, 20), trial.suggest_int("vol3", 20, 30)
            _ema1, _ema2 = trial.suggest_int("ema1", 8, 14), trial.suggest_int("ema2", 40, 44)
            _atr_period, _rsi_length = trial.suggest_int("atr_period", 4, 12), trial.suggest_int("rsi_length", 6, 14)
        else:  # month
            _n_clusters = trial.suggest_int("n_clusters", args.min_clusters, args.max_clusters)
            _pct1, _pct2, _pct3 = trial.suggest_int("pct1", 1, 2), trial.suggest_int("pct2", 2, 4), trial.suggest_int("pct3", 4, 8)
            _vol1, _vol2, _vol3 = trial.suggest_int("vol1", 1, 3), trial.suggest_int("vol2", 3, 6), trial.suggest_int("vol3", 6, 18)
            _ema1, _ema2 = trial.suggest_int("ema1", 2, 6), trial.suggest_int("ema2", 12, 24)
            _atr_period, _rsi_length = trial.suggest_int("atr_period", 3, 8), trial.suggest_int("rsi_length", 4, 10)

        _clustering_algo = trial.suggest_categorical("clustering_algo", ["kmeans", "gaussian_mixture"])

        # Time-series split (prevent look-ahead bias)
        df_train, df_test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

        # Build features & target
        def _build(df_subset):
            feats = build_features(df_subset, _pct1, _pct2, _pct3, _vol1, _vol2, _vol3,
                                   _rsi_length, _ema1, _ema2, _atr_period,
                                   _add_enhanced_features=args.add_enhanced_features)
            targ = build_target(df_subset, _bullish_threshold=args.bullish_threshold)
            return pd.concat([feats, targ.rename('target')], axis=1).dropna()

        train_data, test_data = _build(df_train), _build(df_test)
        # Validate minimum data requirements
        if len(train_data) < 1 or len(test_data) < 1:
            return 0.0  # length of "0" make Optuna crash

        # Scale features (fit on train only)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(train_data.drop(columns=['target']))
        X_test = scaler.transform(test_data.drop(columns=['target']))

        # Train clustering model
        if _clustering_algo == 'kmeans':
            model = KMeans(n_clusters=_n_clusters, init='k-means++', n_init=10, max_iter=900,
                           tol=1e-4, random_state=random_seed, algorithm='lloyd', verbose=0)
        elif _clustering_algo == 'gaussian_mixture':
            model = GaussianMixture(n_components=_n_clusters, covariance_type='full', tol=1e-3,
                                    reg_covar=1e-5, max_iter=900, n_init=20, init_params='k-means++',
                                    random_state=random_seed, verbose=0)
        elif _clustering_algo == 'birch':
            _thresh = trial.suggest_float("birch_threshold", 0.1, 1.0, log=True)
            _branch = trial.suggest_int("birch_branching", 20, 100)
            model = Birch(threshold=_thresh, branching_factor=_branch, n_clusters=_n_clusters)
        else:
            raise ValueError(f"Unknown algorithm: {_clustering_algo}")

        regimes_train = model.fit_predict(X_train)
        regimes_test = model.predict(X_test)
        test_data = test_data.copy()
        test_data['regime'] = regimes_test

        # Evaluate each regime on test data
        scores, valid = [], 0
        for r in range(_n_clusters):
            subset = test_data[test_data["regime"] == r]["target"]
            if len(subset) < min_n_in_cluster:
                continue

            prob_bullish = subset.mean()
            prob_score = prob_bullish
            consistency = 1 - (prob_bullish * (1 - prob_bullish))
            sample_bonus = np.clip(len(subset) / (2 * min_n_in_cluster), 0, 1)

            # Separation from other regimes
            other = [test_data[test_data["regime"] == i]["target"].mean()
                     for i in range(_n_clusters) if i != r and len(test_data[test_data["regime"] == i]["target"]) >= min_n_in_cluster]
            separation = abs(prob_bullish - np.average(other, weights=[len(test_data[test_data["regime"] == i]["target"])
                                                                       for i in range(_n_clusters) if i != r and len(test_data[test_data["regime"] == i]["target"]) >= min_n_in_cluster])) if len(other) >= 2 else (abs(prob_bullish - other[0]) if other else 0)

            combined = 0.50 * prob_score + 0.25 * consistency + 0.15 * sample_bonus + 0.10 * separation
            scores.append(combined)
            valid += 1

        if not scores:
            return 0.0
        base = np.mean(scores)
        return base * (valid / _n_clusters) if args.penalize_invalid_cluster else base

    # =========================================================
    # OPTUNA STUDY SETUP
    # =========================================================
    print(f"\n🔬 Initializing Optuna study: '{study_name}'")
    study = optuna.create_study(study_name=study_name, direction="maximize",
                                sampler=TPESampler(seed=random_seed),
                                pruner=MedianPruner(n_startup_trials=10),
                                storage=storage_url, load_if_exists=True)

    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n📊 Status: {len(completed)}/{len(trials)} completed")
    if completed:
        print(f"   Best score: {study.best_value:.6f} | Params: {study.best_params}")
    print("─" * 40)

    if args.confirmation_before_run:
        input("Appuyez sur Entrée pour continuer...")

    print("⚙️  Running optimization...\n")
    study.optimize(objective, n_trials=max_n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=True)

    # =========================================================
    # RETRAIN WITH BEST PARAMETERS
    # =========================================================
    print(f"\n✅ Optimization complete! Best score: {study.best_value:.6f}")
    best_params = study.best_params.copy()
    _n_clusters = best_params['n_clusters']
    _clustering_algo = best_params['clustering_algo']
    print(f"🎯 Algorithm: {_clustering_algo} | Clusters: {_n_clusters}")

    # Build final model on full data
    features = build_features(df.copy(), best_params.get('pct1', 5), best_params.get('pct2', 10), best_params.get('pct3', 20),
                              best_params.get('vol1', 10), best_params.get('vol2', 20), best_params.get('vol3', 60),
                              best_params.get('rsi_length', 14), best_params.get('ema1', 50), best_params.get('ema2', 200),
                              best_params.get('atr_period', 14), _add_enhanced_features=args.add_enhanced_features)
    target = build_target(df.copy(), _bullish_threshold=args.bullish_threshold)
    _aligned = pd.concat([features, target.rename('target')], axis=1).dropna()

    if len(_aligned) == 0:
        print("❌ ERROR: No valid data after alignment")
        return

    scaler = RobustScaler()
    X = scaler.fit_transform(_aligned.drop(columns=['target']))

    # Train final model
    if _clustering_algo == 'kmeans':
        model = KMeans(n_clusters=_n_clusters, random_state=random_seed, n_init=10, algorithm='lloyd', max_iter=900, tol=1e-4)
    elif _clustering_algo == 'gaussian_mixture':
        model = GaussianMixture(n_components=_n_clusters, random_state=random_seed, n_init=20, covariance_type='full', max_iter=900, tol=1e-3, reg_covar=1e-5, init_params='k-means++')
    elif _clustering_algo == 'birch':
        model = Birch(threshold=best_params.get('birch_threshold', 0.5), branching_factor=best_params.get('birch_branching', 50), n_clusters=_n_clusters)
    else:
        raise ValueError(f"Unknown algorithm: {_clustering_algo}")

    regimes = model.fit_predict(X)
    _aligned["regime"] = regimes

    # Compute regime statistics
    _stats = {}
    for r in range(_n_clusters):
        subset = _aligned[_aligned["regime"] == r]["target"].dropna()
        if len(subset) < min_n_in_cluster:
            continue
        prob_bullish = subset.mean()
        ci_low, ci_upp = proportion_confint(count=subset.sum(), nobs=len(subset), alpha=0.05, method='wilson')
        _stats[r] = {
            "count": len(subset), "total_count": total_rows,
            "prob_bullish": prob_bullish, "prob_itm": 1 - prob_bullish,
            "std_outcome": np.sqrt(prob_bullish * (1 - prob_bullish)),
            "ci_95_lower": ci_low, "ci_95_upper": ci_upp,
            "dataset_id": dataset_id,
        }

    if not _stats:
        print(f"⚠️  WARNING: No clusters met minimum sample requirement ({min_n_in_cluster})")
        return

    print_all_cluster_characteristics(_stats, features, model, scaler, _n_clusters)

    # Predict latest regime
    result = predict_latest(features, model, scaler, _stats)
    if result is None:
        print("⚠️  Could not predict regime for latest data point")
    else:
        regime, regime_stats = result
        print_report(regime, regime_stats)

        # Trade evaluation (generalized)
        print("💰 TRADE DECISION EVALUATION")
        print("─" * 40)
        trade_decision = should_trade_binary_outcome(
            _regime_stats=regime_stats,
            _win_payout=win_payout,
            _loss_amount=loss_amount,
            _min_edge_ratio=min_edge_ratio
        )
        print(f"Configuration: Win +${win_payout:.2f} | Loss -${loss_amount:.2f} | Min Edge {min_edge_ratio * 100:.1f}%")
        print(f"Break-Even Win Rate: {trade_decision['break_even_prob'] * 100:.2f}%")
        print(f"Expected Value: ${trade_decision['expectancy']:.3f} | Edge Ratio: {trade_decision['edge_ratio'] * 100:.2f}%")
        print(f"🎯 DECISION: {trade_decision['message']}")

    # Save model
    print(f"\n💾 Saving model...")
    metadata_extra = {'add_enhanced_features': args.add_enhanced_features}
    model_filename = generate_model_filename(ticker, study_name, best_params, metadata_extra)
    model_path = f"{args.output_dir}/{model_filename}"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model, "scaler": scaler, "stats": _stats, "params": best_params,
            "metadata": {
                "ticker": ticker,
                "target": "bullish_close_gt_open",
                "bullish_threshold": args.bullish_threshold,
                "add_enhanced_features": args.add_enhanced_features,
                "timestamp": datetime.now(), "study_name": study_name,
                "best_score": study.best_value, "dataset_id": dataset_id
            }
        }, f)

    print(f"✅ Saved: {model_path}")
    print("\n" + "═" * 60 + "\n✨ PIPELINE COMPLETE\n" + "═" * 60)


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize regime clustering for bullish day prediction (Close > Open)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        TRADE LOGIC:
        • Target: Binary outcome where 1 = Close > Open (bullish day)
        • Regimes are clusters of similar market conditions
        • Trade approval requires:
          1. Edge ratio (expectancy/risk) ≥ min_edge_ratio
          2. Bullish probability > break-even threshold

        FEATURE ENGINEERING:
        • --add-enhanced-features: Include advanced features (skew, kurtosis, Bollinger Bands, 
          ADX, drawdown analysis, volume confirmation, gap analysis, etc.)
        • Enhanced features improve predictive power but increase training time by ~15-40%
        • IMPORTANT: Training and inference MUST use the same enhanced features setting
        """
    )

    # Data config
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Ticker symbol")
    parser.add_argument("--dataset-id", type=str, default="day", choices=DATASET_AVAILABLE, help="Frequency")
    parser.add_argument("--lookback-years", type=int, default=99999, help="Years of history (0 = all)")
    parser.add_argument("--output-dir", type=str, default="models", help="Model save directory")

    # Optuna config
    parser.add_argument("--study-name", type=str, default="bullish_clustering", help="Optuna study name")
    parser.add_argument("--storage-url", type=str, default=None, help="DB URL for persistence, ie: sqlite:///put_5_days.db")
    parser.add_argument("--max-n-trials", type=int, default=999999, help="Max trials")
    parser.add_argument("--min-clusters", type=int, default=4, help="Min clusters")
    parser.add_argument("--max-clusters", type=int, default=9, help="Max clusters")
    parser.add_argument("--timeout", type=int, default=86400, help="Max runtime (seconds)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Reproducibility seed")

    # Clustering config
    parser.add_argument("--min-n-in-cluster", type=int, default=33, help="Min samples per cluster")
    parser.add_argument("--penalize-invalid-cluster", action="store_true", help="Penalize unused clusters")

    # Trade evaluation (generalized for binary outcome)
    parser.add_argument("--credit-received", type=float, default=1.0, help="Profit if bullish (win payout)")
    parser.add_argument("--spread-width", type=float, default=2.0, help="Total risk (win + loss amount)")
    parser.add_argument("--min-edge-ratio", type=float, default=0.04, help="Min edge/risk to approve trade")
    parser.add_argument("--bullish-threshold", type=float, default=0.0,
                        help="Minimum %% gain required for bullish classification (e.g., 0.005 = 0.5%%). Target: Close > Open * (1 + threshold)")

    # Feature engineering config
    parser.add_argument("--add-enhanced-features", action="store_true",
                        help="Include enhanced features (skew, kurtosis, Bollinger Bands, ADX, drawdown, volume confirmation, gap analysis, etc.)")

    # Execution mode
    parser.add_argument("--real-time", action="store_true", help="Inference mode (skip optimization)")
    parser.add_argument("--short-verbose", action="store_true", help="Minimal output in real-time mode")
    parser.add_argument("--hypershort-verbose", action="store_true", help="Very minimal output")
    parser.add_argument("--confirmation-before-run", action="store_true", help="Prompt before running")

    # Regime filtering (real-time only)
    parser.add_argument("--model-filename", type=str, default=None, help="Specific model to load")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.real_time:
            run_real_time_inference(args, args.ticker, args.list_models, args.model_filename)
        else:
            entry_main(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)