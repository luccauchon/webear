#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Credit Spread Regime Optimization Pipeline
==========================================

Optimizes clustering-based market regime detection to evaluate credit spread
trading opportunities using Optuna for hyperparameter tuning.

Key Features:
• Regime detection via K-Means or Gaussian Mixture clustering
• Binary target: probability of spread expiring OTM (profitable)
• Expectancy-based trade filtering with risk-adjusted edge ratio
• Optuna optimization with SQLite/PostgreSQL persistence
• CLI interface for flexible configuration

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
from utils import DATASET_AVAILABLE
# Machine Learning
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN,
    Birch, OPTICS
)
from sklearn.neighbors import NearestNeighbors  # For DBSCAN prediction

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
# MODEL NAMING HELPER
# =========================================================
def generate_model_filename(_ticker, _study_name, _params, _metadata_extra=None):
    """
    Generate a unique, descriptive model filename based on experiment configuration.

    Parameters:
    -----------
    _ticker : str
        Asset ticker symbol
    _study_name : str
        Optuna study name
    _params : dict
        Best parameters from optimization
    _metadata_extra : dict, optional
        Additional metadata to include in filename

    Returns:
    --------
    str : Safe filename for model storage
    """
    import re
    from datetime import datetime

    # Base components
    ticker_clean = re.sub(r'[^\w\-]', '', _ticker.replace('^', ''))
    study_clean = re.sub(r'[^\w\-]', '', _study_name)

    # Key experiment identifiers
    algo = _params.get('clustering_algo', 'unknown')
    n_clusters = _params.get('n_clusters', 'NA')
    spread_type = _metadata_extra.get('spread_type', 'put') if _metadata_extra else 'put'
    strike_dist = _metadata_extra.get('strike_distance', 0.03) if _metadata_extra else 0.03
    forward = _metadata_extra.get('forward_days', 20) if _metadata_extra else 20

    # Timestamp for uniqueness within same config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename (keep it readable but under ~200 chars for filesystem compatibility)
    filename = (
        f"{ticker_clean}__{study_clean}__"
        f"{algo}c{n_clusters}__"
        f"{spread_type}_sd{int(strike_dist * 100)}pct_dte{forward}__"
        f"{timestamp}.pkl"
    )

    return filename


# =========================================================
#
# =========================================================
def compute_oos_regime_stats(_model, _scaler, _features_test, _target_test, _n_clusters, min_n):
    """Compute regime stats on true out-of-sample data"""
    X_test = _scaler.transform(_features_test.dropna())
    regimes_test = _model.predict(X_test)
    test_df = _features_test.dropna().copy()
    test_df['regime'] = regimes_test
    test_df['target'] = _target_test.dropna()

    stats_oos = {}
    for r in range(_n_clusters):
        subset = test_df[test_df['regime'] == r]['target'].dropna()
        if len(subset) >= min_n:
            prob = subset.mean()
            ci_low, ci_upp = proportion_confint(subset.sum(), len(subset), method='wilson')
            stats_oos[r] = {'prob_otm': prob, 'prob_itm': 1-prob, 'count': len(subset), 'ci_lower_oos': ci_low, 'ci_upper_oos': ci_upp}
    return stats_oos


# =========================================================
#
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
            'prob_otm': _stats[r]['prob_otm'],
            'avg_vol_10': subset['vol_10'].mean(),
            'avg_rsi': subset['rsi'].mean(),
            'avg_trend_strength': subset['trend_strength'].mean(),
            'avg_atr_ratio': subset['atr_ratio'].mean(),
            'dominant_month': subset['month'].mode().iloc[0] if not subset['month'].empty else None,
            # Add more interpretable features as needed
        }
    return cluster_chars


# =========================================================
#
# =========================================================
def print_all_cluster_characteristics(_stats, _features, _model, _scaler, _n_clusters):
    """Print comprehensive characteristics for ALL discovered clusters"""
    print("\n" + "═" * 70)
    print(" " * 20 + "🔍 ALL REGIME CHARACTERISTICS")
    print("═" * 70 + "\n")

    # Get feature means per cluster for interpretability
    X = _scaler.transform(_features.dropna())
    regimes = _model.predict(X)
    df_clustered = _features.dropna().copy()
    df_clustered['regime'] = regimes

    summary_rows = []

    for r in range(_n_clusters):
        if r not in _stats:
            print(f"⚠️  Regime #{r}: Skipped (insufficient samples)\n")
            continue

        stats = _stats[r]
        cluster_data = df_clustered[df_clustered['regime'] == r]

        # Core stats
        print(f"📦 REGIME #{r}")
        print(f"   ├─ Samples: {stats['count']}")
        print(f"   ├─ OTM Probability: {stats['prob_otm'] * 100:.2f}%")
        print(f"   ├─ 95% CI (Wilson): [{stats['ci_95_lower'] * 100:.2f}%, {stats['ci_95_upper'] * 100:.2f}%]")
        print(f"   ├─ Outcome Std Dev: {stats['std_outcome']:.3f}")

        # Feature fingerprints (what defines this regime?)
        print(f"   ├─ 🎯 Feature Profile:")
        print(f"   │  • Avg Volatility (10d): {cluster_data['vol_10'].mean():.4f}")
        print(f"   │  • Avg RSI: {cluster_data['rsi'].mean():.2f}")
        print(f"   │  • Avg Trend Strength: {cluster_data['trend_strength'].mean():.4f}")
        print(f"   │  • Avg ATR Ratio: {cluster_data['atr_ratio'].mean():.4f}")
        print(f"   │  • Avg Volume Ratio: {cluster_data['volume_ratio'].mean():.2f}")
        print(f"   │  • Dominant Month: {cluster_data['month'].mode().iloc[0] if not cluster_data['month'].empty else 'N/A'}")

        # Trading implication
        ev_per_dollar = stats['prob_otm'] - stats['prob_itm']
        recommendation = "✅ FAVORABLE" if stats['prob_otm'] > 0.6 else "⚠️ NEUTRAL" if stats['prob_otm'] > 0.45 else "❌ AVOID"
        print(f"   └─ 💡 Trading Implication: {recommendation} | EV per $1: ${ev_per_dollar:.3f}")
        print()

        # Collect for CSV export
        summary_rows.append({
            'regime_id': r,
            'n_samples': stats['count'],
            'prob_otm': stats['prob_otm'],
            'ci_lower': stats['ci_95_lower'],
            'ci_upper': stats['ci_95_upper'],
            'avg_vol_10': cluster_data['vol_10'].mean(),
            'avg_rsi': cluster_data['rsi'].mean(),
            'avg_trend': cluster_data['trend_strength'].mean(),
            'ev_per_dollar': ev_per_dollar,
            'recommendation': recommendation
        })


# =========================================================
# TRADE DECISION HELPER - Credit Spread Expectancy Filter
# =========================================================
def should_trade_credit_spread(_regime_stats, _credit_received, _max_loss, _min_edge_ratio):
    """
    Evaluate whether a credit spread trade has positive expectancy given regime probabilities.

    Parameters:
    -----------
    _regime_stats : dict
        Output from regime stats computation (must contain 'prob_otm')
    _credit_received : float
        Premium received per share (e.g., 0.40 for $0.40)
    _max_loss : float
        Max loss per share if spread expires ITM (spread_width - credit)
    _min_edge_ratio : float, default=0.05
        Minimum edge per dollar risked to approve trade

    Returns:
    --------
    dict with keys:
        - 'trade': bool, whether to take the trade
        - 'expectancy': float, expected P&L per spread in dollars
        - 'edge_ratio': float, expectancy / max_loss (risk-adjusted edge)
        - 'break_even_prob': float, minimum prob_otm needed for positive EV
        - 'message': str, human-readable summary
    """
    prob_otm = _regime_stats.get('prob_otm')

    if prob_otm is None:
        return {
            'trade': False,
            'expectancy': None,
            'edge_ratio': None,
            'break_even_prob': None,
            'message': "❌ Missing 'prob_otm' in regime stats"
        }

    prob_itm = 1 - prob_otm

    # Expected value in dollar terms per share
    expectancy = (prob_otm * _credit_received) - (prob_itm * _max_loss)

    # Risk-adjusted edge: expectancy per $1 risked
    edge_ratio = expectancy / _max_loss if _max_loss > 0 else 0

    # Break-even probability: minimum prob_otm for EV >= 0
    # Solve: p*credit - (1-p)*max_loss = 0  →  p = max_loss / (credit + max_loss)
    denominator = _credit_received + _max_loss
    break_even_prob = _max_loss / denominator if denominator > 0 else 1.0

    # Decision: require both edge threshold AND win rate above break-even
    trade = (edge_ratio >= _min_edge_ratio) and (prob_otm > break_even_prob)

    # Build human-readable message
    if not trade:
        if edge_ratio < _min_edge_ratio:
            reason = f"Edge ratio {edge_ratio * 100:.2f}% < minimum {_min_edge_ratio * 100:.1f}%"
        elif prob_otm <= break_even_prob:
            reason = f"Win rate {prob_otm * 100:.1f}% ≤ break-even {break_even_prob * 100:.1f}%"
        else:
            reason = "Unknown filter failed"
        message = f"⛔ SKIP: {reason}"
    else:
        message = f"✅ APPROVE: Edge {edge_ratio * 100:.2f}% | EV ${expectancy:.3f}/share"

    return {
        'trade': trade,
        'expectancy': expectancy,
        'edge_ratio': edge_ratio,
        'break_even_prob': break_even_prob,
        'message': message
    }


# =========================================================
# DATA LOADING
# =========================================================
def load_data(_ticker, _dataset_id):
    """
    Load preprocessed price data from pickle cache.

    Parameters:
    -----------
    _ticker : str
        Asset ticker symbol
    _dataset_id : str
        Dataset frequency identifier (e.g., 'day', 'hour')

    Returns:
    --------
    pd.DataFrame : Cleaned price data with OHLCV columns
    """
    try:
        from utils import get_filename_for_dataset
        filename = get_filename_for_dataset(_dataset_id, older_dataset=None)
    except ImportError:
        # Fallback if utils module not available
        filename = f"data/{_dataset_id}_cache.pkl"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")

    with open(filename, "rb") as f:
        cache = pickle.load(f)

    if _ticker not in cache:
        raise KeyError(f"Ticker '{_ticker}' not found in dataset cache. Available: {list(cache.keys())}")

    df = cache[_ticker].copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure chronological order and remove missing values
    df = df.sort_index().dropna()

    if len(df) == 0:
        raise ValueError(f"No valid data remaining for ticker {_ticker} after cleaning")

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def build_features(_df, _pct1, _pct2, _pct3, _vol1, _vol2, _vol3, _rsi_length, _ema1, _ema2, _atr_period):
    """
    Build technical features for regime classification.

    Parameters:
    -----------
    _df : pd.DataFrame
        Input dataframe with OHLCV columns
    _pct1, _pct2, _pct3 : int
        Lookback periods for return calculations
    _vol1, _vol2, _vol3 : int
        Rolling windows for volatility calculations
    _rsi_length : int
        Period for RSI calculation
    _ema1, _ema2 : int
        Periods for short/long EMAs
    _atr_period : int
        Period for ATR calculation

    Returns:
    --------
    pd.DataFrame : Feature matrix indexed by date
    """
    close = _df["Close"]
    _features = pd.DataFrame(index=_df.index)

    # ===== Return Features =====
    assert all(p > 0 for p in [_pct1, _pct2, _pct3]), "Return periods must be positive"
    _features["ret_5"] = close.pct_change(_pct1)
    _features["ret_10"] = close.pct_change(_pct2)
    _features["ret_20"] = close.pct_change(_pct3)

    # ===== Volatility Features =====
    assert all(v > 0 for v in [_vol1, _vol2, _vol3]), "Volatility windows must be positive"
    returns = close.pct_change()
    _features["vol_10"] = returns.rolling(_vol1, min_periods=_vol1).std()
    _features["vol_20"] = returns.rolling(_vol2, min_periods=_vol2).std()
    _features["vol_60"] = returns.rolling(_vol3, min_periods=_vol3).std()

    # ===== Momentum & Trend Features =====
    assert _rsi_length > 0, "RSI length must be positive"
    _features["rsi"] = ta.rsi(close, _rsi_length)

    assert all(e > 0 for e in [_ema1, _ema2]), "EMA periods must be positive"
    ema_short = ta.ema(close, _ema1)
    ema_long = ta.ema(close, _ema2)

    _features["dist_ema50"] = (close - ema_short) / ema_short
    _features["dist_ema200"] = (close - ema_long) / ema_long
    _features["trend_strength"] = (ema_short - ema_long) / ema_long

    # ===== Volatility Normalization =====
    assert _atr_period > 0, "ATR period must be positive"
    atr = ta.atr(_df["High"], _df["Low"], _df["Close"], length=_atr_period)
    _features["atr_ratio"] = atr / close

    # ===== Volume Features =====
    _features["volume_ratio"] = _df["Volume"] / _df["Volume"].rolling(20).mean()

    # ===== Seasonality Features =====
    _features["month"] = _df.index.month
    _features["day_of_week"] = _df.index.dayofweek
    _features["quarter_end"] = (_df.index.month % 3 == 0).astype(int)

    # ===== MACD =====
    macd_data = macd(close)
    _features["macd_histogram"] = macd_data.get("MACDh_12_26_9", pd.Series(0, index=close.index))

    # ===== Volatility Regime Changes =====
    _features["vol_regime_change"] = (_features["vol_10"] / _features["vol_60"].shift(1)) - 1

    return _features


# =========================================================
# TARGET CONSTRUCTION - OTM Expiration Probability
# =========================================================
def build_target(_df, _forward_days, _strike_distance, _spread_type):
    """
    Build binary target: 1 if credit spread expires OTM (profitable), 0 otherwise.

    Parameters:
    -----------
    _df : pd.DataFrame
        Input dataframe with 'Close' column
    _forward_days : int
        Days to expiration / holding period
    _strike_distance : float
        Distance from current price to short strike (0.05 = 5%)
    _spread_type : str
        One of: 'put', 'call', 'iron_condor'

    Returns:
    --------
    pd.Series : Binary target aligned with input dataframe index
    """
    close = _df["Close"]

    assert _forward_days > 0, "Forward days must be positive"
    assert 0 < _strike_distance < 1, "Strike distance must be between 0 and 1"

    future_close = close.shift(-_forward_days)

    if _spread_type == 'put':
        # Bullish put credit spread: profit if price > short_put_strike
        short_strike = close * (1 - _strike_distance)
        target = (future_close > short_strike).astype(int)

    elif _spread_type == 'call':
        # Bearish call credit spread: profit if price < short_call_strike
        short_strike = close * (1 + _strike_distance)
        target = (future_close < short_strike).astype(int)

    elif _spread_type == 'iron_condor':
        # Iron condor: profit if price stays between both short strikes
        put_strike = close * (1 - _strike_distance)
        call_strike = close * (1 + _strike_distance)
        target = ((future_close > put_strike) & (future_close < call_strike)).astype(int)

    else:
        raise ValueError(
            f"Unknown spread_type: '{_spread_type}'. "
            f"Valid options: 'put', 'call', 'iron_condor'"
        )

    return target


# =========================================================
# LATEST PREDICTION HELPER
# =========================================================
def predict_latest(_features, _model, _scaler, _stats):
    """
    Predict regime for the most recent valid data point.

    Parameters:
    -----------
    _features : pd.DataFrame
        Full feature matrix
    _model : sklearn clustering model
        Trained clustering model
    _scaler : sklearn scaler
        Fitted feature scaler
    _stats : dict
        Pre-computed regime statistics

    Returns:
    --------
    tuple or None : (regime_id, regime_stats) or None if prediction fails
    """
    # Get last valid (non-NaN) row
    valid_features = _features.dropna()

    if len(valid_features) == 0:
        print("⚠️  No valid features available for prediction")
        return None

    latest_date = valid_features.index[-1]
    print(f"Latest valid date: {latest_date.strftime('%Y-%m-%d')}")

    latest = valid_features.iloc[[-1]]  # Keep as DataFrame for transform
    X = _scaler.transform(latest)

    regime = _model.predict(X)[0]

    if regime not in _stats:
        print(f"⚠️  Predicted regime {regime} not found in stats (may have insufficient samples)")
        return None

    return regime, _stats[regime]


# =========================================================
# REPORT PRINTING
# =========================================================
def print_report(_regime, _stats, _strike_distance, _spread_type):
    """Print formatted regime analysis report."""
    print("\n" + "=" * 60)
    print(" " * 15 + "CREDIT SPREAD REGIME ANALYSIS")
    print("=" * 60 + "\n")

    print(f"📊 Detected Regime:          #{_regime}")
    print(f"📈 Spread Type:              {_spread_type.upper()}")
    print(f"🎯 Short Strike Distance:    {_strike_distance * 100:.1f}% from current price")
    print()

    print("📋 Regime Statistics:")
    print(f"   • Historical Samples:      {_stats['count']}")
    print(f"   • OTM Probability:         {_stats['prob_otm'] * 100:.2f}%")
    print(f"   • ITM Probability:         {_stats['prob_itm'] * 100:.2f}%")
    print()

    print("🔒 95% Confidence Interval (Wilson):")
    print(f"   [{_stats['ci_95_lower'] * 100:.2f}% — {_stats['ci_95_upper'] * 100:.2f}%]")
    print()

    print("⚡ Risk Metrics:")
    print(f"   • Outcome Std Dev:         {_stats['std_outcome']:.3f}")
    ev_per_dollar = _stats['prob_otm'] - _stats['prob_itm']
    print(f"   • Expected Value per $1:   ${ev_per_dollar:.3f}")

    print("\n" + "=" * 60 + "\n")


# =========================================================
# REAL-TIME INFERENCE FUNCTION
# =========================================================
def run_real_time_inference(args):
    """
    Load a saved model and run inference on the latest data point.
    """
    print("⚡ REAL-TIME INFERENCE MODE")
    print("-" * 60)

    ticker = args.ticker
    # Handle --list-models flag
    if getattr(args, 'list_models', False):
        models_dir = Path("models")
        if not models_dir.exists():
            print("📁 No models directory found.")
            return
        pattern = f"{ticker.replace('^', '')}__*.pkl"
        models = list(models_dir.glob(pattern))
        if not models:
            print(f"🔍 No models found matching: {pattern}")
            return
        print(f"\n📦 Available models for {ticker}:")
        print("─" * 70)
        for m in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
            stat = m.stat()
            print(f"• {m.name}")
            print(f"  └─ Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')} | Size: {stat.st_size / 1024:.1f} KB")
        print("─" * 70)
        print("\n💡 Use --model-filename <name> to load a specific model")
        return

    # Determine model path to load
    if getattr(args, 'model_filename', None):
        # User specified exact filename
        model_path = f"{args.output_dir}/{args.model_filename}"
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file not found: {model_path}")
            return
        print(f"📂 Loading specified model: {model_path}")
    else:
        # Auto-detect: try latest symlink first, then most recent matching file
        latest_link = f"{args.output_dir}/{ticker.replace('^', '')}_regime_model_latest.pkl"
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            model_path = latest_link
            print(f"📂 Loading latest model symlink: {latest_link}")
        else:
            # Fallback: find most recent model for this ticker
            models_dir = Path("models")
            pattern = f"{ticker.replace('^', '')}__*.pkl"
            models = list(models_dir.glob(pattern))
            if not models:
                print(f"❌ ERROR: No models found for ticker {ticker}")
                print("   → Run optimization first without --real-time to generate a model.")
                return
            model_path = str(max(models, key=lambda x: x.stat().st_mtime))
            print(f"📂 Loading most recent model: {os.path.basename(model_path)}")
    model_filename = model_path
    # 1. Load Saved Model
    if not os.path.exists(model_filename):
        print(f"❌ ERROR: Model file not found: {model_filename}")
        print("   → Run optimization first without --real-time to generate a model.")
        return

    print(f"📂 Loading model: {model_filename}")
    with open(model_filename, "rb") as f:
        model_data = pickle.load(f)

    _model = model_data["model"]
    _scaler = model_data["scaler"]
    _stats = model_data["stats"]
    _params = model_data["params"]
    _metadata = model_data.get("metadata", {})

    print(f"✅ Model loaded successfully (Created: {_metadata.get('timestamp', 'N/A')})")

    # 2. Load Latest Data
    try:
        df = load_data(_ticker=ticker, _dataset_id=args.dataset_id)
        print(f"📦 Loaded {len(df)} rows for {ticker}")
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return

    required_feature_params = ['pct1', 'pct2', 'pct3', 'vol1', 'vol2', 'vol3', 'rsi_length', 'ema1', 'ema2', 'atr_period']
    for param in required_feature_params:
        if param not in _params:
            raise ValueError(f"Missing required param '{param}' in saved model")
    # 3. Build Features using SAVED Hyperparameters
    # Crucial: Must use the same params used during training
    print("🔧 Engineering features with saved hyperparameters...")
    features = build_features(
        _df=df,
        _pct1=_params.get('pct1', 5),
        _pct2=_params.get('pct2', 10),
        _pct3=_params.get('pct3', 20),
        _vol1=_params.get('vol1', 10),
        _vol2=_params.get('vol2', 20),
        _vol3=_params.get('vol3', 60),
        _rsi_length=_params.get('rsi_length', 14),
        _ema1=_params.get('ema1', 50),
        _ema2=_params.get('ema2', 200),
        _atr_period=_params.get('atr_period', 14)
    )

    # 4. Prepare Latest Data Point
    valid_features = features.dropna()
    if len(valid_features) == 0:
        print("❌ ERROR: No valid features available (insufficient history for indicators)")
        return

    latest_date = valid_features.index[-1]
    latest_row = valid_features.iloc[[-1]]

    print(f"📅 Analyzing latest data point: {latest_date.strftime('%Y-%m-%d')}")

    # 5. Scale and Predict
    X_latest = _scaler.transform(latest_row)
    regime = _model.predict(X_latest)[0]

    print(f"🔍 Detected Regime: #{regime}")

    if regime not in _stats:
        print(f"⚠️  WARNING: Regime #{regime} has no statistics (possibly insufficient samples during training)")
        return

    regime_stats = _stats[regime]

    # 6. Print Regime Report
    print_report(
        _regime=regime,
        _stats=regime_stats,
        _strike_distance=args.strike_distance,
        _spread_type=args.spread_type
    )

    # 7. Evaluate Trade Decision
    print("💰 TRADE DECISION EVALUATION")
    print("─" * 40)
    max_loss = args.spread_width - args.credit_received
    trade_decision = should_trade_credit_spread(
        _regime_stats=regime_stats,
        _credit_received=args.credit_received,
        _max_loss=max_loss,
        _min_edge_ratio=args.min_edge_ratio
    )

    print(f"Spread Configuration:")
    print(f"   • Width:           ${args.spread_width:.2f}")
    print(f"   • Credit Received: ${args.credit_received:.2f}")
    print(f"   • Max Loss:        ${max_loss:.2f}")
    print(f"   • Min Edge Ratio:  {args.min_edge_ratio * 100:.1f}%")
    print()
    print(f"🎯 DECISION: {trade_decision['message']}")

    print("\n" + "═" * 60)
    print("✨ INFERENCE COMPLETE")
    print("═" * 60 + "\n")


# =========================================================
# MAIN PIPELINE FUNCTION
# =========================================================
def entry_main(args):
    """
    Main execution pipeline for regime optimization and trade evaluation.

    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Extract arguments with defaults fallback
    ticker = getattr(args, 'ticker', '^GSPC')
    dataset_id = getattr(args, 'dataset_id', 'day')
    study_name = getattr(args, 'study_name', 'spx_clustering_2')
    storage_url = getattr(args, 'storage_url', 'sqlite:///spx_study__target_OTM.db')
    max_n_trials = getattr(args, 'max_n_trials', 99999)
    timeout = getattr(args, 'timeout', 86400)
    n_jobs = getattr(args, 'n_jobs', 1)
    random_seed = getattr(args, 'random_seed', DEFAULT_RANDOM_SEED)

    min_n_in_cluster = getattr(args, 'min_n_in_cluster', 160)

    spread_type = getattr(args, 'spread_type', 'put')
    strike_distance = getattr(args, 'strike_distance', 0.03)
    forward_days = getattr(args, 'forward_days', 20)

    spread_width = getattr(args, 'spread_width', 5.0)
    credit_received = getattr(args, 'credit_received', 2.0)
    min_edge_ratio = getattr(args, 'min_edge_ratio', 0.04)

    # Set global random seed
    np.random.seed(random_seed)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Starting Credit Spread Regime Optimization")
    print(f"   Ticker: {ticker} | Dataset: {dataset_id}")
    print(f"   Spread: {spread_type} @{strike_distance * 100:.1f}% | DTE: {forward_days}")
    print(f"   Min samples/cluster: {min_n_in_cluster} | Trials: {max_n_trials} | Timeout: {timeout} seconds")
    print(f"   Forward : {forward_days} {dataset_id}")
    print("-" * 60)
    assert args.dataset_id in ['day', 'week', 'month']
    # Load data
    df = load_data(_ticker=ticker, _dataset_id=dataset_id)
    if args.lookback_years > 0:
        rows_per_year = {'day': 252, 'week': 52, 'month': 12, 'quarter': 4, 'year': 1}.get(args.dataset_id)
        cutoff = args.lookback_years * rows_per_year
        print(f"📅 Using {args.lookback_years}-year lookback: ~{cutoff} rows")
        df = df.iloc[-cutoff:].copy()
    minimum_train_data, minimum_test_data = 1000, 200
    if args.dataset_id == 'week':
        minimum_train_data, minimum_test_data = minimum_train_data // 5, minimum_test_data // 5
    if args.dataset_id == 'month':
        minimum_train_data, minimum_test_data = minimum_train_data // 20, minimum_test_data // 20
    print(f"📦 Loaded {len(df)} rows of data ({df.index[0].date()} to {df.index[-1].date()})")
    assert len(df) > minimum_train_data + minimum_test_data
    # =========================================================
    # OPTUNA OBJECTIVE FUNCTION
    # =========================================================
    def objective(trial):
        """
        Optuna objective: maximize regime-based trading edge.

        Data is split into Train/Test BEFORE feature/target engineering
        to prevent look-ahead bias in rolling indicators and target calculation.
        """
        # ===== Suggest Hyperparameters =====
        if args.dataset_id == 'day':
            _n_clusters = trial.suggest_int("n_clusters", 3, 6)
            _pct1 = trial.suggest_int("pct1", 1, 5)
            _pct2 = trial.suggest_int("pct2", 5, 10)
            _pct3 = trial.suggest_int("pct3", 10, 20)
            _vol1 = trial.suggest_int("vol1", 1, 10)
            _vol2 = trial.suggest_int("vol2", 10, 20)
            _vol3 = trial.suggest_int("vol3", 20, 60)
            _ema1 = trial.suggest_int("ema1", 40, 60)
            _ema2 = trial.suggest_int("ema2", 180, 220)
            _atr_period = trial.suggest_int("atr_period", 2, 21)
            _rsi_length = trial.suggest_int("rsi_length", 10, 20)
        if args.dataset_id == 'week':
            _n_clusters = trial.suggest_int("n_clusters", 3, 6)
            _pct1 = trial.suggest_int("pct1", 1, 3)
            _pct2 = trial.suggest_int("pct2", 3, 8)
            _pct3 = trial.suggest_int("pct3", 8, 16)
            _vol1 = trial.suggest_int("vol1", 1, 10)
            _vol2 = trial.suggest_int("vol2", 10, 20)
            _vol3 = trial.suggest_int("vol3", 20, 30)
            _ema1 = trial.suggest_int("ema1", 8, 14)
            _ema2 = trial.suggest_int("ema2", 40, 44)
            _atr_period = trial.suggest_int("atr_period", 4, 12)
            _rsi_length = trial.suggest_int("rsi_length", 6, 14)
        if args.dataset_id == 'month':
            _n_clusters = trial.suggest_int("n_clusters", 3, 5)
            _pct1 = trial.suggest_int("pct1", 1, 2)
            _pct2 = trial.suggest_int("pct2", 2, 4)
            _pct3 = trial.suggest_int("pct3", 4, 8)
            _vol1 = trial.suggest_int("vol1", 1, 3)
            _vol2 = trial.suggest_int("vol2", 3, 6)
            _vol3 = trial.suggest_int("vol3", 6, 18)
            _ema1 = trial.suggest_int("ema1", 2, 6)
            _ema2 = trial.suggest_int("ema2", 12, 24)
            _atr_period = trial.suggest_int("atr_period", 3, 8)
            _rsi_length = trial.suggest_int("rsi_length", 4, 10)

        # Suggest clustering algorithm as part of optimization
        _clustering_algo = trial.suggest_categorical("clustering_algo", ["kmeans", "gaussian_mixture"])  # ["kmeans", "gaussian_mixture", "birch"])

        # ===== 1. TIME-SERIES SPLIT (RAW DATA) =====
        # Split BEFORE feature engineering to prevent leakage in rolling windows/targets
        split_idx = int(len(df) * 0.8)

        df_train_raw = df.iloc[:split_idx].copy()
        df_test_raw  = df.iloc[split_idx:].copy()

        # ===== 2. Build Features & Target (Separately for Train & Test) =====
        # Train Set
        _features_train = build_features(
            _df=df_train_raw,
            _pct1=_pct1, _pct2=_pct2, _pct3=_pct3,
            _vol1=_vol1, _vol2=_vol2, _vol3=_vol3,
            _rsi_length=_rsi_length,
            _ema1=_ema1, _ema2=_ema2,
            _atr_period=_atr_period
        )
        _target_train = build_target(
            _df=df_train_raw,
            _forward_days=forward_days,
            _strike_distance=strike_distance,
            _spread_type=spread_type
        )

        # Test Set
        _features_test = build_features(
            _df=df_test_raw,
            _pct1=_pct1, _pct2=_pct2, _pct3=_pct3,
            _vol1=_vol1, _vol2=_vol2, _vol3=_vol3,
            _rsi_length=_rsi_length,
            _ema1=_ema1, _ema2=_ema2,
            _atr_period=_atr_period
        )
        _target_test = build_target(
            _df=df_test_raw,
            _forward_days=forward_days,
            _strike_distance=strike_distance,
            _spread_type=spread_type
        )

        # ===== 3. Align & Clean Data =====
        # Align features and targets within their respective sets
        _train_aligned = pd.concat([_features_train, _target_train.rename('target')], axis=1)
        _test_aligned  = pd.concat([_features_test,  _target_test.rename('target')], axis=1)

        # Drop NaNs (Resulting from rolling windows & forward-looking targets)
        train_data = _train_aligned.dropna().copy()
        test_data  = _test_aligned.dropna().copy()

        # Validate minimum data requirements
        if len(train_data) < minimum_train_data or len(test_data) < minimum_test_data:
            return 0.0

        # ===== 4. Feature Scaling =====
        # Fit ONLY on Train, Transform Test (Prevents data leakage)
        _scaler = RobustScaler()
        X_train = _scaler.fit_transform(train_data.drop(columns=['target']))
        X_test  = _scaler.transform(test_data.drop(columns=['target']))

        # ===== 5. Train Clustering Model =====
        if _clustering_algo == 'kmeans':
            _model = KMeans(
                n_clusters=_n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=900,
                tol=1e-4,
                random_state=random_seed,
                algorithm='lloyd',
                verbose=0
            )
        elif _clustering_algo == 'gaussian_mixture':
            _model = GaussianMixture(
                n_components=_n_clusters,
                covariance_type='full',
                tol=1e-3,
                reg_covar=1e-5,
                max_iter=900,
                n_init=20,
                init_params='k-means++',
                random_state=random_seed,
                verbose=0
            )
        elif _clustering_algo == 'birch':
            _threshold = trial.suggest_float("birch_threshold", 0.1, 1.0, log=True)
            _branching_factor = trial.suggest_int("birch_branching", 20, 100)

            _model = Birch(threshold=_threshold,branching_factor=_branching_factor,n_clusters=_n_clusters)
        else:
            raise ValueError(f"Unknown algorithm: {_clustering_algo}")

        # Fit on Train
        regimes_train = _model.fit_predict(X_train)

        # Predict on Test (Out-of-Sample)
        regimes_test = _model.predict(X_test)

        # Attach regimes to test dataframe for evaluation
        test_data = test_data.copy()
        test_data['regime'] = regimes_test

        # ===== 6. Evaluate Each Regime on TEST Data Only =====
        scores = []
        for r in range(_n_clusters):
            subset = test_data[test_data["regime"] == r]["target"]

            if len(subset) < min_n_in_cluster:
                continue  # Skip underpopulated clusters

            # 1. Primary: OTM Probability (higher = better for credit spreads)
            prob_otm = subset.mean()
            prob_score = prob_otm

            # 2. Consistency: Low variance = more predictable regime
            # Bernoulli variance = p*(1-p), so 1-variance peaks at p=0 or p=1
            consistency = 1 - (prob_otm * (1 - prob_otm))

            # 3. Sample size bonus: More data = more reliable estimate
            sample_bonus = np.clip(len(subset) / (2 * min_n_in_cluster), 0, 1)

            # 4. Separation: How distinct is this regime from others?
            other_probs = [
                test_data[test_data["regime"] == i]["target"].mean()
                for i in range(_n_clusters)
                if i != r and len(test_data[test_data["regime"] == i]["target"]) >= min_n_in_cluster
            ]

            if len(other_probs) >= 2:
                # Use weighted separation by sample size for robustness
                separation = abs(
                    prob_otm - np.average(
                        other_probs,
                        weights=[len(test_data[test_data["regime"] == i]["target"])
                                 for i in range(_n_clusters)
                                 if i != r and len(test_data[test_data["regime"] == i]["target"]) >= min_n_in_cluster]
                    )
                )
            elif other_probs:
                separation = abs(prob_otm - other_probs[0])
            else:
                separation = 0

            # ===== Combined Score (tunable weights) =====
            combined = (
                    0.50 * prob_score +  # Primary: high OTM probability
                    0.25 * consistency +  # Secondary: predictable regime
                    0.15 * sample_bonus +  # Tertiary: statistical confidence
                    0.10 * separation  # Quaternary: distinct from others
            )

            scores.append(combined)

        return np.mean(scores) if scores else 0.0

    # =========================================================
    # OPTUNA STUDY SETUP
    # =========================================================
    print(f"\n🔬 Initializing Optuna study: '{study_name}'")
    print(f"   Storage: {storage_url}")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=random_seed),
        pruner=MedianPruner(n_startup_trials=10),
        storage=storage_url,
        load_if_exists=True,
    )

    # ===== DB Resume Verification =====
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n" + "─" * 40)
    print("📊 OPTUNA PERSISTENCE STATUS")
    print("─" * 40)
    print(f"Total trials:     {len(trials)}")
    print(f"Completed:        {len(completed)}")

    if completed:
        print(f"Best score:       {study.best_value:.6f}")
        print(f"Best params:      {study.best_params}")
        print("→ Resuming optimization...")
    else:
        print("→ Starting fresh optimization...")
    print("─" * 40 + "\n")

    # ===== Run Optimization =====
    print("⚙️  Running Optuna optimization...\n")
    study.optimize(
        objective,
        n_trials=max_n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    # ===== Results Summary =====
    print(f"\n✅ Optimization complete!")
    print(f"🏆 Best Score: {study.best_value:.6f}")
    print(f"🎯 Best Parameters:")
    for k, v in study.best_params.items():
        print(f"   • {k}: {v}")

    # =========================================================
    # RETRAIN WITH BEST PARAMETERS
    # =========================================================
    print("\n" + "═" * 60)
    print("🔄 RETRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("═" * 60)

    best_params = study.best_params.copy()

    # Extract parameters (handle optional clustering_algo)
    _n_clusters = best_params['n_clusters']
    _clustering_algo = best_params['clustering_algo']

    print(f"Selected algorithm: {_clustering_algo}")
    print(f"Clusters: {_n_clusters}")

    # Build features with best params
    features = build_features(
        _df=df.copy(),
        _pct1=best_params.get('pct1', 5),
        _pct2=best_params.get('pct2', 10),
        _pct3=best_params.get('pct3', 20),
        _vol1=best_params.get('vol1', 10),
        _vol2=best_params.get('vol2', 20),
        _vol3=best_params.get('vol3', 60),
        _rsi_length=best_params.get('rsi_length', 14),
        _ema1=best_params.get('ema1', 50),
        _ema2=best_params.get('ema2', 200),
        _atr_period=best_params.get('atr_period', 14)
    )

    # Build target
    target = build_target(
        _df=df.copy(),
        _forward_days=forward_days,
        _strike_distance=strike_distance,
        _spread_type=spread_type
    )

    # ===== Align and Prepare Full Dataset =====
    _aligned = pd.concat([features, target.rename('target')], axis=1)
    _valid = _aligned.dropna().copy()

    if len(_valid) == 0:
        print("❌ ERROR: No valid data after feature/target alignment")
        return

    # ===== Scale Features =====
    _scaler = RobustScaler()
    X = _scaler.fit_transform(_valid.drop(columns=['target']))

    # ===== Train Final Model (with CONSISTENT parameters) =====
    if _clustering_algo == 'kmeans':
        _model = KMeans(
            n_clusters=_n_clusters,
            random_state=random_seed,
            n_init=10,
            algorithm='lloyd',
            max_iter=900,
            tol=1e-4
        )
    elif _clustering_algo == 'gaussian_mixture':
        _model = GaussianMixture(
            n_components=_n_clusters,
            random_state=random_seed,
            n_init=20,  # ← MATCHES objective()
            covariance_type='full',
            max_iter=900,  # ← MATCHES objective()
            tol=1e-3,
            reg_covar=1e-5,
            init_params = 'k-means++',
        )
    elif _clustering_algo == 'birch':
        _threshold = best_params.get('birch_threshold', 0.5)
        _branching = best_params.get('birch_branching', 50)
        _model = Birch(threshold=_threshold,branching_factor=_branching,n_clusters=_n_clusters)
    else:
        raise ValueError(f"Unknown algorithm: {_clustering_algo}")

    regimes = _model.fit_predict(X)
    _valid["regime"] = regimes

    # ===== Compute Regime Statistics (with Wilson CI) =====
    _stats = {}
    for r in range(_n_clusters):
        subset = _valid[_valid["regime"] == r]["target"].dropna()

        if len(subset) < min_n_in_cluster:
            continue

        prob_otm = subset.mean()
        n = len(subset)

        # Wilson score interval (more accurate for proportions)
        ci_low, ci_upp = proportion_confint(
            count=subset.sum(),
            nobs=n,
            alpha=0.05,
            method='wilson'
        )

        _stats[r] = {
            "count": n,
            "prob_otm": prob_otm,
            "prob_itm": 1 - prob_otm,
            "win_rate": prob_otm,
            "std_outcome": np.sqrt(prob_otm * (1 - prob_otm)),  # Bernoulli std
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_upp,
        }

    if not _stats:
        print(f"⚠️  WARNING: No clusters met minimum sample requirement ({min_n_in_cluster})")
        return

    # Show ALL cluster characteristics
    print_all_cluster_characteristics(_stats=_stats, _features=features, _model=_model, _scaler=_scaler, _n_clusters=_n_clusters)

    # ===== Predict Latest Regime =====
    result = predict_latest(_features=features, _model=_model, _scaler=_scaler, _stats=_stats)

    if result is None:
        print("⚠️  Could not predict regime for latest data point")
        return

    regime, regime_stats = result

    # ===== Print Analysis Report =====
    print_report(_regime=regime, _stats=regime_stats, _strike_distance=strike_distance, _spread_type=spread_type)

    # ===== TRADE EVALUATION =====
    print("💰 TRADE DECISION EVALUATION")
    print("─" * 40)

    max_loss = spread_width - credit_received

    trade_decision = should_trade_credit_spread(_regime_stats=regime_stats, _credit_received=credit_received, _max_loss=max_loss, _min_edge_ratio=min_edge_ratio)

    print(f"Spread Configuration:")
    print(f"   • Width:           ${spread_width:.2f}")
    print(f"   • Credit Received: ${credit_received:.2f}")
    print(f"   • Max Loss:        ${max_loss:.2f}")
    print(f"   • Min Edge Ratio:  {min_edge_ratio * 100:.1f}%")
    print()
    print(f"Regime-Based Metrics:")
    print(f"   • Break-Even Win Rate: {trade_decision['break_even_prob'] * 100:.2f}%")
    print(f"   • Expected Value:      ${trade_decision['expectancy']:.3f}/share")
    print(f"   • Edge Ratio:          {trade_decision['edge_ratio'] * 100:.2f}%")
    print()
    print(f"🎯 DECISION: {trade_decision['message']}")

    # ===== Save Model =====
    print(f"\n💾 Saving model artifacts...")

    # Generate unique, descriptive filename
    model_filename = generate_model_filename(
        _ticker=ticker,
        _study_name=study_name,
        _params=best_params,
        _metadata_extra={
            'spread_type': spread_type,
            'strike_distance': strike_distance,
            'forward_days': forward_days
        }
    )
    model_path = f"{args.output_dir}/{model_filename}"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": _model,
            "scaler": _scaler,
            "stats": _stats,
            "params": best_params,
            "metadata": {
                "ticker": ticker,
                "spread_type": spread_type,
                "strike_distance": strike_distance,
                "forward_days": forward_days,
                "clustering_algo": _clustering_algo,
                "timestamp": datetime.now(),
                "study_name": study_name,
                "best_score": study.best_value,
                "model_filename": model_filename  # ← Store filename for easy retrieval
            },
            "trade_context": {
                "credit_received": credit_received,
                "spread_width": spread_width,
                "max_loss": max_loss,
                "edge_ratio": trade_decision['edge_ratio'],
                "expectancy": trade_decision['expectancy'],
                "evaluated_at": datetime.now()
            }
        }, f)

    print(f"✅ Model saved to: {model_path}")

    # Create/update a "latest" symlink for convenience
    latest_link = f"{args.output_dir}/{ticker.replace('^', '')}_regime_model_latest.pkl"
    try:
        # Check if the destination already exists and remove it to avoid errors
        if os.path.exists(latest_link):
            os.remove(latest_link)
        shutil.copy2(model_path, latest_link)
        print(f"💾 Created copy: {latest_link} (from {model_filename})")
    except (OSError, IOError) as e:
        print(f"❌ Failed to copy model: {e}")

    print("\n" + "═" * 60)
    print("✨ PIPELINE COMPLETE")
    print("═" * 60 + "\n")


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    """
    Command-line interface for Credit Spread Regime Optimization.

    EXAMPLES:
    ---------
    # Basic run with defaults:
    python script.py

    # Optimize put spreads on SPY with 30 DTE:
    python script.py --ticker SPY --spread-type put --forward-days 30

    # Use PostgreSQL storage:
    python script.py --storage-url "postgresql://user:pass@localhost/db" \\
                     --max-n-trials 200

    # Conservative trade filtering:
    python script.py --min-edge-ratio 0.08 --credit-received 1.5

    # Quick test run:
    python script.py --max-n-trials 10 --timeout 300 --study-name test_run
    
    # List available models for SPY:
    python script.py --ticker SPY --list-models

    # Run inference with a specific experiment:
    python script.py --ticker SPY --real-time \\
                     --model-filename "SPY__my_study__kmeansc4__put_sd3pct_dte20__20260402_143022.pkl"

    # Use the latest model (default behavior):
    python script.py --ticker SPY --real-time
    """

    parser = argparse.ArgumentParser(
        description="Optimize regime clustering for credit spread trading decisions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        TRADE LOGIC:
        • Regime detection via unsupervised clustering (K-Means or GMM)
        • Target: Binary outcome of spread expiring OTM (profitable)
        • Trade approval requires:
          1. Edge ratio (expectancy/max_loss) ≥ min_edge_ratio
          2. Regime OTM probability > break-even probability
        • Model saved only when trade is approved
        """
    )

    # ─────────────────────────────────────────────────────
    # DATA & ASSET CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--ticker", type=str, default="^GSPC",
        help="Ticker symbol to analyze (e.g., ^GSPC, SPY, QQQ)"
    )
    parser.add_argument(
        "--dataset-id", type=str, default="day", choices=DATASET_AVAILABLE,
        help=f"Dataset frequency: {DATASET_AVAILABLE}"
    )
    parser.add_argument(
        "--lookback-years", type=int, default=15,
        help="Years of historical data to use (0 = full history). Recommended: 10-15 for daily data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help=f"Directory where to save the model"
    )

    # ─────────────────────────────────────────────────────
    # OPTUNA OPTIMIZATION CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--study-name", type=str, default="spx_clustering_2",
        help="Unique Optuna study name (for DB persistence)"
    )
    parser.add_argument(
        "--storage-url", type=str, default=None,
        help="Database URL: sqlite:///path.db or postgresql://..."
    )
    parser.add_argument(
        "--max-n-trials", type=int, default=999999,
        help="Maximum Optuna trials to run"
    )
    parser.add_argument(
        "--timeout", type=int, default=86400,
        help="Max optimization runtime in seconds (86400 = 24h)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs for Optuna (1 recommended for SQLite)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility"
    )

    # ─────────────────────────────────────────────────────
    # CLUSTERING / REGIME MODEL CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--min-n-in-cluster", type=int, default=160,
        help="Minimum samples per cluster to be considered valid"
    )

    # ─────────────────────────────────────────────────────
    # CREDIT SPREAD TRADE CONFIGURATION
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--spread-type", type=str, default="put",
        choices=["put", "call", "iron_condor"],
        help="Spread type: 'put' (bullish), 'call' (bearish), 'iron_condor'"
    )
    parser.add_argument(
        "--strike-distance", type=float, default=0.03,
        help="Short strike distance as fraction (0.03 = 3%% away)"
    )
    parser.add_argument(
        "--forward-days", type=int, default=20,
        help="Days to expiration / holding period. Could be week or month, depends on dataset-id."
    )

    # ─────────────────────────────────────────────────────
    # TRADE EVALUATION PARAMETERS
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--spread-width", type=float, default=5.0,
        help="Total spread width in dollars (e.g., 5.0 = $5 wide)"
    )
    parser.add_argument(
        "--credit-received", type=float, default=2.0,
        help="Premium received per share in dollars"
    )
    parser.add_argument(
        "--min-edge-ratio", type=float, default=0.04,
        help="Minimum edge ratio (expectancy/max_loss) to approve trade"
    )

    # ─────────────────────────────────────────────────────
    # EXECUTION MODE
    # ─────────────────────────────────────────────────────
    parser.add_argument(
        "--real-time", action="store_true",
        help="Load saved model and run inference on latest data (skip optimization)"
    )
    parser.add_argument(
        "--model-filename", type=str, default=None,
        help="Specific model filename to load for inference (overrides auto-detection)"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available trained models for this ticker and exit"
    )

    # ─────────────────────────────────────────────────────
    # EXECUTE
    # ─────────────────────────────────────────────────────
    args = parser.parse_args()

    # Ensure models directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.real_time:
            # Run Inference Mode
            run_real_time_inference(args)
        else:
            # Run Optimization Mode
            entry_main(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Exiting gracefully.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)