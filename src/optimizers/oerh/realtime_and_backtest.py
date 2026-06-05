"""
🎯 RULE-BASED TECHNICAL FORECASTING SYSTEM (JIT-ACCELERATED)
====================================================================================================
🔧 ENHANCED: Added "floor" target mode for put credit spread / defined-risk bullish strategies

📋 OVERVIEW
-----------
This system implements a multi-indicator, rule-based forecasting engine for financial time series.
It combines adaptive signal smoothing, momentum/reversion logic, and flexible target labeling to
generate actionable trading signals with configurable lookahead horizons.

🔧 CORE ALGORITHMS
------------------

1️⃣ ONE-EURO FILTER (Adaptive Smoothing)
   ├─ Purpose: Reduce noise while preserving rapid price movements
   ├─ Mechanism: Dynamic cutoff frequency based on signal derivative
   ├─ Parameters:
   │  • one_euro_min  : Minimum cutoff frequency (Hz) – controls baseline smoothing
   │  • one_euro_factor: Beta factor – adapts smoothing to signal volatility
   └─ Output: Smoothed price series used as dynamic support/resistance reference

2️⃣ RSI – RELATIVE STRENGTH INDEX (Wilder's Method) [✅ JIT-ACCELERATED]
   ├─ Purpose: Measure momentum and identify overbought/oversold conditions
   ├─ Two Operating Modes:
   │  • "momentum"  : Buy when RSI > overbought (trend continuation)
   │  • "reversion" : Buy when RSI < oversold (mean reversion)
   ├─ Parameters:
   │  • rsi_period    : Lookback window for RSI calculation (default: 14)
   │  • rsi_oversold  : Lower threshold for oversold signal (default: 30)
   │  • rsi_overbought: Upper threshold for overbought signal (default: 70)
   └─ Output: Oscillator [0–100] used in conjunction with trend filters

3️⃣ MACD – MOVING AVERAGE CONVERGENCE DIVERGENCE [✅ JIT-ACCELERATED]
   ├─ Purpose: Identify trend direction, strength, and potential reversals
   ├─ Components:
   │  • MACD Line     : EMA(fast) − EMA(slow)
   │  • Signal Line   : EMA of MACD Line
   │  • Histogram     : MACD − Signal (momentum accelerator)
   ├─ Parameters:
   │  • macd_fast   : Fast EMA period (default: 12)
   │  • macd_slow   : Slow EMA period (default: 26)
   │  • macd_signal : Signal line EMA period (default: 9)
   └─ Output: Trend confirmation + momentum divergence signals

4️⃣ TARGET LABELING STRATEGIES (Forecast Ground Truth)
   ├─ Purpose: Define what constitutes a "successful" prediction
   ├─ Four Modes:
   │  • "exact"      : Price must exceed threshold EXACTLY at t + lookahead_bars
   │  • "any"        : Price must exceed threshold ANYWHERE in [t+1, t+lookahead_bars]
   │  • "any_half_B" : Price must exceed threshold in SECOND HALF of window [t+B//2+1, t+B]
   │  • "floor" 🆕   : Price must NEVER fall below threshold in [t+1, t+lookahead_bars]
   ├─ Parameters:
   │  • lookahead_bars : Forecast horizon in bars (default: 5)
   │  • threshold_pct  : Minimum price move to trigger positive label (default: 0.01 = 1%)
   │                     For "floor" mode: use NEGATIVE value (e.g., -0.015 = -1.5% floor)
   └─ Output: Binary labels (1 = target met, 0 = not met) for signal evaluation

⚙️ SIGNAL GENERATION LOGIC [✅ OPTIONAL JIT]
--------------------------
A trading signal is generated when ALL of the following conditions align:

🟢 LONG SIGNAL (Signal = +1):
   ✓ Price > One-Euro Filter (uptrend confirmation)
   ✓ RSI condition (mode-dependent):
       - momentum: RSI > overbought AND rising vs. previous bar
       - reversion: RSI < oversold AND rising vs. previous bar
   ✓ MACD Histogram > 0 AND increasing vs. previous bar (bullish momentum)

🔴 SHORT SIGNAL (Signal = -1):
   ✓ Price < One-Euro Filter (downtrend confirmation)
   ✓ RSI condition (mode-dependent):
       - momentum: RSI < oversold AND falling vs. previous bar
       - reversion: RSI > overbought AND falling vs. previous bar
   ✓ MACD Histogram < 0 AND decreasing vs. previous bar (bearish momentum)

⚪ NO SIGNAL (Signal = 0): Default when conditions are not met

📊 EVALUATION METRICS
---------------------
After signal generation, the system computes:
   • Overall Accuracy     : % of signals where prediction matched actual outcome
   • Long Accuracy        : % of long signals that correctly predicted upward move
   • Short Accuracy       : % of short signals that correctly predicted downward move
   • Signal Frequency     : Total signals generated vs. total bars analyzed

🚀 EXECUTION MODES
------------------
🔹 BATCH MODE (default):
   - Processes entire historical dataset
   - Supports train/validation split via --train-ratio
   - Outputs comprehensive metrics + optional visualization

🔹 REAL-TIME MODE (--real-time):
   - Loads pre-optimized model parameters from .pkl file
   - Evaluates ONLY the most recent data point
   - Outputs minimal, automation-friendly signal format:
       BUY|DATE|PRICE|+BARS|TARGET_PRICE
       SELL|DATE|PRICE|+BARS|TARGET_PRICE
       HOLD|DATE|PRICE

📦 MODEL PERSISTENCE
--------------------
Models saved via pickle contain:
   • best_params    : Optimized algorithm hyperparameters
   • metadata       : Dataset ID, ticker, target_type, metric used for optimization
   • (optional) performance history for audit trails

🎨 VISUALIZATION DASHBOARD
--------------------------
Three-panel synchronized plot:
   1. Price + One-Euro Filter + Signal markers (▲ long / ▼ short)
   2. RSI oscillator with overbought/oversold zones
   3. MACD histogram + signal line with momentum coloring
Features: Zoom/pan synchronization, inset zoom region, interactive Matplotlib backend

⚙️ KEY COMMAND-LINE PARAMETERS
------------------------------
Data & Environment:
   --dataset-id      : Dataset identifier (choices from DATASET_AVAILABLE)
   --ticker          : Symbol to analyze (default: "^GSPC")
   --seed            : Random seed for reproducibility
   --validate-jit    : Run JIT consistency sanity check at startup (default: disabled)

Algorithm Tuning:
   --rsi-period / --rsi-oversold / --rsi-overbought / --rsi-mode
   --macd-fast / --macd-slow / --macd-signal
   --one-euro-min / --one-euro-factor
   --lookahead-bars / --threshold-pct / --target-type

Evaluation & Output:
   --train-ratio     : Fraction of data for training (rest for validation)
   --plot-sample     : Number of recent bars to visualize
   --verbose / --disable-print : Control console output

Real-Time Execution:
   --real-time       : Enable latest-point evaluation mode
   --model-path      : Path to saved .pkl model file (required with --real-time)
   --output-signal-only : Minimal output format for scripting/automation

🧪 EXAMPLE USAGE
---------------
# Put credit spread strategy: floor mode with -1.5% threshold
python forecast.py --ticker SPY --target-type floor --threshold-pct -0.015 --lookahead-bars 5 --rsi-mode reversion

# Batch backtest with custom parameters
python forecast.py --ticker AAPL --rsi-mode reversion --lookahead-bars 10 --threshold-pct 0.02

# Train/validation split evaluation
python forecast.py --train-ratio 0.7 --plot-sample 150

# Real-time signal with saved model
python forecast.py --real-time --model-path models/aapl_optimized.pkl --output-signal-only

# Silent automation mode (parse output programmatically)
python forecast.py --real-time --model-path models/spx.pkl --output-signal-only --disable-print

# Run with JIT validation sanity check
python forecast.py --validate-jit --ticker SPY --disable-plot-sample

🔐 DESIGN PRINCIPLES
--------------------
• Modular: Each indicator is independently testable and replaceable
• Vectorized: Leverages pandas/numpy for efficient batch processing
• JIT-Accelerated: Numba @njit for RSI, MACD, and optional signal generation
• Cache-aware: LRU caching for dataset loading + numba function caching
• Type-safe: Comprehensive type hints for maintainability
• Reproducible: Seed control + deterministic indicator calculations
• Production-ready: Fallback imports, error handling, and clear CLI interface

📚 DEPENDENCIES
--------------
numpy, pandas, numba>=0.56 (JIT acceleration), matplotlib, pickle, argparse

====================================================================================================
🏁 SYSTEM READY — Configure parameters via CLI or load optimized model for real-time deployment
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

import numpy as np
import pandas as pd
from functools import lru_cache
from numba import njit, prange
import math
import copy
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import pickle
import argparse
from utils import get_filename_for_dataset
import os
import warnings

# ============================================
# 1. ONE-EURO FILTER (Adaptive Smoothing)
# ============================================
from algorithms.one_euro_filter import one_euro_filter
from utils import DATASET_AVAILABLE, get_next_step


# ============================================
# JIT-ACCELERATED NUMERICAL CORES
# ============================================

@njit(cache=True, fastmath=True)
def _rsi_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-accelerated RSI using Wilder's smoothing method."""
    n = len(values)
    rsi = np.empty(n, dtype=np.float64)
    rsi[:] = np.nan

    if n < 2:
        return rsi

    # Calculate price deltas
    delta = np.empty(n, dtype=np.float64)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = values[i] - values[i - 1]

    # Separate gains and losses
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if delta[i] > 0:
            gain[i] = delta[i]
        else:
            loss[i] = -delta[i]

    # Initial average (simple mean for first period)
    if n > period:
        avg_gain = np.mean(gain[1:period + 1])
        avg_loss = np.mean(loss[1:period + 1])
    else:
        avg_gain = np.mean(gain[1:])
        avg_loss = np.mean(loss[1:])

    # Wilder's smoothing: RSI calculation
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit(cache=True, fastmath=True)
def _ema_numba(values: np.ndarray, span: int) -> np.ndarray:
    """Numba-accelerated exponential moving average (Wilder's style)."""
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n == 0:
        return result

    # Alpha for EMA: same as pandas ewm(span=span, adjust=False)
    alpha = 2.0 / (span + 1)
    result[0] = values[0]

    for i in range(1, n):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]

    return result


@njit(cache=True, fastmath=True)
def _macd_numba(close: np.ndarray, fast: int, slow: int, signal: int) -> tuple:
    """Compute MACD components - returns (macd_line, signal_line, histogram)."""
    ema_fast = _ema_numba(close, fast)
    ema_slow = _ema_numba(close, slow)
    macd_line = np.empty(len(close), dtype=np.float64)

    for i in range(len(close)):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            macd_line[i] = np.nan
        else:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = _ema_numba(macd_line, signal)
    histogram = np.empty(len(close), dtype=np.float64)

    for i in range(len(close)):
        if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
            histogram[i] = np.nan
        else:
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


@njit(cache=True, parallel=True, fastmath=True)
def _generate_signals_numba(
        close: np.ndarray,
        one_euro: np.ndarray,
        rsi: np.ndarray,
        histogram: np.ndarray,
        rsi_oversold: float,
        rsi_overbought: float,
        rsi_mode_momentum: bool,
) -> np.ndarray:
    """Numba-accelerated signal generation with parallel processing."""
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    for i in prange(1, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(histogram[i]) or
                np.isnan(one_euro[i]) or np.isnan(close[i])):
            continue

        # Handle previous values with fallback
        rsi_prev = rsi[i - 1] if not np.isnan(rsi[i - 1]) else rsi[i]
        hist_prev = histogram[i - 1] if not np.isnan(histogram[i - 1]) else histogram[i]

        # RSI condition based on mode
        if rsi_mode_momentum:
            long_rsi = (rsi[i] > rsi_overbought) and (rsi[i] > rsi_prev)
            short_rsi = (rsi[i] < rsi_oversold) and (rsi[i] < rsi_prev)
        else:  # reversion mode
            long_rsi = (rsi[i] < rsi_oversold) and (rsi[i] > rsi_prev)
            short_rsi = (rsi[i] > rsi_overbought) and (rsi[i] < rsi_prev)

        # Long signal condition
        if ((close[i] > one_euro[i]) and long_rsi and
                (histogram[i] > 0) and (histogram[i] > hist_prev)):
            signals[i] = 1
        # Short signal condition
        elif ((close[i] < one_euro[i]) and short_rsi and
              (histogram[i] < 0) and (histogram[i] < hist_prev)):
            signals[i] = -1

    return signals


# ============================================
# PANDAS WRAPPERS FOR JIT FUNCTIONS
# ============================================

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing - JIT-accelerated core."""
    values = close.to_numpy(dtype=np.float64)
    rsi_array = _rsi_numba(values, period)
    return pd.Series(rsi_array, index=close.index, name='RSI', dtype=float)


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram - JIT-accelerated core."""
    values = close.to_numpy(dtype=np.float64)
    macd_arr, sig_arr, hist_arr = _macd_numba(values, fast, slow, signal)

    return pd.DataFrame({
        'MACD': pd.Series(macd_arr, index=close.index, dtype=float),
        'Signal': pd.Series(sig_arr, index=close.index, dtype=float),
        'Histogram': pd.Series(hist_arr, index=close.index, dtype=float)
    })


# ============================================
# MODEL LOADING & REAL-TIME EXECUTION
# ============================================
def load_model(model_path: str) -> dict:
    """Load a saved model file and return its parameters."""
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def run_real_time(model_path: str, output_signal_only: bool, verbose: bool, clip: bool):
    """Run forecast in real-time mode using saved model parameters."""
    # Load model
    model_data = load_model(model_path)
    params = model_data['best_params']
    metadata = model_data.get('metadata', {})
    assert 'lookahead_bars' in metadata
    lookahead_bars = params.get('lookahead_bars', metadata.get('lookahead_bars', 5))
    dataset_id = metadata.get('dataset_id')
    ticker = metadata.get('ticker')
    master_data_cache = copy.deepcopy(_load_df(_datase_id=dataset_id))
    signal_ratio = model_data['user_attrs']['signal_ratio']
    metric_used = model_data['user_attrs']['metric_used']
    val_acc = model_data['user_attrs']['val_accuracy']
    train_acc = model_data['user_attrs']['raw_accuracy']
    assert 'threshold_pct' in metadata
    threshold_pct = params.get('threshold_pct', metadata.get('threshold_pct', 0.))
    if metric_used == 'long_accuracy':
        train_acc = model_data['user_attrs']['train_long_accuracy']
        val_acc = model_data['user_attrs']['val_long_accuracy']
    elif metric_used == 'short_accuracy':
        train_acc = model_data['user_attrs']['train_short_accuracy']
        val_acc = model_data['user_attrs']['val_short_accuracy']
    else:
        print(model_data)
        assert False, f"TODO: {metric_used}"
    df = master_data_cache[ticker].sort_index()
    if clip:
        df = df.iloc[:-1].copy()
    print(f"\n📊 Dataset Loaded: {ticker} | {dataset_id} | Lookahead {lookahead_bars} bars | Metric used: {metric_used} with threshold: {threshold_pct}%")
    print(f"   Bars: {len(df):,} | Range: {df.index[0].strftime('%Y%m%d')}  ->  {df.index[-1].strftime('%Y%m%d')} | Train Accuracy: {train_acc:.2%} "
          f":: Val Accuracy: {val_acc:.2%}  @{signal_ratio:.2%} signal density")
    price_col = ('Close', ticker)

    # Determine minimum history needed for indicators
    assert 'rsi_period' in params and 'macd_slow' in params
    min_history = max(
        params.get('rsi_period', 14),
        params.get('macd_slow', 26),
        params.get('rsi_period', 14) + params.get('macd_slow', 26),  # safety margin
        100
    )

    # Get latest window of data (needed for indicator calculations)
    latest_df = df.iloc[-min_history:].copy()

    # Extract parameters for run_forecast (filter only valid ones)
    valid_params = {k: v for k, v in params.items() if k in [
        'rsi_period', 'rsi_oversold', 'rsi_overbought', 'rsi_mode',
        'macd_fast', 'macd_slow', 'macd_signal',
        'one_euro_min', 'one_euro_factor',
        'lookahead_bars', 'threshold_pct', 'target_type'
    ]}
    assert 'target_type' in metadata
    target_type = metadata['target_type']
    valid_params.update({'target_type': target_type})

    # Run forecast on the window
    df_results, _ = run_forecast(
        df=latest_df,
        price_col=price_col,
        ticker=str(ticker),
        **valid_params
    )

    # Get the very latest signal
    latest_row = df_results.iloc[-1]
    signal = latest_row['Signal']
    current_price = latest_row[price_col]
    current_date = latest_row.name  # Index is datetime

    # Get config values (prefer params, fallback to metadata)
    assert 'metric' in metadata
    the_metric = params.get('metric', metadata.get('metric', 'long_accuracy'))
    assert the_metric == metric_used

    assert 1 == len(signal)
    signal = signal.values[0]
    if signal == 1    and the_metric in ['long_accuracy', 'accuracy']:  # Buy signal
        signal = 1
    elif signal == -1 and the_metric in ['short_accuracy', 'accuracy']:  # Sell signal
        signal = -1
    else:
        signal = 0

    # Calculate target price and date
    if signal == 1:  # Buy signal
        target_price = current_price * (1 + threshold_pct)
        direction = "BuySignal"
        operator = ">"
    elif signal == -1:  # Sell signal
        target_price = current_price * (1 - threshold_pct)
        direction = "SellSignal"
        operator = "<"
    else:
        target_price = None
        direction = "NoSignal"
        operator = ""

    # Estimate target date
    target_date = get_next_step(the_date=current_date, dataset_id=dataset_id, nn=lookahead_bars)

    # Format output
    ticker_display = ticker if ticker != "^GSPC" else "SPX500"
    if verbose:
        if output_signal_only:
            # Minimal output for automation/scripts
            if signal == 1:
                print(f"BUY|{current_date.strftime('%Y-%m-%d')}|{current_price:.2f}|+{lookahead_bars}bars|{target_price:.2f}")
            elif signal == -1:
                print(f"SELL|{current_date.strftime('%Y-%m-%d')}|{current_price:.2f}|+{lookahead_bars}bars|{target_price:.2f}")
            else:
                print(f"HOLD|{current_date.strftime('%Y-%m-%d')}|{current_price:.2f}")
        else:
            # Human-readable output
            if signal != 0:
                if target_type == "any":
                    target_mode_desc = f"any point in window [t+1, t+{lookahead_bars}]"
                elif target_type == "any_half_B":
                    target_mode_desc = f"any point in second half [t+{lookahead_bars//2+1}, t+{lookahead_bars}]"
                elif target_type == "floor":
                    target_mode_desc = f"NEVER falls below floor [t+1, t+{lookahead_bars}] 🛡️"
                else:
                    target_mode_desc = "exact future point"
                print(f"\n🎯 Real-Time Signal Detected:")
                print(f"On {current_date.strftime('%Y-%m-%d')} {ticker_display} is at {current_price:.2f} and {direction} activated for +{lookahead_bars}Bars , {ticker_display} {operator} {target_price:.2f} on {target_date.strftime('%Y-%m-%d')}")
                print(f"   Threshold: {threshold_pct * 100:.2f}% | Target Mode: '{target_type}' ({target_mode_desc}) | Model: {os.path.basename(model_path)}\n")
            else:
                print(f"\n⏸️  No signal on {current_date.strftime('%Y-%m-%d')}: {ticker_display} at {current_price:.2f}")
                print(f"   (Threshold: {threshold_pct * 100:.4f}%, Lookahead: {lookahead_bars} bars, Target Mode: '{target_type}', Metric: {the_metric})\n")

    return signal, current_price, target_price, target_date


# ============================================
# 4. LOOK-AHEAD LABELING (Forecast Target)
# ============================================
def create_target_exact(close: pd.Series, lookahead_bars: int, threshold_pct: float = 0.0) -> pd.Series:
    """Exact: Checks price exactly at t + lookahead_bars"""
    future_close = close.shift(-lookahead_bars)
    labels = (future_close > close * (1 + threshold_pct)).astype(float)
    labels.iloc[-lookahead_bars:] = np.nan
    return labels


def create_target_any(close: pd.Series, lookahead_bars: int, threshold_pct: float = 0.0) -> pd.Series:
    """
    Any: Checks if price EXCEEDS threshold at ANY point from t to t+lookahead_bars.
    Uses a forward-looking rolling maximum for efficient vectorized computation.
    """
    # Version optimisée (Exclut le prix actuel t, regarde uniquement de t+1 à t+lookahead_bars)
    future_max = close.iloc[::-1].rolling(window=lookahead_bars, min_periods=lookahead_bars).max().iloc[::-1].shift(-1)

    # True if max in window > threshold
    labels = (future_max > close * (1 + threshold_pct)).astype(float)

    # Mask out last lookahead_bars where future data is incomplete
    labels.iloc[-lookahead_bars:] = np.nan

    return labels


def create_target_any_at_half_B(close: pd.Series, lookahead_bars: int, threshold_pct: float = 0.0) -> pd.Series:
    """
    Any at half B: Checks if price EXCEEDS threshold at ANY point from t+B//2+1 to t+B.
    (i.e., the second half of the lookahead window, excluding the midpoint itself)
    Uses a forward-looking rolling maximum for efficient vectorized computation.

    Parameters
    ----------
    close : pd.Series
        Closing prices series
    lookahead_bars : int
        Number of bars to look ahead (B)
    threshold_pct : float, optional
        Percentage threshold for target (default: 0.0)

    Returns
    -------
    pd.Series
        Binary labels (1 if threshold exceeded in window, 0 otherwise), with NaN for incomplete windows
    """
    half_B = lookahead_bars // 2
    window_size = lookahead_bars - half_B  # Size of window from t+half_B+1 to t+B inclusive

    # Handle edge case where window would be empty
    if window_size <= 0:
        return pd.Series(np.nan, index=close.index, dtype=float)

    # Get max over the window [t+half_B+1, t+B] for each t
    # Reverse series, apply rolling max, reverse back, then shift to align window
    future_max = close.iloc[::-1].rolling(window=window_size, min_periods=window_size).max().iloc[::-1].shift(-(half_B + 1))

    # True if max in window > threshold
    labels = (future_max > close * (1 + threshold_pct)).astype(float)

    # Mask out last lookahead_bars where future data is incomplete
    labels.iloc[-lookahead_bars:] = np.nan

    return labels


def create_target_floor(close: pd.Series, lookahead_bars: int, threshold_pct: float = 0.0) -> pd.Series:
    """
    🆕 FLOOR MODE: Label = 1 only if price NEVER falls below threshold
    throughout the entire lookahead window [t+1, t+B].

    Ideal for put credit spreads / defined-risk bullish strategies where you profit
    if price stays ABOVE your short put strike (the "floor").

    ⚠️ IMPORTANT: Use NEGATIVE threshold_pct for floor mode!
       • threshold_pct = -0.015 means floor = entry_price × 0.985 (1.5% below entry)
       • Label = 1 only if price MINIMUM in window > floor level

    Parameters
    ----------
    close : pd.Series
        Closing prices series
    lookahead_bars : int
        Number of bars to look ahead (B)
    threshold_pct : float, optional
        Percentage threshold for floor (default: 0.0)
        For floor protection: use NEGATIVE value (e.g., -0.015 = -1.5% floor)

    Returns
    -------
    pd.Series
        Binary labels (1 if price NEVER fell below floor, 0 otherwise),
        with NaN for incomplete windows
    """
    # Get MINIMUM price in forward window [t+1, t+lookahead_bars]
    # Reverse series → rolling min → reverse back → shift to exclude current bar
    future_min = close.iloc[::-1].rolling(
        window=lookahead_bars, min_periods=lookahead_bars
    ).min().iloc[::-1].shift(-1)

    # Label = 1 if minimum stays ABOVE floor level throughout window
    # floor_level = entry_price × (1 + threshold_pct)
    # For threshold_pct = -0.015: floor = entry × 0.985
    labels = (future_min > close * (1 + threshold_pct)).astype(float)

    # Mask out last lookahead_bars where future data is incomplete
    labels.iloc[-lookahead_bars:] = np.nan

    return labels


# ============================================
# 5. RULE-BASED FORECASTING SYSTEM
# ============================================
class ForecastSystem:
    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 30, rsi_overbought: float = 70,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 one_euro_min: float = 10, one_euro_factor: float = 0.2,
                 lookahead_bars: int = 5, threshold_pct: float = 0.01,
                 rsi_mode: str = "momentum", target_type: str = "any",
                 use_jit_signals: bool = False):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.one_euro_min = one_euro_min
        self.one_euro_factor = one_euro_factor
        self.lookahead_bars = lookahead_bars
        self.threshold_pct = threshold_pct
        self.rsi_mode = rsi_mode
        self.target_type = target_type  # 'exact', 'any', 'any_half_B', or 'floor'
        self.use_jit_signals = use_jit_signals  # Enable JIT for signal generation

    def generate_signals(self, df: pd.DataFrame, price_col, ticker) -> pd.DataFrame:
        df = df.copy()
        close = df[price_col]
        _tmp_array = one_euro_filter(close.values, self.one_euro_min, self.one_euro_factor)
        assert len(_tmp_array) == len(close)
        df['OneEuro'] = _tmp_array
        df['RSI'] = calculate_rsi(close, self.rsi_period)
        macd_df = calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        df['MACD'] = macd_df['MACD']
        df['MACD_Signal'] = macd_df['Signal']
        df['Histogram'] = macd_df['Histogram']

        # ✅ DYNAMIC TARGET SELECTION (now includes "floor" mode)
        if self.target_type == "floor":
            df['FutureLabel'] = create_target_floor(close, self.lookahead_bars, self.threshold_pct)
        elif self.target_type == "any":
            df['FutureLabel'] = create_target_any(close, self.lookahead_bars, self.threshold_pct)
        elif self.target_type == "any_half_B":
            df['FutureLabel'] = create_target_any_at_half_B(close, self.lookahead_bars, self.threshold_pct)
        else:  # "exact"
            df['FutureLabel'] = create_target_exact(close, self.lookahead_bars, self.threshold_pct)

        # ✅ SIGNAL GENERATION: JIT or vectorized pandas
        if self.use_jit_signals and len(df) > 100:  # JIT overhead not worth it for small datasets
            close_vals = close.to_numpy(dtype=np.float64)
            one_euro_vals = df['OneEuro'].to_numpy(dtype=np.float64)
            rsi_vals = df['RSI'].to_numpy(dtype=np.float64)
            hist_vals = df['Histogram'].to_numpy(dtype=np.float64)
            rsi_mode_momentum = (self.rsi_mode == "momentum")

            signal_array = _generate_signals_numba(
                close_vals, one_euro_vals, rsi_vals, hist_vals,
                self.rsi_oversold, self.rsi_overbought, rsi_mode_momentum
            )
            df['Signal'] = signal_array
        else:
            # Original vectorized pandas approach
            df['Signal'] = 0
            rsi_shift = df['RSI'].shift(1).fillna(df['RSI'])
            hist_shift = df['Histogram'].shift(1).fillna(df['Histogram'])

            if self.rsi_mode == "reversion":
                long_rsi_cond = df['RSI'] < self.rsi_oversold
                short_rsi_cond = df['RSI'] > self.rsi_overbought
            else:  # momentum
                long_rsi_cond = df['RSI'] > self.rsi_overbought
                short_rsi_cond = df['RSI'] < self.rsi_oversold

            long_cond = ((close > df['OneEuro']) & long_rsi_cond & (df['RSI'] > rsi_shift) &
                         (df['Histogram'] > 0) & (df['Histogram'] > hist_shift))
            short_cond = ((close < df['OneEuro']) & short_rsi_cond & (df['RSI'] < rsi_shift) &
                          (df['Histogram'] < 0) & (df['Histogram'] < hist_shift))

            df.loc[long_cond, 'Signal'] = 1
            df.loc[short_cond, 'Signal'] = -1
            df['Signal'] = df['Signal'].fillna(0)

        return df

    def evaluate_signals(self, df: pd.DataFrame, price_col) -> dict:
        valid = df[['Signal', 'FutureLabel']].dropna()
        if len(valid) == 0: return {}
        valid = valid[valid['Signal'] != 0].copy()
        if len(valid) == 0: return {'total_signals': 0}

        valid['PredUp'] = (valid['Signal'] == 1).astype(int)
        valid['ActualUp'] = valid['FutureLabel'].astype(int)
        total = len(valid)
        correct = (valid['PredUp'] == valid['ActualUp']).sum()
        long_signals = valid[valid['Signal'] == 1]
        short_signals = valid[valid['Signal'] == -1]

        long_correct = (long_signals['ActualUp'] == 1).sum() if len(long_signals) > 0 else 0
        short_correct = (short_signals['ActualUp'] == 0).sum() if len(short_signals) > 0 else 0

        return {
            'total_signals': total,
            'total_bars': len(df),
            'accuracy': correct / total if total > 0 else 0,
            'long_signals': len(long_signals),
            'long_accuracy': long_correct / len(long_signals) if len(long_signals) > 0 else 0,
            'short_signals': len(short_signals),
            'short_accuracy': short_correct / len(short_signals) if len(short_signals) > 0 else 0,
        }


# ============================================
# JIT WARMUP & VALIDATION UTILITIES
# ============================================

def warmup_jit():
    """Pre-compile JIT functions to avoid first-run overhead."""
    dummy = np.array([100.0, 101.0, 99.5, 102.0, 100.5, 103.0, 101.5, 104.0], dtype=np.float64)
    _ = _rsi_numba(dummy, 14)
    _ = _ema_numba(dummy, 12)
    _ = _macd_numba(dummy, 12, 26, 9)
    _ = _generate_signals_numba(dummy, dummy, dummy, dummy, 30.0, 70.0, True)


def _validate_jit_consistency(verbose: bool = True) -> bool:
    """
    Sanity check: Assert JIT functions produce results consistent with expected behavior.
    Returns True if validation passes, False otherwise.
    """
    if verbose:
        print("🔍 Running JIT consistency validation...")

    try:
        np.random.seed(42)
        # Generate realistic test data
        test_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        test_close = pd.Series(test_prices, name='Close')

        # Test RSI
        rsi_result = calculate_rsi(test_close, period=14)
        assert not rsi_result.isna().all(), "RSI: All values are NaN"
        assert rsi_result.min() >= 0 and rsi_result.max() <= 100, "RSI: Values out of [0, 100] range"

        # Test MACD
        macd_df = calculate_macd(test_close, fast=12, slow=26, signal=9)
        assert not macd_df['MACD'].isna().all(), "MACD: All values are NaN"
        assert len(macd_df) == len(test_close), "MACD: Length mismatch"

        # Test signal generation (both modes)
        for mode in ["momentum", "reversion"]:
            df_test = pd.DataFrame({'Close': test_close})
            df_test['OneEuro'] = one_euro_filter(test_close.values, 10.0, 0.2)
            df_test['RSI'] = calculate_rsi(test_close, 14)
            macd_test = calculate_macd(test_close, 12, 26, 9)
            df_test['Histogram'] = macd_test['Histogram']

            # Test pandas vectorized path
            system_pandas = ForecastSystem(rsi_mode=mode, use_jit_signals=False)
            result_pandas = system_pandas.generate_signals(df_test.copy(), 'Close', 'TEST')

            # Test JIT path
            system_jit = ForecastSystem(rsi_mode=mode, use_jit_signals=True)
            result_jit = system_jit.generate_signals(df_test.copy(), 'Close', 'TEST')

            # Signals should be identical (both paths use same logic)
            assert (result_pandas['Signal'] == result_jit['Signal']).all(), \
                f"Signal mismatch in {mode} mode between pandas and JIT paths"

        if verbose:
            print("✅ JIT validation passed: All functions produce consistent results")
        return True

    except Exception as e:
        print(f"❌ JIT validation FAILED: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


# ============================================
# 6. RUN FORECAST (Updated to accept parameters)
# ============================================
def run_forecast(df: pd.DataFrame, price_col, ticker: str = "^GSPC",
                 rsi_period: int = 14, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0, rsi_mode: str = "momentum",
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 one_euro_min: float = 10.0, one_euro_factor: float = 0.2,
                 lookahead_bars: int = 5, threshold_pct: float = 0.01,
                 target_type: str = "any", use_jit_signals: bool = False) -> Tuple[pd.DataFrame, dict]:
    if not pd.api.types.is_numeric_dtype(df[price_col].to_numpy().dtype if isinstance(df[price_col], pd.DataFrame) else df[price_col].dtype):
        df[price_col] = df[price_col].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[price_col])

    system = ForecastSystem(
        rsi_period=rsi_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought, rsi_mode=rsi_mode,
        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
        one_euro_min=one_euro_min, one_euro_factor=one_euro_factor,
        lookahead_bars=lookahead_bars, threshold_pct=threshold_pct,
        target_type=target_type, use_jit_signals=use_jit_signals
    )
    df_signals = system.generate_signals(df=df, price_col=price_col, ticker=ticker)
    metrics = system.evaluate_signals(df=df_signals, price_col=price_col)
    return df_signals, metrics


# ============================================
# 7. PLOTTING HELPER (Unchanged)
# ============================================
def plot_forecast_results(df: pd.DataFrame, price_col, sample: int = 200, start_idx: int = -1,
                          highlight_signals: bool = True, zoom_region: Optional[Tuple[int, int]] = None):
    if start_idx == -1:
        start_idx = max(0, len(df) - sample)
    plot_df = df.iloc[start_idx:start_idx + sample].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(plot_df.index, plot_df[price_col], label='Close', alpha=0.7, linewidth=1, color='black')
    ax1.plot(plot_df.index, plot_df['OneEuro'], label='One-Euro Filter', color='blue', linewidth=2)
    longs = plot_df[plot_df['Signal'] == 1]
    shorts = plot_df[plot_df['Signal'] == -1]
    ax1.scatter(longs.index, longs[price_col], marker='^', color='green', s=100, label='Long Signal', zorder=6, edgecolors='darkgreen', linewidth=1.5)
    ax1.scatter(shorts.index, shorts[price_col], marker='v', color='red', s=100, label='Short Signal', zorder=6, edgecolors='darkred', linewidth=1.5)
    if highlight_signals:
        for idx in longs.index: ax1.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax1.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax1.set_title('Price + One-Euro Filter + Trading Signals', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10);
    ax1.legend(loc='upper left', fontsize=9);
    ax1.grid(True, alpha=0.3, linestyle='--');
    ax1.set_axisbelow(True)

    ax2.plot(plot_df.index, plot_df['RSI'], label='RSI', color='purple', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Overbought/Oversold')
    ax2.axhline(30, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax2.fill_between(plot_df.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(plot_df.index, 0, 30, color='green', alpha=0.1)
    if highlight_signals:
        for idx in longs.index: ax2.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax2.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax2.set_ylabel('RSI', fontsize=10);
    ax2.set_ylim(0, 100);
    ax2.legend(loc='lower right', fontsize=9);
    ax2.grid(True, alpha=0.3, linestyle='--');
    ax2.set_axisbelow(True)

    colors = ['green' if val >= 0 else 'red' for val in plot_df['Histogram']]
    ax3.bar(plot_df.index, plot_df['Histogram'], color=colors, alpha=0.6, label='Histogram', width=1)
    ax3.plot(plot_df.index, plot_df['MACD'], label='MACD', color='blue', linewidth=1.2)
    ax3.plot(plot_df.index, plot_df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1.2)
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.4, linewidth=0.8)
    if highlight_signals:
        for idx in longs.index: ax3.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax3.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax3.set_ylabel('MACD', fontsize=10);
    ax3.set_xlabel('Date', fontsize=10);
    ax3.legend(loc='lower right', fontsize=9);
    ax3.grid(True, alpha=0.3, linestyle='--');
    ax3.set_axisbelow(True)

    if zoom_region is not None:
        zoom_start, zoom_end = zoom_region
        if zoom_start < len(plot_df) and zoom_end <= len(plot_df) and zoom_start < zoom_end:
            zoom_df = plot_df.iloc[zoom_start:zoom_end]
            from matplotlib.patches import Rectangle
            ax1_inset = ax1.inset_axes([0.62, 0.55, 0.35, 0.35])
            ax1_inset.plot(zoom_df.index, zoom_df[price_col], color='black', linewidth=1.5)
            ax1_inset.plot(zoom_df.index, zoom_df['OneEuro'], color='blue', linewidth=2)
            ax1_inset.scatter(zoom_df[zoom_df['Signal'] == 1].index, zoom_df[zoom_df['Signal'] == 1][price_col], marker='^', color='green', s=50, zorder=5)
            ax1_inset.scatter(zoom_df[zoom_df['Signal'] == -1].index, zoom_df[zoom_df['Signal'] == -1][price_col], marker='v', color='red', s=50, zorder=5)
            ax1_inset.set_xticks([]);
            ax1_inset.set_yticks([]);
            ax1_inset.set_title('Zoom', fontsize=8, fontweight='bold');
            ax1_inset.grid(True, alpha=0.3)
            rect = Rectangle((zoom_df.index[0], plot_df[price_col].min()), zoom_df.index[-1] - zoom_df.index[0], plot_df[price_col].max() - plot_df[price_col].min(), linewidth=1.5, edgecolor='gold', facecolor='none', linestyle='--')
            ax1.add_patch(rect)
            ax1.indicate_inset_zoom(ax1_inset, edgecolor="gold", alpha=0.7)

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for ax in [ax1, ax2]: ax.tick_params(labelbottom=False)
    plt.suptitle('🔗 Linked Technical Analysis Dashboard (Zoom: sharex enabled)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.text(0.5, 0.01, "💡 Tip: Use mouse wheel to zoom, drag to pan — all panels stay synchronized!", ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.show()


# ============================================
# 8. ARGPARSE & ENTRY POINT
# ============================================
def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based Technical Forecasting System (One-Euro + RSI + MACD) [JIT-ACCELERATED]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    data_grp = parser.add_argument_group("Data & Environment")
    data_grp.add_argument("--dataset-id", type=str, default="day", help="Dataset identifier", choices=DATASET_AVAILABLE)
    data_grp.add_argument("--ticker", type=str, default="^GSPC", help="Ticker symbol to analyze")
    data_grp.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    data_grp.add_argument("--disable-print", action="store_true", help="Skip prints")
    data_grp.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Print detailed progress, metrics, and explanations')
    data_grp.add_argument('--validate-jit', action='store_true', default=False,
                          help='Run JIT consistency sanity check at startup (default: disabled)')
    data_grp.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")
    algo_grp = parser.add_argument_group("Algorithm Parameters")
    algo_grp.add_argument("--rsi-period", type=int, default=14, help="RSI calculation window")
    algo_grp.add_argument("--rsi-oversold", type=float, default=30.0, help="RSI oversold threshold")
    algo_grp.add_argument("--rsi-overbought", type=float, default=70.0, help="RSI overbought threshold")
    algo_grp.add_argument("--rsi-mode", type=str, choices=["momentum", "reversion"], default="momentum", help="RSI signal logic")
    algo_grp.add_argument("--macd-fast", type=int, default=12, help="MACD fast EMA period")
    algo_grp.add_argument("--macd-slow", type=int, default=26, help="MACD slow EMA period")
    algo_grp.add_argument("--macd-signal", type=int, default=9, help="MACD signal line period")
    algo_grp.add_argument("--one-euro-min", type=float, default=10.0, help="One-Euro filter min cutoff")
    algo_grp.add_argument("--one-euro-factor", type=float, default=0.2, help="One-Euro filter beta factor")
    algo_grp.add_argument("--lookahead-bars", type=int, default=5, help="Future bars to forecast")
    algo_grp.add_argument("--threshold-pct", type=float, default=0., help="Min %% move to create the target (use NEGATIVE for floor mode)")
    algo_grp.add_argument('--train-ratio', type=float, default=1.0, help='Ratio of data to use for training (rest for validation). Use 1.0 to disable split.')
    algo_grp.add_argument('--use-jit-signals', action='store_true', default=False,
                          help='Enable JIT acceleration for signal generation (recommended for large datasets)')

    # Target labeling mode - UPDATED to include "floor"
    algo_grp.add_argument(
        "--target-type",
        type=str,
        choices=["exact", "any", "any_half_B", "floor"],  # ← Added "floor"
        default="any_half_B",
        help="Target labeling method: 'exact' (price at t+lookahead), 'any' (price > threshold anywhere), 'any_half_B' (second half), or 'floor' 🆕 (price NEVER < threshold - ideal for put credit spreads)"
    )

    viz_grp = parser.add_argument_group("Visualization")
    viz_grp.add_argument("--plot-sample", type=int, default=100, help="Number of recent bars to plot")
    viz_grp.add_argument("--disable-plot-sample", action="store_true", help="Skip plotting section")

    realtime_grp = parser.add_argument_group("Real-Time Mode")
    realtime_grp.add_argument(
        "--real-time",
        action="store_true",
        help="Run in real-time mode: evaluate only the latest data point using a saved model"
    )
    realtime_grp.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model file (.pkl) to load optimized parameters"
    )
    realtime_grp.add_argument(
        "--output-signal-only",
        action="store_true",
        help="In real-time mode, output only the signal decision (for automation)"
    )

    return parser


@lru_cache(maxsize=32)
def _load_df(_datase_id):
    cache_filename = get_filename_for_dataset(_datase_id, older_dataset=None)
    with open(cache_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    return master_data_cache


def entry(args):
    # ✅ JIT VALIDATION (optional, disabled by default)
    if args.validate_jit:
        if not _validate_jit_consistency(verbose=args.verbose):
            print("⚠️  JIT validation failed. Proceeding anyway (results may be inconsistent).")
            return None
        # Warmup JIT after validation to cache compiled functions
        warmup_jit()

    # ✅ REAL-TIME MODE: Load model and evaluate latest point only
    if args.real_time:
        if not args.model_path:
            raise ValueError("--model-path is required when using --real-time")
        return run_real_time(output_signal_only=args.output_signal_only, model_path=args.model_path, verbose=args.verbose, clip=args.clip)

    if args.verbose:
        print("✅ __doc__ length:", len(__doc__ or ""))
        print("✅ First line:", __doc__.split('\n')[0] if __doc__ else "None")
        print(__doc__)

    # ✅ BATCH MODE: Original behavior with optional train/val split
    np.random.seed(args.seed)
    master_data_cache = copy.deepcopy(_load_df(_datase_id=args.dataset_id))
    df_spx500 = master_data_cache[args.ticker].sort_index()
    price_col = ('Close', args.ticker) if isinstance(df_spx500.columns, pd.MultiIndex) else 'Close'
    if 1 == args.lookahead_bars:
        assert args.target_type in ["exact", "any", "floor"]
    if args.target_type in ["floor"]:
        assert args.threshold_pct <= 0.
    # Helper to run forecast and print results
    def run_and_report(df_subset, label, plot_results=False):
        df_results, metrics = run_forecast(
            df=df_subset, price_col=price_col, ticker=args.ticker,
            rsi_period=args.rsi_period, rsi_oversold=args.rsi_oversold, rsi_overbought=args.rsi_overbought, rsi_mode=args.rsi_mode,
            macd_fast=args.macd_fast, macd_slow=args.macd_slow, macd_signal=args.macd_signal,
            one_euro_min=args.one_euro_min, one_euro_factor=args.one_euro_factor,
            lookahead_bars=args.lookahead_bars, threshold_pct=args.threshold_pct,
            target_type=args.target_type,
            use_jit_signals=args.use_jit_signals  # Pass JIT flag
        )
        if not args.disable_print:
            print(f"\n📊 {label} Evaluation")
            print(f"   Ticker: {args.ticker} | Data range: {df_subset.index[0].date()} to {df_subset.index[-1].date()}")
            print(f"   Total Signals: {metrics.get('total_signals', 0)}")
            print(f"   Overall Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
            print(f"   Long Signals: {metrics.get('long_signals', 0)} | Accuracy: {metrics.get('long_accuracy', 0) * 100:.2f}%")
            print(f"   Short Signals: {metrics.get('short_signals', 0)} | Accuracy: {metrics.get('short_accuracy', 0) * 100:.2f}%")
            print(f"   Look-ahead Horizon: {args.lookahead_bars} bars")
            print(f"   Target Mode: '{args.target_type}'")
            print(f"   Threshold For Creating Target: {args.threshold_pct * 100:.2f}%\n")
        if plot_results and not args.disable_plot_sample:
            plot_forecast_results(df=df_results, price_col=price_col, sample=args.plot_sample)
        return df_results, metrics

    # ✅ TRAIN/VALIDATION SPLIT LOGIC
    if args.train_ratio < 1.0:
        # Calculate split index (ensure minimum data for indicators)
        min_history = max(args.rsi_period, args.macd_slow, 100) + args.lookahead_bars + 10
        total_len = len(df_spx500)

        if total_len < min_history:
            raise ValueError(f"Dataset too small ({total_len} bars) for indicators + lookahead. Need at least {min_history}.")

        train_end_idx = int(total_len * args.train_ratio)
        # Ensure training set has enough data for warmup
        train_end_idx = max(train_end_idx, min_history)

        df_train = df_spx500.iloc[:train_end_idx].copy()
        df_val = df_spx500.iloc[train_end_idx:].copy()
        if args.verbose:
            print(f"🔀 Data Split: Train={len(df_train)} bars ({args.train_ratio * 100:.1f}%), Val={len(df_val)} bars ({(1 - args.train_ratio) * 100:.1f}%)")

        # Run on training set (backtest)
        _, metrics_train = run_and_report(df_train, "🔧 TRAINING SET (Backtest)", plot_results=False)

        # Run on validation set (final evaluation)
        df_results_val, metrics_val = run_and_report(df_val, "✅ VALIDATION SET (Final Evaluation)", plot_results=True)

        # Print comparison summary
        if not args.disable_print and metrics_train and metrics_val:
            print("\n" + "=" * 60)
            print("📈 TRAIN vs VALIDATION COMPARISON")
            print("=" * 60)
            print(f"{'Metric':<25} {'Train':>12} {'Val':>12} {'Δ':>10}")
            print("-" * 60)
            for key in ['accuracy', 'long_accuracy', 'short_accuracy']:
                train_val = metrics_train.get(key, 0) * 100
                val_val = metrics_val.get(key, 0) * 100
                delta = val_val - train_val
                print(f"{key:<25} {train_val:>11.2f}% {val_val:>11.2f}% {delta:>+9.2f}%")
            print("=" * 60 + "\n")

        return metrics_val, metrics_train

    else:
        # ✅ Original behavior: run on full dataset
        df_results, metrics = run_forecast(
            df=df_spx500, price_col=price_col, ticker=args.ticker,
            rsi_period=args.rsi_period, rsi_oversold=args.rsi_oversold, rsi_overbought=args.rsi_overbought, rsi_mode=args.rsi_mode,
            macd_fast=args.macd_fast, macd_slow=args.macd_slow, macd_signal=args.macd_signal,
            one_euro_min=args.one_euro_min, one_euro_factor=args.one_euro_factor,
            lookahead_bars=args.lookahead_bars, threshold_pct=args.threshold_pct,
            target_type=args.target_type,
            use_jit_signals=args.use_jit_signals  # Pass JIT flag
        )

        if not args.disable_print:
            print("\n📊 Forecast System Evaluation")
            print(f"   Ticker: {args.ticker}")
            print(f"   Total Signals: {metrics.get('total_signals', 0)}")
            print(f"   Overall Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
            print(f"   Long Signals: {metrics.get('long_signals', 0)} | Accuracy: {metrics.get('long_accuracy', 0) * 100:.2f}%")
            print(f"   Short Signals: {metrics.get('short_signals', 0)} | Accuracy: {metrics.get('short_accuracy', 0) * 100:.2f}%")
            print(f"   Look-ahead Horizon: {args.lookahead_bars} bars")
            print(f"   Target Mode: '{args.target_type}'")
            print(f"   Threshold For Creating Target: {args.threshold_pct * 100:.2f}%\n")

        if not args.disable_plot_sample:
            plot_forecast_results(df=df_results, price_col=price_col, sample=args.plot_sample)

        return metrics, metrics


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)