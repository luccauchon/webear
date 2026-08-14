"""
==============================================================================
#  ▄▄▄█████▓ ██░ ██  ▄████▄   ██ ▄█▀ ▓█████▄  ██▓  ██████  ██▓███
#  ▓  ██▒ ▓▒▓██░ ██▒▒██▀ ▀█   ██▄█▒  ▒██▀ ██▌▓██▒▒██    ▒ ▓██░  ██▒
#  ▒ ▓██░ ▒░▒██▀▀██░▒▓█    ▄ ▓███▄░  ░██   █▌▒██▒░ ▓██▄   ▓██░ ██▓▒
#  ░ ▓██▓ ░ ░▓█ ░██ ▒▓▓▄ ▄██▒▓██ █▄  ░▓█▄   ▌░██░  ▒   ██▒▓██▄█▓▒ ░
#    ▒██▒ ░ ░▓█▒░██▓▒ ▓███▀ ░▒██▒ █▄ ░▒████▓ ░██░▒██████▒▒▓███▒░
#    ▒ ░░    ▒ ░░▒░▒░ ░▒ ▒  ░▒ ▒▒ ▓▒  ▒▒▓  ▒ ░▓  ▒ ▒▓▒ ▒ ░▒ ▒▓▒░
#      ░     ▒ ░▒░ ░  ░  ▒   ░ ░▒ ▒░  ░ ▒  ▒  ▒ ░░ ░▒  ░ ░░ ▒░░
#    ░       ░  ░░ ░░        ░ ░   ░   ░  ░  ▒ ░░  ░  ░  ░ ░░
#            ░  ░  ░░ ░      ░  ░      ░     ░        ░    ░
#                         ░           ░
==============================================================================
🎯 MULTI-SIGNAL TECHNICAL STRATEGY OPTIMIZER & REAL-TIME MONITOR
==============================================================================

OVERVIEW:
---------
This script implements a robust, multi-signal technical analysis strategy
specifically designed for validating options trading setups (specifically
Put and Call Credit Spreads). It combines momentum, mean-reversion, and
Fibonacci confluence indicators to generate high-probability BUY/SELL signals,
which are then rigorously evaluated against forward-looking strike-price
targets to simulate real-world options expiration outcomes.

The system leverages Optuna for hyperparameter optimization, utilizing
TimeSeriesSplit cross-validation to prevent look-ahead bias, and supports
three distinct execution modes: Backtesting/Optimization, Offline Evaluation,
and Real-Time Signal Monitoring.

CORE STRATEGY COMPONENTS:
-------------------------
1. RSI Strategies:
   - Pullback-to-50: Enters when price crosses SMA and RSI crosses back over 50.
   - EMA Crossovers: Enters on RSI and RSI-EMA crossovers in overbought/oversold zones.
   - MA Confluence: Combines SMA trends with RSI momentum shifts.

2. Fibonacci Confluence:
   - Calculates dynamic swing highs/lows over a configurable lookback.
   - Identifies the "Golden Zone" (0.5 to 0.618 retracement).
   - Triggers buys when price enters the Golden Zone alongside RSI momentum shifts.

3. Divergence Detection:
   - Scans for Regular and Hidden Bullish/Bearish divergences between
     Price (Highs/Lows) and RSI over a rolling window.

4. New Advanced Strategies:
   - MACD Crossovers: Signal line crossovers.
   - Bollinger Band Bounces: Reversals off the upper/lower bands.
   - Volume Spikes: High volume directional candles.
   - Stochastic Oscillators: Oversold/overbought %K/%D crossovers.
   - VWAP Retests: Price crossing rolling 50-period VWAP.

EVALUATION LOGIC (OPTIONS SIMULATION):
--------------------------------------
Signals are evaluated based on forward-looking price action over a defined
`lookahead_bars` window to simulate holding an options contract to expiration.

- BUY Signals (Put Credit Spread Simulation):
  Generates a bullish signal. To "win", the underlying asset's price must
  remain ABOVE a specified `put_strike_pct` (e.g., 0.96 * entry_price)
  throughout the lookahead window (or at final close, depending on the method).

- SELL Signals (Call Credit Spread Simulation):
  Generates a bearish signal. To "win", the underlying asset's price must
  remain BELOW a specified `call_strike_pct` (e.g., 1.04 * entry_price)
  throughout the lookahead window (or at final close).

Evaluation methods include:
  - 'final_close': Only the closing price at the end of the lookahead window matters.
  - 'touched': The price must not breach the strike threshold at ANY point
    during the lookahead window (simulates touch/no-touch options or strict risk).

ALGORITHM FLOW:
---------------
1. Data Ingestion & Feature Engineering:
   - Loads cached OHLCV data.
   - Computes technical indicators (RSI, SMA, EMA, Fibonacci retracements)
     using `pandas_ta` wrapped in safe error-handling functions.
   - Uses vectorized Pandas operations to generate boolean masks for individual
     strategy setups (e.g., RSI pullbacks, divergences).

2. Signal Aggregation:
   - Combines individual setup masks. New `min_buy_confluence` and `min_sell_confluence`
     parameters require a minimum number of strategies to agree simultaneously,
     effectively narrowing down signal quantity and filtering out weaker, isolated setups.

3. Forward-Looking Vectorized Evaluation:
   - Extracts signal indices and uses NumPy broadcasting to create a 2D matrix
     of future price indices `[n_signals, lookahead_bars]`.
   - Evaluates win conditions simultaneously across all signals without slow
     Python `for` loops, comparing future Highs/Lows/Closes against the
     calculated strike prices (`entry_price * strike_pct`).
   - Computes Win Rate (Wins / Total Signals) and Signal Density.

4. Hyperparameter Optimization (Optuna):
   - Defines a search space for indicator lengths, windows, and confluence thresholds.
   - For each trial, splits the training data into chronological folds
     (`TimeSeriesSplit`).
   - Calculates the Bayesian Smoothed Win Rate for each fold to balance win rate
     and sample size, applies a penalty if signal density falls below the minimum
     threshold, applies an excess penalty for high densities to push selectivity,
     applies a penalty for overlapping signals (clustering), and averages the scores.
   - Uses Tree-structured Parzen Estimator (TPE) sampling to converge on
     the optimal parameter set.

5. Validation & Inference:
   - Tests the optimized parameters on a hold-out validation set to measure
     the train/validation performance gap (overfitting detection).
   - In Real-Time mode, applies the saved parameters to the latest completed
     bar and outputs precise options trade recommendations (strikes,
     expiration dates, and win conditions).

EXECUTION MODES:
----------------
1. Optimization Mode (`--optimize`):
   - Splits data chronologically into Train/Validation sets (`--train-ratio`).
   - Tunes hyperparameters on the training set using Optuna.
   - Evaluates the best parameters on the hold-out validation set.
   - Saves the optimal parameters and metadata to a `.pkl` file.

2. Evaluation Mode (`--model-path <file>`):
   - Loads a previously saved `.pkl` model.
   - Applies the saved parameters to the full dataset.
   - Outputs comprehensive win-rate statistics, yearly breakdowns, and optional plots.

3. Real-Time Mode (`--real-time --model-path <file>`):
   - Loads a saved model and the latest available market data.
   - Checks the most recent completed bar for active BUY/SELL signals.
   - Outputs specific recommended options trades (strikes, expirations, win conditions).

==============================================================================
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
from numba import njit
import pandas as pd
import pandas_ta as ta
import numpy as np
from utils import get_filename_for_dataset, get_next_step, factory_load_data
import pickle
import argparse
import os
import optuna
import json
from argparse import Namespace
import math
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import sys
# Suppress Optuna & pandas_ta debug logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None


def safe_ta_indicator(func, series, default_fill=np.nan, **kwargs):
    """
    Safely call pandas_ta functions, returning a valid Series even on failure.
    """
    try:
        result = func(series, **kwargs)
        # Handle None return or all-NaN result
        if result is None or (hasattr(result, 'isna') and result.isna().all()):
            return pd.Series(default_fill, index=series.index, name=series.name)
        return result
    except Exception as e:
        print(f"⚠️  TA function {func.__name__} failed with {kwargs}: {e}")
        return pd.Series(default_fill, index=series.index, name=series.name)


# ==============================================================================
# 🎯 STRATEGY OPTIMIZER & REAL-TIME MONITOR
# ==============================================================================
#
#  ▄▄▄█████▓ ██░ ██  ▄████▄   ██ ▄█▀ ▓█████▄  ██▓  ██████  ██▓███
#  ▓  ██▒ ▓▒▓██░ ██▒▒██▀ ▀█   ██▄█▒  ▒██▀ ██▌▓██▒▒██    ▒ ▓██░  ██▒
#  ▒ ▓██░ ▒░▒██▀▀██░▒▓█    ▄ ▓███▄░  ░██   █▌▒██▒░ ▓██▄   ▓██░ ██▓▒
#  ░ ▓██▓ ░ ░▓█ ░██ ▒▓▓▄ ▄██▒▓██ █▄  ░▓█▄   ▌░██░  ▒   ██▒▓██▄█▓▒ ░
#    ▒██▒ ░ ░▓█▒░██▓▒ ▓███▀ ░▒██▒ █▄ ░▒████▓ ░██░▒██████▒▒▓███▒░
#    ▒ ░░    ▒ ░░▒░▒░ ░▒ ▒  ░▒ ▒▒ ▓▒  ▒▒▓  ▒ ░▓  ▒ ▒▓▒ ▒ ░▒ ▒▓▒░
#      ░     ▒ ░▒░ ░  ░  ▒   ░ ░▒ ▒░  ░ ▒  ▒  ▒ ░░ ░▒  ░ ░░ ▒░░
#    ░       ░  ░░ ░░        ░ ░   ░   ░  ░  ▒ ░░  ░  ░  ░ ░░
#            ░  ░  ░░ ░      ░  ░      ░     ░        ░    ░
#                         ░           ░
#
# 📈 PURPOSE:
#   Multi-signal technical strategy optimizer for options trading validation.
#   Combines momentum, mean-reversion, and Fibonacci confluence to generate
#   high-probability BUY/SELL signals evaluated against forward-looking
#   strike-price targets.
#
# 🔧 CORE COMPONENTS:
#   • RSI strategies      : Pullback-to-50, EMA crossovers, MA confluence
#   • Fibonacci Confluence: Golden Zone (0.5–0.618) retracement entries
#   • Divergence Detection: Regular & Hidden Bullish/Bearish divergences
#   • Signal Aggregation  : Logical OR across all bullish/bearish conditions,
#                           now enhanced with `min_buy_confluence` and `min_sell_confluence`
#                           thresholds to narrow down signal quantity and increase selectivity.
#
# 🎯 EVALUATION LOGIC:
#   • BUY Signal  → Price must stay ABOVE put_strike_pct (e.g., 0.96×)
#                   within lookahead_bars → simulates profitable put credit spread
#   • SELL Signal → Price must stay BELOW call_strike_pct (e.g., 1.04×)
#                   within lookahead_bars → simulates profitable call credit spread
#
# ⚙️ OPTIMIZATION (Optuna):
#   • Hyperparameters: RSI length, SMA period, Fib lookback, divergence window,
#                      and Minimum Buy/Sell Confluence (number of strategies that must agree).
#   • Objective: Maximize Smoothed Win Rate (balances Win Rate & Sample Size)
#                with density penalty, overlap penalty, and consistency penalty.
#   • Persistence: Supports SQLite, PostgreSQL, MySQL, etc., via standard database URLs
#
# 🔄 TRAIN/VALIDATION SPLIT:
#   • --train-ratio argument (default: 0.7) for chronological data splitting
#   • Optimize on training set, evaluate best params on validation set
#   • Detects overfitting via train/val performance gap
#
# 🚀 MODES:
#   1. Backtest Mode    : --optimize          → Tune parameters on historical data
#   2. Evaluation Mode  : --model-path FILE   → Test saved config on full dataset
#   3. Real-Time Mode   : --real-time --model-path FILE → Check latest bar for signals
#
# 📦 OUTPUT:
#   • Descriptive model files: {ticker}_{dataset}_{target}_score-{X.XXXX}_{params}.pkl
#   • Optional: Signal dataframes, yearly win-rate tables, matplotlib plots
#
# ==============================================================================

# ==============================================================================
# ORIGINAL STRATEGY FUNCTIONS (Unchanged)
# ==============================================================================
def calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, ticker, lookback, rsi_midline):
    swing_high = df[high_col].rolling(window=lookback).max()
    swing_low = df[low_col].rolling(window=lookback).min()
    diff = swing_high - swing_low

    fib_50_col = ('Fib_50', ticker)
    Fib_618_col = ('Fib_618', ticker)
    df[fib_50_col] = swing_high - 0.5 * diff
    df[Fib_618_col] = swing_high - 0.618 * diff

    golden_zone_col = ('In_Golden_Zone', ticker)
    df[golden_zone_col] = (df[close_col] <= df[fib_50_col]) & (df[close_col] >= df[Fib_618_col])

    strategy_Fib_RSI_Buy_col = ('Strategy_Fib_RSI_Buy', ticker)
    df[strategy_Fib_RSI_Buy_col] = df[golden_zone_col] & \
                                   (df[rsi_col].shift(1) < rsi_midline) & \
                                   (df[rsi_col] > rsi_midline)
    return df, strategy_Fib_RSI_Buy_col


def implement_rsi_strategies(df, close_col, ticker, rsi_length, rsi_ema_10, sma_50, rsi_midline, rsi_oversold, rsi_overbought):
    rsi_col = ('RSI', ticker)
    df[rsi_col] = safe_ta_indicator(ta.rsi, df[close_col], length=rsi_length)

    rsi_ema_10_col = ('RSI_EMA_10', ticker)
    df[rsi_ema_10_col] = safe_ta_indicator(ta.ema, df[rsi_col], length=rsi_ema_10)

    sma_50_col = ('SMA_50', ticker)
    df[sma_50_col] = safe_ta_indicator(ta.sma, df[close_col], length=sma_50)

    Setup_Pullback_50_Buy_col = ('Setup_Pullback_50_Buy', ticker)
    df[Setup_Pullback_50_Buy_col] = (df[close_col] > df[sma_50_col]) & \
                                    (df[rsi_col].shift(1) < rsi_midline) & \
                                    (df[rsi_col] > rsi_midline)

    Setup_Pullback_50_Sell_col = ('Setup_Pullback_50_Sell', ticker)
    df[Setup_Pullback_50_Sell_col] = (df[close_col] < df[sma_50_col]) & \
                                     (df[rsi_col].shift(1) > rsi_midline) & \
                                     (df[rsi_col] < rsi_midline)

    Setup_EMA_Cross_Buy_col = ('Setup_EMA_Cross_Buy', ticker)
    df[Setup_EMA_Cross_Buy_col] = (df[rsi_col].shift(1) < df[rsi_ema_10_col].shift(1)) & \
                                  (df[rsi_col] > df[rsi_ema_10_col]) & \
                                  (df[rsi_col] < rsi_oversold)

    Setup_EMA_Cross_Sell_col = ('Setup_EMA_Cross_Sell', ticker)
    df[Setup_EMA_Cross_Sell_col] = (df[rsi_col].shift(1) > df[rsi_ema_10_col].shift(1)) & \
                                   (df[rsi_col] < df[rsi_ema_10_col]) & \
                                   (df[rsi_col] > rsi_overbought)

    Strategy_MA_Confluence_Buy_col = ('Strategy_MA_Confluence_Buy', ticker)
    df[Strategy_MA_Confluence_Buy_col] = (df[close_col] > df[sma_50_col]) & \
                                         (df[sma_50_col] > df[sma_50_col].shift(1)) & \
                                         (df[rsi_col] > df[rsi_ema_10_col]) & \
                                         (df[rsi_col].shift(1) < df[rsi_ema_10_col].shift(1))

    return df, rsi_col, Setup_Pullback_50_Buy_col, Setup_Pullback_50_Sell_col, Setup_EMA_Cross_Buy_col, Setup_EMA_Cross_Sell_col, Strategy_MA_Confluence_Buy_col


def find_divergences(df, high_col, low_col, rsi_col, ticker, window):
    Price_Low_col = ('Price_Low', ticker)
    df[Price_Low_col] = df[low_col].rolling(window=window).min()
    Price_High_col = ('Price_High', ticker)
    df[Price_High_col] = df[high_col].rolling(window=window).max()
    RSI_Low_col = ('RSI_Low', ticker)
    df[RSI_Low_col] = df[rsi_col].rolling(window=window).min()
    RSI_High_col = ('RSI_High', ticker)
    df[RSI_High_col] = df[rsi_col].rolling(window=window).max()

    Regular_Bullish_Div_col = ('Regular_Bullish_Div', ticker)
    df[Regular_Bullish_Div_col] = (df[Price_Low_col] < df[Price_Low_col].shift(window)) & \
                                  (df[RSI_Low_col] > df[RSI_Low_col].shift(window))

    Regular_Bearish_Div_col = ('Regular_Bearish_Div', ticker)
    df[Regular_Bearish_Div_col] = (df[Price_High_col] > df[Price_High_col].shift(window)) & \
                                  (df[RSI_High_col] < df[RSI_High_col].shift(window))

    Hidden_Bullish_Div_col = ('Hidden_Bullish_Div', ticker)
    df[Hidden_Bullish_Div_col] = (df[Price_Low_col] > df[Price_Low_col].shift(window)) & \
                                 (df[RSI_Low_col] < df[RSI_Low_col].shift(window))

    Hidden_Bearish_Div_col = ('Hidden_Bearish_Div', ticker)
    df[Hidden_Bearish_Div_col] = (df[Price_High_col] < df[Price_High_col].shift(window)) & \
                                 (df[RSI_High_col] > df[RSI_High_col].shift(window))

    return df, Regular_Bullish_Div_col, Regular_Bearish_Div_col, Hidden_Bullish_Div_col, Hidden_Bearish_Div_col


def implement_additional_strategies(df, close_col, high_col, low_col, volume_col, open_col, ticker,
                                    macd_fast=12, macd_slow=26, macd_signal=9,
                                    bb_length=20, bb_std=2.0,
                                    vol_sma_length=20, vol_multiplier=2.0,
                                    stoch_k_period=14, stoch_d_period=3, stoch_smooth_k_period=1, stoch_oversold=20, stoch_overbought=80,
                                    vwap_window=50):
    # 1. MACD Crossovers
    try:
        macd_df = ta.macd(df[close_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        if macd_df is not None and not macd_df.empty and macd_df.shape[1] >= 3:
            macd_line = macd_df.iloc[:, 0]
            signal_line = macd_df.iloc[:, 2]
        else:
            macd_line = pd.Series(np.nan, index=df.index)
            signal_line = pd.Series(np.nan, index=df.index)
    except Exception:
        macd_line = pd.Series(np.nan, index=df.index)
        signal_line = pd.Series(np.nan, index=df.index)

    macd_buy_col = ('Setup_MACD_Buy', ticker)
    macd_sell_col = ('Setup_MACD_Sell', ticker)
    df[macd_buy_col] = ((macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)).fillna(False).astype(bool)
    df[macd_sell_col] = ((macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)).fillna(False).astype(bool)

    # 2. Bollinger Band Bounces
    try:
        bbands_df = ta.bbands(df[close_col], length=bb_length, std=bb_std)
        if bbands_df is not None and not bbands_df.empty and bbands_df.shape[1] >= 3:
            bb_upper = bbands_df.iloc[:, 0]
            bb_lower = bbands_df.iloc[:, 2]
        else:
            bb_lower = pd.Series(np.nan, index=df.index)
            bb_upper = pd.Series(np.nan, index=df.index)
    except Exception:
        bb_lower = pd.Series(np.nan, index=df.index)
        bb_upper = pd.Series(np.nan, index=df.index)

    bb_buy_col = ('Setup_BB_Buy', ticker)
    bb_sell_col = ('Setup_BB_Sell', ticker)
    df[bb_buy_col] = ((df[close_col].shift(1) < bb_lower.shift(1)) & (df[close_col] > bb_lower)).fillna(False).astype(bool)
    df[bb_sell_col] = ((df[close_col].shift(1) > bb_upper.shift(1)) & (df[close_col] < bb_upper)).fillna(False).astype(bool)

    # 3. Volume Spikes
    vol_buy_col = ('Setup_Vol_Buy', ticker)
    vol_sell_col = ('Setup_Vol_Sell', ticker)
    if volume_col in df.columns and open_col in df.columns:
        vol_sma = safe_ta_indicator(ta.sma, df[volume_col], length=vol_sma_length)
        vol_spike = df[volume_col] > (vol_multiplier * vol_sma)
        df[vol_buy_col] = (vol_spike & (df[close_col] > df[open_col])).fillna(False).astype(bool)
        df[vol_sell_col] = (vol_spike & (df[close_col] < df[open_col])).fillna(False).astype(bool)
    else:
        df[vol_buy_col] = False
        df[vol_sell_col] = False

    # 4. Stochastic Oscillators
    try:
        stoch_df = ta.stoch(df[high_col], df[low_col], df[close_col], k=stoch_k_period, d=stoch_d_period, smooth_k=stoch_smooth_k_period)
        if stoch_df is not None and not stoch_df.empty and stoch_df.shape[1] >= 2:
            stoch_k = stoch_df.iloc[:, 0]
            stoch_d = stoch_df.iloc[:, 1]
        else:
            stoch_k = pd.Series(np.nan, index=df.index)
            stoch_d = pd.Series(np.nan, index=df.index)
    except Exception:
        stoch_k = pd.Series(np.nan, index=df.index)
        stoch_d = pd.Series(np.nan, index=df.index)

    stoch_buy_col = ('Setup_Stoch_Buy', ticker)
    stoch_sell_col = ('Setup_Stoch_Sell', ticker)
    df[stoch_buy_col] = ((stoch_k.shift(1) < stoch_d.shift(1)) & (stoch_k > stoch_d) & (stoch_k < stoch_oversold)).fillna(False).astype(bool)
    df[stoch_sell_col] = ((stoch_k.shift(1) > stoch_d.shift(1)) & (stoch_k < stoch_d) & (stoch_k > stoch_overbought)).fillna(False).astype(bool)

    # 5. VWAP Retests (Rolling)
    vwap_buy_col = ('Setup_VWAP_Buy', ticker)
    vwap_sell_col = ('Setup_VWAP_Sell', ticker)
    if volume_col in df.columns:
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        tp_vol = typical_price * df[volume_col]
        rolling_tp_vol = tp_vol.rolling(window=vwap_window).sum()
        rolling_vol = df[volume_col].rolling(window=vwap_window).sum()
        vwap = rolling_tp_vol / rolling_vol

        df[vwap_buy_col] = ((df[close_col].shift(1) < vwap.shift(1)) & (df[close_col] > vwap)).fillna(False).astype(bool)
        df[vwap_sell_col] = ((df[close_col].shift(1) > vwap.shift(1)) & (df[close_col] < vwap)).fillna(False).astype(bool)
    else:
        df[vwap_buy_col] = False
        df[vwap_sell_col] = False

    return (macd_buy_col, macd_sell_col, bb_buy_col, bb_sell_col,
            vol_buy_col, vol_sell_col, stoch_buy_col, stoch_sell_col,
            vwap_buy_col, vwap_sell_col)


# ==============================================================================
# 🆕 ADVANCED METRICS & PENALTIES
# ==============================================================================
def apply_cooldown(signals, cooldown):
    """
    Applies a cooldown period to a boolean array of signals.
    If a signal is True, the next `cooldown` signals are forced to False.
    This diminishes density and helps augment accuracy by filtering clustered signals.
    """
    if cooldown <= 0:
        return signals
    res = signals.copy()
    indices = np.where(res)[0]
    if len(indices) == 0:
        return res

    keep = [indices[0]]
    for idx in indices[1:]:
        if idx - keep[-1] > cooldown:
            keep.append(idx)

    res[:] = False
    res[keep] = True
    return res


def wilson_lower_bound(wins, n, z=1.96):
    """
    Calculates the lower bound of the Wilson score interval (95% confidence).
    This naturally balances Win Rate and Sample Size, penalizing low sample sizes
    much more effectively than a flat density penalty.

    NOTE: For the Optuna objective function, we now use Bayesian (Add-2) smoothing
    instead of Wilson. Wilson's extreme penalty for small sample sizes forces the
    optimizer to choose high-density/low-win-rate setups. Smoothed WR allows it
    to find highly selective (lower density) but higher win-rate signals.
    """
    if n == 0: return 0.0
    p_hat = wins / n
    denominator = 1 + z ** 2 / n
    centre_adjusted = p_hat + z ** 2 / (2 * n)
    adjusted_std = np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
    return max(0.0, (centre_adjusted - z * adjusted_std) / denominator)


def calculate_overlap_penalty(signal_indices, lookahead_bars):
    """
    Calculates the ratio of overlapping signals (clustering during anomalies).
    An overlap occurs when the distance between consecutive signals is < lookahead_bars.
    """
    if len(signal_indices) <= 1:
        return 0.0
    diffs = np.diff(signal_indices)
    overlaps = np.sum(diffs < lookahead_bars)
    # Ratio of overlapping instances to total signals
    overlap_ratio = overlaps / len(signal_indices)
    return overlap_ratio


def calculate_win_rates_vectorized(df, _args, close_col, high_col, low_col):
    """
    Vectorized version: evaluates all signals simultaneously using NumPy broadcasting.
    ~10-100x faster than loop-based version for typical datasets.
    """
    if _args.lookahead_bars <= 0:
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    buy_sig_col = ('Signal_Buy', _args.ticker)
    sell_sig_col = ('Signal_Sell', _args.ticker)

    # Extract numpy arrays for speed
    close_prices = df[close_col].to_numpy()
    high_prices = df[high_col].to_numpy()
    low_prices = df[low_col].to_numpy()

    # Get signal positions as integer indices
    buy_positions = np.where(df[buy_sig_col].to_numpy())[0]
    sell_positions = np.where(df[sell_sig_col].to_numpy())[0]

    lookahead = _args.lookahead_bars
    method = _args.method
    n_rows = len(df)

    # ============ BUY SIGNALS ============
    if len(buy_positions) > 0:
        # Filter positions with sufficient lookahead data
        valid_buy_mask = buy_positions + 1 + lookahead <= n_rows
        valid_buy_pos = buy_positions[valid_buy_mask]

        if len(valid_buy_pos) > 0:
            entry_prices = close_prices[valid_buy_pos]
            strikes = entry_prices * _args.put_strike_pct

            # Create 2D index array: [n_signals, lookahead]
            future_idx = valid_buy_pos[:, None] + np.arange(1, lookahead + 1)

            if method == "final_close":
                future_closes = close_prices[future_idx]
                buy_success = future_closes[:, -1] > strikes  # Shape: (n_signals,)
            else:  # "touched" method
                future_lows = low_prices[future_idx]
                # BUY (Put Spread) wins if price NEVER drops below strike.
                # So the LOW of every future bar must be strictly greater than the strike.
                buy_success = np.all(future_lows > strikes[:, None], axis=1)

            buy_wins = np.count_nonzero(buy_success)
            total_buy = len(valid_buy_pos)
        else:
            buy_wins = total_buy = 0
    else:
        buy_wins = total_buy = 0

    # ============ SELL SIGNALS ============
    if len(sell_positions) > 0:
        valid_sell_mask = sell_positions + 1 + lookahead <= n_rows
        valid_sell_pos = sell_positions[valid_sell_mask]

        if len(valid_sell_pos) > 0:
            entry_prices = close_prices[valid_sell_pos]
            strikes = entry_prices * _args.call_strike_pct

            future_idx = valid_sell_pos[:, None] + np.arange(1, lookahead + 1)

            if method == "final_close":
                future_closes = close_prices[future_idx]
                sell_success = future_closes[:, -1] < strikes
            else:  # "touched" method
                future_highs = high_prices[future_idx]
                # SELL (Call Spread) wins if price NEVER spikes above strike.
                # So the HIGH of every future bar must be strictly less than the strike.
                sell_success = np.all(future_highs < strikes[:, None], axis=1)

            sell_wins = np.count_nonzero(sell_success)
            total_sell = len(valid_sell_pos)
        else:
            sell_wins = total_sell = 0
    else:
        sell_wins = total_sell = 0

    # Calculate win rates
    buy_wr = buy_wins / total_buy if total_buy > 0 else 0.0
    sell_wr = sell_wins / total_sell if total_sell > 0 else 0.0
    total_sig = total_buy + total_sell
    combined_wr = (buy_wins + sell_wins) / total_sig if total_sig > 0 else 0.0

    return buy_wr, sell_wr, combined_wr, buy_wins, sell_wins, total_buy, total_sell


def calculate_win_rates(df, _args, close_col, high_col, low_col):
    if _args.lookahead_bars <= 0:
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    buy_sig_col = ('Signal_Buy', _args.ticker)
    sell_sig_col = ('Signal_Sell', _args.ticker)

    buy_indices = df.index[df[buy_sig_col]].tolist()
    sell_indices = df.index[df[sell_sig_col]].tolist()

    buy_wins = sell_wins = total_buy = total_sell = 0
    lookahead = _args.lookahead_bars
    method = _args.method
    n_rows = len(df)
    idx_to_pos = {idx: i for i, idx in enumerate(df.index)}

    for idx in buy_indices:
        pos = idx_to_pos[idx]
        if pos + 1 + lookahead > n_rows: continue
        total_buy += 1
        price = df[close_col].iloc[pos]
        strike = price * _args.put_strike_pct
        future_df = df.iloc[pos + 1: pos + 1 + lookahead]
        # BUY (Put Spread) wins if price NEVER drops below strike.
        success = future_df[close_col].iloc[-1] > strike if method == "final_close" else (future_df[low_col].min() > strike)
        if success: buy_wins += 1

    for idx in sell_indices:
        pos = idx_to_pos[idx]
        if pos + 1 + lookahead > n_rows: continue
        total_sell += 1
        price = df[close_col].iloc[pos]
        strike = price * _args.call_strike_pct
        future_df = df.iloc[pos + 1: pos + 1 + lookahead]
        # SELL (Call Spread) wins if price NEVER spikes above strike.
        success = future_df[close_col].iloc[-1] < strike if method == "final_close" else (future_df[high_col].max() < strike)
        if success: sell_wins += 1

    buy_wr = buy_wins / total_buy if total_buy > 0 else 0.0
    sell_wr = sell_wins / total_sell if total_sell > 0 else 0.0
    total_sig = total_buy + total_sell
    combined_wr = (buy_wins + sell_wins) / total_sig if total_sig > 0 else 0.0

    return buy_wr, sell_wr, combined_wr, buy_wins, sell_wins, total_buy, total_sell


def calculate_yearly_win_rates_vectorized(df, args, close_col, high_col, low_col):
    """Vectorized yearly win rate calculation."""
    if args.lookahead_bars <= 0:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    buy_sig_col = ('Signal_Buy', args.ticker)
    sell_sig_col = ('Signal_Sell', args.ticker)

    df = df.copy()
    df['_Year'] = df.index.year

    close_prices = df[close_col].to_numpy()
    high_prices = df[high_col].to_numpy()
    low_prices = df[low_col].to_numpy()
    years = df['_Year'].to_numpy()
    n_rows = len(df)
    lookahead = args.lookahead_bars
    method = args.method

    results = []

    for year in sorted(df['_Year'].dropna().unique()):
        year_mask = (years == year)
        year_indices = np.where(year_mask)[0]

        # Get signal positions within this year
        buy_positions = year_indices[df[buy_sig_col].to_numpy()[year_mask]]
        sell_positions = year_indices[df[sell_sig_col].to_numpy()[year_mask]]

        # Vectorized BUY evaluation
        if len(buy_positions) > 0:
            valid_mask = buy_positions + 1 + lookahead <= n_rows
            valid_pos = buy_positions[valid_mask]

            if len(valid_pos) > 0:
                entry_prices = close_prices[valid_pos]
                strikes = entry_prices * args.put_strike_pct
                future_idx = valid_pos[:, None] + np.arange(1, lookahead + 1)

                if method == "final_close":
                    future_closes = close_prices[future_idx]
                    success = future_closes[:, -1] > strikes
                else:
                    future_lows = low_prices[future_idx]
                    # BUY (Put Spread)
                    success = np.all(future_lows > strikes[:, None], axis=1)

                buy_wins = np.count_nonzero(success)
                total_buy = len(valid_pos)
            else:
                buy_wins = total_buy = 0
        else:
            buy_wins = total_buy = 0

        # Vectorized SELL evaluation
        if len(sell_positions) > 0:
            valid_mask = sell_positions + 1 + lookahead <= n_rows
            valid_pos = sell_positions[valid_mask]

            if len(valid_pos) > 0:
                entry_prices = close_prices[valid_pos]
                strikes = entry_prices * args.call_strike_pct
                future_idx = valid_pos[:, None] + np.arange(1, lookahead + 1)

                if method == "final_close":
                    future_closes = close_prices[future_idx]
                    success = future_closes[:, -1] < strikes
                else:
                    future_highs = high_prices[future_idx]
                    # SELL (Call Spread)
                    success = np.all(future_highs < strikes[:, None], axis=1)

                sell_wins = np.count_nonzero(success)
                total_sell = len(valid_pos)
            else:
                sell_wins = total_sell = 0
        else:
            sell_wins = total_sell = 0

        # Calculate rates
        buy_wr = buy_wins / total_buy if total_buy > 0 else 0.0
        sell_wr = sell_wins / total_sell if total_sell > 0 else 0.0
        total_sig = total_buy + total_sell
        combined_wr = (buy_wins + sell_wins) / total_sig if total_sig > 0 else 0.0

        results.append({
            'Year': year, 'Buy Signals': total_buy, 'Buy Wins': buy_wins, 'Buy WR %': buy_wr * 100,
            'Sell Signals': total_sell, 'Sell Wins': sell_wins, 'Sell WR %': sell_wr * 100,
            'Total Signals': total_sig, 'Combined WR %': combined_wr * 100
        })

    df.drop(columns=['_Year'], inplace=True, errors='ignore')
    return pd.DataFrame(results) if results else None


def calculate_yearly_win_rates(df, args, close_col, high_col, low_col):
    if args.lookahead_bars <= 0: return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    buy_sig_col = ('Signal_Buy', args.ticker)
    sell_sig_col = ('Signal_Sell', args.ticker)
    df = df.copy()
    df['_Year'] = df.index.year

    results = []
    lookahead = args.lookahead_bars
    method = args.method
    idx_to_pos = {idx: i for i, idx in enumerate(df.index)}
    n_rows = len(df)

    for year in sorted(df['_Year'].dropna().unique()):
        year_mask = df['_Year'] == year
        year_indices = df.index[year_mask].tolist()
        buy_indices = [idx for idx in year_indices if df.loc[idx, buy_sig_col]]
        sell_indices = [idx for idx in year_indices if df.loc[idx, sell_sig_col]]
        buy_wins = sell_wins = total_buy = total_sell = 0

        for idx in buy_indices:
            pos = idx_to_pos[idx]
            if pos + 1 + lookahead > n_rows: continue
            total_buy += 1
            price = df[close_col].iloc[pos]
            strike = price * args.put_strike_pct
            future_df = df.iloc[pos + 1: pos + 1 + lookahead]
            # BUY (Put Spread)
            success = future_df[close_col].iloc[-1] > strike if method == "final_close" else (future_df[low_col].min() > strike)
            if success: buy_wins += 1

        for idx in sell_indices:
            pos = idx_to_pos[idx]
            if pos + 1 + lookahead > n_rows: continue
            total_sell += 1
            price = df[close_col].iloc[pos]
            strike = price * args.call_strike_pct
            future_df = df.iloc[pos + 1: pos + 1 + lookahead]
            # SELL (Call Spread)
            success = future_df[close_col].iloc[-1] < strike if method == "final_close" else (future_df[high_col].max() < strike)
            if success: sell_wins += 1

        buy_wr = buy_wins / total_buy if total_buy > 0 else 0.0
        sell_wr = sell_wins / total_sell if total_sell > 0 else 0.0
        total_sig = total_buy + total_sell
        combined_wr = (buy_wins + sell_wins) / total_sig if total_sig > 0 else 0.0

        results.append({
            'Year': year, 'Buy Signals': total_buy, 'Buy Wins': buy_wins, 'Buy WR %': buy_wr * 100,
            'Sell Signals': total_sell, 'Sell Wins': sell_wins, 'Sell WR %': sell_wr * 100,
            'Total Signals': total_sig, 'Combined WR %': combined_wr * 100
        })

    df.drop(columns=['_Year'], inplace=True, errors='ignore')
    return pd.DataFrame(results) if results else None


def print_yearly_stats(yearly_df, ticker, overall_wr=None):
    if yearly_df is None or yearly_df.empty:
        print("⚠️  No yearly statistics available.")
        return
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    display_df = yearly_df.copy()
    display_df['Buy WR %'] = display_df['Buy WR %'].map(lambda x: f"{x:6.2f}%")
    display_df['Sell WR %'] = display_df['Sell WR %'].map(lambda x: f"{x:6.2f}%")
    display_df['Combined WR %'] = display_df['Combined WR %'].map(lambda x: f"{x:6.2f}%")

    print(f"\n{'=' * 80}")
    print(f"📅 YEARLY WIN RATE STATISTICS — {ticker}")
    print(f"{'=' * 80}")

    if use_tabulate:
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False, numalign='right', stralign='center'))
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(display_df.to_string(index=False))

    if not yearly_df.empty:
        total_buy = yearly_df['Buy Signals'].sum()
        total_sell = yearly_df['Sell Signals'].sum()
        total_buy_wins = yearly_df['Buy Wins'].sum()
        total_sell_wins = yearly_df['Sell Wins'].sum()
        overall_buy_wr = (total_buy_wins / total_buy * 100) if total_buy > 0 else 0
        overall_sell_wr = (total_sell_wins / total_sell * 100) if total_sell > 0 else 0
        overall_combined = ((total_buy_wins + total_sell_wins) / (total_buy + total_sell) * 100) if (total_buy + total_sell) > 0 else 0

        print(f"\n{'─' * 80}")
        print(f"📊 OVERALL TOTALS")
        print(f"{'─' * 80}")
        print(f"   🟢 Buy:  {total_buy_wins:,} wins / {total_buy:,} signals → {overall_buy_wr:6.2f}%")
        print(f"   🔴 Sell: {total_sell_wins:,} wins / {total_sell:,} signals → {overall_sell_wr:6.2f}%")
        print(f"   🎯 Combined: {total_buy_wins + total_sell_wins:,} wins / {total_buy + total_sell:,} signals → {overall_combined:.2%}")
        print(f"{'=' * 80}\n")


# ==============================================================================
# 🆕 SIGNAL OUTCOMES FOR PLOTTING
# ==============================================================================
def get_signal_outcomes(df, args, close_col, high_col, low_col, buy_sig_col, sell_sig_col):
    """
    Vectorized evaluation of signal outcomes specifically for plotting W/L markers.
    Returns boolean arrays for wins and losses aligned with the dataframe index.
    """
    buy_positions = np.where(df[buy_sig_col].to_numpy())[0]
    sell_positions = np.where(df[sell_sig_col].to_numpy())[0]

    close_prices = df[close_col].to_numpy()
    high_prices = df[high_col].to_numpy()
    low_prices = df[low_col].to_numpy()
    n_rows = len(df)
    lookahead = args.lookahead_bars
    method = args.method

    buy_wins_arr = np.zeros(n_rows, dtype=bool)
    buy_losses_arr = np.zeros(n_rows, dtype=bool)

    if lookahead > 0 and len(buy_positions) > 0:
        valid_buy_mask = buy_positions + 1 + lookahead <= n_rows
        valid_buy_pos = buy_positions[valid_buy_mask]
        if len(valid_buy_pos) > 0:
            entry_prices = close_prices[valid_buy_pos]
            strikes = entry_prices * args.put_strike_pct
            future_idx = valid_buy_pos[:, None] + np.arange(1, lookahead + 1)
            if method == "final_close":
                future_closes = close_prices[future_idx]
                success = future_closes[:, -1] > strikes
            else:
                future_lows = low_prices[future_idx]
                success = np.all(future_lows > strikes[:, None], axis=1)
            buy_wins_arr[valid_buy_pos[success]] = True
            buy_losses_arr[valid_buy_pos[~success]] = True

    sell_wins_arr = np.zeros(n_rows, dtype=bool)
    sell_losses_arr = np.zeros(n_rows, dtype=bool)

    if lookahead > 0 and len(sell_positions) > 0:
        valid_sell_mask = sell_positions + 1 + lookahead <= n_rows
        valid_sell_pos = sell_positions[valid_sell_mask]
        if len(valid_sell_pos) > 0:
            entry_prices = close_prices[valid_sell_pos]
            strikes = entry_prices * args.call_strike_pct
            future_idx = valid_sell_pos[:, None] + np.arange(1, lookahead + 1)
            if method == "final_close":
                future_closes = close_prices[future_idx]
                success = future_closes[:, -1] < strikes
            else:
                future_highs = high_prices[future_idx]
                success = np.all(future_highs < strikes[:, None], axis=1)
            sell_wins_arr[valid_sell_pos[success]] = True
            sell_losses_arr[valid_sell_pos[~success]] = True

    return buy_wins_arr, buy_losses_arr, sell_wins_arr, sell_losses_arr


def plot_results(df, args, params, close_col, high_col, low_col, rsi_col, sma_50_col,
                 fib_50_col, fib_618_col, buy_sig_col, sell_sig_col,
                 reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col,
                 rsi_midline=50, rsi_oversold=30, rsi_overbought=70):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
    except ImportError:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        return

    plot_bars = min(800, len(df))
    df_plot = df.iloc[-plot_bars:].copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    ax1.plot(df_plot.index, df_plot[close_col], label='Close', color='blue', linewidth=1)

    opt_target = args.optimize_target

    # Conditionally plot SMA 50 based on what the model used
    use_sma = False
    if opt_target in ['buy_wr', 'combined_wr']:
        if params.get('use_pullback_buy', 0) or params.get('use_ma_conf_buy', 0): use_sma = True
    if opt_target in ['sell_wr', 'combined_wr']:
        if params.get('use_pullback_sell', 0): use_sma = True

    if use_sma and sma_50_col in df_plot.columns:
        ax1.plot(df_plot.index, df_plot[sma_50_col], label='SMA 50', color='orange', linewidth=1)

    # Conditionally plot Fibonacci based on what the model used
    use_fib = False
    if opt_target in ['buy_wr', 'combined_wr']:
        if params.get('use_fib_rsi_buy', 0): use_fib = True

    if use_fib and fib_50_col in df_plot.columns and fib_618_col in df_plot.columns:
        ax1.fill_between(df_plot.index, df_plot[fib_618_col], df_plot[fib_50_col],
                         color='gold', alpha=0.25, label='Fib Golden Zone (0.5-0.618)')

    # Only show buy signals if target is buy or combined, and only sell if target is sell or combined
    plot_buy = opt_target in ['buy_wr', 'combined_wr']
    plot_sell = opt_target in ['sell_wr', 'combined_wr']

    # Get outcomes for W/L markers
    buy_wins_full, buy_losses_full, sell_wins_full, sell_losses_full = get_signal_outcomes(
        df, args, close_col, high_col, low_col, buy_sig_col, sell_sig_col
    )

    buy_wins_plot = buy_wins_full[-plot_bars:]
    buy_losses_plot = buy_losses_full[-plot_bars:]
    sell_wins_plot = sell_wins_full[-plot_bars:]
    sell_losses_plot = sell_losses_full[-plot_bars:]

    y_range = df_plot[close_col].max() - df_plot[close_col].min()
    if y_range == 0: y_range = 1.0
    y_offset = y_range * 0.02

    if plot_buy:
        buy_mask = df_plot[buy_sig_col].to_numpy() & df_plot[close_col].notna().to_numpy()
        if buy_mask.any():
            ax1.scatter(df_plot.index[buy_mask], df_plot[close_col][buy_mask],
                        marker='^', color='green', s=120, label='Buy Signal', zorder=5, edgecolors='white')
            buy_prices = df_plot[close_col].to_numpy()[buy_mask]
            buy_dates = df_plot.index[buy_mask]
            buy_w = buy_wins_plot[buy_mask]
            buy_l = buy_losses_plot[buy_mask]

            # Write in bold a W just up the triangle if it is a win or a bold L if it is a loss
            for dt, p, w, l in zip(buy_dates, buy_prices, buy_w, buy_l):
                if w:
                    ax1.text(dt, p + y_offset, 'W', color='green', fontweight='bold', fontsize=12, ha='center', va='bottom', zorder=6)
                elif l:
                    ax1.text(dt, p + y_offset, 'L', color='red', fontweight='bold', fontsize=12, ha='center', va='bottom', zorder=6)

    if plot_sell:
        sell_mask = df_plot[sell_sig_col].to_numpy() & df_plot[close_col].notna().to_numpy()
        if sell_mask.any():
            ax1.scatter(df_plot.index[sell_mask], df_plot[close_col][sell_mask],
                        marker='v', color='red', s=120, label='Sell Signal', zorder=5, edgecolors='white')
            sell_prices = df_plot[close_col].to_numpy()[sell_mask]
            sell_dates = df_plot.index[sell_mask]
            sell_w = sell_wins_plot[sell_mask]
            sell_l = sell_losses_plot[sell_mask]

            # Write in bold a W just down the triangle if it is a win or a bold L if it is a loss
            for dt, p, w, l in zip(sell_dates, sell_prices, sell_w, sell_l):
                if w:
                    ax1.text(dt, p - y_offset, 'W', color='green', fontweight='bold', fontsize=12, ha='center', va='top', zorder=6)
                elif l:
                    ax1.text(dt, p - y_offset, 'L', color='red', fontweight='bold', fontsize=12, ha='center', va='top', zorder=6)

    # Conditionally plot divergences based on what the model used
    use_reg_bull = opt_target in ['buy_wr', 'combined_wr'] and params.get('use_reg_bull_div', 0)
    use_reg_bear = opt_target in ['sell_wr', 'combined_wr'] and params.get('use_reg_bear_div', 0)
    use_hid_bull = opt_target in ['buy_wr', 'combined_wr'] and params.get('use_hid_bull_div', 0)
    use_hid_bear = opt_target in ['sell_wr', 'combined_wr'] and params.get('use_hid_bear_div', 0)

    if use_reg_bull and reg_bull_div_col in df_plot.columns:
        bull_div = df_plot[reg_bull_div_col] & df_plot[low_col].notna()
        if bull_div.any(): ax1.scatter(df_plot.index[bull_div], df_plot[low_col][bull_div], marker='*', color='lime', s=200, label='Reg. Bullish Div', zorder=6, edgecolors='black')
    if use_reg_bear and reg_bear_div_col in df_plot.columns:
        bear_div = df_plot[reg_bear_div_col] & df_plot[high_col].notna()
        if bear_div.any(): ax1.scatter(df_plot.index[bear_div], df_plot[high_col][bear_div], marker='*', color='magenta', s=200, label='Reg. Bearish Div', zorder=6, edgecolors='black')
    if use_hid_bull and hid_bull_div_col in df_plot.columns:
        hbull_div = df_plot[hid_bull_div_col] & df_plot[low_col].notna()
        if hbull_div.any(): ax1.scatter(df_plot.index[hbull_div], df_plot[low_col][hbull_div], marker='P', color='cyan', s=100, label='Hidden Bullish Div', zorder=6, edgecolors='black')
    if use_hid_bear and hid_bear_div_col in df_plot.columns:
        hbear_div = df_plot[hid_bear_div_col] & df_plot[high_col].notna()
        if hbear_div.any(): ax1.scatter(df_plot.index[hbear_div], df_plot[high_col][hbear_div], marker='P', color='yellow', s=100, label='Hidden Bearish Div', zorder=6, edgecolors='black')

    ax1.set_ylabel('Price')
    ax1.set_title(f'{args.ticker} - Strategy Signals & Fibonacci Confluence (Last {plot_bars} bars)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

    if rsi_col in df_plot.columns:
        ax2.plot(df_plot.index, df_plot[rsi_col], label='RSI', color='purple', linewidth=1)

        use_rsi_ema = False
        if opt_target in ['buy_wr', 'combined_wr'] and (params.get('use_ema_cross_buy', 0) or params.get('use_ma_conf_buy', 0)):
            use_rsi_ema = True
        if opt_target in ['sell_wr', 'combined_wr'] and params.get('use_ema_cross_sell', 0):
            use_rsi_ema = True

        rsi_ema_10_col = ('RSI_EMA_10', args.ticker)
        if use_rsi_ema and rsi_ema_10_col in df_plot.columns:
            ax2.plot(df_plot.index, df_plot[rsi_ema_10_col], label='RSI EMA(10)', color='cyan', linewidth=0.8)

        ax2.axhline(y=rsi_overbought, color='red', linestyle='--', alpha=0.5, label=f'Overbought ({rsi_overbought})')
        ax2.axhline(y=rsi_oversold, color='green', linestyle='--', alpha=0.5, label=f'Oversold ({rsi_oversold})')
        ax2.axhline(y=rsi_midline, color='gray', linestyle=':', alpha=0.3, label=f'Midline ({rsi_midline})')

        if use_rsi_ema and rsi_ema_10_col in df_plot.columns:
            crossover_buy = (df_plot[rsi_col].shift(1) < df_plot[rsi_ema_10_col].shift(1)) & (df_plot[rsi_col] > df_plot[rsi_ema_10_col]) & (df_plot[rsi_col] < rsi_oversold)
            crossover_sell = (df_plot[rsi_col].shift(1) > df_plot[rsi_ema_10_col].shift(1)) & (df_plot[rsi_col] < df_plot[rsi_ema_10_col]) & (df_plot[rsi_col] > rsi_overbought)
            if plot_buy and crossover_buy.any():
                ax2.scatter(df_plot.index[crossover_buy], df_plot[rsi_col][crossover_buy], marker='^', color='green', s=100, zorder=5, edgecolors='white')
            if plot_sell and crossover_sell.any():
                ax2.scatter(df_plot.index[crossover_sell], df_plot[rsi_col][crossover_sell], marker='v', color='red', s=100, zorder=5, edgecolors='white')

    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)

    if len(df_plot) > 0:
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_path = os.path.join(args.output_dir, f"{args.ticker}_{args.dataset_id}_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        if args.verbose: print(f"📊 Plot saved to: {plot_path}")
    if args.verbose: print("📈 Showing plot window... (close to continue)")
    plt.show()


# ==============================================================================
# 🆕 MODEL NAMING & SAVING UTILITIES
# ==============================================================================
def generate_model_name(args, params, score):
    """
    Generate a descriptive model filename that includes args and final score.
    Format: {ticker}_{dataset_id}_{target}_score-{score:.4f}_{param1}-{val1}_{param2}-{val2}..._{timestamp}.pkl
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build param string (limit to key params to keep filename reasonable)
    param_parts = []
    # for key in ['rsi_length', 'rsi_signal_len', 'sma_len', 'fib_lookback', 'div_window']:
    #     if key in params:
    #         param_parts.append(f"{key}-{params[key]}")

    # Add key args that affect model behavior
    arg_parts = []
    if hasattr(args, 'cooldown_bars'):
        arg_parts.append(f"cd{args.cooldown_bars}")
    if hasattr(args, 'lookahead_bars'):
        arg_parts.append(f"la{args.lookahead_bars}")
    if hasattr(args, 'method'):
        arg_parts.append(f"mt{args.method}")
    if hasattr(args, 'put_strike_pct'):
        arg_parts.append(f"put{args.put_strike_pct:.6f}")
    if hasattr(args, 'call_strike_pct'):
        arg_parts.append(f"call{args.call_strike_pct:.6f}")
    if hasattr(args, 'train_ratio') and args.train_ratio < 1.0:
        arg_parts.append(f"train_ratio-{args.train_ratio:.2f}")
    score = 0. if score is None else score
    # Combine all parts
    name_parts = [
        args.ticker.replace('^', ''),
        args.dataset_id,
        args.optimize_target,
        f"twr{score:.12f}",
        '__'.join(param_parts),
        '__'.join(arg_parts),
        timestamp
    ]

    # Filter out empty parts and join
    name_parts = [p for p in name_parts if p]
    filename = '__'.join(name_parts) + '.pkl'

    # Sanitize filename (remove any problematic characters)
    filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')

    return filename


def save_model(params, score, command_line, args, validation_score, train_val_split):
    """
    Save model parameters and metadata to a file with descriptive name.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = generate_model_name(args, params, validation_score)
    model_path = os.path.join(args.output_dir, model_name)

    model_data = {
        'params': params,
        'score': score,
        'validation_score': validation_score,  # ← Added validation score
        'train_val_split': train_val_split,  # ← Added split metadata
        'args': {
            'ticker': args.ticker,
            'dataset_id': args.dataset_id,
            'optimize_target': args.optimize_target,
            'lookahead_bars': args.lookahead_bars,
            'method': args.method,
            'put_strike_pct': args.put_strike_pct,
            'call_strike_pct': args.call_strike_pct,
            'min_signal_density': args.min_signal_density,
            'train_ratio': getattr(args, 'train_ratio', 1.0),
        },
        'timestamp': datetime.now().isoformat(),
        'command_line': command_line,
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if args.verbose:
        print(f"💾 Model saved to: {model_path}")
        if validation_score is not None:
            print(f"   📊 Test Win Rate      : {validation_score:.8f}")  # Validation is the Test :)
            print(f"   📊 Training Win Rate  : {score:.8f}")

    return model_path


def load_model(model_path):
    """
    Load a saved model and return its parameters and metadata.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


# ==============================================================================
# 🆕 REAL-TIME MODE FUNCTIONS
# ==============================================================================
def run_strategy_on_latest(df_base, params, _args, close_col, high_col, low_col, volume_col, open_col):
    """
    Run the strategy on the latest datapoint to check for signals.
    Returns dict with signal info.
    """
    df = df_base.copy()

    # Retrieve min_buy_confluence and min_sell_confluence parameters (default to 1 for backward compatibility)
    min_buy_confluence = params.get('min_buy_confluence', 1)
    min_sell_confluence = params.get('min_sell_confluence', 1)

    # Apply strategy functions with loaded params
    df, rsi_col, pullback_buy_col, pullback_sell_col, ema_cross_buy_col, ema_cross_sell_col, ma_conf_buy_col = \
        implement_rsi_strategies(df, close_col, _args.ticker,
                                 params['rsi_length'], params['rsi_signal_len'], params['sma_len'],
                                 params.get('rsi_midline', 50), params.get('rsi_oversold', 30), params.get('rsi_overbought', 70))
    df, reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col = \
        find_divergences(df, high_col, low_col, rsi_col, _args.ticker, params['div_window'])
    df, fib_rsi_buy_col = calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, _args.ticker, params['fib_lookback'], params.get('rsi_midline', 50))

    # NEW STRATEGIES
    macd_buy_col, macd_sell_col, bb_buy_col, bb_sell_col, vol_buy_col, vol_sell_col, stoch_buy_col, stoch_sell_col, vwap_buy_col, vwap_sell_col = \
        implement_additional_strategies(df, close_col, high_col, low_col, volume_col, open_col, _args.ticker,
                                        macd_fast=params.get('macd_fast', 12),
                                        macd_slow=params.get('macd_slow', 26),
                                        macd_signal=params.get('macd_signal', 9),
                                        bb_length=params.get('bb_length', 20),
                                        bb_std=params.get('bb_std', 2.0),
                                        vol_sma_length=params.get('vol_sma_length', 20),
                                        vol_multiplier=params.get('vol_multiplier', 2.0),
                                        stoch_k_period=params.get('stoch_k_period', 14),
                                        stoch_d_period=params.get('stoch_d_period', 3),
                                        stoch_smooth_k_period=params.get('stoch_smooth_k_period', 1),
                                        stoch_oversold=params.get('stoch_oversold', 20),
                                        stoch_overbought=params.get('stoch_overbought', 80),
                                        vwap_window=params.get('vwap_window', 50))

    # 0/1 triggers for strategy inclusion (default to 1 for backward compatibility)
    buy_cols = []
    if params.get('use_pullback_buy', 1): buy_cols.append(pullback_buy_col)
    if params.get('use_ema_cross_buy', 1): buy_cols.append(ema_cross_buy_col)
    if params.get('use_ma_conf_buy', 1): buy_cols.append(ma_conf_buy_col)
    if params.get('use_fib_rsi_buy', 1): buy_cols.append(fib_rsi_buy_col)
    if params.get('use_reg_bull_div', 1): buy_cols.append(reg_bull_div_col)
    if params.get('use_hid_bull_div', 1): buy_cols.append(hid_bull_div_col)
    if params.get('use_macd_buy', 1): buy_cols.append(macd_buy_col)
    if params.get('use_bb_buy', 1): buy_cols.append(bb_buy_col)
    if params.get('use_vol_buy', 1): buy_cols.append(vol_buy_col)
    if params.get('use_stoch_buy', 1): buy_cols.append(stoch_buy_col)
    if params.get('use_vwap_buy', 1): buy_cols.append(vwap_buy_col)

    sell_cols = []
    if params.get('use_pullback_sell', 1): sell_cols.append(pullback_sell_col)
    if params.get('use_ema_cross_sell', 1): sell_cols.append(ema_cross_sell_col)
    if params.get('use_reg_bear_div', 1): sell_cols.append(reg_bear_div_col)
    if params.get('use_hid_bear_div', 1): sell_cols.append(hid_bear_div_col)
    if params.get('use_macd_sell', 1): sell_cols.append(macd_sell_col)
    if params.get('use_bb_sell', 1): sell_cols.append(bb_sell_col)
    if params.get('use_vol_sell', 1): sell_cols.append(vol_sell_col)
    if params.get('use_stoch_sell', 1): sell_cols.append(stoch_sell_col)
    if params.get('use_vwap_sell', 1): sell_cols.append(vwap_sell_col)

    # Aggregate signals based on minimum confluence required to narrow down quantity
    if len(buy_cols) > 0:
        df[('Signal_Buy', _args.ticker)] = df[buy_cols].fillna(False).sum(axis=1) >= min_buy_confluence
    else:
        df[('Signal_Buy', _args.ticker)] = False

    if len(sell_cols) > 0:
        df[('Signal_Sell', _args.ticker)] = df[sell_cols].fillna(False).sum(axis=1) >= min_sell_confluence
    else:
        df[('Signal_Sell', _args.ticker)] = False

    cooldown_bars = getattr(_args, 'cooldown_bars', 0)  # Backward comptability
    if cooldown_bars > 0:
        df[('Signal_Buy', _args.ticker)] = apply_cooldown(df[('Signal_Buy', _args.ticker)].to_numpy(), cooldown_bars)
        df[('Signal_Sell', _args.ticker)] = apply_cooldown(df[('Signal_Sell', _args.ticker)].to_numpy(), cooldown_bars)

    # Get latest signal
    latest_idx = df.index[-1]
    latest_buy = df.loc[latest_idx, ('Signal_Buy', _args.ticker)]
    latest_sell = df.loc[latest_idx, ('Signal_Sell', _args.ticker)]
    latest_close = df.loc[latest_idx, close_col]
    latest_rsi = df.loc[latest_idx, rsi_col] if rsi_col in df.columns else None

    return {
        'timestamp': latest_idx,
        'close': latest_close,
        'rsi': latest_rsi,
        'buy_signal': bool(latest_buy),
        'sell_signal': bool(latest_sell),
        'individual_signals': {
            'pullback_buy': bool(df.loc[latest_idx, pullback_buy_col]) if pullback_buy_col in df.columns else False,
            'ema_cross_buy': bool(df.loc[latest_idx, ema_cross_buy_col]) if ema_cross_buy_col in df.columns else False,
            'ma_conf_buy': bool(df.loc[latest_idx, ma_conf_buy_col]) if ma_conf_buy_col in df.columns else False,
            'fib_rsi_buy': bool(df.loc[latest_idx, fib_rsi_buy_col]) if fib_rsi_buy_col in df.columns else False,
            'bullish_div': bool(df.loc[latest_idx, reg_bull_div_col]) if reg_bull_div_col in df.columns else False,
            'hidden_bull_div': bool(df.loc[latest_idx, hid_bull_div_col]) if hid_bull_div_col in df.columns else False,
            'macd_buy': bool(df.loc[latest_idx, macd_buy_col]) if macd_buy_col in df.columns else False,
            'bb_buy': bool(df.loc[latest_idx, bb_buy_col]) if bb_buy_col in df.columns else False,
            'vol_buy': bool(df.loc[latest_idx, vol_buy_col]) if vol_buy_col in df.columns else False,
            'stoch_buy': bool(df.loc[latest_idx, stoch_buy_col]) if stoch_buy_col in df.columns else False,
            'vwap_buy': bool(df.loc[latest_idx, vwap_buy_col]) if vwap_buy_col in df.columns else False,
            'pullback_sell': bool(df.loc[latest_idx, pullback_sell_col]) if pullback_sell_col in df.columns else False,
            'ema_cross_sell': bool(df.loc[latest_idx, ema_cross_sell_col]) if ema_cross_sell_col in df.columns else False,
            'bearish_div': bool(df.loc[latest_idx, reg_bear_div_col]) if reg_bear_div_col in df.columns else False,
            'hidden_bear_div': bool(df.loc[latest_idx, hid_bear_div_col]) if hid_bear_div_col in df.columns else False,
            'macd_sell': bool(df.loc[latest_idx, macd_sell_col]) if macd_sell_col in df.columns else False,
            'bb_sell': bool(df.loc[latest_idx, bb_sell_col]) if bb_sell_col in df.columns else False,
            'vol_sell': bool(df.loc[latest_idx, vol_sell_col]) if vol_sell_col in df.columns else False,
            'stoch_sell': bool(df.loc[latest_idx, stoch_sell_col]) if stoch_sell_col in df.columns else False,
            'vwap_sell': bool(df.loc[latest_idx, vwap_sell_col]) if vwap_sell_col in df.columns else False,
        }
    }


def real_time_mode(model_path, verbose, clip_n, reduce_n, close_col, high_col, low_col, volume_col, open_col):
    """
    Real-time mode: load model from path and test latest datapoint for signals.
    """
    if not model_path:
        print("❌ Error: --model-path is required for real-time mode")
        return None

    # Load the model
    if verbose: print(f"🔍 Loading model from: {model_path}")
    model_data = load_model(model_path)
    put_strike_pct = model_data['args']['put_strike_pct']
    call_strike_pct = model_data['args']['call_strike_pct']
    lookahead = model_data['args']['lookahead_bars']
    method = model_data['args']['method']
    optimize_target = model_data['args']['optimize_target']
    min_signal_density = model_data['args']['min_signal_density']
    params = model_data['params']
    assert 'score' in model_data
    train_score = model_data.get('score', 'N/A')
    val_score = model_data.get('validation_score')
    _cooldown_bars = params["cooldown_bars"] if "cooldown_bars" in params else 0
    _dataset_id = model_data['args']['dataset_id']
    _ticker = model_data['args']['ticker']
    df_base = factory_load_data(_dataset_id=_dataset_id, _ticker=_ticker, _args={"clip_n": clip_n, "reduce_n": reduce_n})
    if verbose: print(f"📂 Dataset ranging from {df_base.index[0].strftime('%Y-%m-%d')} to {df_base.index[-1].strftime('%Y-%m-%d')}")
    command_line = model_data["command_line"] if "command_line" in model_data else ""
    train_ratio = model_data['train_val_split']['train_ratio']
    train_bars = model_data['train_val_split']['train_bars']
    val_bars = model_data['train_val_split']['val_bars']
    train_range = model_data['train_val_split']['train_range']
    val_range = model_data['train_val_split']['val_range']
    if verbose:
        print(f"📊 Loaded model with training score: {train_score:.4%}")
        print(f"📊 Test score: {val_score:.4%}")
        print(f"🧠 Parameters: {params}")
        print(f"🧠 Ratio: {train_ratio} | {train_bars} Train Bars ({train_range}) | {val_bars} Val Bars ({val_range}) | Method: {method} | Optimize Target: {optimize_target} | Minimum Signal Density: {min_signal_density:.2%} | Cooldown: {_cooldown_bars} bars")
    # Run strategy on latest datapoint
    if verbose: print(f"\n⚡ Testing latest datapoint ({df_base.index[-1].strftime('%Y-%m-%d')}) for {_ticker} | Dataset {_dataset_id} | Lookahead: {lookahead} bars")
    result = run_strategy_on_latest(df_base=df_base, params=params, _args=Namespace(ticker=_ticker), close_col=close_col, high_col=high_col, low_col=low_col, volume_col=volume_col, open_col=open_col)

    # ==============================================================================
    # 📊 RECOMPUTE TRAIN & VALIDATION WIN RATES
    # ==============================================================================
    train_win_rate, val_win_rate = 0.0, 0.0
    if train_ratio < 1.0:
        import copy

        # Create an args object that exactly matches the configuration used during training
        eval_args = copy.copy(args)
        for k, v in model_data['args'].items():
            setattr(eval_args, k, v)
        eval_args.sanity_check = getattr(args, 'sanity_check', False)

        # Split df_base using the original train_bars to maintain the exact same training set
        # (If new data was appended to df_base, it will correctly fall into the validation set)
        if train_bars < len(df_base):
            split_idx = train_bars
        else:
            split_idx = int(len(df_base) * train_ratio)

        df_train = df_base.iloc[:split_idx].copy()
        df_val = df_base.iloc[split_idx:].copy()

        # Evaluate on the training set
        buy_wr_train, sell_wr_train, combined_wr_train, _, _, _, _, _, _, _ = \
            run_strategy_and_evaluate(df_train, eval_args, close_col, high_col, low_col, volume_col, open_col, **params)

        # Evaluate on the validation set
        buy_wr_val, sell_wr_val, combined_wr_val, _, _, _, _, _, _, _ = \
            run_strategy_and_evaluate(df_val, eval_args, close_col, high_col, low_col, volume_col, open_col, **params)

        # Extract the specific win rate that was targeted during optimization
        if optimize_target == 'buy_wr':
            train_win_rate = buy_wr_train
            val_win_rate = buy_wr_val
        elif optimize_target == 'sell_wr':
            train_win_rate = sell_wr_train
            val_win_rate = sell_wr_val
        else:
            train_win_rate = combined_wr_train
            val_win_rate = combined_wr_val

        if args.verbose:
            print(f"📈 Recomputed Train Win Rate: {train_win_rate:.2%}")
            print(f"📈 Recomputed Test Win Rate : {val_win_rate:.2%}")

    # Output results
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🔔 REAL-TIME SIGNAL CHECK — {_ticker}")
        print(f"{'=' * 60}")
    assert df_base.index[-1].strftime('%Y-%m-%d') == result['timestamp'].strftime('%Y-%m-%d')
    if verbose: print(f"📅 Last Timestamp: {result['timestamp'].strftime('%Y-%m-%d')}")
    current_price, target_price, target_date = result['close'], None, None
    assert df_base[close_col].iloc[-1] == current_price
    if verbose: print(f"💰 Last Close Price: ${current_price:.2f}")
    buy_signal_detected = result['buy_signal'] and optimize_target in ['combined_wr', 'buy_wr']
    sell_signal_detected = result['sell_signal'] and optimize_target in ['combined_wr', 'sell_wr']
    result['buy_signal_detected'] = buy_signal_detected
    result['sell_signal_detected'] = sell_signal_detected
    if buy_signal_detected:
        if args.verbose:
            print(f"\n🎯 SIGNALS:")
            print(f"   🟢 BUY SIGNAL DETECTED! | Put Threshold: {put_strike_pct:.2%} | @{lookahead} {_dataset_id}")
    if sell_signal_detected:
        if args.verbose:
            print(f"\n🎯 SIGNALS:")
            print(f"   🔴 SELL SIGNAL DETECTED! | Call Threshold: {call_strike_pct:.2%} | @{lookahead} {_dataset_id}")
    if not buy_signal_detected and not sell_signal_detected:
        if args.verbose:
            print(f"   ⚪ No signal at this time")

    if verbose:
        print(f"\n🔍 Individual Signal Components:")
        for name, active in result['individual_signals'].items():
            status = "✅" if active else "❌"
            print(f"   {status} {name}: {active}")

    if verbose: print(f"{'=' * 60}\n")

    # Calculate approximate target/expiration date based on lookahead bars
    entry_date = result['timestamp'].strftime('%Y-%m-%d')
    target_date = get_next_step(the_date=entry_date, dataset_id=_dataset_id, nn=lookahead).strftime('%Y-%m-%d')

    # ==============================================================================
    # 💡 RECOMMENDED TRADE OUTPUT (if signal detected)
    # ==============================================================================
    if buy_signal_detected or sell_signal_detected:
        entry_price = result['close']
        assert entry_price == current_price
        if verbose:
            print(f"\n💡 RECOMMENDED OPTIONS TRADE:")
            print(f"{'─' * 60}")
        if buy_signal_detected:
            strike_price = entry_price * put_strike_pct
            if verbose:
                print(f"   📊 Strategy  : Put Credit Spread")
                print(f"   📅 Entry Date: {entry_date}")
                print(f"   💰 Entry Price: ${entry_price:.2f}")
                print(f"   🎯 Short Put Strike: ${strike_price:.2f} ({put_strike_pct:.2%} of entry)")
                print(f"   📅 Target/Expiration: ~{target_date} ({lookahead} bars)")
                print(f"   ✅ Win Condition: Price stays ABOVE ${strike_price:.2f}")
                print(f"   💡 Premium: Sell OTM put spread below current price")
            target_price = strike_price
        if sell_signal_detected:
            strike_price = entry_price * call_strike_pct
            if verbose:
                print(f"   📊 Strategy  : Call Credit Spread")
                print(f"   📅 Entry Date: {entry_date}")
                print(f"   💰 Entry Price: ${entry_price:.2f}")
                print(f"   🎯 Short Call Strike: ${strike_price:.2f} ({call_strike_pct:.2%} of entry)")
                print(f"   📅 Target/Expiration: ~{target_date} ({lookahead} bars)")
                print(f"   ✅ Win Condition: Price stays BELOW ${strike_price:.2f}")
                print(f"   💡 Premium: Sell OTM call spread above current price")
            target_price = strike_price
        if buy_signal_detected and sell_signal_detected:
            if verbose: print(f"\n   ⚠️  BOTH SIGNALS DETECTED - Review confluence carefully")
        if verbose: print(f"{'─' * 60}\n")
    if verbose: print(f"{'─' * 60}\n")
    if verbose: print(f"Command line use: {command_line}")
    if verbose: print(f"{'─' * 60}\n")
    result['train_score'] = train_score
    result['val_score'] = val_score
    result['train_win_rate'] = train_win_rate
    result['val_win_rate'] = val_win_rate
    result['optimize_target'] = optimize_target
    result['current_price'] = current_price
    result['current_date'] = entry_date
    result['target_price'] = target_price
    result['target_date'] = target_date
    result['dataset_id'] = _dataset_id
    result['ticker'] = _ticker
    result['lookahead'] = lookahead
    result['method'] = method
    result['put_strike_pct'] = put_strike_pct
    result['call_strike_pct'] = call_strike_pct
    return result


# ==============================================================================
# OPTUNA INTEGRATION
# ==============================================================================
def run_strategy_and_evaluate(df_base, _args, close_col, high_col, low_col, volume_col, open_col, rsi_length, rsi_signal_len, sma_len, fib_lookback, div_window, rsi_midline=50, rsi_oversold=30, rsi_overbought=70, min_buy_confluence=1, min_sell_confluence=1,
                              macd_fast=12, macd_slow=26, macd_signal=9,
                              bb_length=20, bb_std=2.0,
                              vol_sma_length=20, vol_multiplier=2.0,
                              stoch_k_period=14, stoch_d_period=3, stoch_smooth_k_period=1, stoch_oversold=20, stoch_overbought=80,
                              vwap_window=50,
                              use_pullback_buy=1, use_ema_cross_buy=1, use_ma_conf_buy=1, use_fib_rsi_buy=1,
                              use_reg_bull_div=1, use_hid_bull_div=1, use_macd_buy=1, use_bb_buy=1,
                              use_vol_buy=1, use_stoch_buy=1, use_vwap_buy=1,
                              use_pullback_sell=1, use_ema_cross_sell=1, use_reg_bear_div=1, use_hid_bear_div=1,
                              use_macd_sell=1, use_bb_sell=1, use_vol_sell=1, use_stoch_sell=1, use_vwap_sell=1, cooldown_bars=0):
    df = df_base.copy()
    df, rsi_col, pullback_buy_col, pullback_sell_col, ema_cross_buy_col, ema_cross_sell_col, ma_conf_buy_col = \
        implement_rsi_strategies(df, close_col, _args.ticker, rsi_length, rsi_signal_len, sma_len, rsi_midline, rsi_oversold, rsi_overbought)
    df, reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col = \
        find_divergences(df, high_col, low_col, rsi_col, _args.ticker, div_window)
    df, fib_rsi_buy_col = calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, _args.ticker, fib_lookback, rsi_midline)

    # NEW STRATEGIES
    macd_buy_col, macd_sell_col, bb_buy_col, bb_sell_col, vol_buy_col, vol_sell_col, stoch_buy_col, stoch_sell_col, vwap_buy_col, vwap_sell_col = \
        implement_additional_strategies(df, close_col, high_col, low_col, volume_col, open_col, _args.ticker,
                                        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                                        bb_length=bb_length, bb_std=bb_std,
                                        vol_sma_length=vol_sma_length, vol_multiplier=vol_multiplier,
                                        stoch_k_period=stoch_k_period, stoch_d_period=stoch_d_period, stoch_smooth_k_period=stoch_smooth_k_period, stoch_oversold=stoch_oversold, stoch_overbought=stoch_overbought,
                                        vwap_window=vwap_window)

    # 0/1 triggers for strategy inclusion
    buy_cols = []
    if use_pullback_buy: buy_cols.append(pullback_buy_col)
    if use_ema_cross_buy: buy_cols.append(ema_cross_buy_col)
    if use_ma_conf_buy: buy_cols.append(ma_conf_buy_col)
    if use_fib_rsi_buy: buy_cols.append(fib_rsi_buy_col)
    if use_reg_bull_div: buy_cols.append(reg_bull_div_col)
    if use_hid_bull_div: buy_cols.append(hid_bull_div_col)
    if use_macd_buy: buy_cols.append(macd_buy_col)
    if use_bb_buy: buy_cols.append(bb_buy_col)
    if use_vol_buy: buy_cols.append(vol_buy_col)
    if use_stoch_buy: buy_cols.append(stoch_buy_col)
    if use_vwap_buy: buy_cols.append(vwap_buy_col)

    sell_cols = []
    if use_pullback_sell: sell_cols.append(pullback_sell_col)
    if use_ema_cross_sell: sell_cols.append(ema_cross_sell_col)
    if use_reg_bear_div: sell_cols.append(reg_bear_div_col)
    if use_hid_bear_div: sell_cols.append(hid_bear_div_col)
    if use_macd_sell: sell_cols.append(macd_sell_col)
    if use_bb_sell: sell_cols.append(bb_sell_col)
    if use_vol_sell: sell_cols.append(vol_sell_col)
    if use_stoch_sell: sell_cols.append(stoch_sell_col)
    if use_vwap_sell: sell_cols.append(vwap_sell_col)

    # Aggregate signals based on minimum confluence required to narrow down quantity
    if len(buy_cols) > 0:
        df[('Signal_Buy', _args.ticker)] = df[buy_cols].fillna(False).sum(axis=1) >= min_buy_confluence
    else:
        df[('Signal_Buy', _args.ticker)] = False

    if len(sell_cols) > 0:
        df[('Signal_Sell', _args.ticker)] = df[sell_cols].fillna(False).sum(axis=1) >= min_sell_confluence
    else:
        df[('Signal_Sell', _args.ticker)] = False

    if cooldown_bars > 0:
        df[('Signal_Buy', _args.ticker)] = apply_cooldown(df[('Signal_Buy', _args.ticker)].to_numpy(), cooldown_bars)
        df[('Signal_Sell', _args.ticker)] = apply_cooldown(df[('Signal_Sell', _args.ticker)].to_numpy(), cooldown_bars)

    buy_wr, sell_wr, combined_wr, buy_wins, sell_wins, total_buy, total_sell = calculate_win_rates_vectorized(df=df, _args=_args, close_col=close_col, high_col=high_col, low_col=low_col)
    if _args.sanity_check:
        buy_wr2, sell_wr2, combined_wr2, buy_wins2, sell_wins2, total_buy2, total_sell2 = calculate_win_rates(df=df, _args=_args, close_col=close_col, high_col=high_col, low_col=low_col)
        assert np.allclose(buy_wr, buy_wr2)
        assert np.allclose(sell_wr, sell_wr2)
        assert np.allclose(combined_wr, combined_wr2)
        assert np.allclose(buy_wins, buy_wins2)
        assert np.allclose(sell_wins, sell_wins2)

    total_bars = len(df.dropna(subset=[close_col]))
    buy_density = int(df[('Signal_Buy', _args.ticker)].sum()) / total_bars if total_bars > 0 else 0
    sell_density = int(df[('Signal_Sell', _args.ticker)].sum()) / total_bars if total_bars > 0 else 0

    return buy_wr, sell_wr, combined_wr, buy_density, sell_density, total_buy, total_sell, buy_wins, sell_wins, df


def optuna_objective(trial, _args, df_base, close_col, high_col, low_col, volume_col, open_col):
    # 0. 0/1 Triggers for Strategy Inclusion (Controlled via argparse)
    use_pullback_buy = 1 if getattr(_args, 'use_pullback_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_ema_cross_buy = 1 if getattr(_args, 'use_ema_cross_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_ma_conf_buy = 1 if getattr(_args, 'use_ma_conf_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_fib_rsi_buy = 1 if getattr(_args, 'use_fib_rsi_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_reg_bull_div = 1 if getattr(_args, 'use_reg_bull_div', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_hid_bull_div = 1 if getattr(_args, 'use_hid_bull_div', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_macd_buy = 1 if getattr(_args, 'use_macd_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_bb_buy = 1 if getattr(_args, 'use_bb_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_vol_buy = 1 if getattr(_args, 'use_vol_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_stoch_buy = 1 if getattr(_args, 'use_stoch_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0
    use_vwap_buy = 1 if getattr(_args, 'use_vwap_buy', False) and _args.optimize_target in ['buy_wr', 'combined_wr'] else 0

    use_pullback_sell = 1 if getattr(_args, 'use_pullback_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_ema_cross_sell = 1 if getattr(_args, 'use_ema_cross_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_reg_bear_div = 1 if getattr(_args, 'use_reg_bear_div', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_hid_bear_div = 1 if getattr(_args, 'use_hid_bear_div', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_macd_sell = 1 if getattr(_args, 'use_macd_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_bb_sell = 1 if getattr(_args, 'use_bb_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_vol_sell = 1 if getattr(_args, 'use_vol_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_stoch_sell = 1 if getattr(_args, 'use_stoch_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0
    use_vwap_sell = 1 if getattr(_args, 'use_vwap_sell', False) and _args.optimize_target in ['sell_wr', 'combined_wr'] else 0

    use_rsi_any = (use_pullback_buy or use_pullback_sell or use_ema_cross_buy or
                   use_ema_cross_sell or use_ma_conf_buy or use_fib_rsi_buy or
                   use_reg_bull_div or use_reg_bear_div or use_hid_bull_div or use_hid_bear_div)
    use_rsi_signal = use_ema_cross_buy or use_ema_cross_sell or use_ma_conf_buy
    use_sma = use_pullback_buy or use_pullback_sell or use_ma_conf_buy
    use_div = use_reg_bull_div or use_reg_bear_div or use_hid_bull_div or use_hid_bear_div
    use_rsi_midline = use_pullback_buy or use_pullback_sell or use_fib_rsi_buy

    # 1. RSI Length: Standard is 14. 7-21 covers short to medium momentum without excessive noise/lag.
    rsi_length = trial.suggest_int('rsi_length', 7, 21) if use_rsi_any else trial.suggest_int('rsi_length', 14, 14)

    # 2. RSI Signal Line (EMA): Must be faster than RSI to generate crossovers. 3-12 is standard.
    rsi_signal_len = trial.suggest_int('rsi_signal_len', 3, 12) if use_rsi_signal else trial.suggest_int('rsi_signal_len', 9, 9)

    # 3. SMA Length: Used for trend direction. 20-100 covers standard swing trading MAs (20, 50, 100).
    sma_len = trial.suggest_int('sma_len', 20, 100) if use_sma else trial.suggest_int('sma_len', 50, 50)

    # 4. Fibonacci Lookback: Window to find swing high/low. 10-60 captures recent, actionable swings.
    fib_lookback = trial.suggest_int('fib_lookback', 10, 60) if use_fib_rsi_buy else trial.suggest_int('fib_lookback', 50, 50)

    # 5. Divergence Window: Lookback for local pivots. 5-15 prevents noise (3 is too small for a real pivot).
    div_window = trial.suggest_int('div_window', 5, 15) if use_div else trial.suggest_int('div_window', 5, 5)

    # 6. RSI Midline: Center threshold. 40-60 is logically sound for trend bias.
    rsi_midline = trial.suggest_int('rsi_midline', 40, 60) if use_rsi_midline else trial.suggest_int('rsi_midline', 50, 50)

    # 7. RSI Oversold/Overbought: 10 and 90 are statistical anomalies. 20-35 and 65-80 cover realistic extremes.
    rsi_oversold = trial.suggest_int('rsi_oversold', 20 - 5, 35 + 5) if use_ema_cross_buy else trial.suggest_int('rsi_oversold', 30, 30)
    rsi_overbought = trial.suggest_int('rsi_overbought', 65 - 5, 80 + 5) if use_ema_cross_sell else trial.suggest_int('rsi_overbought', 70, 70)

    # 8. Minimum Confluence: Number of strategies that must agree to trigger a signal.
    # Higher values drastically reduce signal quantity, increasing selectivity
    # and filtering out weaker, isolated setups. Dynamically capped by enabled strategies.
    total_buy_strategies_enabled = (use_pullback_buy + use_ema_cross_buy + use_ma_conf_buy + use_fib_rsi_buy +
                                    use_reg_bull_div + use_hid_bull_div + use_macd_buy + use_bb_buy +
                                    use_vol_buy + use_stoch_buy + use_vwap_buy)
    total_sell_strategies_enabled = (use_pullback_sell + use_ema_cross_sell + use_reg_bear_div + use_hid_bear_div +
                                     use_macd_sell + use_bb_sell + use_vol_sell + use_stoch_sell + use_vwap_sell)

    min_buy_confluence = trial.suggest_int('min_buy_confluence', _args.buy_confluence_range[0], _args.buy_confluence_range[1])
    min_sell_confluence = trial.suggest_int('min_sell_confluence', _args.sell_confluence_range[0], _args.sell_confluence_range[1])

    # 9. MACD Crossovers
    macd_fast = trial.suggest_int('macd_fast', 5, 15) if use_macd_buy or use_macd_sell else trial.suggest_int('macd_fast', 5, 5)
    macd_slow = trial.suggest_int('macd_slow', 15, 35) if use_macd_buy or use_macd_sell else trial.suggest_int('macd_slow', 15, 15)
    macd_signal = trial.suggest_int('macd_signal', 5, 15) if use_macd_buy or use_macd_sell else trial.suggest_int('macd_signal', 5, 5)

    # 10. Bollinger Bands
    bb_length = trial.suggest_int('bb_length', 10, 50) if use_bb_buy or use_bb_sell else trial.suggest_int('bb_length', 10, 10)
    bb_std = trial.suggest_float('bb_std', 1.0, 3.0) if use_bb_buy or use_bb_sell else trial.suggest_float('bb_std', 1.0, 1.0)

    # 11. Volume Spikes
    vol_sma_length = trial.suggest_int('vol_sma_length', 10, 50) if use_vol_buy or use_vol_sell else trial.suggest_int('vol_sma_length', 10, 10)
    vol_multiplier = trial.suggest_float('vol_multiplier', 1.0, 3.0) if use_vol_buy or use_vol_sell else trial.suggest_float('vol_multiplier', 1.0, 1.0)

    # 12. Stochastic Oscillators
    stoch_k_period = trial.suggest_int('stoch_k_period', 5, 21) if use_stoch_buy or use_stoch_sell else trial.suggest_int('stoch_k_period', 5, 5)
    stoch_d_period = trial.suggest_int('stoch_d_period', 2, 9) if use_stoch_buy or use_stoch_sell else trial.suggest_int('stoch_d_period', 2, 2)
    stoch_smooth_k_period = trial.suggest_int('stoch_smooth_k_period', 1, 5) if use_stoch_buy or use_stoch_sell else trial.suggest_int('stoch_smooth_k_period', 1, 1)
    stoch_oversold = trial.suggest_int('stoch_oversold', 10, 30) if use_stoch_buy or use_stoch_sell else trial.suggest_int('stoch_oversold', 10, 10)
    stoch_overbought = trial.suggest_int('stoch_overbought', 70, 90) if use_stoch_buy or use_stoch_sell else trial.suggest_int('stoch_overbought', 70, 70)

    # 13. VWAP Retests (Rolling window)
    vwap_window = trial.suggest_int('vwap_window', 10, 100) if use_vwap_buy or use_vwap_sell else trial.suggest_int('vwap_window', 10, 10)

    # 🚨 CONSTRAINT PRUNING: Instantly discard invalid trials to speed up convergence
    # A signal line EMA must be shorter than the RSI length to create meaningful crossovers.
    if rsi_signal_len >= rsi_length:
        raise optuna.exceptions.TrialPruned()

    # Overbought must be strictly greater than oversold (already guaranteed by ranges, but good practice)
    if rsi_overbought <= rsi_oversold:
        raise optuna.exceptions.TrialPruned()

    # MACD fast period must be strictly less than slow period to form valid signals
    if macd_fast >= macd_slow:
        raise optuna.exceptions.TrialPruned()

    # Prune if min_confluence is greater than the number of enabled strategies (redundant now due to dynamic max, but kept for safety)
    if (min_buy_confluence > total_buy_strategies_enabled or total_buy_strategies_enabled == 0) and _args.optimize_target in ['buy_wr', 'combined_wr']:
        raise optuna.exceptions.TrialPruned()
    if (min_sell_confluence > total_sell_strategies_enabled or total_sell_strategies_enabled == 0) and _args.optimize_target in ['sell_wr', 'combined_wr']:
        raise optuna.exceptions.TrialPruned()
    if total_buy_strategies_enabled + total_sell_strategies_enabled == 0:
        raise optuna.exceptions.TrialPruned()

    try:
        # Compute indicators on the full df_base once per trial for efficiency
        df = df_base.copy()
        df, rsi_col, pullback_buy_col, pullback_sell_col, ema_cross_buy_col, ema_cross_sell_col, ma_conf_buy_col = \
            implement_rsi_strategies(df, close_col, _args.ticker, rsi_length, rsi_signal_len, sma_len, rsi_midline, rsi_oversold, rsi_overbought)
        df, reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col = \
            find_divergences(df, high_col, low_col, rsi_col, _args.ticker, div_window)
        df, fib_rsi_buy_col = calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, _args.ticker, fib_lookback, rsi_midline)

        # NEW STRATEGIES
        macd_buy_col, macd_sell_col, bb_buy_col, bb_sell_col, vol_buy_col, vol_sell_col, stoch_buy_col, stoch_sell_col, vwap_buy_col, vwap_sell_col = \
            implement_additional_strategies(df, close_col, high_col, low_col, volume_col, open_col, _args.ticker,
                                            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
                                            bb_length=bb_length, bb_std=bb_std,
                                            vol_sma_length=vol_sma_length, vol_multiplier=vol_multiplier,
                                            stoch_k_period=stoch_k_period, stoch_d_period=stoch_d_period, stoch_smooth_k_period=stoch_smooth_k_period, stoch_oversold=stoch_oversold, stoch_overbought=stoch_overbought,
                                            vwap_window=vwap_window)

        # 0/1 triggers for strategy inclusion
        buy_cols = []
        if use_pullback_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(pullback_buy_col)
        if use_ema_cross_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(ema_cross_buy_col)
        if use_ma_conf_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(ma_conf_buy_col)
        if use_fib_rsi_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(fib_rsi_buy_col)
        if use_reg_bull_div and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(reg_bull_div_col)
        if use_hid_bull_div and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(hid_bull_div_col)
        if use_macd_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(macd_buy_col)
        if use_bb_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(bb_buy_col)
        if use_vol_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(vol_buy_col)
        if use_stoch_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(stoch_buy_col)
        if use_vwap_buy and _args.optimize_target in ['buy_wr', 'combined_wr']: buy_cols.append(vwap_buy_col)

        sell_cols = []
        if use_pullback_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(pullback_sell_col)
        if use_ema_cross_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(ema_cross_sell_col)
        if use_reg_bear_div and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(reg_bear_div_col)
        if use_hid_bear_div and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(hid_bear_div_col)
        if use_macd_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(macd_sell_col)
        if use_bb_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(bb_sell_col)
        if use_vol_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(vol_sell_col)
        if use_stoch_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(stoch_sell_col)
        if use_vwap_sell and _args.optimize_target in ['sell_wr', 'combined_wr']: sell_cols.append(vwap_sell_col)

        buy_sig_col = ('Signal_Buy', _args.ticker)
        sell_sig_col = ('Signal_Sell', _args.ticker)

        # Aggregate signals based on minimum confluence required to narrow down quantity
        if len(buy_cols) > 0:
            df[buy_sig_col] = df[buy_cols].fillna(False).sum(axis=1) >= min_buy_confluence
        else:
            df[buy_sig_col] = False

        if len(sell_cols) > 0:
            df[sell_sig_col] = df[sell_cols].fillna(False).sum(axis=1) >= min_sell_confluence
        else:
            df[sell_sig_col] = False

        cooldown_bars = getattr(_args, 'cooldown_bars', 0)
        if cooldown_bars > 0:
            df[buy_sig_col] = apply_cooldown(df[buy_sig_col].to_numpy(), cooldown_bars)
            df[sell_sig_col] = apply_cooldown(df[sell_sig_col].to_numpy(), cooldown_bars)

        # Initialize TimeSeriesSplit with 10 folds
        tscv = TimeSeriesSplit(n_splits=10)
        fold_scores = []

        # Evaluate score across all chronological passes
        for train_index, test_index in tscv.split(df):
            # Temporarily mask signals outside the current test fold
            orig_buy = df[buy_sig_col].to_numpy().copy()
            orig_sell = df[sell_sig_col].to_numpy().copy()

            mask = np.zeros(len(df), dtype=bool)
            mask[test_index] = True

            df.loc[~mask, buy_sig_col] = False
            df.loc[~mask, sell_sig_col] = False

            # Calculate win rates only for the signals occurring in the test fold
            buy_wr, sell_wr, combined_wr, buy_wins, sell_wins, total_buy, total_sell = \
                calculate_win_rates_vectorized(df=df, _args=_args, close_col=close_col, high_col=high_col, low_col=low_col)

            # Calculate densities based on the masked signals (only those in the test fold)
            total_bars = len(test_index)
            buy_density = int(df[buy_sig_col].sum()) / total_bars if total_bars > 0 else 0
            sell_density = int(df[sell_sig_col].sum()) / total_bars if total_bars > 0 else 0

            # Restore original signals for the next fold iteration
            df[buy_sig_col] = orig_buy
            df[sell_sig_col] = orig_sell

            # 🎯 MODIFIED OBJECTIVE: Use Smoothed Win Rate + High-Density Penalty
            # Wilson Score heavily penalizes small sample sizes, which inadvertently forces
            # the optimizer to select high-density (many signals) setups to narrow the confidence
            # interval, often sacrificing Win Rate.
            # By using Bayesian (Add-2) smoothing, we allow the optimizer to discover highly
            # selective (lower density) setups that have high Win Rates.
            def smoothed_wr(wins, n):
                return (wins + 1.0) / (n + 2.0) if n > 0 else 0.0

            if _args.optimize_target == 'buy_wr':
                raw_target = smoothed_wr(buy_wins, total_buy)
                density = buy_density
            elif _args.optimize_target == 'sell_wr':
                raw_target = smoothed_wr(sell_wins, total_sell)
                density = sell_density
            else:
                raw_target = smoothed_wr(buy_wins + sell_wins, total_buy + total_sell)
                density = min(buy_density, sell_density)

            # Replace the hard minimum with a smooth approximation
            # softplus(x) = ln(1 + exp(x))
            # We scale it with a temperature hyperparameter 'k' (higher k = sharper turn)
            k = 20.0
            diff = _args.min_signal_density - density

            if diff > 100 / k:  # Prevent overflow in math.exp
                smooth_max_diff = diff
            else:
                smooth_max_diff = math.log(1.0 + math.exp(k * diff)) / k

            # Apply the fourth root with correct operator precedence
            density_penalty_val = smooth_max_diff ** 0.25
            target = raw_target - density_penalty_val

            # Penalize overlapping signals (clustering during anomalies)
            overlap_penalty = 0.0
            if _args.optimize_target in ['buy_wr', 'combined_wr']:
                buy_indices = np.where(df[buy_sig_col].to_numpy())[0]
                overlap_penalty += calculate_overlap_penalty(buy_indices, _args.lookahead_bars)
            if _args.optimize_target in ['sell_wr', 'combined_wr']:
                sell_indices = np.where(df[sell_sig_col].to_numpy())[0]
                overlap_penalty += calculate_overlap_penalty(sell_indices, _args.lookahead_bars)

            if _args.optimize_target == 'combined_wr':
                overlap_penalty /= 2.0  # Average the penalty

            # Reduce the target score based on the overlap ratio.
            # If 50% of signals overlap, the score is halved.
            target = target * (1.0 - overlap_penalty)

            fold_scores.append(target)

        # Return the mean score of all the passes minus a consistency penalty
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        # Alpha controls how much we punish inconsistency across different time periods
        alpha = 0.125
        final_score = mean_score - (alpha * std_score)

        return final_score

    except Exception as ee:
        print(ee)
        return -1.0  # Discard invalid trials


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strategy Optimizer & Real-Time Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    data_group = parser.add_argument_group('Data & Symbol')
    data_group.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    data_group.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    data_group.add_argument("--clip-n", type=int, default=0, help="Number of most recent bars to clip from the dataset.")
    data_group.add_argument("--reduce-n", type=int, default=0, help="Number of most oldest bars to clip from the dataset.")

    strat_group = parser.add_argument_group('Strategy & P&L Parameters')
    strat_group.add_argument('--lookahead-bars', type=int, default=1, dest='lookahead_bars', help='Forward-looking window')
    strat_group.add_argument('--cooldown-bars', type=int, default=0, dest='cooldown_bars', help='Minimum number of bars to wait between signals (cooldown period)')
    strat_group.add_argument('--method', type=str, default='final_close', choices=['touched', 'final_close'], help='Strike evaluation method')
    strat_group.add_argument('--min-signal-density', type=float, default=0.01, help='Min signal frequency threshold')
    strat_group.add_argument('--put-strike-pct', type=float, default=0.9999, help='Base put strike multiplier')
    strat_group.add_argument('--call-strike-pct', type=float, default=1.0001, help='Base call strike multiplier')
    strat_group.add_argument('--wr-weight', type=float, default=0.9, help='Weight for Win-Rate')
    strat_group.add_argument('--td-weight', type=float, default=0.1, help='Weight for Trade-Density')
    strat_group.add_argument('--buy-confluence-range', type=int, nargs=2, default=[2, 3], metavar=('MIN', 'MAX'), help='Min and max range for buy confluence optimization (default: 2 3)')
    strat_group.add_argument('--sell-confluence-range', type=int, nargs=2, default=[2, 3], metavar=('MIN', 'MAX'), help='Min and max range for sell confluence optimization (default: 2 3)')

    opt_group = parser.add_argument_group('Optimization & Execution')
    opt_group.add_argument('--optimize', action='store_true', help='Run Optuna hyperparameter optimization')
    opt_group.add_argument('--optimize-target', type=str, default='buy_wr', choices=['combined_wr', 'buy_wr', 'sell_wr'],
                           help='Metric to maximize during optimization')
    opt_group.add_argument('--n-trials', type=int, default=9999, help='Optuna trials per run')
    opt_group.add_argument('--timeout', type=int, default=3600, help='Max runtime (seconds)')
    opt_group.add_argument('--output-dir', type=str, default='models', help='Output directory')
    opt_group.add_argument('--optuna-db', type=str, default=None, help='Database URL for Optuna persistence (e.g., sqlite:///optuna.db or postgresql://user:pass@host/db)')
    opt_group.add_argument('--train-ratio', type=float, default=0.8,
                           help='Ratio of data to use for training (rest for validation). Use 1.0 to disable split.')

    flag_group = parser.add_argument_group('Execution Flags')
    flag_group.add_argument('--real-time', action=argparse.BooleanOptionalAction, default=False,
                            help='Real-time mode: test latest datapoint with specified model')
    flag_group.add_argument('--model-path', type=str, default=None,
                            help='Path to saved model .pkl file (required for real-time mode, optional for evaluation)')
    flag_group.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose output')
    flag_group.add_argument('--verbose-short', action=argparse.BooleanOptionalAction, default=False, help='Short real-time output')
    flag_group.add_argument('--verbose-optuna-progression', action=argparse.BooleanOptionalAction, default=False, help='')
    flag_group.add_argument('--seed', type=int, default=123, help='Random seed')
    flag_group.add_argument('--plot', action='store_true', default=False, help='Plot results with matplotlib')
    flag_group.add_argument('--sanity-check', action='store_true', default=False, help='Check vectorized implementation consistency')

    # Strategy Toggles: Allow enabling/disabling specific strategies via argparse
    strat_toggles = parser.add_argument_group('Strategy Toggles')

    # Buy Strategies
    strat_toggles.add_argument('--use-pullback-buy', action='store_true', default=True, help='Enable Pullback Buy strategy')
    strat_toggles.add_argument('--use-ema-cross-buy', action='store_true', default=True, help='Enable EMA Cross Buy strategy')
    strat_toggles.add_argument('--use-ma-conf-buy', action='store_true', default=True, help='Enable MA Confluence Buy strategy')
    strat_toggles.add_argument('--use-fib-rsi-buy', action='store_true', default=True, help='Enable Fibonacci RSI Buy strategy')
    strat_toggles.add_argument('--use-reg-bull-div', action='store_true', default=True, help='Enable Regular Bullish Divergence strategy')
    strat_toggles.add_argument('--use-hid-bull-div', action='store_true', default=True, help='Enable Hidden Bullish Divergence strategy')
    strat_toggles.add_argument('--use-macd-buy', action='store_true', default=True, help='Enable MACD Buy strategy')
    strat_toggles.add_argument('--use-bb-buy', action='store_true', default=True, help='Enable Bollinger Bands Buy strategy')
    strat_toggles.add_argument('--use-vol-buy', action='store_true', default=True, help='Enable Volume Spike Buy strategy')
    strat_toggles.add_argument('--use-stoch-buy', action='store_true', default=True, help='Enable Stochastic Buy strategy')
    strat_toggles.add_argument('--use-vwap-buy', action='store_true', default=True, help='Enable VWAP Retest Buy strategy')

    # Sell Strategies
    strat_toggles.add_argument('--use-pullback-sell', action='store_true', default=True, help='Enable Pullback Sell strategy')
    strat_toggles.add_argument('--use-ema-cross-sell', action='store_true', default=True, help='Enable EMA Cross Sell strategy')
    strat_toggles.add_argument('--use-reg-bear-div', action='store_true', default=True, help='Enable Regular Bearish Divergence strategy')
    strat_toggles.add_argument('--use-hid-bear-div', action='store_true', default=True, help='Enable Hidden Bearish Divergence strategy')
    strat_toggles.add_argument('--use-macd-sell', action='store_true', default=True, help='Enable MACD Sell strategy')
    strat_toggles.add_argument('--use-bb-sell', action='store_true', default=True, help='Enable Bollinger Bands Sell strategy')
    strat_toggles.add_argument('--use-vol-sell', action='store_true', default=True, help='Enable Volume Spike Sell strategy')
    strat_toggles.add_argument('--use-stoch-sell', action='store_true', default=True, help='Enable Stochastic Sell strategy')
    strat_toggles.add_argument('--use-vwap-sell', action='store_true', default=True, help='Enable VWAP Retest Sell strategy')

    return parser


def print_startup_banner(args):
    """Print a visible banner when the program runs."""
    train_info = ""
    if hasattr(args, 'train_ratio') and args.optimize and args.train_ratio < 1.0:
        train_info = f" | Train/Val: {args.train_ratio * 100:.0f}%/{(1 - args.train_ratio) * 100:.0f}%"

    banner = f"""
╔{'═' * 78}╗
║  🎯 STRATEGY OPTIMIZER & REAL-TIME MONITOR  {' ' * 33}║
╠{'═' * 78}╣
║  📈 Multi-signal technical strategy for options validation                   ║
║  🔧 RSI • Fibonacci • Divergences • Optuna Optimization                      ║
╠{'─' * 78}╣
║  🔹 Ticker       : {args.ticker:<58}║
║  🔹 Dataset      : {args.dataset_id:<58}║
║  🔹 Mode         : {'REAL-TIME' if args.real_time else 'OPTIMIZATION' if args.optimize else 'EVALUATION' if args.model_path else 'DEFAULT BACKTEST':<58}║
║  🔹 Lookahead    : {args.lookahead_bars:02d} bars | Cooldown: {args.cooldown_bars:02d} bars{' ' * 31}║
║  🔹 Method       : {args.method:<58}║
║  🔹 Target       : {args.optimize_target:<58}║
║  🔹 Min Density  : {args.min_signal_density:<58}║
║  🔹 Strike Pct   : Put {args.put_strike_pct:.2%} | Call {args.call_strike_pct:.2%}{' ' * 1}{train_info if train_info else ' ' * 78}{' ' * 11}║
╚{'═' * 78}╝
"""
    print(banner)


def early_stop_on_perfect_success(study, trial):
    """
    Callback to stop optimization when 100% success rate is achieved.
    Objective returns -success_rate, so -100.0 = 100% success.
    """
    try:
        if study.best_value is not None and study.best_value >= 0.999:
            print(f"\n🎯 100% success rate achieved at trial #{trial.number}! Stopping optimization early...")
            study.stop()
    except:
        pass


def entry(args):
    np.random.seed(args.seed)
    close_col = ('Close', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)
    volume_col = ('Volume', args.ticker)
    open_col = ('Open', args.ticker)
    command_line = "python " + " ".join(sys.argv)

    if not args.real_time:
        df_base = factory_load_data(_dataset_id=args.dataset_id, _ticker=args.ticker, _args={"clip_n": args.clip_n, "reduce_n": args.reduce_n})
        if args.verbose: print(f"📂 Dataset ranging from {df_base.index[0].strftime('%Y-%m-%d')} to {df_base.index[-1].strftime('%Y-%m-%d')}")
        if args.verbose:
            print(f"\n✨ Loaded {args.ticker} | Dataset: {args.dataset_id} | Bars: {len(df_base)}")
            if args.real_time:
                print("🔹 Mode: REAL-TIME SIGNAL CHECK")
            elif args.optimize:
                print(f"🔹 Mode: OPTIMIZATION → Target: {args.optimize_target} | Trials: {args.n_trials}")
                if hasattr(args, 'train_ratio') and args.train_ratio < 1.0:
                    print(f"🔹 Train/Val Split: {args.train_ratio * 100:.1f}% / {(1 - args.train_ratio) * 100:.1f}%")
            elif args.model_path:
                print(f"🔹 Mode: EVALUATION → Model: {os.path.basename(args.model_path)}")
            else:
                print("🔹 Mode: DEFAULT BACKTEST → Params: {rsi:14, sma:50, fib:50, div:5}")
            print("─" * 80 + "\n")

    # 🔹 Handle real-time mode first
    if args.real_time:
        return real_time_mode(model_path=args.model_path, verbose=args.verbose, clip_n=args.clip_n, reduce_n=args.reduce_n,
                              close_col=close_col, high_col=high_col, low_col=low_col, volume_col=volume_col, open_col=open_col)
    assert args.put_strike_pct > 0.89 and args.call_strike_pct < 1.11, f"Just to make sure one does not use 0.05 instead 0.95 , for example."
    if args.verbose: print_startup_banner(args)

    # Default params (used if not optimizing)
    params = {
        'rsi_length': 14, 'rsi_signal_len': 10, 'sma_len': 50, 'fib_lookback': 50, 'div_window': 5,
        'rsi_midline': 50, 'rsi_oversold': 30, 'rsi_overbought': 70,
        'min_buy_confluence': args.buy_confluence_range[0], 'min_sell_confluence': args.sell_confluence_range[0],
        'max_buy_confluence': args.buy_confluence_range[1], 'max_sell_confluence': args.sell_confluence_range[1],
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'bb_length': 20, 'bb_std': 2.0,
        'vol_sma_length': 20, 'vol_multiplier': 2.0,
        'stoch_k_period': 14, 'stoch_d_period': 3, 'stoch_smooth_k_period': 1, 'stoch_oversold': 20, 'stoch_overbought': 80,
        'vwap_window': 50, 'cooldown_bar': args.cooldown_bars,
        # 0/1 triggers for strategy inclusion (read from argparse, default to 1)
        'use_pullback_buy': 1 if getattr(args, 'use_pullback_buy', False) else 0,
        'use_ema_cross_buy': 1 if getattr(args, 'use_ema_cross_buy', False) else 0,
        'use_ma_conf_buy': 1 if getattr(args, 'use_ma_conf_buy', False) else 0,
        'use_fib_rsi_buy': 1 if getattr(args, 'use_fib_rsi_buy', False) else 0,
        'use_reg_bull_div': 1 if getattr(args, 'use_reg_bull_div', False) else 0,
        'use_hid_bull_div': 1 if getattr(args, 'use_hid_bull_div', False) else 0,
        'use_macd_buy': 1 if getattr(args, 'use_macd_buy', False) else 0,
        'use_bb_buy': 1 if getattr(args, 'use_bb_buy', False) else 0,
        'use_vol_buy': 1 if getattr(args, 'use_vol_buy', False) else 0,
        'use_stoch_buy': 1 if getattr(args, 'use_stoch_buy', False) else 0,
        'use_vwap_buy': 1 if getattr(args, 'use_vwap_buy', False) else 0,
        'use_pullback_sell': 1 if getattr(args, 'use_pullback_sell', False) else 0,
        'use_ema_cross_sell': 1 if getattr(args, 'use_ema_cross_sell', False) else 0,
        'use_reg_bear_div': 1 if getattr(args, 'use_reg_bear_div', False) else 0,
        'use_hid_bear_div': 1 if getattr(args, 'use_hid_bear_div', False) else 0,
        'use_macd_sell': 1 if getattr(args, 'use_macd_sell', False) else 0,
        'use_bb_sell': 1 if getattr(args, 'use_bb_sell', False) else 0,
        'use_vol_sell': 1 if getattr(args, 'use_vol_sell', False) else 0,
        'use_stoch_sell': 1 if getattr(args, 'use_stoch_sell', False) else 0,
        'use_vwap_sell': 1 if getattr(args, 'use_vwap_sell', False) else 0,
    }

    # 🔹 Load model from path if specified (for evaluation without optimization)
    if args.model_path and not args.optimize:
        print(f"🔍 Loading model from: {args.model_path}")
        model_data = load_model(args.model_path)
        params = model_data['params']
        if args.verbose:
            print(f"📊 Loaded model with score: {model_data.get('score', 'N/A')}")
            if model_data.get('validation_score') is not None:
                print(f"📊 Test score: {model_data['validation_score']:.4f}")
            print(f"🧠 Optimized Parameters from model: {params}")

    # ==============================================================================
    # 🔄 TRAIN/VALIDATION SPLIT (for optimization only)
    # ==============================================================================
    df_train = df_base
    df_val = None
    train_val_split_info = None

    if args.optimize and args.train_ratio < 1.0:
        # Chronological split to avoid look-ahead bias (critical for time-series)
        split_idx = int(len(df_base) * args.train_ratio)

        # Ensure minimum data in each split
        min_bars = 1  # Adjust based on your strategy's warmup needs
        if split_idx < min_bars:
            if args.verbose: print(f"⚠️  Train split too small ({split_idx} < {min_bars}), using full dataset")
        elif len(df_base) - split_idx < min_bars:
            if args.verbose: print(f"⚠️  Validation split too small ({len(df_base) - split_idx} < {min_bars}), using full dataset")
        else:
            df_train = df_base.iloc[:split_idx].copy()
            df_val = df_base.iloc[split_idx:].copy()
            train_val_split_info = {
                'train_ratio': args.train_ratio,
                'train_bars': len(df_train),
                'val_bars': len(df_val),
                'train_range': (df_train.index[0].strftime('%Y-%m-%d'), df_train.index[-1].strftime('%Y-%m-%d')),
                'val_range': (df_val.index[0].strftime('%Y-%m-%d'), df_val.index[-1].strftime('%Y-%m-%d'))
            }
            if args.verbose:
                print(f"📊 Data Split: Train={len(df_train)} bars ({args.train_ratio * 100:.1f}%), "
                      f"Val={len(df_val)} bars ({(1 - args.train_ratio) * 100:.1f}%)     Note: consider Val as Test")
                print(f"   📅 Train: {train_val_split_info['train_range'][0]} → {train_val_split_info['train_range'][1]}")
                print(f"   📅 Val:   {train_val_split_info['val_range'][0]} → {train_val_split_info['val_range'][1]}")

    if args.optimize:
        # ==============================================================================
        # 🔧 OPTUNA STORAGE SETUP: In-memory if --optuna-db is None, URL/SQLite otherwise
        # ==============================================================================
        if args.optuna_db is None:
            # Use in-memory storage (no persistence across runs)
            storage = None
            db_path_display = "in-memory (no persistence)"
        else:
            # Use the provided URL directly. Optuna supports postgresql://, mysql://, sqlite://, etc.
            # For backward compatibility, if it doesn't contain '://', assume it's a local sqlite path.
            if "://" not in args.optuna_db:
                db_path = args.optuna_db
                db_dir = os.path.dirname(db_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)
                storage = f"sqlite:///{db_path}"
            else:
                def parse_storage_url(url: str):
                    """Convertit une chaîne d'URL en objet Storage Optuna adapté."""
                    if url.startswith("journal://"):
                        # Extrait le chemin après "journal://"
                        file_path = url.replace("journal://", "", 1)
                        return JournalStorage(JournalFileBackend(file_path))

                    # Pour sqlite://, postgresql://, redis://, etc.
                    return url

                storage = parse_storage_url(args.optuna_db)
            db_path_display = storage

        study_name = (
            f"{args.ticker}_"
            f"{args.dataset_id}_"
            f"{args.optimize_target}_"
            f"la{args.lookahead_bars}_"
            f"{args.method}"
        )
        if args.verbose:
            print(f"\n🔍 Initializing Optuna study: {study_name}")
            print(f"   📂 Storage: {db_path_display}")
        if df_val is not None:
            if args.verbose: print(f"   🔄 Optimizing on TRAINING set only (validation held out)")

        n_startup_trials = max(99, min(10000, int(0.001 * args.n_trials)))
        if args.verbose: print(f"Starting Optuna optimization with TimeSeriesSplit and {n_startup_trials} random trials...")
        sampler = optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=n_startup_trials,
        )

        # Create or load the study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,  # Safe: does nothing for in-memory, loads for SQLite/PostgreSQL
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(), sampler=sampler
        )

        # ==============================================================================
        # 📦 CHECK FOR EXISTING TRIALS AND PRINT BEFORE OPTIMIZATION
        # ==============================================================================
        completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if completed_trials:
            best_trial = study.best_trial
            print("\n" + "─" * 80)
            print("📦 STUDY PERSISTENCE DETECTED — Resuming optimization")
            print("─" * 80)
            print(f"   🏆  Best Recorded {args.optimize_target.replace('_', ' ').title()}: {best_trial.value:.4f} ({best_trial.value * 100:.2f}%)")
            print(f"   📊  Completed Trials in Storage: {len(completed_trials)}")
            print(f"   🧠  Optimal Configuration Found So Far:")
            for k, v in best_trial.params.items():
                print(f"      • {k}: {v}")
            print("─" * 80)
            print(f"   🔄 Will append {args.n_trials} additional trials to the existing study...\n")
        else:
            if args.verbose: print("📦 No stored trials found. Initializing fresh optimization run.\n")
        # ==============================================================================
        # 🚀 RUN OPTIMIZATION
        # ==============================================================================
        study.optimize(
            lambda trial: optuna_objective(
                trial=trial,
                _args=args,
                df_base=df_train,
                close_col=close_col,
                high_col=high_col,
                low_col=low_col,
                volume_col=volume_col,
                open_col=open_col
            ),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True if args.verbose_optuna_progression else False,
            callbacks=[early_stop_on_perfect_success],
        )
        if args.verbose:
            print(f"\n✅ Optimization Complete!")
        best_trial = study.best_trial
        params = best_trial.params
        params.update({'cooldown_bars': args.cooldown_bars})
        if args.verbose:
            print(f"   🏆 Best {args.optimize_target}: {best_trial.value:.4f}")
            print("   🧠 Best Hyperparameters:")
            for k, v in best_trial.params.items():
                if args.verbose: print(f"      {k}: {v}")
        # ==============================================================================
        # 🎯 VALIDATION SET EVALUATION (if split was used)
        # ==============================================================================
        if df_val is not None and len(df_val) > args.lookahead_bars:
            if args.verbose:
                print(f"\n{'=' * 80}")
                print(f"🔍 EVALUATING BEST PARAMETERS ON VALIDATION SET")
                print(f"{'=' * 80}")

            buy_wr_train, sell_wr_train, combined_wr_train, _, _, _, _, _, _, _ = \
                run_strategy_and_evaluate(
                    df_base=df_train,
                    _args=args,
                    close_col=close_col,
                    high_col=high_col,
                    low_col=low_col,
                    volume_col=volume_col,
                    open_col=open_col,
                    **params
                )

            # Evaluate on validation set with best params
            buy_wr_val, sell_wr_val, combined_wr_val, _, _, eval_buy_val, eval_sell_val, buy_wins_val, sell_wins_val, df_val_final = \
                run_strategy_and_evaluate(
                    df_base=df_val,
                    _args=args,
                    close_col=close_col,
                    high_col=high_col,
                    low_col=low_col,
                    volume_col=volume_col,
                    open_col=open_col,
                    **params
                )

            if args.verbose:
                print(f"📊 Validation Results ({len(df_val)} bars):")
                if args.optimize_target in ['buy_wr', 'combined_wr']:
                    print(f"   🟢 Buy:  {buy_wins_val}/{eval_buy_val} → {buy_wr_val:6.2%}")
                if args.optimize_target in ['sell_wr', 'combined_wr']:
                    print(f"   🔴 Sell: {sell_wins_val}/{eval_sell_val} → {sell_wr_val:6.2%}")
                if args.optimize_target in ['combined_wr']:
                    print(f"   🎯 Combined: {(buy_wins_val + sell_wins_val)}/{(eval_buy_val + eval_sell_val)} → {combined_wr_val:6.2%}")

                # Compare train vs val if both available
                train_score = combined_wr_train if args.optimize_target == 'combined_wr' else (buy_wr_train if args.optimize_target == 'buy_wr' else sell_wr_train)
                val_score = combined_wr_val if args.optimize_target == 'combined_wr' else (buy_wr_val if args.optimize_target == 'buy_wr' else sell_wr_val)
                # Actually get training score from best trial
                gap = train_score - val_score
                status = "✅ Good generalization" if abs(gap) < 0.1 else "⚠️  Potential overfitting/underfitting"
                print(f"\n📈 Train vs Validation Comparison:")
                print(f"   Target Metric ({args.optimize_target}):")
                print(f"      Train: {train_score:.2%} | Test: {val_score:.2%} | Gap: {gap:+.2%} {status}")

            # Store validation score for model saving
            validation_score = combined_wr_val if args.optimize_target == 'combined_wr' else (buy_wr_val if args.optimize_target == 'buy_wr' else sell_wr_val)
        else:
            validation_score = None

    else:
        validation_score = None

    # Final Evaluation & Output on FULL dataset (unless in real-time mode)
    if args.verbose: print(f"\n⚙️  Running final evaluation with optimized params: {params}")
    buy_wr, sell_wr, combined_wr, buy_density, sell_density, eval_buy, eval_sell, buy_wins, sell_wins, df_final = \
        run_strategy_and_evaluate(df_base, args, close_col, high_col, low_col, volume_col, open_col, **params)

    total_bars = len(df_final.dropna(subset=[close_col]))

    if args.verbose:
        yearly_stats = calculate_yearly_win_rates_vectorized(df_final, args, close_col, high_col, low_col)
        if args.sanity_check:
            yearly_stats2 = calculate_yearly_win_rates(df_final, args, close_col, high_col, low_col)
            print(f"{yearly_stats}  vs  {yearly_stats2}")
        # print_yearly_stats(yearly_stats, args.ticker)

        print(f"\n📊 {args.ticker} | Valid Bars: {total_bars}")
        if args.optimize_target in ['buy_wr', 'combined_wr']:
            print(f"🟢 Buy Signals: {int(df_final[('Signal_Buy', args.ticker)].sum())} (Density: {buy_density:.4f}) | Evaluated: {eval_buy} | Wins: {buy_wins} | Win Rate: {buy_wr:.2%}\n"
                  f"\t Consider: Put Credit Spread\n"
                  f"\t Short Put Strike ≈ $latest_close * {args.put_strike_pct:.6f}")
        if args.optimize_target in ['sell_wr', 'combined_wr']:
            print(f"🔴 Sell Signals: {int(df_final[('Signal_Sell', args.ticker)].sum())} (Density: {sell_density:.4f}) | Evaluated: {eval_sell} | Wins: {sell_wins} | Win Rate: {sell_wr:.2%}\n"
                  f"\t Consider: Call Credit Spread\n"
                  f"\t Short Call Strike ≈ $latest_close * {args.call_strike_pct:.6f}")
        if args.optimize_target in ['combined_wr']:
            print(f"🎯 Combined Win Rate: {combined_wr:.2%} ({buy_wins + sell_wins}/{eval_buy + eval_sell})")
        if args.optimize_target in ['buy_wr', 'combined_wr']:
            print(f"🎯   Buy Win Rate : {buy_wr:.2%} ({buy_wins}/{eval_buy})")
        if args.optimize_target in ['sell_wr', 'combined_wr']:
            print(f"🎯   Sell Win Rate: {sell_wr:.2%} ({sell_wins}/{eval_sell})")
        print(f"📉 Min Density Threshold: {args.min_signal_density:.2%}")
        if buy_density < args.min_signal_density and args.optimize_target in ['buy_wr', 'combined_wr']: print("⚠️  Buy signal density below threshold.")
        if sell_density < args.min_signal_density and args.optimize_target in ['sell_wr', 'combined_wr']: print("⚠️  Sell signal density below threshold.")
    # 🔹 Save model with descriptive name including params and score
    if args.optimize or args.model_path is None:  # Only save new model if we optimized or didn't load one
        score = combined_wr if args.optimize_target == 'combined_wr' else (buy_wr if args.optimize_target == 'buy_wr' else sell_wr)
        saved_path = save_model(
            params=params,
            score=score,
            args=args,
            validation_score=validation_score,
            train_val_split=train_val_split_info, command_line=command_line
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot:
        fib_50_col = ('Fib_50', args.ticker)
        fib_618_col = ('Fib_618', args.ticker)
        sma_50_col = ('SMA_50', args.ticker)
        buy_sig_col = ('Signal_Buy', args.ticker)
        sell_sig_col = ('Signal_Sell', args.ticker)
        rsi_col = ('RSI', args.ticker)
        plot_results(
            df_final, args, params, close_col, high_col, low_col, rsi_col, sma_50_col,
            fib_50_col, fib_618_col, buy_sig_col, sell_sig_col,
            ('Regular_Bullish_Div', args.ticker), ('Regular_Bearish_Div', args.ticker),
            ('Hidden_Bullish_Div', args.ticker), ('Hidden_Bearish_Div', args.ticker),
            params.get('rsi_midline', 50), params.get('rsi_oversold', 30), params.get('rsi_overbought', 70)
        )

    return df_final


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)