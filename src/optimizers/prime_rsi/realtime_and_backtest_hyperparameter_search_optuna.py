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
from utils import get_filename_for_dataset, get_next_step
import pickle
import argparse
import os
import optuna
import json
from datetime import datetime

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
#   • RSI Strategies      : Pullback-to-50, EMA crossovers, MA confluence
#   • Fibonacci Confluence: Golden Zone (0.5–0.618) retracement entries
#   • Divergence Detection: Regular & Hidden Bullish/Bearish divergences
#   • Signal Aggregation  : Logical OR across all bullish/bearish conditions
#
# 🎯 EVALUATION LOGIC:
#   • BUY Signal  → Price must stay ABOVE put_strike_pct (e.g., 0.96×)
#                   within lookahead_bars → simulates profitable put credit spread
#   • SELL Signal → Price must stay BELOW call_strike_pct (e.g., 1.04×)
#                   within lookahead_bars → simulates profitable call credit spread
#
# ⚙️ OPTIMIZATION (Optuna):
#   • Hyperparameters: RSI length, SMA period, Fib lookback, divergence window
#   • Objective: Maximize Win Rate (buy/sell/combined) with density penalty
#   • Persistence: SQLite-backed studies for resumable, distributed optimization
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
def calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, ticker, lookback=50):
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
                                   (df[rsi_col].shift(1) < 50) & \
                                   (df[rsi_col] > 50)
    return df, strategy_Fib_RSI_Buy_col


def implement_rsi_strategies(df, close_col, ticker, rsi_length=14, rsi_ema_10=10, sma_50=50):
    rsi_col = ('RSI', ticker)
    df[rsi_col] = safe_ta_indicator(ta.rsi, df[close_col], length=rsi_length)

    rsi_ema_10_col = ('RSI_EMA_10', ticker)
    df[rsi_ema_10_col] = safe_ta_indicator(ta.ema, df[rsi_col], length=rsi_ema_10)

    sma_50_col = ('SMA_50', ticker)
    df[sma_50_col] = safe_ta_indicator(ta.sma, df[close_col], length=sma_50)

    Setup_Pullback_50_Buy_col = ('Setup_Pullback_50_Buy', ticker)
    df[Setup_Pullback_50_Buy_col] = (df[close_col] > df[sma_50_col]) & \
                                    (df[rsi_col].shift(1) < 50) & \
                                    (df[rsi_col] > 50)

    Setup_Pullback_50_Sell_col = ('Setup_Pullback_50_Sell', ticker)
    df[Setup_Pullback_50_Sell_col] = (df[close_col] < df[sma_50_col]) & \
                                     (df[rsi_col].shift(1) > 50) & \
                                     (df[rsi_col] < 50)

    Setup_EMA_Cross_Buy_col = ('Setup_EMA_Cross_Buy', ticker)
    df[Setup_EMA_Cross_Buy_col] = (df[rsi_col].shift(1) < df[rsi_ema_10_col].shift(1)) & \
                                  (df[rsi_col] > df[rsi_ema_10_col]) & \
                                  (df[rsi_col] < 30)

    Setup_EMA_Cross_Sell_col = ('Setup_EMA_Cross_Sell', ticker)
    df[Setup_EMA_Cross_Sell_col] = (df[rsi_col].shift(1) > df[rsi_ema_10_col].shift(1)) & \
                                   (df[rsi_col] < df[rsi_ema_10_col]) & \
                                   (df[rsi_col] > 70)

    Strategy_MA_Confluence_Buy_col = ('Strategy_MA_Confluence_Buy', ticker)
    df[Strategy_MA_Confluence_Buy_col] = (df[close_col] > df[sma_50_col]) & \
                                         (df[sma_50_col] > df[sma_50_col].shift(1)) & \
                                         (df[rsi_col] > df[rsi_ema_10_col]) & \
                                         (df[rsi_col].shift(1) < df[rsi_ema_10_col].shift(1))

    return df, rsi_col, Setup_Pullback_50_Buy_col, Setup_Pullback_50_Sell_col, Setup_EMA_Cross_Buy_col, Setup_EMA_Cross_Sell_col, Strategy_MA_Confluence_Buy_col


def find_divergences(df, high_col, low_col, rsi_col, ticker, window=5):
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
                future_highs = high_prices[future_idx]
                buy_success = np.any(future_highs > strikes[:, None], axis=1)

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
                future_lows = low_prices[future_idx]
                sell_success = np.any(future_lows < strikes[:, None], axis=1)

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
        success = future_df[close_col].iloc[-1] > strike if method == "final_close" else (future_df[high_col] > strike).any()
        if success: buy_wins += 1

    for idx in sell_indices:
        pos = idx_to_pos[idx]
        if pos + 1 + lookahead > n_rows: continue
        total_sell += 1
        price = df[close_col].iloc[pos]
        strike = price * _args.call_strike_pct
        future_df = df.iloc[pos + 1: pos + 1 + lookahead]
        success = future_df[close_col].iloc[-1] < strike if method == "final_close" else (future_df[low_col] < strike).any()
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
                    future_highs = high_prices[future_idx]
                    success = np.any(future_highs > strikes[:, None], axis=1)

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
                    future_lows = low_prices[future_idx]
                    success = np.any(future_lows < strikes[:, None], axis=1)

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
            success = future_df[close_col].iloc[-1] > strike if method == "final_close" else (future_df[high_col] > strike).any()
            if success: buy_wins += 1

        for idx in sell_indices:
            pos = idx_to_pos[idx]
            if pos + 1 + lookahead > n_rows: continue
            total_sell += 1
            price = df[close_col].iloc[pos]
            strike = price * args.call_strike_pct
            future_df = df.iloc[pos + 1: pos + 1 + lookahead]
            success = future_df[close_col].iloc[-1] < strike if method == "final_close" else (future_df[low_col] < strike).any()
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
        print(f"   🎯 Combined: {total_buy_wins + total_sell_wins:,} wins / {total_buy + total_sell:,} signals → {overall_combined:6.2f}%")
        print(f"{'=' * 80}\n")


def plot_results(df, args, close_col, high_col, low_col, rsi_col, sma_50_col,
                 fib_50_col, fib_618_col, buy_sig_col, sell_sig_col,
                 reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
    except ImportError:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        return

    plot_bars = min(200, len(df))
    df_plot = df.iloc[-plot_bars:].copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    ax1.plot(df_plot.index, df_plot[close_col], label='Close', color='blue', linewidth=1)
    if sma_50_col in df_plot.columns:
        ax1.plot(df_plot.index, df_plot[sma_50_col], label='SMA 50', color='orange', linewidth=1)
    if fib_50_col in df_plot.columns and fib_618_col in df_plot.columns:
        ax1.fill_between(df_plot.index, df_plot[fib_618_col], df_plot[fib_50_col],
                         color='gold', alpha=0.25, label='Fib Golden Zone (0.5-0.618)')

    buy_signals = df_plot[buy_sig_col] & df_plot[close_col].notna()
    if buy_signals.any():
        ax1.scatter(df_plot.index[buy_signals], df_plot[close_col][buy_signals],
                    marker='^', color='green', s=120, label='Buy Signal', zorder=5, edgecolors='white')
    sell_signals = df_plot[sell_sig_col] & df_plot[close_col].notna()
    if sell_signals.any():
        ax1.scatter(df_plot.index[sell_signals], df_plot[close_col][sell_signals],
                    marker='v', color='red', s=120, label='Sell Signal', zorder=5, edgecolors='white')

    if reg_bull_div_col in df_plot.columns:
        bull_div = df_plot[reg_bull_div_col] & df_plot[low_col].notna()
        if bull_div.any(): ax1.scatter(df_plot.index[bull_div], df_plot[low_col][bull_div], marker='*', color='lime', s=200, label='Reg. Bullish Div', zorder=6, edgecolors='black')
    if reg_bear_div_col in df_plot.columns:
        bear_div = df_plot[reg_bear_div_col] & df_plot[high_col].notna()
        if bear_div.any(): ax1.scatter(df_plot.index[bear_div], df_plot[high_col][bear_div], marker='*', color='magenta', s=200, label='Reg. Bearish Div', zorder=6, edgecolors='black')
    if hid_bull_div_col in df_plot.columns:
        hbull_div = df_plot[hid_bull_div_col] & df_plot[low_col].notna()
        if hbull_div.any(): ax1.scatter(df_plot.index[hbull_div], df_plot[low_col][hbull_div], marker='P', color='cyan', s=100, label='Hidden Bullish Div', zorder=6, edgecolors='black')
    if hid_bear_div_col in df_plot.columns:
        hbear_div = df_plot[hid_bear_div_col] & df_plot[high_col].notna()
        if hbear_div.any(): ax1.scatter(df_plot.index[hbear_div], df_plot[high_col][hbear_div], marker='P', color='yellow', s=100, label='Hidden Bearish Div', zorder=6, edgecolors='black')

    ax1.set_ylabel('Price')
    ax1.set_title(f'{args.ticker} - Strategy Signals & Fibonacci Confluence (Last {plot_bars} bars)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

    if rsi_col in df_plot.columns:
        ax2.plot(df_plot.index, df_plot[rsi_col], label='RSI(14)', color='purple', linewidth=1)
        rsi_ema_10_col = ('RSI_EMA_10', args.ticker)
        if rsi_ema_10_col in df_plot.columns:
            ax2.plot(df_plot.index, df_plot[rsi_ema_10_col], label='RSI EMA(10)', color='cyan', linewidth=0.8)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.3, label='Midline (50)')
        if rsi_ema_10_col in df_plot.columns:
            crossover_buy = (df_plot[rsi_col].shift(1) < df_plot[rsi_ema_10_col].shift(1)) & (df_plot[rsi_col] > df_plot[rsi_ema_10_col]) & (df_plot[rsi_col] < 30)
            crossover_sell = (df_plot[rsi_col].shift(1) > df_plot[rsi_ema_10_col].shift(1)) & (df_plot[rsi_col] < df_plot[rsi_ema_10_col]) & (df_plot[rsi_col] > 70)
            if crossover_buy.any(): ax2.scatter(df_plot.index[crossover_buy], df_plot[rsi_col][crossover_buy], marker='^', color='green', s=100, zorder=5, edgecolors='white')
            if crossover_sell.any(): ax2.scatter(df_plot.index[crossover_sell], df_plot[rsi_col][crossover_sell], marker='v', color='red', s=100, zorder=5, edgecolors='white')

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
    if hasattr(args, 'lookahead_bars'):
        arg_parts.append(f"lookahead-{args.lookahead_bars}")
    if hasattr(args, 'method'):
        arg_parts.append(f"m-{args.method}")
    if hasattr(args, 'put_strike_pct'):
        arg_parts.append(f"put-{args.put_strike_pct:.6f}")
    if hasattr(args, 'call_strike_pct'):
        arg_parts.append(f"call-{args.call_strike_pct:.6f}")
    if hasattr(args, 'train_ratio') and args.train_ratio < 1.0:
        arg_parts.append(f"train_ratio-{args.train_ratio:.2f}")

    # Combine all parts
    name_parts = [
        args.ticker.replace('^', ''),
        args.dataset_id,
        args.optimize_target,
        f"score-{score:.12f}",
        '_'.join(param_parts),
        '_'.join(arg_parts),
        timestamp
    ]

    # Filter out empty parts and join
    name_parts = [p for p in name_parts if p]
    filename = '_'.join(name_parts) + '.pkl'

    # Sanitize filename (remove any problematic characters)
    filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')

    return filename


def save_model(params, score, args, df_final=None, validation_score=None, train_val_split=None):
    """
    Save model parameters and metadata to a file with descriptive name.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = generate_model_name(args, params, score)
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
        'final_df': df_final  # Optional: include final dataframe if needed
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if args.verbose:
        print(f"💾 Model saved to: {model_path}")
        if validation_score is not None:
            print(f"   📊 Validation Score: {validation_score:.4f}")

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
def run_strategy_on_latest(df_base, params, _args, close_col, high_col, low_col):
    """
    Run the strategy on the latest datapoint to check for signals.
    Returns dict with signal info.
    """
    df = df_base.copy()

    # Apply strategy functions with loaded params
    df, rsi_col, pullback_buy_col, pullback_sell_col, ema_cross_buy_col, ema_cross_sell_col, ma_conf_buy_col = \
        implement_rsi_strategies(df, close_col, _args.ticker,
                                 params['rsi_length'], params['rsi_signal_len'], params['sma_len'])
    df, reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col = \
        find_divergences(df, high_col, low_col, rsi_col, _args.ticker, params['div_window'])
    df, fib_rsi_buy_col = calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, _args.ticker, params['fib_lookback'])

    buy_cols = [pullback_buy_col, ema_cross_buy_col, ma_conf_buy_col, fib_rsi_buy_col, reg_bull_div_col, hid_bull_div_col]
    sell_cols = [pullback_sell_col, ema_cross_sell_col, reg_bear_div_col, hid_bear_div_col]
    df[('Signal_Buy', _args.ticker)] = df[buy_cols].fillna(False).any(axis=1)
    df[('Signal_Sell', _args.ticker)] = df[sell_cols].fillna(False).any(axis=1)

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
            'pullback_sell': bool(df.loc[latest_idx, pullback_sell_col]) if pullback_sell_col in df.columns else False,
            'ema_cross_sell': bool(df.loc[latest_idx, ema_cross_sell_col]) if ema_cross_sell_col in df.columns else False,
            'bearish_div': bool(df.loc[latest_idx, reg_bear_div_col]) if reg_bear_div_col in df.columns else False,
            'hidden_bear_div': bool(df.loc[latest_idx, hid_bear_div_col]) if hid_bear_div_col in df.columns else False,
        }
    }


def real_time_mode(args, close_col, high_col, low_col):
    """
    Real-time mode: load model from path and test latest datapoint for signals.
    """
    if not args.model_path:
        print("❌ Error: --model-path is required for real-time mode")
        return None

    # Load the model
    print(f"🔍 Loading model from: {args.model_path}")
    model_data = load_model(args.model_path)
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

    _dataset_id = model_data['args']['dataset_id']
    _ticker     = model_data['args']['ticker']
    _cache_filename = get_filename_for_dataset(model_data['args']['dataset_id'], older_dataset=None)
    with open(_cache_filename, 'rb') as f:
        _master_data_cache = pickle.load(f)
    df_base = _master_data_cache[model_data['args']['ticker']].sort_index()
    if args.clip:
        df_base = df_base.iloc[:-1].copy()
    if args.verbose: print(f"📂 Dataset ranging from {df_base.index[0].strftime('%Y-%m-%d')} to {df_base.index[-1].strftime('%Y-%m-%d')}")

    train_ratio = model_data['train_val_split']['train_ratio']
    train_bars = model_data['train_val_split']['train_bars']
    val_bars = model_data['train_val_split']['val_bars']
    train_range = model_data['train_val_split']['train_range']
    val_range = model_data['train_val_split']['val_range']
    if args.verbose:
        print(f"📊 Loaded model with training score: {train_score:.4%}")
        print(f"📊 Validation score: {val_score:.4%}")
        print(f"🧠 Parameters: {params}")
        print(f"🧠 Ratio: {train_ratio} | {train_bars} Train Bars ({train_range}) | {val_bars} Val Bars ({val_range}) | Method: {method} | Optimize Target: {optimize_target} | Minimum Signal Density: {min_signal_density:.2%}")
    # Run strategy on latest datapoint
    print(f"\n⚡ Testing latest datapoint ({df_base.index[-1].strftime('%Y-%m-%d')}) for {_ticker} | Dataset {_dataset_id} | Lookahead: {lookahead} bars")
    result = run_strategy_on_latest(df_base=df_base, params=params, _args=args, close_col=close_col, high_col=high_col, low_col=low_col)

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
            run_strategy_and_evaluate(df_train, eval_args, close_col, high_col, low_col, **params)

        # Evaluate on the validation set
        buy_wr_val, sell_wr_val, combined_wr_val, _, _, _, _, _, _, _ = \
            run_strategy_and_evaluate(df_val, eval_args, close_col, high_col, low_col, **params)

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
            print(f"📈 Recomputed Val Win Rate  : {val_win_rate:.2%}")

    # Output results
    print(f"\n{'=' * 60}")
    print(f"🔔 REAL-TIME SIGNAL CHECK — {_ticker}")
    print(f"{'=' * 60}")
    assert df_base.index[-1].strftime('%Y-%m-%d') == result['timestamp'].strftime('%Y-%m-%d')
    print(f"📅 Last Timestamp: {result['timestamp'].strftime('%Y-%m-%d')}")
    current_price, target_price, target_date = result['close'], None, None
    assert df_base[close_col].iloc[-1] == current_price
    print(f"💰 Last Close Price: ${current_price:.2f}")
    buy_signal_detected   = result['buy_signal'] and optimize_target in ['combined_wr', 'buy_wr']
    sell_signal_detected  = result['sell_signal'] and optimize_target in ['combined_wr', 'sell_wr']
    result['buy_signal_detected']  = buy_signal_detected
    result['sell_signal_detected'] = sell_signal_detected
    if buy_signal_detected:
        print(f"\n🎯 SIGNALS:")
        print(f"   🟢 BUY SIGNAL DETECTED! | Put Threshold: {put_strike_pct:.2%} | @{lookahead} {_dataset_id}")
    if sell_signal_detected:
        print(f"\n🎯 SIGNALS:")
        print(f"   🔴 SELL SIGNAL DETECTED! | Call Threshold: {call_strike_pct:.2%} | @{lookahead} {_dataset_id}")
    if not buy_signal_detected and not sell_signal_detected:
        print(f"   ⚪ No signal at this time")

    if args.verbose and not args.verbose_short:
        print(f"\n🔍 Individual Signal Components:")
        for name, active in result['individual_signals'].items():
            status = "✅" if active else "❌"
            print(f"   {status} {name}: {active}")

    print(f"{'=' * 60}\n")

    # Calculate approximate target/expiration date based on lookahead bars
    entry_date = result['timestamp'].strftime('%Y-%m-%d')
    target_date = get_next_step(the_date=entry_date, dataset_id=_dataset_id, nn=lookahead).strftime('%Y-%m-%d')

    # ==============================================================================
    # 💡 RECOMMENDED TRADE OUTPUT (if signal detected)
    # ==============================================================================
    if buy_signal_detected or sell_signal_detected:
        entry_price = result['close']
        assert entry_price == current_price

        print(f"\n💡 RECOMMENDED OPTIONS TRADE:")
        print(f"{'─' * 60}")

        if buy_signal_detected:
            strike_price = entry_price * put_strike_pct
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
            print(f"   📊 Strategy  : Call Credit Spread")
            print(f"   📅 Entry Date: {entry_date}")
            print(f"   💰 Entry Price: ${entry_price:.2f}")
            print(f"   🎯 Short Call Strike: ${strike_price:.2f} ({call_strike_pct:.2%} of entry)")
            print(f"   📅 Target/Expiration: ~{target_date} ({lookahead} bars)")
            print(f"   ✅ Win Condition: Price stays BELOW ${strike_price:.2f}")
            print(f"   💡 Premium: Sell OTM call spread above current price")
            target_price = strike_price

        if buy_signal_detected and sell_signal_detected:
            print(f"\n   ⚠️  BOTH SIGNALS DETECTED - Review confluence carefully")

        print(f"{'─' * 60}\n")

    result['train_score']     = train_score
    result['val_score']       = val_score
    result['train_win_rate']  = train_win_rate
    result['val_win_rate']    = val_win_rate
    result['optimize_target'] = optimize_target
    result['current_price']   = current_price
    result['current_date']    = entry_date
    result['target_price']    = target_price
    result['target_date']     = target_date
    result['dataset_id']      = _dataset_id
    result['ticker']          = _ticker
    result['lookahead']       = lookahead
    result['method']          = method
    result['put_strike_pct']  = put_strike_pct
    result['call_strike_pct'] = call_strike_pct
    return result


# ==============================================================================
# OPTUNA INTEGRATION
# ==============================================================================
def run_strategy_and_evaluate(df_base, _args, close_col, high_col, low_col, rsi_length, rsi_signal_len, sma_len, fib_lookback, div_window):
    df = df_base.copy()
    df, rsi_col, pullback_buy_col, pullback_sell_col, ema_cross_buy_col, ema_cross_sell_col, ma_conf_buy_col = \
        implement_rsi_strategies(df, close_col, _args.ticker, rsi_length, rsi_signal_len, sma_len)
    df, reg_bull_div_col, reg_bear_div_col, hid_bull_div_col, hid_bear_div_col = \
        find_divergences(df, high_col, low_col, rsi_col, _args.ticker, div_window)
    df, fib_rsi_buy_col = calculate_fibonacci_confluence(df, close_col, high_col, low_col, rsi_col, _args.ticker, fib_lookback)

    buy_cols = [pullback_buy_col, ema_cross_buy_col, ma_conf_buy_col, fib_rsi_buy_col, reg_bull_div_col, hid_bull_div_col]
    sell_cols = [pullback_sell_col, ema_cross_sell_col, reg_bear_div_col, hid_bear_div_col]
    df[('Signal_Buy', _args.ticker)] = df[buy_cols].fillna(False).any(axis=1)
    df[('Signal_Sell', _args.ticker)] = df[sell_cols].fillna(False).any(axis=1)

    buy_wr, sell_wr, combined_wr, buy_wins, sell_wins, total_buy, total_sell = calculate_win_rates_vectorized(df=df, _args=_args, close_col=close_col, high_col=high_col, low_col=low_col)
    if _args.sanity_check:
        buy_wr2, sell_wr2, combined_wr2, buy_wins2, sell_wins2, total_buy2, total_sell2 = calculate_win_rates(df=df, _args=_args, close_col=close_col, high_col=high_col, low_col=low_col)
        assert np.allclose(buy_wr, buy_wr2)
        assert np.allclose(sell_wr, sell_wr2)
        assert np.allclose(combined_wr, combined_wr2)
        assert np.allclose(buy_wins, buy_wins2)
        assert np.allclose(sell_wins, sell_wins2)
    eval_buy, eval_sell = total_buy, total_sell
    total_bars = len(df.dropna(subset=[close_col]))
    buy_density = int(df[('Signal_Buy', _args.ticker)].sum()) / total_bars if total_bars > 0 else 0
    sell_density = int(df[('Signal_Sell', _args.ticker)].sum()) / total_bars if total_bars > 0 else 0

    return buy_wr, sell_wr, combined_wr, buy_density, sell_density, eval_buy, eval_sell, buy_wins, sell_wins, df


def optuna_objective(trial, _args, df_base, close_col, high_col, low_col):
    rsi_length = trial.suggest_int('rsi_length', 5, 30)
    rsi_signal_len = trial.suggest_int('rsi_signal_len', 2, 60)
    sma_len = trial.suggest_int('sma_len', 10, 200)
    fib_lookback = trial.suggest_int('fib_lookback', 2, 200)
    div_window = trial.suggest_int('div_window', 3, 20)
    try:
        buy_wr, sell_wr, combined_wr, buy_density, sell_density, eval_buy, eval_sell, buy_wins, sell_wins, df = \
            run_strategy_and_evaluate(df_base=df_base, _args=_args, close_col=close_col, high_col=high_col, low_col=low_col, rsi_length=rsi_length,
                                      rsi_signal_len=rsi_signal_len, sma_len=sma_len, fib_lookback=fib_lookback, div_window=div_window)
    except Exception as ee:
        print(ee)
        return -1.0  # Discard invalid trials

    if _args.optimize_target == 'buy_wr':
        target = buy_wr
        density = buy_density
    elif _args.optimize_target == 'sell_wr':
        target = sell_wr
        density = sell_density
    else:
        target = combined_wr
        density = min(buy_density, sell_density)

    # Density penalty
    if density < _args.min_signal_density:
        return target - _args.density_penalty

    return target


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strategy Optimizer & Real-Time Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    data_group = parser.add_argument_group('Data & Symbol')
    data_group.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    data_group.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    data_group.add_argument('--length-dataset', type=int, default=999999, help='Trailing data points')
    data_group.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")

    strat_group = parser.add_argument_group('Strategy & P&L Parameters')
    strat_group.add_argument('--lookahead-bars', type=int, default=20, dest='lookahead_bars', help='Forward-looking window')
    strat_group.add_argument('--method', type=str, default='final_close', choices=['touched', 'final_close'], help='Strike evaluation method')
    strat_group.add_argument('--min-signal-density', type=float, default=0.04, help='Min signal frequency threshold')
    strat_group.add_argument('--put-strike-pct', type=float, default=0.96, help='Base put strike multiplier')
    strat_group.add_argument('--call-strike-pct', type=float, default=1.04, help='Base call strike multiplier')
    strat_group.add_argument('--wr-weight', type=float, default=0.9, help='Weight for Win-Rate')
    strat_group.add_argument('--td-weight', type=float, default=0.1, help='Weight for Trade-Density')

    opt_group = parser.add_argument_group('Optimization & Execution')
    opt_group.add_argument('--optimize', action='store_true', help='Run Optuna hyperparameter optimization')
    opt_group.add_argument('--optimize-target', type=str, default='combined_wr', choices=['combined_wr', 'buy_wr', 'sell_wr'],
                           help='Metric to maximize during optimization')
    opt_group.add_argument('--density-penalty', type=float, default=0.5,
                           help='Penalty subtracted from objective if density < min-signal-density')
    opt_group.add_argument('--n-trials', type=int, default=100, help='Optuna trials per run')
    opt_group.add_argument('--timeout', type=int, default=3600, help='Max runtime (seconds)')
    opt_group.add_argument('--output-dir', type=str, default='models', help='Output directory')
    opt_group.add_argument('--optuna-db', type=str, default=None, help='SQLite path for Optuna persistence.')
    opt_group.add_argument('--train-ratio', type=float, default=0.7,
                           help='Ratio of data to use for training (rest for validation). Use 1.0 to disable split.')

    flag_group = parser.add_argument_group('Execution Flags')
    flag_group.add_argument('--real-time', action=argparse.BooleanOptionalAction, default=False,
                            help='Real-time mode: test latest datapoint with specified model')
    flag_group.add_argument('--model-path', type=str, default=None,
                            help='Path to saved model .pkl file (required for real-time mode, optional for evaluation)')
    flag_group.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')
    flag_group.add_argument('--verbose-short', action=argparse.BooleanOptionalAction, default=False, help='Short real-time output')
    flag_group.add_argument('--seed', type=int, default=123, help='Random seed')
    flag_group.add_argument('--plot', action='store_true', default=False, help='Plot results with matplotlib')
    flag_group.add_argument('--sanity-check', action='store_true', default=False, help='Check vectorized implementation consistency')

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
║  🔹 Lookahead    : {args.lookahead_bars:02d} bars{' ' * 51}║
║  🔹 Method       : {args.method:<58}║
║  🔹 Min Density  : {args.min_signal_density:<58}║
║  🔹 Strike Pct   : Put {args.put_strike_pct:.2%} | Call {args.call_strike_pct:.2%}{' ' * 33}║{train_info if train_info else ' ' * 78}║
╚{'═' * 78}╝
"""
    print(banner)


def early_stop_on_perfect_success(study, trial):
    """
    Callback to stop optimization when 100% success rate is achieved.
    Objective returns -success_rate, so -100.0 = 100% success.
    """
    if study.best_value is not None and study.best_value >= 0.999:
        print(f"\n🎯 100% success rate achieved at trial #{trial.number}! Stopping optimization early...")
        study.stop()


def entry(args):
    np.random.seed(args.seed)
    close_col = ('Close', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)
    if not args.real_time:
        cache_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
        if args.verbose: print(f"📂 Loading dataset from: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            master_data_cache = pickle.load(f)
        if args.ticker not in master_data_cache:
            raise KeyError(f"Ticker '{args.ticker}' not found in cache. Available: {list(master_data_cache.keys())}")
        df_base = master_data_cache[args.ticker].sort_index()
        if args.length_dataset != 999999:
            df_base = df_base.iloc[-args.length_dataset:].copy()
        if args.clip and args.real_time:
            df_base = df_base.iloc[:-1].copy()
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
        return real_time_mode(args, close_col, high_col, low_col)
    assert args.put_strike_pct > 0.89 and args.call_strike_pct < 1.11, f"Just to make sure one does not use 0.05 instead 0.95 , for example."
    print_startup_banner(args)
    # Default params (used if not optimizing)
    params = {'rsi_length': 14, 'rsi_signal_len': 10, 'sma_len': 50, 'fib_lookback': 50, 'div_window': 5}

    # 🔹 Load model from path if specified (for evaluation without optimization)
    if args.model_path and not args.optimize:
        print(f"🔍 Loading model from: {args.model_path}")
        model_data = load_model(args.model_path)
        params = model_data['params']
        if args.verbose:
            print(f"📊 Loaded model with score: {model_data.get('score', 'N/A')}")
            if model_data.get('validation_score') is not None:
                print(f"📊 Validation score: {model_data['validation_score']:.4f}")
            print(f"🧠 Parameters: {params}")

    # ==============================================================================
    # 🔄 TRAIN/VALIDATION SPLIT (for optimization only)
    # ==============================================================================
    df_train = df_base
    df_val = None
    train_val_split_info = None

    if args.optimize and hasattr(args, 'train_ratio') and args.train_ratio < 1.0:
        # Chronological split to avoid look-ahead bias (critical for time-series)
        split_idx = int(len(df_base) * args.train_ratio)

        # Ensure minimum data in each split
        min_bars = 1  # Adjust based on your strategy's warmup needs
        if split_idx < min_bars:
            print(f"⚠️  Train split too small ({split_idx} < {min_bars}), using full dataset")
        elif len(df_base) - split_idx < min_bars:
            print(f"⚠️  Validation split too small ({len(df_base) - split_idx} < {min_bars}), using full dataset")
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
                      f"Val={len(df_val)} bars ({(1 - args.train_ratio) * 100:.1f}%)")
                print(f"   📅 Train: {train_val_split_info['train_range'][0]} → {train_val_split_info['train_range'][1]}")
                print(f"   📅 Val:   {train_val_split_info['val_range'][0]} → {train_val_split_info['val_range'][1]}")

    if args.optimize:
        # ==============================================================================
        # 🔧 OPTUNA STORAGE SETUP: In-memory if --optuna-db is None, SQLite otherwise
        # ==============================================================================
        if args.optuna_db is None:
            # Use in-memory storage (no persistence across runs)
            storage = None
            db_path_display = "in-memory (no persistence)"
        else:
            # Use SQLite persistence with the specified path
            db_path = args.optuna_db
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            storage = f"sqlite:///{db_path}"
            db_path_display = db_path
        study_name = (
            f"{args.ticker}_"            
            f"{args.dataset_id}_"
            f"{args.optimize_target}_"
            f"la{args.lookahead_bars}_"
            f"{args.method}"
        )
        print(f"\n🔍 Initializing Optuna study: {study_name}")
        print(f"   📂 Storage: {db_path_display}")
        if df_val is not None:
            print(f"   🔄 Optimizing on TRAINING set only (validation held out)")

        # Create or load the study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,  # Safe: does nothing for in-memory, loads for SQLite
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
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
            print("📦 No stored trials found. Initializing fresh optimization run.\n")

        # ==============================================================================
        # 🚀 RUN OPTIMIZATION
        # ==============================================================================
        study.optimize(
            lambda trial: optuna_objective(
                trial=trial,
                _args=args,
                df_base=df_train,  # ← Use training set for optimization
                close_col=close_col,
                high_col=high_col,
                low_col=low_col
            ),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=args.verbose,
            callbacks=[early_stop_on_perfect_success],
        )

        best_trial = study.best_trial
        print(f"\n✅ Optimization Complete!")
        print(f"   🏆 Best {args.optimize_target}: {best_trial.value:.4f}")
        print("   🧠 Best Hyperparameters:")
        for k, v in best_trial.params.items():
            print(f"      {k}: {v}")
            params[k] = v

        # ==============================================================================
        # 🎯 VALIDATION SET EVALUATION (if split was used)
        # ==============================================================================
        if df_val is not None and len(df_val) > args.lookahead_bars:
            if args.verbose:
                print(f"\n{'=' * 80}")
                print(f"🔍 EVALUATING BEST PARAMETERS ON VALIDATION SET")
                print(f"{'=' * 80}")

            # Evaluate on validation set with best params
            buy_wr_val, sell_wr_val, combined_wr_val, _, _, eval_buy_val, eval_sell_val, buy_wins_val, sell_wins_val, df_val_final = \
                run_strategy_and_evaluate(
                    df_base=df_val,
                    _args=args,
                    close_col=close_col,
                    high_col=high_col,
                    low_col=low_col,
                    **params
                )

            if args.verbose:
                print(f"📊 Validation Results ({len(df_val)} bars):")
                print(f"   🟢 Buy:  {buy_wins_val}/{eval_buy_val} → {buy_wr_val:6.2f}%")
                print(f"   🔴 Sell: {sell_wins_val}/{eval_sell_val} → {sell_wr_val:6.2f}%")
                print(f"   🎯 Combined: {(buy_wins_val + sell_wins_val)}/{(eval_buy_val + eval_sell_val)} → {combined_wr_val:6.2f}%")

                # Compare train vs val if both available
                train_score = combined_wr_val if args.optimize_target == 'combined_wr' else (buy_wr_val if args.optimize_target == 'buy_wr' else sell_wr_val)
                val_score = combined_wr_val if args.optimize_target == 'combined_wr' else (buy_wr_val if args.optimize_target == 'buy_wr' else sell_wr_val)
                # Actually get training score from best trial
                train_score = best_trial.value
                gap = train_score - val_score
                status = "✅ Good generalization" if abs(gap) < 0.1 else "⚠️  Potential overfitting"
                print(f"\n📈 Train vs Validation Comparison:")
                print(f"   Target Metric ({args.optimize_target}):")
                print(f"      Train: {train_score:.2%} | Val: {val_score:.2%} | Gap: {gap:+.2%} {status}")

            # Store validation score for model saving
            validation_score = combined_wr_val if args.optimize_target == 'combined_wr' else (buy_wr_val if args.optimize_target == 'buy_wr' else sell_wr_val)
        else:
            validation_score = None

    else:
        validation_score = None

    # Final Evaluation & Output on FULL dataset (unless in real-time mode)
    print(f"\n⚙️  Running final evaluation with params: {params}")
    buy_wr, sell_wr, combined_wr, buy_density, sell_density, eval_buy, eval_sell, buy_wins, sell_wins, df_final = \
        run_strategy_and_evaluate(df_base, args, close_col, high_col, low_col, **params)

    total_bars = len(df_final.dropna(subset=[close_col]))

    if args.verbose:
        yearly_stats = calculate_yearly_win_rates_vectorized(df_final, args, close_col, high_col, low_col)
        if args.sanity_check:
            yearly_stats2 = calculate_yearly_win_rates(df_final, args, close_col, high_col, low_col)
            print(f"{yearly_stats}  vs  {yearly_stats2}")
        print_yearly_stats(yearly_stats, args.ticker)

        print(f"\n📊 {args.ticker} | Valid Bars: {total_bars}")
        print(f"🟢 Buy Signals: {int(df_final[('Signal_Buy', args.ticker)].sum())} (Density: {buy_density:.4f}) | Evaluated: {eval_buy} | Wins: {buy_wins} | Win Rate: {buy_wr:.2%}\n"
              f"\t Consider: Put Credit Spread\n"
              f"\t Short Put Strike ≈ $latest_close * {args.put_strike_pct:.2f}")
        print(f"🔴 Sell Signals: {int(df_final[('Signal_Sell', args.ticker)].sum())} (Density: {sell_density:.4f}) | Evaluated: {eval_sell} | Wins: {sell_wins} | Win Rate: {sell_wr:.2%}\n"
              f"\t Consider: Call Credit Spread\n"
              f"\t Short Call Strike ≈ $latest_close * {args.call_strike_pct:.2f}")
        print(f"🎯 Combined Win Rate: {combined_wr:.2%} ({buy_wins + sell_wins}/{eval_buy + eval_sell})")
        print(f"🎯   Buy Win Rate : {buy_wr:.2%} ({buy_wins}/{eval_buy})")
        print(f"🎯   Sell Win Rate: {sell_wr:.2%} ({sell_wins}/{eval_sell})")
        print(f"📉 Min Density Threshold: {args.min_signal_density:.2%}")
        if buy_density < args.min_signal_density: print("⚠️  Buy signal density below threshold.")
        if sell_density < args.min_signal_density: print("⚠️  Sell signal density below threshold.")
    # 🔹 Save model with descriptive name including params and score
    if args.optimize or args.model_path is None:  # Only save new model if we optimized or didn't load one
        score = combined_wr if args.optimize_target == 'combined_wr' else (buy_wr if args.optimize_target == 'buy_wr' else sell_wr)
        saved_path = save_model(
            params,
            score,
            args,
            df_final if args.verbose else None,
            validation_score=validation_score,
            train_val_split=train_val_split_info
        )

    os.makedirs(args.output_dir, exist_ok=True)
    # output_path = os.path.join(args.output_dir, f"{args.ticker}_{args.dataset_id}_signals.pkl")
    # with open(output_path, 'wb') as f:
    #     pickle.dump(df_final, f)
    # if args.verbose: print(f"💾 Updated dataframe saved to: {output_path}")

    if args.plot:
        fib_50_col = ('Fib_50', args.ticker)
        fib_618_col = ('Fib_618', args.ticker)
        sma_50_col = ('SMA_50', args.ticker)
        buy_sig_col = ('Signal_Buy', args.ticker)
        sell_sig_col = ('Signal_Sell', args.ticker)
        rsi_col = ('RSI', args.ticker)
        plot_results(
            df_final, args, close_col, high_col, low_col, rsi_col, sma_50_col,
            fib_50_col, fib_618_col, buy_sig_col, sell_sig_col,
            ('Regular_Bullish_Div', args.ticker), ('Regular_Bearish_Div', args.ticker),
            ('Hidden_Bullish_Div', args.ticker), ('Hidden_Bearish_Div', args.ticker)
        )

    return df_final


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)