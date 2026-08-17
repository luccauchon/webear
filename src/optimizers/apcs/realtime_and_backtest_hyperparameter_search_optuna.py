"""
Asymmetric Pivot Credit Strategy
================================

This module implements a quantitative trading strategy based on identifying
asymmetric pivot patterns (Higher Lows and Lower Highs) in financial time series
data. It is designed to trade credit spreads (Put Credit Spreads for bullish
setups and Call Credit Spreads for bearish setups).

Key Features:
- **Pattern Recognition**: Uses confirmed pivot logic to identify significant
  market turns (peaks and valleys) and evaluates 3-turn patterns.
- **Filtering**: Incorporates Exponential Moving Average (EMA) and Relative
  Strength Index (RSI) to filter out low-probability setups and avoid
  overbought/oversold exhaustion points.
- **Cooldown**: Adds a configurable cooldown period (minimum bars between signals).
- **Optimization**: Utilizes `optuna` with `TimeSeriesSplit` cross-validation
  to find optimal parameters while preventing look-ahead bias.
- **Scoring Mechanism**: Employs Laplace smoothing for win rate calculation and
  applies strict proportional penalties for trade density violations.
- **Real-Time Execution**: Includes a dedicated mode to evaluate the most recent
  market bar against a saved, optimized model for live signal generation.

Option B Execution Model:
- A pivot at bar i is confirmed on bar i + 1.
- Entry is executed at the open of bar i + 2.
- This avoids entering at the open of the confirmation bar, which would be
  non-executable because the confirmation is only known at the close of that bar.

Dependencies:
- pandas, numpy, scipy
- yfinance (for data fetching if applicable)
- optuna (for hyperparameter optimization)
- scikit-learn (for TimeSeriesSplit)
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

import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks  # kept for compatibility; confirmed pivot logic is used by default
from datetime import datetime, timedelta
from utils import get_next_step, factory_load_data
import pickle
import optuna  # Added Optuna import
import random
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
import argparse  # Added for command-line arguments
from fetchers.serialize_fyahoo import realtime as fyahoo_realtime
import math
import traceback

# Suppress Optuna & pandas debug logs for cleaner console output
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None


def is_closed_bar(ts, dataset_id):
    """
    Placeholder for bar-close validation.

    This function should later return True only if the bar timestamp `ts`
    is a finalized/closed bar for the given dataset_id.

    For now, it returns True unconditionally so the script remains runnable.
    """
    return True


def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.

    The RSI is a momentum oscillator that measures the speed and change of price
    movements. In this strategy, it is used as a regime filter to prevent entering
    trades when the market is in an extreme overbought or oversold state.

    Args:
        prices (pd.Series): A pandas Series containing the historical price data
            (typically closing prices).
        period (int): The lookback window (number of periods) for calculating the RSI.

    Returns:
        pd.Series: A pandas Series containing the calculated RSI values, bounded
        between 0 and 100. NaN values resulting from flat markets are forward-filled
        with a neutral value of 50.0.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's smoothing method (equivalent to EMA with com=period-1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)  # Fallback for NaNs (e.g., perfectly flat market)
    return rsi


def prepare_clean_ohlc(df, close_col, open_col, high_col, low_col):
    """
    Clean OHLC data by dropping rows where any required price column is missing.

    Enhancement:
    The original script dropped NaNs separately for each column. That could
    misalign Close vs Open/High/Low arrays if missing bars differed by column.
    This helper ensures all price arrays share the same index and bar alignment.
    """
    cols = [close_col, open_col, high_col, low_col]
    df_clean = df.dropna(subset=cols).copy()

    # Time-series logic assumes ascending chronological order.
    df_clean = df_clean.sort_index()

    prices_series = df_clean[close_col].astype(float)
    open_prices = df_clean[open_col].astype(float).values
    high_prices = df_clean[high_col].astype(float).values
    low_prices = df_clean[low_col].astype(float).values

    return df_clean, prices_series, open_prices, high_prices, low_prices


def detect_confirmed_turns(prices, dates, min_distance):
    """
    Detect confirmed peaks and valleys without repainting.

    Enhancement:
    The original script used scipy.signal.find_peaks on the full price series with
    distance=min_distance. With distance > 1, a pivot can be removed later when a
    future pivot appears nearby. That introduces look-ahead / repainting.

    This implementation uses a causal confirmed-pivot rule:
    - A pivot at bar i is confirmed only after bar i+1 exists.
    - Once accepted, it is never removed by future data.
    - min_distance is enforced causally per pivot type.

    This changes pivot selection compared with SciPy's offline distance handling,
    but makes backtest and realtime behavior consistent and executable.
    """
    n = len(prices)
    if n < 3:
        return []

    min_distance = max(1, int(min_distance))
    turns = []

    # Initialize far enough in the past so the first valid pivot can be accepted.
    last_peak_idx = -min_distance - 1
    last_valley_idx = -min_distance - 1

    # A pivot at i is confirmed by bar i+1.
    # Therefore we can only detect pivots up to n-2.
    for i in range(1, n - 1):
        # Strict confirmed local peak.
        is_peak = prices[i] > prices[i - 1] and prices[i] > prices[i + 1]

        # Strict confirmed local valley.
        is_valley = prices[i] < prices[i - 1] and prices[i] < prices[i + 1]

        if is_peak:
            # Enforce minimum distance causally.
            # If a future peak is closer than min_distance, it is rejected instead
            # of retroactively removing this accepted peak.
            if i - last_peak_idx >= min_distance:
                turns.append(("Peak", i, float(prices[i]), dates[i]))
                last_peak_idx = i

        if is_valley:
            # Enforce minimum distance causally for valleys as well.
            if i - last_valley_idx >= min_distance:
                turns.append(("Valley", i, float(prices[i]), dates[i]))
                last_valley_idx = i

    turns.sort(key=lambda x: x[1])
    return turns


def classify_pattern(
    t1,
    t2,
    t3,
    prices,
    ema,
    rsi,
    trade_direction,
    rsi_buy_max,
    rsi_sell_min
):
    """
    Classify a 3-turn pattern as BUY, SELL, or None.

    This is shared by backtest and realtime so the exact same rules are used.
    """
    neckline_price = t2[2]
    trade_type = None

    # --- SELL LOGIC (Call Credit Spread) – Lower High pattern ---
    if t1[0] == "Peak" and t2[0] == "Valley" and t3[0] == "Peak":
        if trade_direction in ["sell", "both"]:
            if t1[2] > neckline_price and t3[2] > neckline_price:
                if t3[2] < t1[2]:
                    # EMA Filter: Price must be below EMA for SELL setup
                    if prices[t3[1]] < ema[t3[1]]:
                        # RSI Filter: Prevent selling when oversold
                        if not np.isnan(rsi[t3[1]]) and rsi[t3[1]] >= rsi_sell_min:
                            trade_type = "SELL"

    # --- BUY LOGIC (Put Credit Spread) – Higher Low pattern ---
    elif t1[0] == "Valley" and t2[0] == "Peak" and t3[0] == "Valley":
        if trade_direction in ["buy", "both"]:
            if t1[2] < neckline_price and t3[2] < neckline_price:
                if t3[2] > t1[2]:
                    # EMA Filter: Price must be above EMA for BUY setup
                    if prices[t3[1]] > ema[t3[1]]:
                        # RSI Filter: Prevent buying when overbought
                        if not np.isnan(rsi[t3[1]]) and rsi[t3[1]] <= rsi_buy_max:
                            trade_type = "BUY"

    return trade_type


def calculate_entry_details(
    trade_type,
    entry_idx,
    neckline_price,
    open_prices,
    high_prices,
    low_prices,
    buy_offset,
    sell_offset
):
    """
    Calculate entry strike and execution type for a confirmed signal.

    Option B:
    - A pivot at bar i is confirmed on bar i + 1.
    - Entry is taken at the open of bar i + 2.
    - Therefore only the entry bar open is used.
    - High/Low are kept in the signature for compatibility but are not used
      for entry qualification under Option B.

    Enhancement:
    - Added a guard to avoid zero strikes for low-priced instruments.
    """
    if entry_idx < 0 or entry_idx >= len(open_prices):
        return None, None

    entry_price = None
    entry_execution = None

    if trade_type == "BUY":
        # Option B: entry is valid only if the next bar open is already above neckline.
        if open_prices[entry_idx] >= neckline_price:
            # Sell Credit Put Spread – use neckline rounded DOWN to nearest 5
            entry_price = int(math.floor(neckline_price * buy_offset / 5) * 5)

            # Enhancement: prevent invalid zero strike.
            entry_price = max(5, entry_price)

            # Option B execution label.
            entry_execution = "NextOpen"

    elif trade_type == "SELL":
        # Option B: entry is valid only if the next bar open is already below neckline.
        if open_prices[entry_idx] <= neckline_price:
            # Sell Credit Call Spread – use neckline rounded UP to nearest 5
            entry_price = int(math.ceil(neckline_price * sell_offset / 5) * 5)

            # Enhancement: prevent invalid zero strike.
            entry_price = max(5, entry_price)

            # Option B execution label.
            entry_execution = "NextOpen"

    return entry_price, entry_execution


class EarlyStoppingThresholdCallback:
    """
    Custom Optuna callback to terminate the optimization study early.

    This callback monitors the objective value of each trial. If a trial achieves
    a score that meets or exceeds a predefined threshold, it stops the study
    immediately, saving computational resources when a "good enough" or perfect
    parameter set is found.

    Attributes:
        threshold (float): The target objective value that triggers early stopping.
    """

    def __init__(self, threshold: float = 0.99):
        """
        Initialize the callback with a specific score threshold.

        Args:
            threshold (float): The minimum score required to stop the study.
                Defaults to 0.99.
        """
        self.threshold = threshold

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Execute the callback logic after each trial.

        Args:
            study (optuna.study.Study): The current Optuna study object.
            trial (optuna.trial.FrozenTrial): The trial object that just finished.
        """
        if trial.value is not None and trial.value >= self.threshold:
            print(
                f"\n🎯 Early stopping triggered! Trial {trial.number} reached score "
                f"{trial.value:.4f} (>= {self.threshold}). Stopping study."
            )
            study.stop()


def check_live_signal(
    df,
    close_col,
    open_col,
    low_col,
    high_col,
    min_distance,
    ema_period,
    rsi_period,
    rsi_buy_max,
    rsi_sell_min,
    buy_offset,
    sell_offset,
    trade_direction="both",
    cooldown_bar=0,
    dataset_id=None
):
    """
    Evaluate the most recent bar in a real-time dataframe for a valid trading signal.

    Option B:
    - A pivot at bar i is confirmed on bar i + 1.
    - Entry is taken at the open of bar i + 2.
    - In live mode, this function looks for a setup where the current last bar
      is the entry bar, and the previous bar is the closed confirmation bar.

    This function replicates the exact pattern recognition and filtering logic
    used in the historical backtest, but restricts its evaluation to signals
    whose entry bar is the final bar of the provided dataframe.

    Enhancement:
    - Uses shared helpers with backtest.
    - Uses cleaned OHLC rows to prevent Close/Open/High/Low misalignment.
    - Scans all confirmed 3-turn windows that can enter on the last bar, instead
      of assuming the last three turns are always the only possible candidate.
    - Uses confirmed causal pivots to reduce repainting.

    Args:
        df (pd.DataFrame): The historical and current OHLCV dataframe.
        close_col (tuple): Column identifier for the Close price.
        open_col (tuple): Column identifier for the Open price.
        low_col (tuple): Column identifier for the Low price.
        high_col (tuple): Column identifier for the High price.
        min_distance (int): Minimum number of bars between identified peaks/valleys.
        ema_period (int): Lookback period for the Exponential Moving Average filter.
        rsi_period (int): Lookback period for the RSI calculation.
        rsi_buy_max (int): Maximum RSI value allowed to enter a BUY (Put Credit Spread).
        rsi_sell_min (int): Minimum RSI value allowed to enter a SELL (Call Credit Spread).
        buy_offset (float): Multiplier applied to the neckline for BUY strike selection.
        sell_offset (float): Multiplier applied to the neckline for SELL strike selection.
        trade_direction (str): Whether to evaluate "buy", "sell", or "both".
        cooldown_bar (int): Minimum number of bars to wait between signals (cooldown period).
        dataset_id (str or None): Dataset identifier used by is_closed_bar().

    Returns:
        dict: A dictionary containing signal details if a valid entry condition is met
              on the last bar.
        dict: A dictionary containing an explanatory message under 'reason' when no
              signal was generated.
    """
    # Normalize cooldown input for robustness, especially when loading old models.
    cooldown_bar = 0 if cooldown_bar is None else max(0, int(cooldown_bar))

    # ------------------------------------------------------------------
    # 1. Extract price arrays cleanly and identically to backtest
    # ------------------------------------------------------------------
    df_clean, prices_series, open_prices, high_prices, low_prices = prepare_clean_ohlc(
        df=df,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col
    )

    if len(prices_series) < 10:
        return {"reason": "Not enough data points."}

    ema_series = prices_series.ewm(span=ema_period, adjust=False).mean()
    ema = ema_series.values

    rsi_series = calculate_rsi(prices_series, rsi_period)
    rsi = rsi_series.values

    prices = prices_series.values
    dates = prices_series.index

    last_bar_idx = len(prices) - 1

    # ------------------------------------------------------------------
    # 2. Confirmed Peak and Valley Detection
    # ------------------------------------------------------------------
    turns = detect_confirmed_turns(prices, dates, min_distance)

    # Enhancement: original realtime required only 3 turns, while backtest required 4.
    # The pattern logic uses 3 turns, so keep the rule consistent at 3.
    if len(turns) < 3:
        return {"reason": "Not enough significant turns identified to evaluate a signal."}

    # ------------------------------------------------------------------
    # 3. Find all candidate windows that enter on the last bar.
    #    Enhancement: do not assume turns[-3:] is always the only candidate.
    #
    #    Option B:
    #    If t3 is at bar i:
    #      - confirmation bar = i + 1
    #      - entry bar        = i + 2
    # ------------------------------------------------------------------
    candidates = []

    for i in range(len(turns) - 2):
        t1, t2, t3 = turns[i], turns[i + 1], turns[i + 2]
        neckline_price = t2[2]

        confirmation_idx = t3[1] + 1
        entry_idx = t3[1] + 2

        # We only want signals whose entry bar is the last realtime bar.
        if entry_idx != last_bar_idx:
            continue

        if confirmation_idx < 0 or confirmation_idx >= len(prices):
            continue

        # The confirmation bar must be closed.
        # The entry bar may be the currently forming bar because its open is known.
        if not is_closed_bar(dates[confirmation_idx], dataset_id):
            continue

        trade_type = classify_pattern(
            t1=t1,
            t2=t2,
            t3=t3,
            prices=prices,
            ema=ema,
            rsi=rsi,
            trade_direction=trade_direction,
            rsi_buy_max=rsi_buy_max,
            rsi_sell_min=rsi_sell_min
        )

        if trade_type is None:
            continue

        entry_price, entry_execution = calculate_entry_details(
            trade_type=trade_type,
            entry_idx=entry_idx,
            neckline_price=neckline_price,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            buy_offset=buy_offset,
            sell_offset=sell_offset
        )

        if entry_price is None:
            continue

        candidates.append(
            {
                "trade_type": trade_type,
                "entry_price": entry_price,
                "entry_execution": entry_execution,
                "neckline_price": neckline_price,
                "t3": t3,
                "entry_idx": entry_idx,
            }
        )

    if not candidates:
        return {
            "reason": "No valid SELL or BUY pattern entered on the last bar "
                      "(or failed EMA/RSI/entry-condition filter)."
        }

    # Usually there is only one candidate. If multiple exist due to unusual pivot
    # definitions, choose the latest candidate deterministically.
    candidate = candidates[-1]

    trade_type = candidate["trade_type"]
    entry_price = candidate["entry_price"]
    entry_execution = candidate["entry_execution"]
    neckline_price = candidate["neckline_price"]
    t3 = candidate["t3"]
    entry_idx = candidate["entry_idx"]

    # ------------------------------------------------------------------
    # 5.5 Cooldown check:
    #     Ensure the live entry is not too close to the previously executed
    #     signal, using the same entry rules as the backtest.
    #
    #     Convention used here:
    #     If an entry occurs on bar X and cooldown_bar=N, the next entry is
    #     only allowed on bar X + N + 1 or later. In other words, N full bars
    #     must pass without a new executed signal.
    # ------------------------------------------------------------------
    if cooldown_bar > 0:
        last_executed_entry_idx = None

        for i in range(len(turns) - 2):
            c_t1, c_t2, c_t3 = turns[i], turns[i + 1], turns[i + 2]
            c_neckline_price = c_t2[2]

            c_confirmation_idx = c_t3[1] + 1
            c_entry_idx = c_t3[1] + 2

            # Only historical entries before the current live candidate matter.
            if c_entry_idx >= entry_idx:
                break

            if c_entry_idx >= len(prices):
                break

            if c_confirmation_idx < 0 or c_confirmation_idx >= len(prices):
                continue

            # Historical confirmation bars should be closed.
            if not is_closed_bar(dates[c_confirmation_idx], dataset_id):
                continue

            c_trade_type = classify_pattern(
                t1=c_t1,
                t2=c_t2,
                t3=c_t3,
                prices=prices,
                ema=ema,
                rsi=rsi,
                trade_direction=trade_direction,
                rsi_buy_max=rsi_buy_max,
                rsi_sell_min=rsi_sell_min
            )

            if c_trade_type is None:
                continue

            # Apply cooldown to the historical simulation as well, so the
            # "last executed signal" respects the same cooldown rules.
            if last_executed_entry_idx is not None and c_entry_idx <= last_executed_entry_idx + cooldown_bar:
                continue

            c_entry_price, _ = calculate_entry_details(
                trade_type=c_trade_type,
                entry_idx=c_entry_idx,
                neckline_price=c_neckline_price,
                open_prices=open_prices,
                high_prices=high_prices,
                low_prices=low_prices,
                buy_offset=buy_offset,
                sell_offset=sell_offset
            )

            if c_entry_price is not None:
                last_executed_entry_idx = c_entry_idx

        if last_executed_entry_idx is not None and entry_idx <= last_executed_entry_idx + cooldown_bar:
            # Robust date formatting: avoid crashing if index is not datetime-like.
            try:
                last_date_str = dates[last_executed_entry_idx].strftime('%Y-%m-%d_%H%M')
            except Exception:
                last_date_str = str(dates[last_executed_entry_idx])

            try:
                current_date_str = dates[entry_idx].strftime('%Y-%m-%d_%H%M')
            except Exception:
                current_date_str = str(dates[entry_idx])

            return {
                "reason":
                    f"Signal suppressed by cooldown_bar={cooldown_bar}. "
                    f"Last executed signal at bar {last_executed_entry_idx} "
                    f"({last_date_str}), "
                    f"current entry bar {entry_idx} "
                    f"({current_date_str})."
            }

    # ------------------------------------------------------------------
    # 6. Return the live signal dictionary
    # ------------------------------------------------------------------
    return {
        'Signal': trade_type,
        'Price': entry_price,
        'Date': dates[entry_idx],
        'T3_Type': t3[0],
        'T3_Date': dates[t3[1]],
        'Neckline': neckline_price,
        'Entry_Execution': entry_execution,
        'reason': None,
    }


def format_trade_samples(trades_df, first_n=5, last_n=5):
    """
    Format and print first/last sample trades with explicit entry/exit timing and profit.

    This helper was added to make the final optimization output easier to read.
    It displays entry price, entry bar date/hour, whether entry was confirmed at
    the open or by the close of the confirmation candle, outcome, exit bar
    date/hour, exit candle basis, exit price, and a price-unit profit proxy.

    Args:
        trades_df (pd.DataFrame): Trade log produced by `backtest_asymmetric_strategy`.
        first_n (int): Number of first trades to display.
        last_n (int): Number of last trades to display.

    Returns:
        None
    """
    if trades_df is None or len(trades_df) == 0:
        print("No trades available to display.")
        return

    df = trades_df.copy()

    # Ensure newly added display columns exist even if an older trade frame is passed.
    optional_columns = {
        "Entry_Execution": "Unknown",
        "Exit_Execution": "Close",
        "Profit": np.nan,
    }
    for col, default in optional_columns.items():
        if col not in df.columns:
            df[col] = default

    # Preserve the original trade order and give the user a stable trade number.
    df.insert(0, "Trade #", range(1, len(df) + 1))

    entry_dt = pd.to_datetime(df["Entry_Date"], errors="coerce")
    exit_dt = pd.to_datetime(df["Exit_Date"], errors="coerce")

    entry_price_num = pd.to_numeric(df["Entry_Price"], errors="coerce")
    exit_price_num = pd.to_numeric(df["Exit_Price"], errors="coerce")

    df["Entry Price"] = entry_price_num.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    df["Entry Date"] = entry_dt.dt.strftime("%Y-%m-%d").fillna("")
    df["Entry Hour"] = entry_dt.dt.strftime("%H:%M").fillna("")
    df["Entry Candle"] = df["Entry_Execution"].fillna("Unknown")

    df["Exit Date"] = exit_dt.dt.strftime("%Y-%m-%d").fillna("")
    df["Exit Hour"] = exit_dt.dt.strftime("%H:%M").fillna("")
    df["Exit Candle"] = df["Exit_Execution"].fillna("Close")
    df["Exit Price"] = exit_price_num.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    def _format_profit(value):
        if pd.isna(value):
            return "N/A"
        try:
            return f"{float(value):+.2f}"
        except (TypeError, ValueError):
            return str(value)

    df["Profit"] = df["Profit"].apply(_format_profit)

    display_columns = [
        "Trade #",
        "Entry Price",
        "Entry Date",
        "Entry Hour",
        "Entry Candle",
        "Type",
        "Outcome",
        "Exit Date",
        "Exit Hour",
        "Exit Candle",
        "Exit Price",
        "Profit",
    ]

    display_df = df[display_columns]

    print("Notes:")
    print("- Entry Candle: 'NextOpen' means entry is at the open of the bar after confirmation.")
    print("- Exit Candle: exit is taken at the close of the exit bar.")
    print("- Bar Date/Hour are the candle timestamps supplied by the data.")
    print("- Profit is a price-unit proxy: favorable underlying move minus the delta threshold, not option premium.")

    if len(display_df) <= first_n + last_n:
        print(display_df.to_string(index=False))
    else:
        print(f"First {first_n}:")
        print(display_df.head(first_n).to_string(index=False))
        print(f"\nLast {last_n}:")
        print(display_df.tail(last_n).to_string(index=False))


def plot_test_trades(
    df_test,
    test_results,
    close_col,
    ticker,
    dataset_id,
    lookahead,
    trade_direction
):
    """
    Plot the TEST set after optimization and annotate closed trade outcomes.

    Marker behavior:
    - BUY trades are shown as triangles.
    - SELL trades are shown as squares.
    - Winning trades are green.
    - Losing trades are red.
    - A bold "W" or "L" is placed above the entry price.

    Args:
        df_test (pd.DataFrame): Test dataframe containing OHLC data.
        test_results (dict or str): Test backtest results produced by
            `backtest_asymmetric_strategy`, or an error string if the backtest failed.
        close_col (tuple): Column identifier for the Close price.
        ticker (str): Ticker symbol being plotted.
        dataset_id (str): Dataset identifier used for optimization.
        lookahead (int): Number of bars used for trade outcome evaluation.
        trade_direction (str): Trade direction used during optimization.

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from pandas.plotting import register_matplotlib_converters

        register_matplotlib_converters()
    except ImportError as exc:
        print(f"❌ Unable to plot because matplotlib could not be imported: {exc}")
        return

    # Use a nicer style when available, but remain compatible with older Matplotlib versions.
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        if style in plt.style.available:
            plt.style.use(style)
            break

    if df_test is None or len(df_test) == 0:
        print("❌ No test data available to plot.")
        return

    close_series = df_test[close_col].dropna().copy()
    if close_series.empty:
        print("❌ No close data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the test close series as the main market context.
    ax.plot(
        close_series.index,
        close_series.values,
        color="#1f77b4",
        linewidth=1.8,
        label="Close"
    )

    ax.set_title(
        f"APCS - Asymmetric Pivot Credit Strategy - TEST Set\n"
        f"{ticker} | dataset={dataset_id} | lookahead={lookahead} | direction={trade_direction}",
        fontsize=16,
        fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.35, linestyle="--")

    # Improve date rendering when the index is datetime-like.
    if isinstance(close_series.index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        fig.autofmt_xdate(rotation=30, ha="right")

    # These will be used to adjust the vertical range so that W/L labels remain visible.
    entry_x = None
    entry_prices = None
    outcomes = None
    trade_types = None

    # Extract closed trades from the test results, if available.
    if isinstance(test_results, dict) and "df_trades" in test_results:
        trades_df = test_results["df_trades"]

        if trades_df is not None and len(trades_df) > 0:
            closed_trades = trades_df[trades_df["Outcome"].isin(["Win", "Loss"])].copy()

            if not closed_trades.empty:
                entry_x = closed_trades["Entry_Date"]

                # Convert entry dates to datetime only when the main price index is datetime-like.
                # This avoids accidentally converting numeric index values into timestamps.
                if isinstance(close_series.index, pd.DatetimeIndex):
                    entry_x = pd.to_datetime(entry_x, errors="coerce")

                entry_prices = pd.to_numeric(closed_trades["Entry_Price"], errors="coerce")
                outcomes = closed_trades["Outcome"].astype(str)

                # Trade side: BUY or SELL.
                # Fallback to empty string for older trade frames without a Type column.
                if "Type" in closed_trades.columns:
                    trade_types = closed_trades["Type"].astype(str)
                else:
                    trade_types = pd.Series("", index=closed_trades.index)

                valid_mask = (
                    entry_x.notna()
                    & entry_prices.notna()
                    & outcomes.notna()
                    & trade_types.notna()
                )

                entry_x = entry_x[valid_mask]
                entry_prices = entry_prices[valid_mask]
                outcomes = outcomes[valid_mask]
                trade_types = trade_types[valid_mask]

    # Determine a comfortable y-range that leaves space above entries for the bold W/L labels.
    y_min = float(close_series.min())
    y_max = float(close_series.max())

    if entry_prices is not None and not entry_prices.empty:
        y_min = min(y_min, float(entry_prices.min()))
        y_max = max(y_max, float(entry_prices.max()))

    y_range = max(y_max - y_min, 1e-9)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.10 * y_range)

    # Annotate each closed trade at the entry level.
    if (
        entry_x is not None
        and entry_prices is not None
        and outcomes is not None
        and trade_types is not None
        and not entry_x.empty
    ):
        text_offset = 0.02 * y_range

        for x, entry_price, outcome, trade_type in zip(
            entry_x,
            entry_prices,
            outcomes,
            trade_types
        ):
            if outcome == "Win":
                label = "W"
                color = "#1b7f3b"
            else:
                label = "L"
                color = "#b62836"

            # Marker denotes trade side:
            # BUY  -> triangle
            # SELL -> square
            if str(trade_type).upper().startswith("SELL"):
                marker = "s"
            else:
                marker = "^"

            # Mark the entry itself.
            ax.scatter(
                x,
                entry_price,
                color=color,
                marker=marker,
                s=85,
                zorder=5,
                edgecolors="black",
                linewidths=0.7
            )

            # Place the bold W/L just above the entry.
            ax.text(
                x,
                entry_price + text_offset,
                label,
                fontsize=13,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
                zorder=6
            )

        # Build a clear legend for the annotated entries.
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0], [0],
                color="#1f77b4",
                linewidth=1.8,
                label="Close"
            ),
            Line2D(
                [0], [0],
                marker="^",
                linestyle="None",
                color="none",
                markerfacecolor="#1b7f3b",
                markeredgecolor="black",
                markersize=9,
                label="Buy win (triangle)"
            ),
            Line2D(
                [0], [0],
                marker="^",
                linestyle="None",
                color="none",
                markerfacecolor="#b62836",
                markeredgecolor="black",
                markersize=9,
                label="Buy loss (triangle)"
            ),
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                color="none",
                markerfacecolor="#1b7f3b",
                markeredgecolor="black",
                markersize=9,
                label="Sell win (square)"
            ),
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                color="none",
                markerfacecolor="#b62836",
                markeredgecolor="black",
                markersize=9,
                label="Sell loss (square)"
            ),
        ]

        ax.legend(handles=legend_elements, loc="best")
    else:
        ax.legend()

    plt.tight_layout()

    try:
        plt.show()
    except Exception as exc:
        print(f"❌ Unable to display the plot: {exc}")


def backtest_asymmetric_strategy(
    ticker,
    df,
    close_col,
    open_col,
    low_col,
    high_col,
    min_distance,
    lookahead,
    sell_offset,
    buy_offset,
    ema_period,
    rsi_period,
    rsi_buy_max,
    rsi_sell_min,
    trade_direction="both",
    delta=0.0,
    cooldown_bar=0,
    record_start_idx=None,
    record_end_idx=None
):
    """
    Perform a historical backtest of the Asymmetric Pivot Credit Strategy.

    This function processes historical OHLC data to identify 3-turn pivot patterns
    (Higher Lows for bullish setups, Lower Highs for bearish setups). It applies
    trend (EMA) and momentum (RSI) filters, simulates trade entries based on
    price action relative to the pattern's neckline, and tracks trade outcomes
    over a specified lookahead period.

    Option B:
    - A pivot at bar i is confirmed on bar i + 1.
    - Entry is executed at the open of bar i + 2.

    Enhancements:
    - Cleans OHLC rows together to prevent misaligned price arrays.
    - Uses confirmed causal pivots to reduce look-ahead / repainting.
    - Supports record_start_idx and record_end_idx so a test period can be
      evaluated with full historical context while only recording trades inside
      the desired window.
    - Uses consistent minimum turn count with realtime.

    Args:
        ticker (str): The ticker symbol being backtested (used for metadata/logging).
        df (pd.DataFrame): The historical dataframe containing OHLC data.
        close_col (tuple): Column identifier for the Close price.
        open_col (tuple): Column identifier for the Open price.
        low_col (tuple): Column identifier for the Low price.
        high_col (tuple): Column identifier for the High price.
        min_distance (int): Minimum distance between peaks/valleys for pivot detection.
        lookahead (int): Number of bars to hold the trade before evaluating the exit.
        sell_offset (float): Multiplier for Call Credit Spread strike selection.
        buy_offset (float): Multiplier for Put Credit Spread strike selection.
        ema_period (int): Period for the EMA trend filter.
        rsi_period (int): Period for the RSI momentum filter.
        rsi_buy_max (int): Upper RSI threshold for buying.
        rsi_sell_min (int): Lower RSI threshold for selling.
        trade_direction (str): Whether to trade "buy", "sell", or "both".
        delta (float): Minimum percentage gain required from entry_price to count as a Win.
        cooldown_bar (int): Minimum number of bars to wait between signals (cooldown period).
        record_start_idx (int or None): If provided, only record entries at or after this row index.
        record_end_idx (int or None): If provided, only record entries at or before this row index.

    Returns:
        dict: A dictionary containing backtest results.
        str: An error/info message if the backtest could not be executed.
    """
    # Normalize cooldown input.
    cooldown_bar = 0 if cooldown_bar is None else max(0, int(cooldown_bar))

    # Enhancement: validate lookahead. Original formula allowed lookahead=0,
    # which evaluates exit on the same bar as entry and is usually unintended.
    if lookahead < 1:
        return "lookahead must be >= 1."

    # ------------------------------------------------------------------
    # Clean OHLC data once so Close/Open/High/Low remain aligned.
    # ------------------------------------------------------------------
    df_clean, prices_series, open_prices, high_prices, low_prices = prepare_clean_ohlc(
        df=df,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col
    )

    if len(prices_series) < 10:
        return "Not enough data points."

    # Calculate Indicators
    ema_series = prices_series.ewm(span=ema_period, adjust=False).mean()
    ema = ema_series.values

    rsi_series = calculate_rsi(prices_series, rsi_period)
    rsi = rsi_series.values

    prices = prices_series.values
    dates = prices_series.index
    n_prices = len(prices)

    # Normalize recording window.
    if record_start_idx is None:
        record_start_idx = 0
    if record_end_idx is None:
        record_end_idx = n_prices - 1

    record_start_idx = max(0, int(record_start_idx))
    record_end_idx = min(n_prices - 1, int(record_end_idx))

    if record_start_idx > record_end_idx:
        record_bar_count = 0
    else:
        record_bar_count = record_end_idx - record_start_idx + 1

    # Peak and Valley Detection using confirmed causal pivots.
    turns = detect_confirmed_turns(prices, dates, min_distance)

    # Enhancement: original backtest required 4 turns while realtime required 3.
    # The strategy pattern uses 3 turns, so this is now consistent.
    if len(turns) < 3:
        return "Not enough significant turns identified to map patterns."

    # Pre-allocate lists for faster DataFrame creation (Columnar format)
    entry_dates = []
    trade_types = []
    entry_prices = []
    outcomes = []
    exit_dates = []
    exit_prices = []

    # Added columns to make the final trade sample easier to interpret.
    entry_executions = []
    exit_executions = []
    profits = []

    # Track the last executed entry bar to enforce the cooldown period.
    last_entry_idx = None

    # Sliding Window Signal Generation
    for i in range(len(turns) - 2):
        t1, t2, t3 = turns[i], turns[i + 1], turns[i + 2]
        neckline_price = t2[2]

        # Option B:
        # A peak/valley at t3[1] is confirmed on bar t3[1] + 1.
        # Entry is taken at the open of bar t3[1] + 2.
        entry_idx = t3[1] + 2

        # Prevent out-of-bounds for entry
        if entry_idx >= n_prices:
            continue

        trade_type = classify_pattern(
            t1=t1,
            t2=t2,
            t3=t3,
            prices=prices,
            ema=ema,
            rsi=rsi,
            trade_direction=trade_direction,
            rsi_buy_max=rsi_buy_max,
            rsi_sell_min=rsi_sell_min
        )

        # Trade Outcome Tracking
        if trade_type is not None:
            # ------------------------------------------------------------------
            # Cooldown period:
            # After an executed signal on bar X, ignore new executed signals
            # until bar X + cooldown_bar + 1. This means cooldown_bar full bars
            # must pass without a new executed signal.
            # ------------------------------------------------------------------
            if last_entry_idx is not None and entry_idx <= last_entry_idx + cooldown_bar:
                continue

            entry_price, entry_execution = calculate_entry_details(
                trade_type=trade_type,
                entry_idx=entry_idx,
                neckline_price=neckline_price,
                open_prices=open_prices,
                high_prices=high_prices,
                low_prices=low_prices,
                buy_offset=buy_offset,
                sell_offset=sell_offset
            )

            if entry_price is None:
                # No entry detected. We skip this trade.
                continue

            # Enhancement:
            # Update cooldown state for every executed entry, even entries outside
            # the requested recording window. This allows test-period evaluation
            # to inherit cooldown state from prior history.
            last_entry_idx = entry_idx

            # If this entry is outside the requested recording window, do not
            # record it in the trade log or metrics.
            if entry_idx < record_start_idx or entry_idx > record_end_idx:
                continue

            # The actual holding period (Time For Expiration of my credit spread) starts at entry.
            # Calculate exact exit index.
            exit_idx = entry_idx + lookahead

            # Handle end-of-dataframe edge case
            if exit_idx >= n_prices:
                outcome = "Open"
                exit_date = dates[-1]
                exit_price = prices[-1]
                exit_execution = "Close (end of data)"
            else:
                exit_date = dates[exit_idx]
                exit_price = prices[exit_idx]
                exit_execution = "Close"

                if trade_type == "SELL":
                    outcome = "Win" if exit_price < entry_price - delta * entry_price else "Loss"
                else:  # BUY
                    outcome = "Win" if exit_price > entry_price + delta * entry_price else "Loss"

            # Calculate a transparent price-based profit proxy.
            # For SELL: favorable when exit is below entry strike.
            # For BUY: favorable when exit is above entry strike.
            # The delta threshold is subtracted so the profit aligns with the Win/Loss rule.
            if outcome == "Open":
                profit = np.nan
            elif trade_type == "SELL":
                profit = entry_price - exit_price - delta * entry_price
            else:
                profit = exit_price - entry_price - delta * entry_price

            # Append to lists
            entry_dates.append(dates[entry_idx])
            trade_types.append(trade_type)
            entry_prices.append(entry_price)
            outcomes.append(outcome)
            exit_dates.append(exit_date)
            exit_prices.append(exit_price)
            entry_executions.append(entry_execution)
            exit_executions.append(exit_execution)
            profits.append(profit)

    total_trades = len(outcomes)
    if total_trades == 0:
        return "No trades executed."

    # Create DataFrame once at the end (Much faster than appending dicts)
    df_trades = pd.DataFrame({
        "Entry_Date": entry_dates,
        "Type": trade_types,
        "Entry_Price": entry_prices,
        "Entry_Execution": entry_executions,
        "Outcome": outcomes,
        "Exit_Date": exit_dates,
        "Exit_Execution": exit_executions,
        "Exit_Price": exit_prices,
        "Profit": profits
    })

    # Compute Metrics (Using fast list counting instead of DF filtering)
    wins = outcomes.count("Win")
    losses = outcomes.count("Loss")
    open_trades = outcomes.count("Open")
    closed_trades_count = wins + losses

    win_rate = (wins / closed_trades_count) if closed_trades_count > 0 else 0.0

    # Enhancement:
    # Density is computed over the requested recording window, not necessarily
    # the whole dataframe. This allows evaluating test-period density correctly
    # while using full historical context for indicators/pivots/cooldown.
    density = total_trades / record_bar_count if record_bar_count > 0 else 0.0

    # Filter closed trades for the return dict
    closed_trades_df = df_trades[df_trades["Outcome"].isin(["Win", "Loss"])]

    return {
        'df_trades': df_trades,
        'df': df_clean,
        'total_trades': total_trades,
        'density': density,
        'closed_trades': closed_trades_df,
        'open_trades': open_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate
    }


def entry(args):
    """
    Main execution entry point for the trading strategy pipeline.

    This function acts as the controller for the script, routing execution based
    on command-line arguments. It supports two primary modes:
    1. **Real-Time Mode (`--realtime`)**: Loads a previously optimized model and
       evaluates the latest market data to generate actionable live signals.
    2. **Optimization & Backtest Mode**: Splits historical data into train/test sets,
       runs an Optuna hyperparameter optimization using TimeSeries cross-validation,
       evaluates the best parameters on both sets, and serializes the winning
       model configuration to disk.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            configuration for tickers, data paths, optimization constraints,
            and execution modes.

    Returns:
        None: The function operates via side effects (printing to stdout,
              saving files, or exiting the system).
    """
    realtime = args.realtime
    use_realtime_dataset = args.use_realtime_data
    clip_n = args.clip_n
    model_file = args.model_file
    verbose = args.verbose
    command_line = "python " + " ".join(sys.argv)

    # ==========================================
    # --- REAL-TIME PROCESSING SECTION ---
    # ==========================================
    if realtime:
        values_returned = {
            'target_date': None,
            'signal': 0.,
            'current_price': None,
            'current_date': None,
            'target_price': 0.,
            'train_score': None,
            'val_score': None,
            'train_win_rate': None,
            'val_win_rate': None,
            'optimization_metric': 'buy_wr',
            'method': None,
            'threshold': None,
            'ticker': None,
            'dataset_id': None,
            'lookahead': None,
            'local_results': {'reason': "no yet processed"}
        }

        if verbose:
            print("\n" + "=" * 80)
            print(" REAL-TIME SIGNAL CHECK")
            print("=" * 80)

        if model_file:
            # User specified a specific model file
            if not os.path.exists(model_file):
                print(f"❌ Specified model file not found: {model_file}")
                return values_returned
        else:
            # Fallback to latest model logic
            print("❌ No saved model found. Please run training first.")
            return values_returned

        if verbose:
            print(f"Loading model from: {model_file}")

        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)

        model_trade_direction = model_info.get('trade_direction', 'both')
        command_line = model_info["command_line"] if "command_line" in model_info else ""

        target_date = get_next_step(
            the_date=datetime.now(),
            dataset_id=model_info['dataset_id'],
            nn=model_info['lookahead']
        )

        values_returned.update({'ticker': model_info['ticker']})
        values_returned.update({'dataset_id': model_info['dataset_id']})
        values_returned.update({'lookahead': model_info['lookahead']})
        values_returned.update({'target_date': target_date})
        values_returned.update({'current_date': str(datetime.now().strftime('%Y-%m-%d_%H%M'))})
        values_returned.update({'train_score': model_info['train_wr']})
        values_returned.update({'train_win_rate': model_info['train_wr']})
        values_returned.update({'val_score': model_info['test_wr']})
        values_returned.update({'val_win_rate': model_info['test_wr']})

        df_realtime, df_realtime_not_clipped = None, None

        try:
            if verbose:
                print(f"Command line used: {command_line}")

            df_realtime = factory_load_data(
                _dataset_id=model_info['dataset_id'],
                _ticker=model_info['ticker'],
                _args={"clip_n": clip_n, "realtime": use_realtime_dataset}
            )

            if df_realtime is None or 0 == len(df_realtime):
                print(f"❌ No more data with a clip of {clip_n}")
                values_returned['local_results'].update({'reason': f"no more data"})
                return values_returned

            df_realtime_not_clipped = factory_load_data(
                _dataset_id=model_info['dataset_id'],
                _ticker=model_info['ticker'],
                _args={"clip_n": 0, "realtime": use_realtime_dataset}
            )

            close_col = ('Close', model_info['ticker'])
            open_col = ('Open', model_info['ticker'])
            high_col = ('High', model_info['ticker'])
            low_col = ('Low', model_info['ticker'])

            # Enhancement:
            # Clean realtime data before printing or checking signals.
            df_realtime, _, _, _, _ = prepare_clean_ohlc(
                df=df_realtime,
                close_col=close_col,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col
            )

            if df_realtime_not_clipped is not None and len(df_realtime_not_clipped) > 0:
                df_realtime_not_clipped, _, _, _, _ = prepare_clean_ohlc(
                    df=df_realtime_not_clipped,
                    close_col=close_col,
                    open_col=open_col,
                    high_col=high_col,
                    low_col=low_col
                )

            if len(df_realtime) == 0:
                print("❌ No realtime rows remained after OHLC cleaning.")
                values_returned['local_results'].update({'reason': "no realtime rows after cleaning"})
                return values_returned

            if verbose:
                # Guard against None metrics from failed training/backtest.
                train_wr = model_info.get('train_wr')
                test_wr = model_info.get('test_wr')
                train_den = model_info.get('train_den')
                test_den = model_info.get('test_den')

                if train_wr is not None and test_wr is not None:
                    print(
                        f"Win Rate - Train: {train_wr:.2%} | Test: {test_wr:.2%} | "
                        f"Difference: {test_wr - train_wr:+.2%}"
                    )

                if train_den is not None and test_den is not None:
                    print(
                        f"Density  - Train: {train_den:.2%} | Test: {test_den:.2%} | "
                        f"Difference: {test_den - train_den:+.2%}"
                    )

            model_params = model_info['best_params']

            # Support older saved models that did not include cooldown_bar.
            cooldown_bar = model_params.get('cooldown_bar', model_info.get('cooldown_bar', 0))

            # Check live signal directly on the latest bar
            if verbose:
                try:
                    last_bar_str = df_realtime.index[-1].strftime('%Y-%m-%d_%H%M')
                except Exception:
                    last_bar_str = str(df_realtime.index[-1])

                print(
                    f"Last bar: {last_bar_str} "
                    f"({len(df_realtime)} bars)  "
                    f"Open/High/Low/Close: "
                    f"{df_realtime.iloc[-1][open_col]:.2f}/"
                    f"{df_realtime.iloc[-1][high_col]:.2f}/"
                    f"{df_realtime.iloc[-1][low_col]:.2f}/"
                    f"{df_realtime.iloc[-1][close_col]:.2f}"
                )

            values_returned.update({'current_price': df_realtime[close_col].iloc[-1]})

            live_result = check_live_signal(
                df=df_realtime.copy(),
                buy_offset=model_info['buy_offset'],
                sell_offset=model_info['sell_offset'],
                close_col=close_col,
                open_col=open_col,
                low_col=low_col,
                high_col=high_col,
                min_distance=model_params['min_distance'],
                ema_period=model_params['ema_period'],
                rsi_period=model_params['rsi_period'],
                rsi_buy_max=model_params['rsi_buy_max'],
                rsi_sell_min=model_params['rsi_sell_min'],
                trade_direction=model_trade_direction,
                cooldown_bar=cooldown_bar,
                dataset_id=model_info['dataset_id']
            )

            if not isinstance(live_result, dict):
                raise ValueError("check_live_signal() returned an unexpected object.")

            if live_result['reason'] is None:
                type_option = None

                if live_result['Signal'] == "SELL":
                    if model_info['sell_offset'] < 1.0:
                        raise ValueError("sell_offset must be >= 1.0 for Call Credit Spreads.")
                    type_option = "Call Credit Spread"
                    # Enhancement: expose signal to external consumers.
                    values_returned['signal'] = -1.0

                elif live_result['Signal'] == "BUY":
                    if model_info['buy_offset'] > 1.0:
                        raise ValueError("buy_offset must be <= 1.0 for Put Credit Spreads.")
                    type_option = "Put Credit Spread"
                    # Enhancement: expose signal to external consumers.
                    values_returned['signal'] = 1.0

                # Enhancement: expose target strike/price.
                values_returned['target_price'] = float(live_result.get('Price', 0.0))

                live_result.update(
                    {
                        'type_option': type_option,
                        'df_realtime': df_realtime,
                        'df_realtime_not_clipped': df_realtime_not_clipped,
                        'close_col': close_col,
                        'open_col': open_col,
                        'high_col': high_col,
                        'low_col': low_col,
                        'model_info': model_info
                    }
                )

                if verbose:
                    print(f"\n🚨 LIVE SIGNAL DETECTED FOR {live_result['Date']}!")
                    print(f"   Action       : {live_result['Signal']}  ({type_option})")
                    print(f"   Entry Price  : {live_result['Price']:.2f}")
                    print(f"   Execution    : {live_result['Entry_Execution']}")
            else:
                if verbose:
                    print(f"\nℹ️ Result: {live_result}")

            values_returned.update({'local_results': live_result})

        except Exception as e:
            print(f"❌ Error during real-time processing: {e}")
            traceback.print_exc()

        return values_returned

    # ==========================================
    # --- OPTIMIZATION & BACKTEST SECTION ---
    # ==========================================

    # Map parsed arguments to variables
    ticker = args.ticker
    output_dir = args.output_dir
    lookahead = args.lookahead
    dataset_id = args.dataset_id
    n_trials = args.n_trials
    timeout = args.timeout
    verbose_list_trades = args.verbose_list_trades
    min_density_threshold = args.min_density_threshold
    test_split_n = args.test_split_n
    sell_offset = args.sell_offset
    buy_offset = args.buy_offset
    trade_direction = args.trade_direction
    delta = args.delta
    do_plot = args.plot

    # Enhancement:
    # Original optimization allowed sell_offset slightly below 1 and buy_offset
    # slightly above 1, but realtime asserted the opposite. This aligns the rules.
    # Use explicit validation instead of assert for production robustness.
    if sell_offset < 1.0:
        raise ValueError("Sell offset must be >= 1.0 for Call Credit Spreads.")
    if buy_offset > 1.0:
        raise ValueError("Buy offset must be <= 1.0 for Put Credit Spreads.")
    if lookahead < 1:
        raise ValueError("lookahead must be >= 1.")

    # Depend on ticker
    open_col = ('Open', ticker)
    close_col = ('Close', ticker)
    high_col = ('High', ticker)
    low_col = ('Low', ticker)

    if verbose:
        print(
            f"Dataset: {dataset_id} | Lookahead: {lookahead} bars | "
            f"Minimum Density: {min_density_threshold} | Trade Direction: {trade_direction} | "
            f"Delta: {delta} | Sell Offset: {sell_offset:.6} | Buy Offset: {buy_offset:.6}"
        )

    df_main = factory_load_data(
        _dataset_id=dataset_id,
        _ticker=ticker,
        _args={"clip_n": clip_n}
    )

    if df_main is None or len(df_main) == 0:
        print("❌ No data loaded for optimization.")
        return

    # Enhancement:
    # Clean the master dataframe before train/test splitting so all later
    # positional indices and time-series splits use aligned OHLC rows only.
    df_main, _, _, _, _ = prepare_clean_ohlc(
        df=df_main,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col
    )

    if len(df_main) < 20:
        print("❌ Not enough cleaned data rows for optimization.")
        return

    n = int(len(df_main) * test_split_n)

    if n <= 0 or n >= len(df_main):
        print("❌ Invalid train/test split. Adjust --test-split-n or provide more data.")
        return

    df_train_ticker = df_main.iloc[:n].copy()
    df_test_ticker = df_main.iloc[n:].copy()

    if verbose:
        print(
            f"Train data: {df_train_ticker.index[0].strftime('%Y-%m-%d_%H:%M')}::"
            f"{df_train_ticker.index[-1].strftime('%Y-%m-%d_%H:%M')} ({len(df_train_ticker)} bars)"
        )
        print(
            f"Test data : {df_test_ticker.index[0].strftime('%Y-%m-%d_%H:%M')}::"
            f"{df_test_ticker.index[-1].strftime('%Y-%m-%d_%H:%M')} ({len(df_test_ticker)} bars)"
        )

    # --- OPTUNA OPTIMIZATION BLOCK WITH TIME SERIES CROSS VALIDATION ---
    def objective(trial):
        """
        Optuna objective function for hyperparameter optimization.

        Evaluates a specific set of strategy parameters using TimeSeriesSplit
        cross-validation to prevent look-ahead bias. It calculates a smoothed
        win rate and applies a proportional penalty if the trade density falls
        below the minimum required threshold.

        Enhancements:
        - Uses positional recording windows instead of date-based filtering.
        - Evaluates only entries that can be closed inside the validation window.
        - Stores density violation as a user attribute for feasibility inspection.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object used to suggest
                parameter values and report constraints.

        Returns:
            float: The mean cross-validated score (Laplace-smoothed win rate,
                   potentially penalized for low density) across all folds.
        """
        # 1. Suggest parameter values within a specific search space
        min_distance = trial.suggest_int('min_distance', 2, 30)
        ema_period = trial.suggest_int('ema_period', 2, 200)
        rsi_period = trial.suggest_int('rsi_period', 5, 50)
        cooldown_bar = trial.suggest_int('cooldown_bar', 0, 12)

        # Conditionally optimize RSI parameters based on trade direction to save compute time
        if trade_direction in ["buy", "both"]:
            rsi_buy_max = trial.suggest_int('rsi_buy_max', 10, 90)
        else:
            rsi_buy_max = trial.suggest_int('rsi_buy_max', 50, 50)  # Dummy value

        if trade_direction in ["sell", "both"]:
            rsi_sell_min = trial.suggest_int('rsi_sell_min', 10, 90)
        else:
            rsi_sell_min = trial.suggest_int('rsi_sell_min', 50, 50)  # Dummy value

        # 2. Setup Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=20)
        fold_scores = []

        # 3. Iterate through folds to prevent look-ahead bias
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_train_ticker)):
            if len(val_idx) == 0:
                continue

            val_start = int(val_idx[0])
            val_end = int(val_idx[-1])

            # Enhancement:
            # Only score entries that can be closed within this validation window.
            # Exit index is entry_idx + lookahead, so entry_idx must be <= val_end - lookahead.
            evaluable_end = val_end - lookahead

            if evaluable_end < val_start:
                fold_scores.append(0.0)
                continue

            available_data = df_train_ticker.iloc[:val_end + 1].copy()

            results_dict = backtest_asymmetric_strategy(
                ticker=ticker,
                df=available_data,
                min_distance=min_distance,
                lookahead=lookahead,
                close_col=close_col,
                open_col=open_col,
                low_col=low_col,
                high_col=high_col,
                sell_offset=sell_offset,
                buy_offset=buy_offset,
                ema_period=ema_period,
                rsi_period=rsi_period,
                rsi_buy_max=rsi_buy_max,
                rsi_sell_min=rsi_sell_min,
                trade_direction=trade_direction,
                delta=delta,
                cooldown_bar=cooldown_bar,
                record_start_idx=val_start,
                record_end_idx=evaluable_end
            )
            if isinstance(results_dict, str):
                fold_scores.append(0.0)
                continue

            closed_trades_count = len(results_dict['closed_trades'])
            if closed_trades_count == 0:
                fold_scores.append(0.0)
                continue

            wins = results_dict['wins']

            # ==========================================
            # IMPROVEMENT A: Better Score Generation
            # ==========================================
            # Laplace Smoothing (Bayesian Average).
            # Formula: (Wins + 1) / (Total + 2).
            # This prevents the optimizer from favoring 1 lucky trade (100% WR).
            # It forces the strategy to have a high trade volume to overcome the statistical penalty.
            smoothed_win_rate = (wins + 1) / (closed_trades_count + 2)

            # Enhancement:
            # results_dict['density'] is computed only over the evaluable entry
            # window, not the entire available history.
            density = results_dict['density']

            # ==========================================
            # IMPROVEMENT C: Strict Non-Linear Penalty
            # ==========================================
            # Using a cubic penalty drastically lowers the score of infeasible trials.
            # This prevents low-density/high-win-rate trials from outscoring feasible ones
            # before Optuna's internal constraint model has fully learned the search space.
            if density >= min_density_threshold:
                score = smoothed_win_rate
            else:
                if min_density_threshold > 0:
                    score = smoothed_win_rate * ((density / min_density_threshold) ** 3)
                else:
                    score = smoothed_win_rate

            fold_scores.append(score)

        if not fold_scores:
            return 0.0

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        alpha = 0.5
        final_score = mean_score - (alpha * std_score)
        return final_score

    # Ensure at least 10 startup trials to prevent crashes with low n_trials
    n_startup_trials = max(10, min(1000, int(0.0125 * n_trials)))

    if verbose:
        print(f"Starting Optuna optimization with TimeSeriesSplit and {n_startup_trials} random trials...")

    # Initialize early stopping threshold callback
    early_stopping_cb = EarlyStoppingThresholdCallback(threshold=0.99)

    # Initialize TPE Sampler with constraint handling enabled
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=n_startup_trials,
    )

    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True if verbose else False,
        n_jobs=1,
        callbacks=[early_stopping_cb]
    )

    # --- SAFELY EXTRACT BEST TRIAL (Handling Infeasible Scenarios) ---
    try:
        # Try to get the best trial.
        best_trial = study.best_trial
    except ValueError:
        print("❌ Optuna did not produce any completed trials.")
        return

    if verbose:
        print("\n" + "=" * 80)
        print(" OPTUNA OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Best Cross-Validation Score: {best_trial.value:.8f}")
        print("Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")

    # --- FINAL BACKTEST WITH BEST PARAMETERS ---
    best_params = best_trial.params

    # If cooldown was fixed on the command line, it was not suggested by Optuna.
    # Inject it into best_params so downstream code and saved models stay consistent.
    min_distance = best_params['min_distance']
    ema_period = best_params['ema_period']
    rsi_period = best_params['rsi_period']
    rsi_buy_max = best_params['rsi_buy_max']
    rsi_sell_min = best_params['rsi_sell_min']
    cooldown_bar = best_params['cooldown_bar']

    if verbose:
        print("\n" + "=" * 80)
        print(" FINAL EVALUATION WITH BEST PARAMETERS")
        print("=" * 80)
        print(f"Trade Direction         : {trade_direction.upper()}")
        print(f"Min Distance            : {min_distance}")
        print(f"Cooldown Bar            : {cooldown_bar}")
        print(f"EMA Period              : {ema_period}")
        print(f"RSI Period              : {rsi_period}")
        print(f"RSI Buy Max (Put Spread): {rsi_buy_max}")
        print(f"RSI Sell Min (Call Sprd): {rsi_sell_min}")
        print("=" * 80)

    # Evaluate on Train Data
    train_results = backtest_asymmetric_strategy(
        ticker=ticker,
        df=df_train_ticker,
        min_distance=min_distance,
        lookahead=lookahead,
        close_col=close_col,
        open_col=open_col,
        low_col=low_col,
        high_col=high_col,
        buy_offset=buy_offset,
        sell_offset=sell_offset,
        ema_period=ema_period,
        rsi_period=rsi_period,
        rsi_buy_max=rsi_buy_max,
        rsi_sell_min=rsi_sell_min,
        trade_direction=trade_direction,
        delta=delta,
        cooldown_bar=cooldown_bar,
        record_start_idx=0,
        record_end_idx=len(df_train_ticker) - 1
    )

    # Enhancement:
    # Evaluate TEST using the full historical dataframe so EMA, RSI, pivots, and
    # cooldown inherit training context. Only trades entering inside the test
    # window are recorded.
    test_results = backtest_asymmetric_strategy(
        ticker=ticker,
        df=df_main,
        min_distance=min_distance,
        lookahead=lookahead,
        close_col=close_col,
        open_col=open_col,
        low_col=low_col,
        high_col=high_col,
        buy_offset=buy_offset,
        sell_offset=sell_offset,
        ema_period=ema_period,
        rsi_period=rsi_period,
        rsi_buy_max=rsi_buy_max,
        rsi_sell_min=rsi_sell_min,
        trade_direction=trade_direction,
        delta=delta,
        cooldown_bar=cooldown_bar,
        record_start_idx=n,
        record_end_idx=len(df_main) - 1
    )

    def print_metrics(results_dict, set_name, df_used, verbose):
        """
        Format and print performance metrics for a given dataset split.

        Args:
            results_dict (dict or str): The output dictionary from `backtest_asymmetric_strategy`,
                or an error string if the backtest failed.
            set_name (str): Identifier for the dataset split (e.g., "TRAIN", "TEST").
            df_used (pd.DataFrame): The dataframe used for this specific backtest,
                used to determine the date range.
            verbose (bool): Flag to determine if metrics should be printed to stdout.

        Returns:
            tuple: (win_rate, density) extracted from the results dictionary.
                   Returns (None, None) if the backtest failed.
        """
        if df_used is None or len(df_used) == 0:
            print(f"\n{set_name} Set: no data available.")
            return None, None

        if isinstance(results_dict, str):
            print(f"\n{set_name} Set: {results_dict}")
            return None, None

        df_trades = results_dict['df_trades']
        total_trades = results_dict['total_trades']
        density = results_dict['density']
        closed_trades = results_dict['closed_trades']
        open_trades = results_dict['open_trades']
        wins = results_dict['wins']
        losses = results_dict['losses']
        win_rate = results_dict['win_rate']

        if verbose:
            print(f"\n--- {set_name} SET PERFORMANCE METRICS ({ticker}) ---")
            print(
                f"# Bars                  : {len(df_used)}  "
                f"({df_used.index[0].strftime('%Y-%m-%d_%H:%M')}::"
                f"{df_used.index[-1].strftime('%Y-%m-%d_%H:%M')})"
            )
            print(f"Total Signals Generated : {total_trades}")
            print(f"Density                 : {density:.2%}")
            print(f"Closed Trades           : {len(closed_trades)}")
            print(f"Open Trades (Active)    : {open_trades}")
            print(f"Wins ✅                 : {wins}")
            print(f"Losses ❌               : {losses}")
            print(f"Win Rate (Closed)       : {win_rate:.2%}")

        return win_rate, density

    train_wr, train_den = print_metrics(train_results, "TRAIN", df_train_ticker, verbose)
    test_wr, test_den = print_metrics(test_results, "TEST", df_test_ticker, verbose)

    if verbose:
        print("\n" + "=" * 80)
        print(" COMPARISON: TRAIN vs TEST")
        print("=" * 80)
        if train_wr is not None and test_wr is not None:
            print(
                f"Win Rate - Train: {train_wr:.2%} | Test: {test_wr:.2%} | "
                f"Difference: {test_wr - train_wr:+.2%}"
            )
            print(
                f"Density  - Train: {train_den:.2%} | Test: {test_den:.2%} | "
                f"Difference: {test_den - train_den:+.2%}"
            )
        print("=" * 80 + "\n")

    if not isinstance(test_results, str) and len(test_results['df_trades']) > 0:
        if verbose_list_trades:
            print(
                f"Train data: {df_train_ticker.index[0].strftime('%Y-%m-%d_%H%M')}::"
                f"{df_train_ticker.index[-1].strftime('%Y-%m-%d_%H%M')}  ({len(df_train_ticker)} bars)"
            )
            print(
                f"Test data : {df_test_ticker.index[0].strftime('%Y-%m-%d_%H%M')}::"
                f"{df_test_ticker.index[-1].strftime('%Y-%m-%d_%H%M')}  ({len(df_test_ticker)} bars)"
            )
            print("Sample TEST Trades (First & Last 5):")
            # Added: use a formatted display that makes entry/exit timing and profit clearer.
            format_trade_samples(test_results['df_trades'], first_n=5, last_n=5)

    if verbose:
        # ==========================================
        # --- SAVE BEST MODEL SECTION ---
        # ==========================================
        print("\n" + "=" * 80)
        print(" SAVING BEST MODEL")
        print("=" * 80)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a pertinent filename based on ticker and timestamp
    safe_ticker = ticker.replace('^', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guard against rare failed backtests so filename generation does not crash.
    safe_test_wr = test_wr if test_wr is not None else 0.0
    safe_test_den = test_den if test_den is not None else 0.0

    model_filename = (
        f"apcs_{safe_ticker}__dir{trade_direction}__ds{dataset_id}__bo{buy_offset}__so{sell_offset}__"
        f"d{delta}__la{lookahead}__cb{cooldown_bar}__md{min_density_threshold}__"
        f"twr{safe_test_wr:.8f}__td{safe_test_den:.4f}___{timestamp}.pkl"
    )
    model_path = os.path.join(output_dir, model_filename)

    # Prepare the model data to save
    best_model_data = {
        "best_params": best_params,
        "best_cv_score": best_trial.value,
        "ticker": ticker,
        "lookahead": lookahead,
        "cooldown_bar": cooldown_bar,
        "sell_offset": sell_offset,
        "buy_offset": buy_offset,
        "trade_direction": trade_direction,
        "train_wr": train_wr,
        "test_wr": test_wr,
        "train_den": train_den,
        "test_den": test_den,
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "train_length": len(df_train_ticker),
        "test_length": len(df_test_ticker),
        "command_line": command_line,
    }

    with open(model_path, 'wb') as f:
        pickle.dump(best_model_data, f)

    if verbose:
        print(f"✅ Successfully saved the best model parameters and metadata to:")
        print(f"   {os.path.abspath(model_path)}")
        print("=" * 80 + "\n")

    # ==========================================
    # --- OPTIONAL PLOTTING SECTION ---
    # ==========================================
    # Plot only after optimization/backtest, and only if the user requested it.
    # By default, plotting is disabled.
    if do_plot:
        if verbose:
            print("\n" + "=" * 80)
            print(" PLOTTING TEST SET")
            print("=" * 80)

        plot_test_trades(
            df_test=df_test_ticker,
            test_results=test_results,
            close_col=close_col,
            ticker=ticker,
            dataset_id=dataset_id,
            lookahead=lookahead,
            trade_direction=trade_direction
        )


if __name__ == '__main__':
    """
    Script initialization and CLI argument parsing.

    Sets global random seeds to ensure reproducibility of Optuna's TPE sampler
    and any stochastic processes. Configures the argparse CLI to accept strategy
    parameters, optimization constraints, and execution mode flags.
    """
    # Set global random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # ==========================================
    # --- ARGPARSE CLI CONFIGURATION ---
    # ==========================================
    parser = argparse.ArgumentParser(
        description="Optimize and backtest an asymmetric trading strategy using Optuna and TimeSeriesSplit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core trading parameters
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Ticker symbol to backtest (e.g., ^GSPC for S&P 500).")
    parser.add_argument("--dataset-id", type=str, default="day", help="Dataset ID used for fetching cached master data.")
    parser.add_argument("--use-realtime-data", action="store_true", default=False, help="Use FYahoo! to get realtime data.")
    parser.add_argument("--sell-offset", type=float, default=1.0001, help="Multiplier for sell trade win condition (Call Credit Spread).")
    parser.add_argument("--buy-offset", type=float, default=0.9999, help="Multiplier for buy trade win condition (Put Credit Spread).")

    # Backtest and Optimization parameters
    parser.add_argument("--lookahead", type=int, default=1, help="Number of bars to look ahead for determining trade outcome.")
    parser.add_argument("--n-trials", type=int, default=99, help="Number of trials for Optuna optimization.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for Optuna optimization.")
    parser.add_argument("--min-density-threshold", type=float, default=0.2, help="Minimum trade density threshold for scoring without penalty.")
    parser.add_argument("--test-split-n", type=float, default=0.8, help="Proportion of data to use for training (the rest is test).")
    parser.add_argument("--clip-n", type=int, default=0, help="Number of most recent bars to clip from the dataset.")

    parser.add_argument("--trade-direction", type=str, choices=["buy", "sell", "both"], default="both", help="Optimize and trade only 'buy', only 'sell', or 'both' (default).")
    parser.add_argument("--delta", type=float, default=0.0, help="Minimum percentage gain required from entry price for a trade to be considered a Win (e.g. 0.01 for 1%%).")

    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save and load the trained models.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output (e.g., Optuna progress bar).")
    parser.add_argument("--verbose-list-trades", action="store_true", default=False, help="Enable verbose for first/last trades")

    # Plotting parameter
    parser.add_argument("--plot", action="store_true", default=False, help="After optimization, plot the TEST set with winning trades marked by a bold 'W' and losing trades marked by a bold 'L'.")

    # Real-time mode parameters
    parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time mode to check for live signals using a saved model.")
    parser.add_argument("--model-file", type=str, default=None, help="Specific model filename to load in real-time mode. If not provided, loads the latest model.")

    the_args = parser.parse_args()

    entry(args=the_args)