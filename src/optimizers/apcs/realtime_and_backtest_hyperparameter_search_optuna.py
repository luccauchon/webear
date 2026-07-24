"""
Asymmetric Pivot Credit Strategy
================================

This module implements a quantitative trading strategy based on identifying
asymmetric pivot patterns (Higher Lows and Lower Highs) in financial time series
data. It is designed to trade credit spreads (Put Credit Spreads for bullish
setups and Call Credit Spreads for bearish setups).

Key Features:
- **Pattern Recognition**: Uses `scipy.signal.find_peaks` to identify significant
  market turns (peaks and valleys) and evaluates 3-turn patterns.
- **Filtering**: Incorporates Exponential Moving Average (EMA) and Relative
  Strength Index (RSI) to filter out low-probability setups and avoid
  overbought/oversold exhaustion points.
- **Optimization**: Utilizes `optuna` with `TimeSeriesSplit` cross-validation
  to find optimal parameters while preventing look-ahead bias.
- **Scoring Mechanism**: Employs Laplace smoothing for win rate calculation and
  applies strict proportional penalties for trade density violations.
- **Real-Time Execution**: Includes a dedicated mode to evaluate the most recent
  market bar against a saved, optimized model for live signal generation.

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
from scipy.signal import find_peaks
from datetime import datetime, timedelta
from utils import get_filename_for_dataset, get_next_step
import pickle
import optuna  # Added Optuna import
import random
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
import argparse  # Added for command-line arguments
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from fetchers.serialize_fyahoo import realtime as fyahoo_realtime
import math
import traceback

# Suppress Optuna & pandas debug logs for cleaner console output
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None


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
            print(f"\n🎯 Early stopping triggered! Trial {trial.number} reached score {trial.value:.4f} (>= {self.threshold}). Stopping study.")
            study.stop()


def check_live_signal(df, close_col, open_col, low_col, high_col, min_distance, ema_period, rsi_period, rsi_buy_max, rsi_sell_min, buy_offset, sell_offset):
    """
    Evaluate the most recent bar in a real-time dataframe for a valid trading signal.

    This function replicates the exact pattern recognition and filtering logic
    used in the historical backtest, but restricts its evaluation to the final
    bar of the provided dataframe. It is designed to be used in a live or
    paper-trading environment to check if a new position should be opened today.

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

    Returns:
        dict: A dictionary containing signal details ('Signal', 'Price', 'Date',
              'T3_Type', 'T3_Date', 'Neckline') if a valid entry condition is met
              on the last bar.
        str: An explanatory message detailing why no signal was generated (e.g.,
             insufficient data, pattern mismatch, or failed filters).
    """
    # ------------------------------------------------------------------
    # 1. Extract price arrays (identical to backtest_asymmetric_strategy)
    # ------------------------------------------------------------------
    prices_series = df[close_col].dropna().copy()
    ema_series = prices_series.ewm(span=ema_period, adjust=False).mean()
    ema = ema_series.values

    rsi_series = calculate_rsi(prices_series, rsi_period)
    rsi = rsi_series.values

    prices = prices_series.values
    dates = prices_series.index
    open_prices = df[open_col].dropna().copy().values
    low_prices = df[low_col].dropna().copy().values
    high_prices = df[high_col].dropna().copy().values

    assert len(prices) == len(open_prices) == len(low_prices) == len(high_prices), \
        "Mismatched lengths among OHLC columns after dropna."

    if len(prices) < 10:
        return {"reason": "Not enough data points."}

    last_bar_idx = len(prices) - 1

    # ------------------------------------------------------------------
    # 2. Peak and Valley Detection (identical to backtest)
    # ------------------------------------------------------------------
    peaks_idx, _ = find_peaks(prices, distance=min_distance)
    valleys_idx, _ = find_peaks(-prices, distance=min_distance)

    # Combine and sort turns chronologically
    turns = []
    for idx in peaks_idx:
        turns.append(("Peak", idx, prices[idx], dates[idx]))
    for idx in valleys_idx:
        turns.append(("Valley", idx, prices[idx], dates[idx]))

    turns.sort(key=lambda x: x[1])

    if len(turns) < 3:
        return {"reason": "Not enough significant turns identified to evaluate a signal."}

    # ------------------------------------------------------------------
    # 3. Evaluate the LAST window of 3 consecutive turns
    #    (exactly as the final iteration of the backtest loop would)
    # ------------------------------------------------------------------
    t1, t2, t3 = turns[-3], turns[-2], turns[-1]
    neckline_price = t2[2]

    # A peak/valley at t3[1] is only confirmed on the NEXT bar (t3[1] + 1).
    entry_idx = t3[1] + 1

    # The entry bar MUST be the last bar of the dataframe for a live signal
    if entry_idx != last_bar_idx:
        return {"reason":
            f"No signal on the last bar. Last confirmed turn at index {t3[1]} "
            f"(entry would be bar {entry_idx}), but last bar is {last_bar_idx}."}


    # ------------------------------------------------------------------
    # 4. Pattern classification (identical conditions to backtest)
    # ------------------------------------------------------------------
    trade_type = None

    # --- SELL LOGIC (Call Credit Spread) – Lower High pattern ---
    if t1[0] == "Peak" and t2[0] == "Valley" and t3[0] == "Peak":
        if t1[2] > neckline_price and t3[2] > neckline_price:
            if t3[2] < t1[2]:
                # EMA Filter: Price must be below EMA for SELL setup
                if prices[t3[1]] < ema[t3[1]]:
                    # RSI Filter: Prevent selling when oversold
                    if not np.isnan(rsi[t3[1]]) and rsi[t3[1]] >= rsi_sell_min:
                        trade_type = "SELL"

    # --- BUY LOGIC (Put Credit Spread) – Higher Low pattern ---
    elif t1[0] == "Valley" and t2[0] == "Peak" and t3[0] == "Valley":
        if t1[2] < neckline_price and t3[2] < neckline_price:
            if t3[2] > t1[2]:
                # EMA Filter: Price must be above EMA for BUY setup
                if prices[t3[1]] > ema[t3[1]]:
                    # RSI Filter: Prevent buying when overbought
                    if not np.isnan(rsi[t3[1]]) and rsi[t3[1]] <= rsi_buy_max:
                        trade_type = "BUY"

    if trade_type is None:
        return {"reason": "Last 3 turns do not form a valid SELL or BUY pattern (or failed EMA/RSI filter)."}

    # ------------------------------------------------------------------
    # 5. Entry condition check on the last bar (identical to backtest)
    # ------------------------------------------------------------------
    entry_price = None

    if trade_type == "BUY":
        if open_prices[entry_idx] >= neckline_price or high_prices[entry_idx] >= neckline_price:
            # Sell Credit Put Spread – use neckline rounded DOWN to nearest 5
            entry_price = int(math.floor(neckline_price * buy_offset / 5) * 5)
    elif trade_type == "SELL":
        if open_prices[entry_idx] <= neckline_price or low_prices[entry_idx] <= neckline_price:
            # Sell Credit Call Spread – use neckline rounded UP to nearest 5
            entry_price = int(math.ceil(neckline_price * sell_offset / 5) * 5)

    if entry_price is None:
        return {"reason":
            f"{trade_type} pattern detected but entry price conditions "
            f"not met on the last bar (bar {last_bar_idx})."}

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
        'reason': None,
    }


def backtest_asymmetric_strategy(ticker, df, close_col, open_col, low_col, high_col, min_distance, lookahead, sell_offset, buy_offset, ema_period, rsi_period, rsi_buy_max, rsi_sell_min):
    """
    Perform a historical backtest of the Asymmetric Pivot Credit Strategy.

    This function processes historical OHLC data to identify 3-turn pivot patterns
    (Higher Lows for bullish setups, Lower Highs for bearish setups). It applies
    trend (EMA) and momentum (RSI) filters, simulates trade entries based on
    price action relative to the pattern's neckline, and tracks trade outcomes
    over a specified lookahead period.

    Args:
        ticker (str): The ticker symbol being backtested (used for metadata/logging).
        df (pd.DataFrame): The historical dataframe containing OHLC data.
        close_col (tuple): Column identifier for the Close price.
        open_col (tuple): Column identifier for the Open price.
        low_col (tuple): Column identifier for the Low price.
        high_col (tuple): Column identifier for the High price.
        min_distance (int): Minimum distance between peaks/valleys for `find_peaks`.
        lookahead (int): Number of bars to hold the trade before evaluating the exit.
        sell_offset (float): Multiplier for Call Credit Spread strike selection.
        buy_offset (float): Multiplier for Put Credit Spread strike selection.
        ema_period (int): Period for the EMA trend filter.
        rsi_period (int): Period for the RSI momentum filter.
        rsi_buy_max (int): Upper RSI threshold for buying.
        rsi_sell_min (int): Lower RSI threshold for selling.

    Returns:
        dict: A dictionary containing backtest results, including:
            - 'df_trades' (pd.DataFrame): Detailed log of all generated trades.
            - 'df' (pd.DataFrame): The original input dataframe.
            - 'total_trades' (int): Total number of signals generated.
            - 'density' (float): Ratio of total trades to total bars.
            - 'closed_trades' (pd.DataFrame): Subset of trades with definitive Win/Loss.
            - 'open_trades' (int): Number of trades that reached the end of the data.
            - 'wins' (int): Count of winning trades.
            - 'losses' (int): Count of losing trades.
            - 'win_rate' (float): Win percentage of closed trades.
        str: An error/info message if the backtest could not be executed
             (e.g., insufficient data or turns).
    """
    prices_series = df[close_col].dropna().copy()

    # Calculate Indicators
    ema_series = prices_series.ewm(span=ema_period, adjust=False).mean()
    ema = ema_series.values
    rsi_series = calculate_rsi(prices_series, rsi_period)
    rsi = rsi_series.values

    prices = prices_series.values
    dates = prices_series.index
    open_prices = df[open_col].dropna().copy().values
    low_prices = df[low_col].dropna().copy().values
    high_prices = df[high_col].dropna().copy().values

    assert len(prices_series) == len(open_prices) == len(low_prices) == len(high_prices)
    if len(prices) < 10:
        return "Not enough data points."

    # Peak and Valley Detection
    peaks_idx, _ = find_peaks(prices, distance=min_distance)
    valleys_idx, _ = find_peaks(-prices, distance=min_distance)

    # Combine and sort turns chronologically
    turns = []
    for idx in peaks_idx:
        turns.append(("Peak", idx, prices[idx], dates[idx]))
    for idx in valleys_idx:
        turns.append(("Valley", idx, prices[idx], dates[idx]))

    turns.sort(key=lambda x: x[1])

    if len(turns) < 4:
        return "Not enough significant turns identified to map patterns."

    # Pre-allocate lists for faster DataFrame creation (Columnar format)
    entry_dates = []
    trade_types = []
    entry_prices = []
    outcomes = []
    exit_dates = []
    exit_prices = []

    # Sliding Window Signal Generation
    for i in range(len(turns) - 2):
        t1, t2, t3 = turns[i], turns[i + 1], turns[i + 2]
        neckline_price = t2[2]

        # A peak/valley at t3[1] is only confirmed on the NEXT bar (t3[1] + 1).
        # We must enter the trade on the confirmation bar to avoid look-ahead bias.
        entry_idx = t3[1] + 1

        # Prevent out-of-bounds for entry
        if entry_idx >= len(prices):
            continue

        trade_type = None

        # --- SELL LOGIC (Call Credit Spread) ---
        # Lower High pattern
        if t1[0] == "Peak" and t2[0] == "Valley" and t3[0] == "Peak":
            if t1[2] > neckline_price and t3[2] > neckline_price:
                if t3[2] < t1[2]:
                    # EMA Filter: Price must be below EMA for SELL setup
                    if prices[t3[1]] < ema[t3[1]]:
                        # RSI Filter: Prevent selling when oversold
                        if not np.isnan(rsi[t3[1]]) and rsi[t3[1]] >= rsi_sell_min:
                            trade_type = "SELL"

        # --- BUY LOGIC (Put Credit Spread) ---
        # Higher Low pattern
        elif t1[0] == "Valley" and t2[0] == "Peak" and t3[0] == "Valley":
            if t1[2] < neckline_price and t3[2] < neckline_price:
                if t3[2] > t1[2]:
                    # EMA Filter: Price must be above EMA for BUY setup
                    if prices[t3[1]] > ema[t3[1]]:
                        # RSI Filter: Prevent buying when overbought
                        if not np.isnan(rsi[t3[1]]) and rsi[t3[1]] <= rsi_buy_max:
                            trade_type = "BUY"

        # Trade Outcome Tracking
        if trade_type is not None:
            entry_price = None
            entry_date = None

            # Determine Entry Execution
            if trade_type == "BUY":
                if open_prices[entry_idx] >= neckline_price or high_prices[entry_idx] >= neckline_price:
                    # We sell Credit Put Spread, we gonna use the neckline for our strike price
                    entry_price = int(math.floor(neckline_price * buy_offset / 5) * 5)
                    entry_date = dates[entry_idx]
            elif trade_type == "SELL":
                if open_prices[entry_idx] <= neckline_price or low_prices[entry_idx] <= neckline_price:
                    # We sell Credit Call Spread, we gonna use the neckline for our strike price
                    entry_price = int(math.ceil(neckline_price * sell_offset / 5) * 5)
                    entry_date = dates[entry_idx]

            if entry_price is None:
                # No entry detected. We could for next bar? For now, we skip this trade.
                continue

            # The actual holding period (Time For Expiration of my credit spread) starts the bar AFTER confirmation
            # Calculate exact exit index
            exit_idx = (entry_idx + 1) + lookahead - 1

            # Handle end-of-dataframe edge case
            if exit_idx >= len(prices):
                outcome = "Open"
                exit_date = dates[-1]
                exit_price = prices[-1]
            else:
                exit_date = dates[exit_idx]
                exit_price = prices[exit_idx]

                if trade_type == "SELL":
                    outcome = "Win" if exit_price < entry_price else "Loss"
                else:  # BUY
                    outcome = "Win" if exit_price > entry_price else "Loss"

            # Append to lists
            entry_dates.append(entry_date)
            trade_types.append(trade_type)
            entry_prices.append(entry_price)
            outcomes.append(outcome)
            exit_dates.append(exit_date)
            exit_prices.append(exit_price)

    total_trades = len(outcomes)
    if total_trades == 0:
        return "No trades executed."

    # Create DataFrame once at the end (Much faster than appending dicts)
    df_trades = pd.DataFrame({
        "Entry_Date": entry_dates,
        "Type": trade_types,
        "Entry_Price": entry_prices,
        "Outcome": outcomes,
        "Exit_Date": exit_dates,
        "Exit_Price": exit_prices
    })

    # Compute Metrics (Using fast list counting instead of DF filtering)
    wins = outcomes.count("Win")
    losses = outcomes.count("Loss")
    open_trades = outcomes.count("Open")
    closed_trades_count = wins + losses

    win_rate = (wins / closed_trades_count) if closed_trades_count > 0 else 0.0
    density = total_trades / len(df)

    # Filter closed trades for the return dict
    closed_trades_df = df_trades[df_trades["Outcome"].isin(["Win", "Loss"])]

    return {
        'df_trades': df_trades,
        'df': df,
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

    # ==========================================
    # --- REAL-TIME PROCESSING SECTION ---
    # ==========================================
    if realtime:
        values_returned = {'target_date': None, 'signal': 0., 'current_price': None, 'current_date': None, 'target_price': 0., 'train_score': None, 'val_score': None,
                           'train_win_rate': None, 'val_win_rate': None, 'optimization_metric': 'buy_wr', 'method': None, 'threshold': None, 'ticker': None,
                           'dataset_id': None, 'lookahead': None, 'local_results': {'reason': "no yet processed"}}
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

        if verbose: print(f"Loading model from: {model_file}")
        with open(model_file, 'rb') as f:
            model_info = pickle.load(f)

        target_date = get_next_step(the_date=datetime.now(), dataset_id=model_info['dataset_id'], nn=model_info['lookahead'])
        values_returned.update({'ticker': model_info['ticker']})
        values_returned.update({'dataset_id': model_info['dataset_id']})
        values_returned.update({'lookahead': model_info['lookahead']})
        values_returned.update({'target_date': target_date})
        values_returned.update({'current_date': str(datetime.now().strftime('%Y-%m-%d'))})
        values_returned.update({'train_score': model_info['train_wr']})
        values_returned.update({'train_win_rate': model_info['train_wr']})
        values_returned.update({'val_score': model_info['test_wr']})
        values_returned.update({'val_win_rate': model_info['test_wr']})
        df_realtime, df_realtime_not_clipped = None, None
        try:
            if use_realtime_dataset:
                if verbose: print(f"Fetching latest real-time data for {model_info['ticker']}...")
                assert args.ticker in ["^GSPC"]
                daily_data_cache, weekly_data_cache, monthly_data_cache, quaterly_data_cache, yearly_data_cache = fyahoo_realtime()
                df_realtime = daily_data_cache[model_info['ticker']].sort_index().copy()
            else:
                if verbose: print(f"Loading data for {model_info['ticker']}...")
                with open(get_filename_for_dataset(model_info['dataset_id'], older_dataset=None), 'rb') as f:
                    _master_data_cache = pickle.load(f)
                assert _master_data_cache is not None
                df_realtime = _master_data_cache[model_info['ticker']].sort_index().copy()
                df_realtime_not_clipped = df_realtime.copy()
                if clip_n > 0:
                    df_realtime = df_realtime.iloc[:-clip_n].copy()
                    if verbose: print(f"Clipping to {df_realtime.index[-1]}")
                    if 0 == len(df_realtime):
                        values_returned['local_results'].update({'reason': f'no more data with clip of {clip_n}'})
                        return values_returned
            if verbose: print(f"Win Rate - Train: {model_info['train_wr']:.2%} | Test: {model_info['test_wr']:.2%} | Difference: {model_info['test_wr'] - model_info['train_wr']:+.2%}")
            if verbose: print(f"Density  - Train: {model_info['train_den']:.2%} | Test: {model_info['test_den']:.2%} | Difference: {model_info['test_den'] - model_info['train_den']:+.2%}")
            model_params = model_info['best_params']
            # Check live signal directly on the latest bar
            close_col = ('Close', model_info['ticker'])
            open_col = ('Open', model_info['ticker'])
            high_col = ('High', model_info['ticker'])
            low_col = ('Low', model_info['ticker'])
            values_returned.update({'current_price': df_realtime[close_col].iloc[-1]})
            live_result = check_live_signal(
                df=df_realtime.copy(),
                buy_offset=model_info['buy_offset'], sell_offset=model_info['sell_offset'],
                close_col=close_col, open_col=open_col, high_col=high_col, low_col=low_col,
                min_distance=model_params['min_distance'],
                ema_period=model_params['ema_period'],
                rsi_period=model_params['rsi_period'],
                rsi_buy_max=model_params['rsi_buy_max'],
                rsi_sell_min=model_params['rsi_sell_min'],
            )

            assert isinstance(live_result, dict)
            if live_result['reason'] is None:
                type_option = None
                if live_result['Signal'] == "SELL":
                    assert model_info['sell_offset'] >= 1.
                    type_option = "Call Credit Spread"
                elif live_result['Signal'] == "BUY":
                    assert model_info['buy_offset'] <= 1.
                    type_option = "Put Credit Spread"
                live_result.update({'type_option': type_option, 'df_realtime': df_realtime, 'df_realtime_not_clipped':df_realtime_not_clipped,
                                    'close_col': close_col, 'open_col': open_col, 'high_col': high_col, 'low_col': low_col, 'model_info': model_info})
                if verbose:
                    print(f"\n🚨 LIVE SIGNAL DETECTED FOR {live_result['Date'].strftime('%Y-%m-%d')}!")
                    print(f"   Action       : {live_result['Signal']}  ({type_option})")
                    print(f"   Entry Price  : {live_result['Price']:.2f}")
            else:
                if verbose: print(f"\nℹ️ Result: {live_result}")
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

    assert sell_offset > 0.999 and buy_offset < 1.001, "Sell offset must be > 0.999 and buy offset < 1.001"

    # Depend on ticker
    open_col = ('Open', ticker)
    close_col = ('Close', ticker)
    high_col = ('High', ticker)
    low_col = ('Low', ticker)

    if verbose: print(f"Using Sell Offset of {sell_offset:.2%} and Buy Offset of {buy_offset:.2%}")
    with open(get_filename_for_dataset(dataset_id, older_dataset=None), 'rb') as f:
        _master_data_cache = pickle.load(f)
    assert _master_data_cache is not None
    df_main = _master_data_cache[ticker].sort_index().copy()
    if clip_n > 0:
        df_main = df_main.iloc[:-clip_n].copy()
    n = int(len(df_main) * test_split_n)
    df_train_ticker = df_main.iloc[:n].copy()
    df_test_ticker = df_main.iloc[n:].copy()
    if verbose: print(f"Train data: {df_train_ticker.index[0].strftime('%Y-%m-%d')}::{df_train_ticker.index[-1].strftime('%Y-%m-%d')}")
    if verbose: print(f"Test data : {df_test_ticker.index[0].strftime('%Y-%m-%d')}::{df_test_ticker.index[-1].strftime('%Y-%m-%d')}")

    # --- OPTUNA OPTIMIZATION BLOCK WITH TIME SERIES CROSS VALIDATION ---
    def objective(trial):
        """
        Optuna objective function for hyperparameter optimization.

        Evaluates a specific set of strategy parameters using TimeSeriesSplit
        cross-validation to prevent look-ahead bias. It calculates a smoothed
        win rate and applies a proportional penalty if the trade density falls
        below the minimum required threshold.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object used to suggest
                parameter values and report constraints.

        Returns:
            float: The mean cross-validated score (Laplace-smoothed win rate,
                   potentially penalized for low density) across all folds.
        """
        # 1. Suggest parameter values within a specific search space
        min_distance = trial.suggest_int('min_distance', 3, 30)
        ema_period = trial.suggest_int('ema_period', 2, 200)
        rsi_period = trial.suggest_int('rsi_period', 5, 50)
        rsi_buy_max = trial.suggest_int('rsi_buy_max', 50, 90)
        rsi_sell_min = trial.suggest_int('rsi_sell_min', 10, 50)

        # 2. Setup Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=10)
        fold_scores = []
        max_density_violation = -float('inf')  # Track the worst density violation across folds

        # 3. Iterate through folds to prevent look-ahead bias
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_train_ticker)):
            available_data = df_train_ticker.iloc[:val_idx[-1] + 1]

            results_dict = backtest_asymmetric_strategy(
                ticker=ticker, df=available_data, min_distance=min_distance, lookahead=lookahead,
                close_col=close_col, open_col=open_col, low_col=low_col, high_col=high_col,
                sell_offset=sell_offset, buy_offset=buy_offset, ema_period=ema_period,
                rsi_period=rsi_period, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min,
            )

            if isinstance(results_dict, str):
                fold_scores.append(0.0)
                max_density_violation = max(max_density_violation, min_density_threshold)
                continue

            df_trades = results_dict['df_trades']
            val_dates = df_train_ticker.iloc[val_idx].index
            fold_trades = df_trades[df_trades['Entry_Date'].isin(val_dates)]
            closed_trades = fold_trades[fold_trades['Outcome'].isin(["Win", "Loss"])]
            closed_trades_count = len(closed_trades)

            if closed_trades_count == 0:
                fold_scores.append(0.0)
                max_density_violation = max(max_density_violation, min_density_threshold)
                continue

            wins = len(fold_trades[fold_trades['Outcome'] == "Win"])

            # ==========================================
            # IMPROVEMENT A: Better Score Generation
            # ==========================================
            # Laplace Smoothing (Bayesian Average).
            # Formula: (Wins + 1) / (Total + 2).
            # This prevents the optimizer from favoring 1 lucky trade (100% WR).
            # It forces the strategy to have a high trade volume to overcome the statistical penalty.
            smoothed_win_rate = (wins + 1) / (closed_trades_count + 2)

            density = len(fold_trades) / len(val_idx)

            # ==========================================
            # IMPROVEMENT B: Strict Density Constraint
            # ==========================================
            # Record the constraint violation. Optuna requires violation <= 0 to be "feasible".
            violation = min_density_threshold - density
            if violation > max_density_violation:
                max_density_violation = violation

            # ==========================================
            # IMPROVEMENT C: Strict Proportional Penalty
            # ==========================================
            # Linear penalty: If you achieve 50% of the required density, you get 50% of the score.
            if density >= min_density_threshold:
                score = smoothed_win_rate
            else:
                score = smoothed_win_rate * (density / min_density_threshold)

            fold_scores.append(score)

        # Tell Optuna the constraints for this trial (Optuna will treat violation > 0 as INFEASIBLE)
        trial.set_user_attr("constraint", [max_density_violation])

        return np.mean(fold_scores) if fold_scores else 0.0

    if verbose: print("Starting Optuna optimization with TimeSeriesSplit...")
    # Initialize early stopping threshold callback
    early_stopping_cb = EarlyStoppingThresholdCallback(threshold=0.99)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True if verbose else False, n_jobs=1, callbacks=[early_stopping_cb])
    if verbose:
        print("\n" + "=" * 80)
        print(" OPTUNA OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Best Cross-Validation Score: {study.best_value:.8f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 80 + "\n")

    # --- FINAL BACKTEST WITH BEST PARAMETERS ---
    best_params = study.best_params
    min_distance = best_params['min_distance']
    ema_period = best_params['ema_period']
    rsi_period = best_params['rsi_period']
    rsi_buy_max = best_params['rsi_buy_max']
    rsi_sell_min = best_params['rsi_sell_min']

    if verbose:
        print("\n" + "=" * 80)
        print(" FINAL EVALUATION WITH BEST PARAMETERS")
        print("=" * 80)
        print(f"Min Distance            : {min_distance}")
        print(f"EMA Period              : {ema_period}")
        print(f"RSI Period              : {rsi_period}")
        print(f"RSI Buy Max (Put Spread): {rsi_buy_max}")
        print(f"RSI Sell Min (Call Sprd): {rsi_sell_min}")
        print("=" * 80)

    # Evaluate on Train Data
    train_results = backtest_asymmetric_strategy(
        ticker=ticker, df=df_train_ticker, min_distance=min_distance, lookahead=lookahead, close_col=close_col, open_col=open_col, low_col=low_col, high_col=high_col,
        buy_offset=buy_offset, sell_offset=sell_offset, ema_period=ema_period,
        rsi_period=rsi_period, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min,
    )

    test_results = backtest_asymmetric_strategy(
        ticker=ticker, df=df_test_ticker, min_distance=min_distance, lookahead=lookahead, close_col=close_col, open_col=open_col, low_col=low_col, high_col=high_col,
        buy_offset=buy_offset, sell_offset=sell_offset, ema_period=ema_period,
        rsi_period=rsi_period, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min,
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
            print(f"# Bars                  : {len(df_used)}  ({df_used.index[0].strftime('%Y-%m-%d')}::{df_used.index[-1].strftime('%Y-%m-%d')})")
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
            print(f"Win Rate - Train: {train_wr:.2%} | Test: {test_wr:.2%} | Difference: {test_wr - train_wr:+.2%}")
            print(f"Density  - Train: {train_den:.2%} | Test: {test_den:.2%} | Difference: {test_den - train_den:+.2%}")
        print("=" * 80 + "\n")

    if not isinstance(test_results, str) and len(test_results['df_trades']) > 0:
        if verbose_list_trades:
            print(f"Train data: {df_train_ticker.index[0].strftime('%Y-%m-%d')}::{df_train_ticker.index[-1].strftime('%Y-%m-%d')}")
            print(f"Test data : {df_test_ticker.index[0].strftime('%Y-%m-%d')}::{df_test_ticker.index[-1].strftime('%Y-%m-%d')}")
            print("Sample TEST Trades (First & Last 5):")
            print(test_results['df_trades'].head(5))
            print(test_results['df_trades'].tail(5))

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
    model_filename = f"acps_{safe_ticker}__bo{buy_offset}__so{sell_offset}__la{lookahead}__md{min_density_threshold}__twr{test_wr:.8f}__{timestamp}.pkl"
    model_path = os.path.join(output_dir, model_filename)

    # Prepare the model data to save
    best_model_data = {
        "best_params": study.best_params,
        "best_cv_score": study.best_value,
        "ticker": ticker,
        "lookahead": lookahead,
        "sell_offset": sell_offset,
        "buy_offset": buy_offset,
        "train_wr": train_wr,
        "test_wr": test_wr,
        "train_den": train_den,
        "test_den": test_den,
        "timestamp": datetime.now().isoformat(),
        "dataset_id": dataset_id,
    }

    with open(model_path, 'wb') as f:
        pickle.dump(best_model_data, f)

    if verbose:
        print(f"✅ Successfully saved the best model parameters and metadata to:")
        print(f"   {os.path.abspath(model_path)}")
        print("=" * 80 + "\n")


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
    parser.add_argument("--sell-offset", type=float, default=1.005, help="Multiplier for sell trade win condition (Call Credit Spread).")
    parser.add_argument("--buy-offset", type=float, default=0.995, help="Multiplier for buy trade win condition (Put Credit Spread).")

    # Backtest and Optimization parameters
    parser.add_argument("--lookahead", type=int, default=1, help="Number of bars to look ahead for determining trade outcome.")
    parser.add_argument("--n-trials", type=int, default=9, help="Number of trials for Optuna optimization.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds for Optuna optimization.")
    parser.add_argument("--min-density-threshold", type=float, default=0.1, help="Minimum trade density threshold for scoring without penalty.")
    parser.add_argument("--test-split-n", type=float, default=0.8, help="Proportion of data to use for training (the rest is test).")
    parser.add_argument("--clip-n", type=int, default=0, help="Number of most recent bars to clip from the dataset (for out-of-sample).")

    # Output and Execution
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save and load the trained models.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output (e.g., Optuna progress bar).")
    parser.add_argument("--verbose-list-trades", action="store_true", default=False, help="Enable verbose for fisrt/last trades")

    # Real-time mode parameters
    parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time mode to check for live signals using a saved model.")
    parser.add_argument("--model-file", type=str, default=None, help="Specific model filename to load in real-time mode. If not provided, loads the latest model.")

    the_args = parser.parse_args()

    entry(args=the_args)