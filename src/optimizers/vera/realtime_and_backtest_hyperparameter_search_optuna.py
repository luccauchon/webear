# ==============================================================================
# IMPORTS & PATH FALLBACK
# ==============================================================================
try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found.
    # This allows the script to run directly from nested directories without installation.
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from multiprocessing import freeze_support, Lock, Process, Queue, Value
from argparse import Namespace
from runners.atr import entry as atr
from fetchers.data_factory import factory_load_data
from runners.streak_probability import new_main as streak_prob
from runners.streak_probability import add_streak_columns
import numpy as np
from datetime import datetime, time, timedelta
from tqdm import tqdm
import math
import argparse
import pickle
import os
from collections import defaultdict  # Added for grouping results
from time import sleep # Added to prevent conflict with datetime.time


# ==============================================================================
# HELPER FUNCTIONS: INTRADAY METRICS
# ==============================================================================

def calculate_vix_intraday_change(df_vix_1min_day, vix_close_col):
    """
    Calculates the intraday change in VIX from the open to 15:45,
    plus the specific change in the last hour (14:45 to 15:45).

    Returns:
    - vix_total_pct: Total percentage change from open to 15:45.
    - vix_late_abs: Absolute point change in VIX during the last hour (14:45-15:45).
    """
    # Fallback if data is missing or incomplete
    if len(df_vix_1min_day) < 10:
        return 0.0, 0.0

    # 1. Total Intraday Change (Open to 15:45)
    vix_open = df_vix_1min_day[vix_close_col].iloc[0]
    vix_current = df_vix_1min_day[vix_close_col].iloc[-1]

    vix_total_abs = vix_current - vix_open
    vix_total_pct = (vix_total_abs / vix_open) * 100.0 if vix_open != 0 else 0.0

    # 2. Late Day Change (14:45 to 15:45)
    # Crucial metric: catches sudden afternoon fear spikes that ruin Iron Condors
    late_day_vix = df_vix_1min_day.between_time('14:45', '15:45')[vix_close_col]

    if len(late_day_vix) >= 2:
        vix_late_abs = late_day_vix.iloc[-1] - late_day_vix.iloc[0]
    else:
        vix_late_abs = 0.0

    return vix_total_pct, vix_late_abs


def calculate_volatility_decay(df_1min_day, close_col):
    """
    Calculates the ratio of afternoon volatility to morning volatility.
    A ratio < 1.0 indicates volatility decay (market going to sleep).
    A ratio > 1.0 indicates volatility expansion (market waking up).

    Returns:
    - volatility_ratio: Afternoon Vol / Morning Vol
    - morning_vol: Std dev of morning 1-min returns
    - afternoon_vol: Std dev of afternoon 1-min returns
    """
    # 1. Slice the data for the two 1-hour windows
    # Morning: First hour of trading (09:30 to 10:30)
    morning_data = df_1min_day.between_time('09:30', '10:30')[close_col]

    # Afternoon: The hour right before the 15:45 execution (14:45 to 15:45)
    afternoon_data = df_1min_day.between_time('14:45', '15:45')[close_col]

    # Fallback if there isn't enough data (e.g., holiday early close)
    if len(morning_data) < 10 or len(afternoon_data) < 10:
        return 1.0, 0.0, 0.0

    # 2. Calculate 1-minute percentage returns
    morning_returns = morning_data.pct_change().dropna()
    afternoon_returns = afternoon_data.pct_change().dropna()

    # 3. Calculate standard deviation (our proxy for volatility)
    morning_vol = morning_returns.std()
    afternoon_vol = afternoon_returns.std()

    # Prevent division by zero if the morning was perfectly flat
    if morning_vol == 0 or np.isnan(morning_vol):
        return 1.0, morning_vol, afternoon_vol

    # 4. Calculate the ratio
    volatility_ratio = afternoon_vol / morning_vol

    return volatility_ratio, morning_vol, afternoon_vol


def calculate_intraday_trend_slope(df_1min_window, close_col):
    """
    Calculates the linear regression slope and R-squared of the close prices
    over a specific time window (e.g., the last 2 hours).

    Returns:
    - slope_points: Raw slope in points per minute.
    - slope_pct: Normalized slope in percentage per minute.
    - r_squared: Trend strength (0.0 to 1.0).
    """
    # Fallback if there isn't enough data (e.g., early close or missing data)
    if len(df_1min_window) < 10:
        return 0.0, 0.0, 0.0

    # Extract close prices as a numpy array
    closes = df_1min_window[close_col].values

    # X values: 0, 1, 2... representing each minute in the window
    x = np.arange(len(closes))

    # 1. Calculate Linear Regression (Degree 1)
    # np.polyfit returns [slope, intercept]
    slope_points, intercept = np.polyfit(x, closes, 1)

    # 2. Calculate R-squared (Coefficient of Determination)
    # Measures how well the regression line approximates the real data points
    y_pred = slope_points * x + intercept
    ss_res = np.sum((closes - y_pred) ** 2)
    ss_tot = np.sum((closes - np.mean(closes)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # 3. Normalize slope to percentage per minute
    # (Makes it easier to compare across different price levels/days)
    start_price = closes[0]
    slope_pct = (slope_points / start_price) * 100.0 if start_price != 0 else 0.0

    return slope_points, slope_pct, r_squared


def calculate_vwap_and_distance(df_1min_day, ticker, close_col, high_col, low_col, volume_col):
    """
    Calculates cumulative intraday VWAP and the distance of the current price from VWAP.

    Parameters:
    - df_1min_day: DataFrame filtered for a single trading day (1-min data).
    - ticker: Ticker symbol.
    - close_col, high_col, low_col, volume_col: Column names/tuples.

    Returns:
    - vwap_series: The full VWAP series for the day (useful for charting).
    - vwap_distance_abs: Absolute point distance from VWAP at the last timestamp.
    - vwap_distance_pct: Percentage distance from VWAP at the last timestamp.
    """
    # Extract the necessary columns
    close = df_1min_day[close_col]
    high = df_1min_day[high_col]
    low = df_1min_day[low_col]
    volume = df_1min_day[volume_col]

    # 1. Calculate Typical Price
    typical_price = (high + low + close) / 3.0

    # 2. Calculate cumulative VWAP
    tp_volume = typical_price * volume

    # Cumulative sums (this creates the running VWAP for every minute)
    cum_tp_volume = tp_volume.cumsum()
    cum_volume = volume.cumsum()

    # Prevent division by zero just in case
    cum_volume = cum_volume.replace(0, np.nan)

    vwap_series = cum_tp_volume / cum_volume

    # 3. Calculate Distance at the last row (15:45)
    last_close = close.iloc[-1]
    last_vwap = vwap_series.iloc[-1]

    # Absolute distance (in points, e.g., +2.5 means price is 2.5 points above VWAP)
    vwap_distance_abs = last_close - last_vwap

    # Percentage distance (e.g., 0.0015 means price is 0.15% above VWAP)
    vwap_distance_pct = (last_close - last_vwap) / last_vwap

    return vwap_series, vwap_distance_abs, vwap_distance_pct


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def setup_argparse() -> argparse.ArgumentParser:
    """
    Configures and returns the ArgumentParser for the Volatility Decay &
    Regression Analysis engine.

    This script is designed to backtest or run in realtime an SPX Iron Condor /
    Credit Spread strategy executed at 15:45 EST, utilizing ATR, streak
    probabilities, VWAP, trend slopes, and VIX metrics as filters.
    """
    # Initialize parser with default help formatter showing default values
    parser = argparse.ArgumentParser(
        description="Volatility Decay & Regression Analysis for SPX Credit Spreads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol to analyze.')
    parser.add_argument('--back-n-days', type=int, default=100, help='Number of historical trading days to simulate.')
    parser.add_argument('--intraday-candle-space', type=int, default=15, help='Timeframe in minutes for the execution candle.')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of simulation trials.')
    parser.add_argument('--tightness-weight', type=float, default=0.33, help='Weighting factor for ATR tightness.')
    parser.add_argument("--n-split", type=float, default=0.9, help='Data split ratio for probability models.')
    parser.add_argument('--execution-mode', type=str, default='backtest', choices=['backtest', 'realtime', 'optimize'], help='Operational mode.')
    parser.add_argument('--save-ml-dataset', action=argparse.BooleanOptionalAction, default=False, help='Save ML dataset.')
    parser.add_argument('--verbose-print-progress-bar', action=argparse.BooleanOptionalAction, default=False, help='Display tqdm progress bar.')
    parser.add_argument('--verbose-print-continously-trade', action=argparse.BooleanOptionalAction, default=False, help='Print detailed trade metrics continuously.')
    parser.add_argument('--verbose-print-continously-only-losing-trade', action=argparse.BooleanOptionalAction, default=False, help='Print metrics ONLY for losing trades.')
    parser.add_argument('--verbose-print-results', action=argparse.BooleanOptionalAction, default=True, help='Print final summary report.')
    return parser


# ==============================================================================
# BACKTESTING & REALTIME ENGINE
# ==============================================================================

def realtime_and_backtesting_mode(args):
    ###########################################################################
    # 1. Variables & Configuration Setup
    ###########################################################################
    # Define MultiIndex column names for the dataframe (Ticker, Metric)
    volume_col = ('Volume', args.ticker)
    close_col = ('Close', args.ticker)
    open_col = ('Open', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)
    assert 1 <= args.intraday_candle_space <= 384
    dataset_id = f"intraday-{args.intraday_candle_space}min"
    realtime = args.execution_mode in ["realtime"]

    # Tracking variables for PnL and results
    str_resulting = ""
    put_spread_results, call_spread_results, all_loss_in_dollars, all_credit_in_dollars = [], [], [], []
    n_clip_value = 0
    credit_spread_size = 500  # Maximum loss per spread in dollars

    # Timestamp for unique ML dataset filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ml_dataset, ml_file_path = [], f"ml_dataset__tw{args.tightness_weight}__nt{args.n_trials}__ns{args.n_split}__rt{realtime}__{timestamp}.pkl"

    # Initial console output
    realtime_str_tmp = '' if realtime else f"| Roll back: {args.back_n_days} days"
    if getattr(args, 'verbose_print_results', False):
        print(f"[{args.ticker}] | ATR: Tightness-weight={args.tightness_weight} , n-trials={args.n_trials} , n-split={args.n_split} | Candle: {args.intraday_candle_space}min {realtime_str_tmp} ")
    if getattr(args, 'verbose_print_results', False) and realtime:
        print(f"Realtime mode activated")

    ###########################################################################
    # 2. Load 1-minute Historical/Realtime Data
    ###########################################################################
    df_ticker_1min_data, df_vix_1min_data = factory_load_data(
        _dataset_id="intraday-1min", _ticker=args.ticker, _args={"get_vix": True, "realtime": realtime}
    )

    ###########################################################################
    # 3. Main Backtesting / Realtime Loop
    ###########################################################################
    # Iterate through the specified number of historical days
    loop_iter = tqdm(range(0, args.back_n_days), desc="Backtesting") if getattr(args, 'verbose_print_progress_bar', False) else range(0, args.back_n_days)

    for n_days in loop_iter:
        # --- ATR & Streak Probability Calculations ---
        # Calculate expected range (ATR) and directional streak probabilities
        atr_results = atr(Namespace(
            ticker=args.ticker, dataframe=None, dataset_id=dataset_id, use_realtime_data=realtime, verbose=False,
            atr_window=14, tightness_weight=args.tightness_weight, clip_n=n_clip_value,
            n_split=args.n_split, n_trials=args.n_trials, timeout=9999, use_close_for_range=True
        ))
        df_bt, vix_col, atr_col = atr_results['dataframe_and_cols']

        # Calculate probabilities for positive and negative streaks
        pos_direction = streak_prob(args=Namespace(ticker=args.ticker, dataset_id=dataset_id, direction="pos", max_n=15, min_n=0, delta=0., verbose=False, debug_verify_speeding=False, forward_steps=1, epsilon=0.), bring_my_own_df=df_bt)
        neg_direction = streak_prob(args=Namespace(ticker=args.ticker, dataset_id=dataset_id, direction="neg", max_n=15, min_n=0, delta=0., verbose=False, debug_verify_speeding=False, forward_steps=1, epsilon=0.), bring_my_own_df=df_bt)

        # Extract current streak state
        dict_streak = add_streak_columns(df=df_bt, col_name=close_col, ticker=args.ticker)
        previous_candle_streak_information = dict_streak['df'][[dict_streak['col_streak_number'], dict_streak['col_streak_direction']]].iloc[-2]
        previous_candle_streak_number = previous_candle_streak_information.iloc[0] - 1
        previous_candle_streak_dir = previous_candle_streak_information.iloc[1]
        this_candle_streak_number = previous_candle_streak_number + 1

        # Determine current and inverse streak probabilities based on previous direction
        streak_proba, inv_streak_proba = None, None
        if previous_candle_streak_dir == 'pos':
            try:
                streak_proba = pos_direction[this_candle_streak_number]['prob']
            except:
                streak_proba = 0.001 # this_candle_streak_number is to high! so fix a dump value of 0.1%
            inv_streak_proba = neg_direction[0]['prob']
        else:
            try:
                streak_proba = neg_direction[this_candle_streak_number]['prob']
            except:
                streak_proba = 0.001 # this_candle_streak_number is to high! so fix a dump value of 0.1%
            inv_streak_proba = pos_direction[0]['prob']

        # --- Extract Regime Metrics & Expected Bounds ---
        regime = atr_results['realtime']['vix_regime']
        atr_prob_up = atr_results['regime_metrics'][regime]['Borne Haute Respectée'] / 100.
        atr_prob_down = atr_results['regime_metrics'][regime]['Borne Basse Respectée'] / 100.

        # Calculate expected credit/breakeven in dollars
        be_up = credit_spread_size * (1. - atr_prob_up)
        be_down = credit_spread_size * (1. - atr_prob_down)
        all_credit_in_dollars.append(be_up)
        all_credit_in_dollars.append(be_down)

        # Target date for slicing 1-min data
        target_date = df_bt.index[-1].strftime('%Y-%m-%d')

        atr_predicted_high = atr_results['realtime']['predicted_high']
        atr_predicted_low = atr_results['realtime']['predicted_low']
        open_value__last_candle_of_current_day   = df_bt.loc[target_date].iloc[-1][open_col]
        target_value__last_candle_of_current_day = df_bt.loc[target_date].iloc[-1][close_col]

        #######################################################################
        # FILTER 1: VWAP Distance (The "Extended" vs "Hugging" Filter)
        #######################################################################
        # Slice 1-min data from market open to execution time (15:45)
        df_1min_day = df_ticker_1min_data.loc[f"{target_date} 09:30:00":f"{target_date} 15:45:00"].copy()
        vwap_series, vwap_dist_abs, vwap_dist_pct = calculate_vwap_and_distance(df_1min_day, args.ticker, close_col, high_col, low_col, volume_col)

        vwap_str = f"VWAP:{vwap_dist_pct:+.2%}"
        vwap_barrier = 0.

        # If price is extended > 0.20% from VWAP, flag for mean-reversion risk
        if abs(vwap_dist_pct) > 0.0020:
            vwap_barrier += 1.

        #######################################################################
        # FILTER 2: Intraday Trend Slope (The "Chop" vs "Strong Trend" Filter)
        #######################################################################
        # Slice the last 2 hours (13:45 to 15:45) to evaluate late-day momentum
        df_1min_last_2h = df_ticker_1min_data.loc[f"{target_date} 13:45:00":f"{target_date} 15:46:00"].copy()
        slope_pts, slope_pct, r_sq = calculate_intraday_trend_slope(df_1min_last_2h, close_col)

        trend_str = f"T:{slope_pct:+.3f}%/m (R²:{r_sq:.2f})"
        slope_barrier = 0.

        # If strong trend (R² > 0.75) AND high slope (> 0.20 pts/min), flag as danger zone
        if r_sq > 0.75 and abs(slope_pts) > 0.20:
            slope_barrier += 1.

        #######################################################################
        # FILTER 3: Volatility Decay (The "Death Spiral" vs "Awakening" Filter)
        #######################################################################
        # Compare afternoon vol to morning vol to gauge market exhaustion
        volatility_ratio_barrier = 0.
        df_1min_full_day = df_ticker_1min_data.loc[f"{target_date} 09:30:00":f"{target_date} 15:45:00"].copy()
        vol_ratio, m_vol, a_vol = calculate_volatility_decay(df_1min_full_day, close_col)

        vol_str = f"V: {vol_ratio:.2f}x"

        # If afternoon vol is > 1.5x morning vol, flag as late-day awakening (danger)
        if vol_ratio > 1.50:
            volatility_ratio_barrier += 1.

        #######################################################################
        # FILTER 4: VIX Intraday Change (The "IV Crush" vs "Fear Spike" Filter)
        #######################################################################
        # Analyze VIX behavior to detect institutional hedging/fear
        vix_change_barrier = 0.
        df_vix_1min_day = df_vix_1min_data.loc[f"{target_date} 09:30:00":f"{target_date} 15:45:00"].copy()

        # Define VIX close column (Adjust to 'Close' if dataframe doesn't use MultiIndex)
        vix_close_col = ('Close', '^VIX')
        vix_total_pct, vix_late_abs = calculate_vix_intraday_change(df_vix_1min_day, vix_close_col)

        vix_str = f"VIX:{vix_total_pct:+.1f}% ({vix_late_abs:+.1f})"

        # If VIX is up for the day AND spiking in the last hour, flag as fear spike
        if vix_total_pct > 0. and vix_late_abs > 0.5:
            vix_change_barrier += 1

        #######################################################################
        # TRADE EXECUTION & PNL CALCULATION
        #######################################################################
        # Determine day direction and format metadata string for console output
        open_value__first_candle_of_current_day = df_bt.loc[target_date][open_col].iloc[0]
        direction_of_day = "🔼" if open_value__last_candle_of_current_day - open_value__first_candle_of_current_day < 0 else "🔻"
        day_meta_data_str = f"{direction_of_day}|{vwap_str} {trend_str} {vol_str} {vix_str}|{int(vwap_barrier)},{int(slope_barrier)},{int(volatility_ratio_barrier)},{int(vix_change_barrier)}"

        it_is_a_green_candle = True if target_value__last_candle_of_current_day >= open_value__last_candle_of_current_day else False
        tmp_streak_str = ""

        # ANSI Color codes for console output formatting
        if previous_candle_streak_dir == 'pos':
            up_style = "\033[4m" if it_is_a_green_candle else ""
            up_reset = "\033[24m" if it_is_a_green_candle else ""
            down_style = "" if it_is_a_green_candle else "\033[4m"
            down_reset = "" if it_is_a_green_candle else "\033[24m"
            tmp_streak_str += (f" \033[92m{up_style}▲ {streak_proba:05.1%}{up_reset}\033[0m | \033[91m{down_style}▼ {inv_streak_proba:05.1%}{down_reset}\033[0m ")
        else:
            down_style = "" if it_is_a_green_candle else "\033[4m"
            down_reset = "" if it_is_a_green_candle else "\033[24m"
            up_style = "\033[4m" if it_is_a_green_candle else ""
            up_reset = "\033[24m" if it_is_a_green_candle else ""
            tmp_streak_str += (f" \033[91m{down_style}▼ {streak_proba:05.1%}{down_reset}\033[0m | \033[92m{up_style}▲ {inv_streak_proba:05.1%}{up_reset}\033[0m")

        # --- Strike Adjustment based on Filters ---
        # Sum all triggered barriers. Each barrier widens the expected range by 5 points.
        barrier_sum = vwap_barrier + slope_barrier + volatility_ratio_barrier + vix_change_barrier

        # Widen Put strike (lower) and Call strike (higher) based on risk flags
        atr_predicted_low = math.floor((atr_predicted_low - barrier_sum) / 5) * 5
        atr_predicted_high = math.ceil((atr_predicted_high + barrier_sum) / 5) * 5

        # --- Evaluate Trade Success ---
        put_success = target_value__last_candle_of_current_day >= atr_predicted_low
        call_success = target_value__last_candle_of_current_day <= atr_predicted_high

        put_spread_results.append(put_success)
        call_spread_results.append(call_success)

        put_icon = "🟢" if put_success else "🔴"
        call_icon = "🟢" if call_success else "🔴"

        # Calculate financial loss if strikes are breached
        loss_money = 100. * (atr_predicted_low - target_value__last_candle_of_current_day) if not put_success else 0.
        loss_money += 100. * (target_value__last_candle_of_current_day - atr_predicted_high) if not call_success else 0.
        loss_money = min(credit_spread_size, loss_money)  # Cap max loss at credit spread size
        all_loss_in_dollars.append(loss_money)

        loss_str = "" if 0. == loss_money else f"L:{loss_money:.0f}$"

        # Format the daily trade string
        str_tmp = (f"{df_bt.index[-1].strftime('%Y-%m-%d_%H%M')} O@{open_value__last_candle_of_current_day:.1f} C@{target_value__last_candle_of_current_day:.1f} "
                   f"{put_icon}Put@{atr_predicted_low:.0f}:{atr_prob_down:.0%}:{be_down:.0f}$ "
                   f"|{call_icon}Call@{atr_predicted_high:.0f}:{atr_prob_up:.0%}:{be_up:.0f}$ "
                   f"{tmp_streak_str} {day_meta_data_str} {loss_str}")

        str_resulting += str_tmp + "\n"

        # Handle continuous console printing based on verbosity flags
        if getattr(args, 'verbose_print_continously_trade', False) or getattr(args, 'verbose_print_continously_only_losing_trade', False):
            if getattr(args, 'verbose_print_continously_only_losing_trade', False):
                if not put_success or not call_success: print(str_tmp)
            else:
                print(str_tmp)

        # --- Data Integrity Assertions ---
        last_datetime = df_bt.index[-1]
        last_time = df_bt.index[-1].time()

        # Verify the last candle aligns perfectly with market close minus candle space
        market_close_datetime = datetime.combine(last_datetime.date(), time(16, 0))
        expected_datetime = market_close_datetime - timedelta(minutes=args.intraday_candle_space)
        expected_last_time = expected_datetime.time()
        if last_time != expected_last_time:
            raise Exception(f"Realtime mode activated. Please run after {expected_last_time}.")
        assert last_time == expected_last_time, f"{last_time=}, {expected_last_time=}"

        if args.intraday_candle_space < 60:
            assert last_time == time(15, 60 - args.intraday_candle_space), f"{last_time=}"

        # Update clip value for next iteration
        dernier_jour = last_datetime.date().isoformat()
        nb_lignes = len(df_bt.loc[dernier_jour])
        n_clip_value += nb_lignes

        # --- Append to ML Dataset ---
        # Collect features (f__) and targets (t__) for future machine learning training
        ml_dataset.append({
            "f__put_value": atr_predicted_low / open_value__last_candle_of_current_day,
            "f__call_value": atr_predicted_high / open_value__last_candle_of_current_day,
            "f__streak_probability": streak_proba,
            "f__inverse_streak_probability": inv_streak_proba,
            "f__streak_direction": 1 if previous_candle_streak_dir == 'pos' else 0,
            "f__direction_of_day": direction_of_day,
            "f__vwap_dist_pct": vwap_dist_pct,
            "f__slope_pct": slope_pct,
            "f__r_sq": r_sq,
            "f__volatility_ratio": vol_ratio,
            "f__vix_total_pct": vix_total_pct,
            "f__vix_late_abs": vix_late_abs,
            "t__put_success": put_success,
            "t__call_success": call_success
        })  # Ajouter MA5 + Slope of MA5

        # Break loop if running in realtime mode (only 1 day processed)
        if realtime:
            break

    ###########################################################################
    # 4. Final Reporting & Output
    ###########################################################################
    if getattr(args, 'verbose_print_results', False):
        print(f"{'$' * 80}")
        print(str_resulting)

    # Calculate overall win rates
    put_win_rate = np.mean(put_spread_results)
    call_win_rate = np.mean(call_spread_results)

    # Generate visual progress bars for the console
    put_bar = "█" * int(put_win_rate * 10) + "░" * (10 - int(put_win_rate * 10))
    call_bar = "█" * int(call_win_rate * 10) + "░" * (10 - int(call_win_rate * 10))

    # Determine performance status icons
    put_status = "🟢" if put_win_rate >= 0.70 else "⚠️"
    call_status = "🟢" if call_win_rate >= 0.70 else "⚠️"

    if not realtime:
        if getattr(args, 'verbose_print_results', False):
            print("\n" + "=" * 45)
            print(f"📊   RAPPORT DE PERFORMANCE {'REALTIME' if realtime else 'BACKTESTING'} SPX500   📊")
            print("=" * 45)
            print(f"Trading days       : {'1' if realtime else args.back_n_days}")
            print(f"Put Credit Spread  : {put_win_rate:6.1%}  [{put_bar}]  {put_status}")
            print(f"Call Credit Spread : {call_win_rate:6.1%}  [{call_bar}]  {call_status}")
            print(f"Total credit $$$   : {np.sum(all_credit_in_dollars):>5.0f}$")
            print(f"Total loss $$$     : {np.sum(all_loss_in_dollars):>5.0f}$")
            print("=" * 45 + "\n")

    # Save ML dataset if requested
    if not realtime:
        if getattr(args, 'save_ml_dataset', False):
            if getattr(args, 'verbose_print_results', False): print(f"Saving results for ML analysis in {ml_file_path}")
            with open(ml_file_path, "wb") as f:
                pickle.dump(ml_dataset, f)

    return {"put_win_rate": put_win_rate, "call_win_rate": call_win_rate, "combined_win_rate": (put_win_rate+call_win_rate)/2,}


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        sleep(0.333)

    # Traitement des requêtes
    all_results_computed, run_count = [], 0
    while True:
        use_case_batch = []
        try:
            item = use_cases__shared.get(timeout=0.1)
            use_case_batch.append(item)
        except:
            break  # Queue is empty or no more items within timeout
        if 0 == len(use_case_batch):
            break
        assert 1 == len(use_case_batch)
        a_config, total_runs = use_case_batch[0]
        result = realtime_and_backtesting_mode(args=a_config)
        all_results_computed.append({
            "candle_size": a_config.intraday_candle_space,
            "tightness_weight": a_config.tightness_weight,
            "n_trials": a_config.n_trials,
            "put_win_rate": result["put_win_rate"],
            "call_win_rate": result["call_win_rate"],
            "combined_win_rate": result["combined_win_rate"]
        })
        run_count +=1
        print(f"[{os.getpid()}]  -> Run {run_count}/~{total_runs} completed (Candle: {a_config.intraday_candle_space}m, TW: {a_config.tightness_weight}, Trials: {a_config.n_trials}, "
              f"Win Rate: {result['combined_win_rate']:.2%})")

    out__shared.put(all_results_computed)


def optimization_mode(args):
    list_of_candle_sizes     = [15, 20, 25, 30, 45, 60, 75, 90, 120]
    list_of_tightness_weight = [0, 0.11, 0.33, 0.99, 4.]
    list_of_n_trials         = [1, 5, 10, 25, 50, 100, 250, 500, 999]
    back_n_days              = 100
    nb_worker                = 12

    total_runs = len(list_of_candle_sizes) * len(list_of_tightness_weight) * len(list_of_n_trials)
    print(f"🚀 Starting Optimization: {total_runs} total runs...")

    # Construction des cas à traiter
    use_cases = []
    for candle_size in list_of_candle_sizes:
        for tightness_weight in list_of_tightness_weight:
            for n_trials in list_of_n_trials:
                a_config = Namespace(ticker=args.ticker,intraday_candle_space=candle_size,execution_mode="backtest",tightness_weight=tightness_weight,
                                     n_trials=n_trials,n_split=args.n_split,back_n_days=back_n_days,)
                use_cases.append((a_config, total_runs//nb_worker))
    data_from_workers = []
    # Variables partagées
    use_cases__shared, master_cmd__shared = Queue(256000), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, nb_worker)]
    # Lancement des workers
    for k in range(0, nb_worker):
        p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k],))
        p.start()
    # Envoie les informations aux workers pour traitement
    # Préparation des lots de travail
    for use_case in use_cases:
        use_cases__shared.put(use_case)
    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1
    # Récupération des résultats
    for k in range(0, nb_worker):
        data_from_workers.extend(out__shared[k].get())

    # ==========================================================================
    # NICE PRINTING OF RESULTS
    # ==========================================================================
    print("\n" + "=" * 90)
    print("🏆 OPTIMIZATION RESULTS SUMMARY 🏆")
    print("=" * 90)
    all_results = data_from_workers
    # Sort all results by combined win rate descending
    all_results.sort(key=lambda x: x["combined_win_rate"], reverse=True)
    n_best = 100
    # 1. Overall Top n_best Best Runs
    print(f"\n🌟 TOP {n_best} OVERALL BEST RUNS (Sorted by Combined Win Rate) 🌟")
    print("-" * 90)
    print(f"{'Rank':<5} | {'Candle':<7} | {'Tightness':<10} | {'Trials':<7} | {'Put WR':<8} | {'Call WR':<8} | {'Combined WR':<11}")
    print("-" * 90)

    for i, res in enumerate(all_results[:n_best]):
        print(f"{i + 1:<5} | {res['candle_size']:<7} | {res['tightness_weight']:<10.2f} | {res['n_trials']:<7} | {res['put_win_rate']:<8.1%} | {res['call_win_rate']:<8.1%} | {res['combined_win_rate']:<11.1%}")

    # 2. Best Run for Each Candle Size & Tightness
    print("\n📊 BEST RUN FOR EACH CANDLE SIZE & TIGHTNESS 📊")
    print("-" * 90)
    print(f"{'Candle':<7} | {'Tightness':<10} | {'Trials':<7} | {'Put WR':<8} | {'Call WR':<8} | {'Combined WR':<11}")
    print("-" * 90)

    by_candle_tightness = defaultdict(list)
    for res in all_results:
        group_key = (res['candle_size'], res['tightness_weight'])
        by_candle_tightness[group_key].append(res)

    for group_key in sorted(by_candle_tightness.keys()):
        # Find the best combined win rate for this specific candle size and tightness
        best_for_group = max(by_candle_tightness[group_key], key=lambda x: x['combined_win_rate'])
        print(f"{best_for_group['candle_size']:<7} | {best_for_group['tightness_weight']:<10.2f} | {best_for_group['n_trials']:<7} | {best_for_group['put_win_rate']:<8.1%} | {best_for_group['call_win_rate']:<8.1%} | {best_for_group['combined_win_rate']:<11.1%}")

    # 3. Absolute Best Run Details
    best = all_results[0]
    print("\n" + "=" * 90)
    print(f"🥇 ABSOLUTE BEST PERFORMING RUN 🥇")
    print(f"Candle Size      : {best['candle_size']} min")
    print(f"Tightness Weight : {best['tightness_weight']}")
    print(f"N Trials         : {best['n_trials']}")
    print(f"Put Win Rate     : {best['put_win_rate']:.1%}")
    print(f"Call Win Rate    : {best['call_win_rate']:.1%}")
    print(f"Combined Win Rate: {best['combined_win_rate']:.1%}")
    print("=" * 90 + "\n")

    return None


# ==============================================================================
# MAIN EXECUTION LOGIC
# ==============================================================================
def entry(args):
    freeze_support()
    if args.execution_mode in ["realtime", "backtest"]:
        return realtime_and_backtesting_mode(args=args)
    elif args.execution_mode in ["optimize"]:
        return optimization_mode(args=args)
    return None


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)