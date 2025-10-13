import pandas as pd
import numpy as np


def half_trend(df, ticker_name, high_label, low_label, close_label, amplitude=2, channel_deviation=2):
    """
    Python implementation of the HalfTrend indicator (Pine Script v6).

    Parameters:
    - df: pandas DataFrame with columns ['high', 'low', 'close']
    - amplitude: int, lookback for highest/lowest and SMA
    - channel_deviation: int, multiplier for ATR channel width

    Returns:
    - df with added columns:
        'ht', 'trend', 'arrow_up', 'arrow_down', 'buy_signal', 'sell_signal',
        'atr_high', 'atr_low'
    """
    df = df.copy()
    n = len(df)

    # Initialize arrays
    trend = np.full(n, 0, dtype=int)
    next_trend = np.full(n, 0, dtype=int)
    max_low_price = np.full(n, np.nan)
    min_high_price = np.full(n, np.nan)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    atr_high = np.full(n, np.nan)
    atr_low = np.full(n, np.nan)
    arrow_up = np.full(n, np.nan)
    arrow_down = np.full(n, np.nan)
    buy_signal = np.full(n, False, dtype=bool)
    sell_signal = np.full(n, False, dtype=bool)

    # Precompute ATR(100) / 2
    # ATR calculation
    high = df[high_label].values
    low = df[low_label].values
    close = df[close_label].values

    tr = np.maximum(high - low,
                    np.abs(high - np.concatenate([[close[0]], close[:-1]])),
                    np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    atr = pd.Series(tr).rolling(window=100, min_periods=1).mean().values
    atr2 = atr / 2.0
    dev = channel_deviation * atr2

    # Precompute SMA(high, amplitude) and SMA(low, amplitude)
    high_sma = pd.Series(high).rolling(window=amplitude, min_periods=1).mean().values
    low_sma = pd.Series(low).rolling(window=amplitude, min_periods=1).mean().values

    # Precompute highestbars and lowestbars logic
    # We'll compute highPrice = high[highestbars(amplitude)], etc.
    high_price = np.full(n, np.nan)
    low_price = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - amplitude + 1)
        window_high = high[start:i + 1]
        window_low = low[start:i + 1]
        if len(window_high) > 0:
            # Find index of highest high in window (relative to current)
            highest_idx = np.argmax(window_high)
            high_price[i] = high[start + highest_idx]
            lowest_idx = np.argmin(window_low)
            low_price[i] = low[start + lowest_idx]

    # Initialize state variables
    max_low_price[0] = low[0] if n > 0 else np.nan
    min_high_price[0] = high[0] if n > 0 else np.nan

    for i in range(1, n):
        # Initialize state from previous
        trend[i] = trend[i - 1]
        next_trend[i] = next_trend[i - 1]
        max_low_price[i] = max_low_price[i - 1]
        min_high_price[i] = min_high_price[i - 1]

        if next_trend[i] == 1:
            max_low_price[i] = max(low_price[i], max_low_price[i])
            if high_sma[i] < max_low_price[i] and close[i] < low[i - 1]:
                trend[i] = 1
                next_trend[i] = 0
                min_high_price[i] = high_price[i]
        else:
            min_high_price[i] = min(high_price[i], min_high_price[i])
            if low_sma[i] > min_high_price[i] and close[i] > high[i - 1]:
                trend[i] = 0
                next_trend[i] = 1
                max_low_price[i] = low_price[i]

        # Compute up/down
        if trend[i] == 0:
            if i > 0 and trend[i - 1] != 0:
                up[i] = down[i - 1] if not np.isnan(down[i - 1]) else down[i - 1]  # fallback
                if not np.isnan(up[i]):
                    arrow_up[i] = up[i] - atr2[i]
            else:
                if i == 0:
                    up[i] = max_low_price[i]
                else:
                    up[i] = max_low_price[i] if np.isnan(up[i - 1]) else max(max_low_price[i], up[i - 1])
            atr_high[i] = up[i] + dev[i]
            atr_low[i] = up[i] - dev[i]
        else:
            if i > 0 and trend[i - 1] != 1:
                down[i] = up[i - 1] if not np.isnan(up[i - 1]) else up[i - 1]
                if not np.isnan(down[i]):
                    arrow_down[i] = down[i] + atr2[i]
            else:
                if i == 0:
                    down[i] = min_high_price[i]
                else:
                    down[i] = min_high_price[i] if np.isnan(down[i - 1]) else min(min_high_price[i], down[i - 1])
            atr_high[i] = down[i] + dev[i]
            atr_low[i] = down[i] - dev[i]

        # Signals
        if not np.isnan(arrow_up[i]) and trend[i] == 0 and (i == 0 or trend[i - 1] == 1):
            buy_signal[i] = True
        if not np.isnan(arrow_down[i]) and trend[i] == 1 and (i == 0 or trend[i - 1] == 0):
            sell_signal[i] = True

    # Final HalfTrend line
    ht = np.where(trend == 0, up, down)

    # Assign to DataFrame
    df[(f'ht{amplitude}', ticker_name)] = ht
    df[(f'trend{amplitude}', ticker_name)] = trend
    df[(f'arrow_up{amplitude}', ticker_name)] = arrow_up
    df[(f'arrow_down{amplitude}', ticker_name)] = arrow_down
    df[(f'buy_signal{amplitude}', ticker_name)] = buy_signal
    df[(f'sell_signal{amplitude}', ticker_name)] = sell_signal
    df[(f'atr_high{amplitude}', ticker_name)] = atr_high
    df[(f'atr_low{amplitude}', ticker_name)] = atr_low

    return df


def trade_prime_half_trend_strategy(ticker, ticker_name, buy_setup=True, print_signals=False):
    close_label, high_label, low_label = ('Close', ticker_name), ('High', ticker_name), ('Low', ticker_name)
    if print_signals:
        print(f"Computing HalfTrend indicator ({ticker.index[0]} => {ticker.index[-1]})...")
    ticker = half_trend(ticker, ticker_name=ticker_name, close_label=close_label, high_label=high_label, low_label=low_label, amplitude=10, channel_deviation=2)
    ticker = half_trend(ticker, ticker_name=ticker_name, close_label=close_label, high_label=high_label, low_label=low_label, amplitude=1, channel_deviation=2)
    ticker['ticker_name'] = ticker_name
    df = ticker
    assert ('ht1', ticker_name) in ticker.columns
    assert ('ht10', ticker_name) in ticker.columns
    assert ('trend1', ticker_name) in ticker.columns
    assert ('trend10', ticker_name) in ticker.columns

    # ----------------------------
    # Scanning for Buy/Sell Setup
    # ----------------------------
    if print_signals:
        print("Scanning for Setup...")
    n, i = len(df), 0
    custom_signal = np.full(n, False, dtype=bool)
    setup_triggered = np.full(n, False, dtype=bool)
    triggered_distance = np.full(n, 0, dtype=int)
    while i < n:
        if buy_setup:
            # Conditions for setup at candle i:
            # 1. ht10 is in uptrend (trend10 == 0)
            # 2. ht1 is in downtrend (trend1 == 1)
            # 3. Price (low or close) at i <= ht10[i] (touches or below)
            cond1 = df[('trend10', ticker_name)].iloc[i] == 0
            cond2 = df[('trend1', ticker_name)].iloc[i] == 1
            cond3 = df[('Low', ticker_name)].iloc[i] <= df[('ht10', ticker_name)].iloc[i]  # touches or below
            setup = cond1 and cond2 and cond3
            setup_triggered[i] = setup
            if setup:
                new_i, trigger_buy_distance = i+1, 0
                for j in range(i+1, n):
                    new_i = j
                    if df[('trend10', ticker_name)].iloc[j] == 1:
                        break
                    flip_to_up = (df[('trend1', ticker_name)].iloc[j] == 0) and (df[('trend10', ticker_name)].iloc[j] == 0)
                    if flip_to_up:
                        custom_signal[j] = True
                        triggered_distance[j] = j - i
                        break
                i = new_i
            else:
                i += 1
        else:
            # Conditions for setup at candle i:
            # 1. ht10 is in downtrend (trend10 == 1)
            # 2. ht1 is in uptrend (trend1 == 0)
            # 3. Price (high or close) at i >= ht10[i] (touches or below)
            cond1 = df[('trend10', ticker_name)].iloc[i] == 1
            cond2 = df[('trend1', ticker_name)].iloc[i] == 0
            cond3 = df[('High', ticker_name)].iloc[i] >= df[('ht10', ticker_name)].iloc[i]  # touches or above
            setup = cond1 and cond2 and cond3
            setup_triggered[i] = setup
            if setup:
                new_i, trigger_buy_distance = i+1, 0
                for j in range(i + 1, n):
                    new_i = j
                    if df[('trend10', ticker_name)].iloc[j] == 0:
                        break
                    flip_to_down = (df[('trend1', ticker_name)].iloc[j] == 1) and (df[('trend10', ticker_name)].iloc[j] == 1)
                    if flip_to_down:
                        custom_signal[j] = True
                        triggered_distance[j] = j - i
                        break
                i = new_i
            else:
                i += 1
    df[('custom_signal', ticker_name)] = custom_signal
    df[('setup_triggered', ticker_name)] = setup_triggered
    df[('triggered_distance', ticker_name)] = triggered_distance

    # ----------------------------
    # Print recent signals
    # ----------------------------
    if print_signals:
        recent_setup_triggered = df[df[('setup_triggered', ticker_name)]].tail(5)
        if not recent_setup_triggered.empty:
            print("\nRecent Setup Triggered:")
            print(recent_setup_triggered[[('Close', ticker_name), ('ht10', ticker_name), ('ht1', ticker_name)]])
        else:
            print("\nNo recent setup triggered.")

        recent_signals = df[df[('custom_signal', ticker_name)]].tail(5)
        if not recent_signals.empty:
            print("\nRecent Custom Signals:")
            print(recent_signals[[('Close', ticker_name), ('triggered_distance', ticker_name)]])
        else:
            print("\nNo recent custom signals found.")

    return df



def trade_prime_half_trend_strategy_plus_volume_confirmation_and_atr_stop_loss(ticker, ticker_name, buy_setup=True, print_signals=False):
    '''
     Enhancements:
        Volume Confirmation:
            Require that the buy candle (where HT1 flips up) has above-average volume (e.g., > 20-period SMA of volume).
            Helps filter out weak breakouts.

        ATR-Based Stop Loss:
            Compute ATR(14) (standard lookback).
            Set stop-loss at:
                entry_price - 1.5 × ATR (for longs).
            Also compute a take-profit level (e.g., entry_price + 2 × ATR) if desired.

        Store trade management levels:
            Add columns: stop_loss, take_profit, position_size (optional), and valid_signal (after volume filter).
    :param ticker:
    :param ticker_name:
    :return:
    '''
    close_label, high_label, low_label = ('Close', ticker_name), ('High', ticker_name), ('Low', ticker_name)
    print(f"Computing HalfTrend indicator ({ticker.index[0]} => {ticker.index[-1]})...")
    ticker = half_trend(ticker, ticker_name=ticker_name, close_label=close_label, high_label=high_label, low_label=low_label, amplitude=10, channel_deviation=2)
    ticker = half_trend(ticker, ticker_name=ticker_name, close_label=close_label, high_label=high_label, low_label=low_label, amplitude=1, channel_deviation=2)
    # ----------------------------
    # Volume SMA for confirmation
    # ----------------------------
    ticker[('Volume_SMA_20', ticker_name)] = ticker[('Volume', ticker_name)].rolling(window=20, min_periods=1).mean()

    df = ticker
    assert ('ht1', ticker_name) in ticker.columns
    assert ('ht10', ticker_name) in ticker.columns
    assert ('trend1', ticker_name) in ticker.columns
    assert ('trend10', ticker_name) in ticker.columns

    # ----------------------------
    # Scanning for Buy/Sell Setup
    # ----------------------------
    print("Scanning for Setup...")
    n, i = len(df), 0
    custom_signal = np.full(n, False, dtype=bool)
    setup_triggered = np.full(n, False, dtype=bool)
    triggered_distance = np.full(n, 0, dtype=int)
    while i < n:
        if buy_setup:
            # Conditions for setup at candle i:
            # 1. ht10 is in uptrend (trend10 == 0)
            # 2. ht1 is in downtrend (trend1 == 1)
            # 3. Price (low or close) at i <= ht10[i] (touches or below)
            cond1 = df[('trend10', ticker_name)].iloc[i] == 0
            cond2 = df[('trend1', ticker_name)].iloc[i] == 1
            cond3 = df[('Low', ticker_name)].iloc[i] <= df[('ht10', ticker_name)].iloc[i]  # touches or below
            setup = cond1 and cond2 and cond3
            setup_triggered[i] = setup
            if setup:
                new_i, trigger_buy_distance = i+1, 0
                for j in range(i + 1, n):
                    new_i = j
                    if df[('trend10', ticker_name)].iloc[j] == 1:
                        break
                    flip_to_up = (df[('trend1', ticker_name)].iloc[j] == 0) and (df[('trend10', ticker_name)].iloc[j] == 0)
                    # --- Volume confirmation on trigger candle (i) ---
                    vol_confirmed = df[('Volume', ticker_name)].iloc[i] > df[('Volume_SMA_20', ticker_name)].iloc[i]
                    if flip_to_up and vol_confirmed:
                        custom_signal[j] = True
                        triggered_distance[j] = j - i
                        break
                i = new_i
            else:
                i += 1
        else:
            # Conditions for setup at candle i:
            # 1. ht10 is in downtrend (trend10 == 1)
            # 2. ht1 is in uptrend (trend1 == 0)
            # 3. Price (high or close) at i >= ht10[i] (touches or below)
            cond1 = df[('trend10', ticker_name)].iloc[i] == 1
            cond2 = df[('trend1', ticker_name)].iloc[i] == 0
            cond3 = df[('High', ticker_name)].iloc[i] >= df[('ht10', ticker_name)].iloc[i]  # touches or above
            setup = cond1 and cond2 and cond3
            setup_triggered[i] = setup
            if setup:
                new_i, trigger_buy_distance = i+1, 0
                for j in range(i + 1, n):
                    new_i = j
                    if df[('trend10', ticker_name)].iloc[j] == 0:
                        break
                    flip_to_down = (df[('trend1', ticker_name)].iloc[j] == 1) and (df[('trend10', ticker_name)].iloc[j] == 1)
                    # --- Volume confirmation on trigger candle (i) ---
                    vol_confirmed = df[('Volume', ticker_name)].iloc[i] > df[('Volume_SMA_20', ticker_name)].iloc[i]
                    if flip_to_down and vol_confirmed:
                        custom_signal[j] = True
                        triggered_distance[j] = j - i
                        break
                i = new_i
            else:
                i += 1
    df[('custom_signal', ticker_name)] = custom_signal
    df[('setup_triggered', ticker_name)] = setup_triggered
    df[('triggered_distance', ticker_name)] = triggered_distance

    # ----------------------------
    # Print recent signals
    # ----------------------------
    if print_signals:
        recent_setup_triggered = df[df[('setup_triggered', ticker_name)]].tail(5)
        if not recent_setup_triggered.empty:
            print("\nRecent Setup Triggered:")
            print(recent_setup_triggered[[('Close', ticker_name), ('ht10', ticker_name), ('ht1', ticker_name)]])
        else:
            print("\nNo recent setup triggered.")

        recent_signals = df[df[('custom_signal', ticker_name)]].tail(5)
        if not recent_signals.empty:
            print("\nRecent Custom Signals:")
            print(recent_signals[[('Close', ticker_name), ('triggered_distance', ticker_name)]])
        else:
            print("\nNo recent custom signals found.")

    return df