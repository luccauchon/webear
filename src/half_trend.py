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
    df[('ht', ticker_name)] = ht
    df[('trend', ticker_name)] = trend
    df[('arrow_up', ticker_name)] = arrow_up
    df[('arrow_down', ticker_name)] = arrow_down
    df[('buy_signal', ticker_name)] = buy_signal
    df[('sell_signal', ticker_name)] = sell_signal
    df[('atr_high', ticker_name)] = atr_high
    df[('atr_low', ticker_name)] = atr_low

    return df