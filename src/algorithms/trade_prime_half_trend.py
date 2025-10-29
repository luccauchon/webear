import pandas as pd
import numpy as np

HALF_TREND_DEFAULT_CONFIG = {
    # ✅ ATR-Based Stop Loss:
    # Compute ATR(14) (standard lookback).
    # Set stop-loss at:
    # entry_price - 1.5 × ATR (for longs).
    # Also compute a take-profit level (e.g., entry_price + 2 × ATR) if desired.

    # ✅ 1. Price Must Close Beyond the Trend Line (Not Just Touch)
    #       Why: Wicks can fake you out; closes reflect conviction.
    'use__entry_type': 'Hard',

    #✅ Volume Confirmation:
    # Require that the buy candle (where HT1 flips up) has above-average volume (e.g., > 20-period SMA of volume).
    # Helps filter out weak breakouts.
    'use__volume_confirmed': {'enable': False, 'period': 20},

    #✅ 2. Higher Timeframe (HT10) Must Be "Strong" Uptrend
    #       Why: Not all uptrends are equal — avoid weak or flat HT10 lines.
    # Filter:
    #
    # HT10 line must be rising over last N bars (e.g., ht10[i] > ht10[i-3]).
    # Or: HT10 slope > 0 (use linear regression or simple difference).
    'use__higher_timeframe_strong_trend': {'enable': False, 'length': 21},

    # ✅ 3. Relative Strength vs. Benchmark (e.g., SPX vs. VIX or 10Y Yield)
    #       Why: Breakouts fail in high-fear or rising-rate environments.
    # For SPX:
    # Only take longs when VIX is falling or below its 10-day SMA.
    #
    # Or: SPX > 200-period SMA (long-term bullish bias).
    'use__relative_strength_vs_benchmark': {'enable_vix': False, 'enable_spx': False, 'period_vix': 10*7, 'period_spx': 200*7, 'vix_dataframe': None, 'spx_dataframe': None},

    # ✅ 4. Candlestick Confirmation Pattern
    #       Why: Adds price-action context to the reversal.
    #     Use: Require the HT1 reversal candle (the buy candle) to be:
    #
    # A bullish engulfing, hammer, or simply close in top 50% of its range.
    'use__candlestick_confirmation_pattern': {'enable': False}
}

def get_entry_type(**kwargs):
    return _get_config('use__entry_type', **kwargs)


def get_volume_confirmed(**kwargs):
    volume_confirmed__enabled = bool(_get_config(('use__volume_confirmed', 'enable'), **kwargs))
    volume_confirmed__window_size = int(_get_config(('use__volume_confirmed', 'period'), **kwargs))
    return volume_confirmed__enabled, volume_confirmed__window_size


def set_volumed_confirmed(periode, **args):
    args['use__volume_confirmed'].update({'enable': True, 'period': periode})
    return args


def get_higher_timeframe_strong_trend(**kwargs):
    higher_timeframe_strong_trend__enabled = bool(_get_config(('use__higher_timeframe_strong_trend', 'enable'), **kwargs))
    ht10_strong_n = int(_get_config(('use__higher_timeframe_strong_trend', 'length'), **kwargs))
    return higher_timeframe_strong_trend__enabled, ht10_strong_n


def get_relative_strength_vs_benchmark(**kwargs):
    use_vix, use_spx = _get_config(('use__relative_strength_vs_benchmark', 'enable_vix'), **kwargs), _get_config(('use__relative_strength_vs_benchmark', 'enable_spx'), **kwargs)
    pd_v = _get_config(('use__relative_strength_vs_benchmark', 'period_vix'), **kwargs)
    vix_values = _get_config(('use__relative_strength_vs_benchmark', 'vix_dataframe'), **kwargs)
    pd_s = _get_config(('use__relative_strength_vs_benchmark', 'period_spx'), **kwargs)
    spx_values = _get_config(('use__relative_strength_vs_benchmark', 'spx_dataframe'), **kwargs)
    return (use_vix, pd_v, vix_values), (use_spx, pd_s, spx_values)


def get_candlestick_confirmation_pattern(**kwargs):
    use__candlestick_formation_pattern = _get_config(('use__candlestick_confirmation_pattern', 'enable'), **kwargs)
    return use__candlestick_formation_pattern


def _get_config(keys, **kwargs):
    if not isinstance(keys, list) and not isinstance(keys, tuple):
        keys = [keys]
    if keys[0] in kwargs:
        val_1 = kwargs.get(keys[0], HALF_TREND_DEFAULT_CONFIG[keys[0]])
    else:
        val_1 = HALF_TREND_DEFAULT_CONFIG[keys[0]]

    if isinstance(val_1, dict):
        if keys[1] in val_1:
            return val_1.get(keys[1], HALF_TREND_DEFAULT_CONFIG[keys[0]][keys[1]])
        else:
            return HALF_TREND_DEFAULT_CONFIG[keys[0]][keys[1]]
    else:
        return val_1


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


def trade_prime_half_trend_strategy(ticker_df, ticker_name, buy_setup=True, **kwargs):
    """
    ...

    Configuration Options:
    {}
    """.format('\n'.join(f'    - {key}: {value}' for key, value in HALF_TREND_DEFAULT_CONFIG.items()))

    close_label, high_label, low_label = ('Close', ticker_name), ('High', ticker_name), ('Low', ticker_name)

    use__entry_type = kwargs.get('use__entry_type', HALF_TREND_DEFAULT_CONFIG['use__entry_type'])
    assert use__entry_type in ['Soft', 'Hard']
    buy_setup_trigger_colname, sell_setup_trigger_colname = ('Close', ticker_name), ('Close', ticker_name)
    if use__entry_type == 'Soft':
        buy_setup_trigger_colname, sell_setup_trigger_colname = ('Low', ticker_name), ('High', ticker_name)
    ticker_df = half_trend(ticker_df, ticker_name=ticker_name, close_label=close_label, high_label=high_label, low_label=low_label, amplitude=10, channel_deviation=2)
    ticker_df = half_trend(ticker_df, ticker_name=ticker_name, close_label=close_label, high_label=high_label, low_label=low_label, amplitude=1, channel_deviation=2)
    ticker_df['ticker_name'] = ticker_name

    assert ('ht1', ticker_name) in ticker_df.columns
    assert ('ht10', ticker_name) in ticker_df.columns
    assert ('trend1', ticker_name) in ticker_df.columns
    assert ('trend10', ticker_name) in ticker_df.columns
    short_trend_colname = ('trend1', ticker_name)
    long_trend_colname = ('trend10', ticker_name)
    long_ht_colname = ('ht10', ticker_name)
    volume_colname = ('Volume', ticker_name)
    close_colname = ('Close', ticker_name)
    open_colname = ('Open', ticker_name)
    high_colname = ('High', ticker_name)
    low_colname = ('Low', ticker_name)
    # Extraction of parameters
    volume_confirmed__enabled, volume_confirmed__window_size = get_volume_confirmed(**kwargs)
    volume_confirmed__colname = (f'Volume_SMA_{volume_confirmed__window_size}', ticker_name)
    higher_timeframe_strong_trend__enabled, higher_timeframe_strong_trend__length = get_higher_timeframe_strong_trend(**kwargs)
    (use_vix, pd_v, vix_values), (use_spx, pd_s, spx_values) = get_relative_strength_vs_benchmark(**kwargs)
    use__relative_strength_vs_benchmark__enabled = use_vix or use_spx
    use__candlestick_formation_pattern = get_candlestick_confirmation_pattern(**kwargs)

    # ----------------------------
    # 1. Volume SMA for confirmation
    # 2. VIX SMA for high-fear env
    # ----------------------------
    ticker_df[volume_confirmed__colname] = ticker_df[volume_colname].rolling(window=volume_confirmed__window_size, min_periods=1).mean()
    if use__relative_strength_vs_benchmark__enabled:
        if use_vix:
            assert vix_values is not None
            vix_fear_env__colname     = (f'rs__Close', 'VIX')
            sma_vix_fear_env__colname = (f'rs__Close_SMA{pd_v}', 'VIX')
            ticker_df[vix_fear_env__colname]     = vix_values[('Close', '^VIX')]
            ticker_df[sma_vix_fear_env__colname] = ticker_df[vix_fear_env__colname].rolling(pd_v).mean()
            # ema_vix_fear_env__colname = (f'rs__Close_EMA{pd_v}', 'VIX')  # Updated name to reflect EMA
            # ticker_df[ema_vix_fear_env__colname] = ticker_df[vix_fear_env__colname].ewm(span=pd_v, adjust=False, min_periods=1).mean()
        if use_spx:
            assert spx_values is not None
            spx_env__colname = (f'rs__Close', 'SPX')
            sma_spx_env__colname = (f'rs__Close_SMA{pd_s}', 'SPX')
            ticker_df[spx_env__colname] = spx_values[('Close', '^GSPC')]
            ticker_df[sma_spx_env__colname] = ticker_df[spx_env__colname].rolling(pd_v).mean()
            # ema_spx_env__colname = (f'rs__Close_EMA{pd_v}', 'SPX')  # Updated name to reflect EMA
            # ticker_df[spx_env__colname] = spx_values[('Close', '^GSPC')]
            # ticker_df[ema_spx_env__colname] = ticker_df[spx_env__colname].ewm(span=pd_v, adjust=False, min_periods=1).mean()

    # ----------------------------
    # Compute ATR(14) for stops
    # ----------------------------
    high = ticker_df[high_label].values
    low = ticker_df[low_label].values
    close = ticker_df[close_label].values
    tr = np.maximum(high - low,
                    np.abs(high - np.concatenate([[close[0]], close[:-1]])),
                    np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    ticker_df[(atr14_colname := ('ATR_14', ticker_name))] = atr_14

    # ----------------------------
    # Scanning for Buy/Sell Setup
    # ----------------------------
    df = ticker_df
    setup_trigger_colname = buy_setup_trigger_colname if buy_setup else sell_setup_trigger_colname
    n, i = len(df), 0
    custom_signal = np.full(n, False, dtype=bool)
    setup_triggered = np.full(n, False, dtype=bool)
    triggered_distance = np.full(n, 0, dtype=int)
    entry_price = np.full(n, np.nan, dtype=float)
    stop_loss = np.full(n, np.nan, dtype=float)
    take_profit = np.full(n, np.nan, dtype=float)
    while i < n:
        # Conditions for BUY setup at candle i:
        # 1. ht10 is in uptrend (trend10 == 0)
        # 2. ht1 is in downtrend (trend1 == 1)
        # 3. Price (low or close) at i <= ht10[i] (touches or below)
        #
        # Conditions for SELL setup at candle i:
        # 1. ht10 is in downtrend (trend10 == 1)
        # 2. ht1 is in uptrend (trend1 == 0)
        # 3. Price (high or close) at i >= ht10[i] (touches or below)
        cond1 = df[long_trend_colname].iloc[i]  == (0 if buy_setup else 1)
        cond2 = df[short_trend_colname].iloc[i] == (1 if buy_setup else 0)
        if buy_setup:
            cond3 = df[setup_trigger_colname].iloc[i] <= df[long_ht_colname].iloc[i]
        else:
            cond3 = df[setup_trigger_colname].iloc[i] >= df[long_ht_colname].iloc[i]
        setup_triggered[i] = cond1 and cond2 and cond3
        if setup_triggered[i]:
            new_i, trigger_buy_distance = i + 1, 0
            for j in range(i + 1, n):
                new_i = j
                if df[long_trend_colname].iloc[j] == (1 if buy_setup else 0):
                    break  # Long trend reversal; cancel trade
                # --- Volume confirmation on trigger candle (j) ---
                crt__vol_confirmed = df[volume_colname].iloc[j] >= df[volume_confirmed__colname].iloc[j] if volume_confirmed__enabled else True

                # Higher Timeframe (HT10) Must Be "Strong" trend
                crt__ht10_strong = True
                if higher_timeframe_strong_trend__enabled:
                    try:
                        val_1, val_2 = df[long_ht_colname].iloc[j], df[long_ht_colname].iloc[j-higher_timeframe_strong_trend__length]
                        if val_1 >= val_2 if buy_setup else val_1 <= val_2:
                            crt__ht10_strong = True
                        else:
                            crt__ht10_strong = False
                    except Exception:  # Not enough data
                        pass

                # Does the short term trend change direction?
                assert df[long_trend_colname].iloc[j] == (0 if buy_setup else 1)  # Long trend always in the good direction
                crt__flip_to__up_or_down = (df[short_trend_colname].iloc[j] == (0 if buy_setup else 1))

                # Relative Strength vs. Benchmark
                crt__vix_fear, crt__bull_market = True, True
                if buy_setup and use__relative_strength_vs_benchmark__enabled:
                    if use_vix:
                        # Only take longs when VIX is falling or below its N-day SMA.
                        crt__vix_fear = df[vix_fear_env__colname].iloc[j] < df[sma_vix_fear_env__colname].iloc[j]
                    if use_spx:
                        # SPX > N-period SMA (long-term bullish bias).
                        crt__bull_market = df[spx_env__colname].iloc[j] > df[sma_spx_env__colname].iloc[j]
                elif not buy_setup and use__relative_strength_vs_benchmark__enabled:
                    print(f"TODO *********************************")

                # Candlestick confirmation pattern
                crt__candlestick_confirmation_pattern = True
                if use__candlestick_formation_pattern:
                    if buy_setup:
                        # Bullish momentum: close in upper half of candle
                        bullish_candle = (df[close_colname].iloc[i] - df[low_colname].iloc[i]) > (df[high_colname].iloc[i] - df[close_colname].iloc[i])

                        # Or: strong green candle
                        strong_green = df[close_colname].iloc[i] > df[open_colname].iloc[i] and (df[close_colname].iloc[i] - df[open_colname].iloc[i]) > 0.5 * (df[high_colname].iloc[i] - df[low_colname].iloc[i])
                        crt__candlestick_confirmation_pattern = bullish_candle or strong_green
                    else:
                        # Bearish momentum: close in lower half of candle
                        # Checks if the distance from the close to the high is greater than from the low to the close, meaning the close is in the bottom 50% of the candle’s range.
                        bearish_candle = (df[high_colname].iloc[i] - df[close_colname].iloc[i]) > (df[close_colname].iloc[i] - df[low_colname].iloc[i])

                        # Or: strong red candle
                        # Confirms a bearish (red) candle where the body (open – close) is more than 50% of the total range (high – low), indicating strong selling pressure.
                        strong_red = df[close_colname].iloc[i] < df[open_colname].iloc[i] and (df[open_colname].iloc[i] - df[close_colname].iloc[i]) > 0.5 * (df[high_colname].iloc[i] - df[low_colname].iloc[i])
                        crt__candlestick_confirmation_pattern = bearish_candle or strong_red

                # Combine all criteria
                if crt__flip_to__up_or_down and crt__vol_confirmed and crt__ht10_strong and crt__vix_fear and crt__bull_market and crt__candlestick_confirmation_pattern:
                    custom_signal[j] = True
                    triggered_distance[j] = j - i
                    entry_price[j] = df[close_colname].iloc[j]
                    atr_val = df[atr14_colname].iloc[j]
                    stop_loss[j]   = entry_price[j] - 1.5 * atr_val if buy_setup else entry_price[j] + 1.5 * atr_val
                    take_profit[j] = entry_price[j] + 2.0 * atr_val if buy_setup else entry_price[j] - 2.0 * atr_val
                    break
            i = new_i
        else:
            i += 1
    df[('custom_signal', ticker_name)] = custom_signal
    df[('setup_triggered', ticker_name)] = setup_triggered
    df[('triggered_distance', ticker_name)] = triggered_distance
    df[('entry_price', ticker_name)] = entry_price
    df[('stop_loss', ticker_name)] = stop_loss
    df[('take_profit', ticker_name)] = take_profit
    return df
