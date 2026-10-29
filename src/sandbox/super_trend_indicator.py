import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()

    # ---------- 1) True Range + ATR (Wilder smoothing) ----------
    # ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period
    # (same as the description's "([Prior ATR x 13] + Current TR) / 14", generalized to period 10)
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # ---------- 2) Basic bands ----------
    # Upper Band = (High + Low)/2 + Multiplier * ATR
    # Lower Band = (High + Low)/2 - Multiplier * ATR
    hl2 = (df["High"] + df["Low"]) / 2.0
    mid = hl2.to_numpy()
    basic_ub = (hl2 + multiplier * atr).to_numpy()
    basic_lb = (hl2 - multiplier * atr).to_numpy()
    close = df["Close"].to_numpy()

    n = len(df)
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)          # 1 = bullish, -1 = bearish

    # ---------- 3) Recursive final bands + SuperTrend line ----------
    start = period - 1                          # first bar with a valid ATR
    final_ub[start] = basic_ub[start]
    final_lb[start] = basic_lb[start]
    st[start] = final_lb[start] if close[start] >= mid[start] else final_ub[start]
    direction[start] = 1 if close[start] > st[start] else -1

    for i in range(start + 1, n):
        # Final upper band: can only fall, unless price closed above it
        if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # Final lower band: can only rise, unless price closed below it
        if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # SuperTrend line flips when close crosses the active band
        if st[i - 1] == final_ub[i - 1]:        # previous bar bearish
            st[i] = final_ub[i] if close[i] < final_ub[i] else final_lb[i]
        else:                                   # previous bar bullish
            st[i] = final_lb[i] if close[i] > final_lb[i] else final_ub[i]

        direction[i] = 1 if close[i] > st[i] else -1

    df["ATR"] = atr
    df["Upper Band"] = final_ub
    df["Lower Band"] = final_lb
    df["SuperTrend"] = st
    df["Direction"] = direction
    return df


def main():
    symbol = "^GSPC"

    # Daily data from Yahoo Finance
    data = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)

    # Your platform's defaults: 10 and 3
    result = supertrend(data, period=10, multiplier=3)

    cols = ["Close", "ATR", "Upper Band", "Lower Band", "SuperTrend", "Direction"]
    print(result[cols].tail(10).round(2))

    last = result.iloc[-1]
    print(f"\n{symbol} current trend: {'BULLISH' if last['Direction'] == 1 else 'BEARISH'}")
    print(f"Close = {last['Close']:,.2f} | SuperTrend = {last['SuperTrend']:,.2f}")

    # When did the trend last flip?
    valid = result[result["Direction"] != 0]
    flips = valid[valid["Direction"] != valid["Direction"].shift(1)]
    print(f"Last trend flip: {flips.index[-1].date()} -> "
          f"{'BULLISH' if flips['Direction'].iloc[-1] == 1 else 'BEARISH'}")

    # ---- Plot last 3 years ----
    cutoff = result.index[-1] - pd.DateOffset(years=3)
    plot_df = result[result.index >= cutoff]
    up = plot_df[plot_df["Direction"] == 1]
    dn = plot_df[plot_df["Direction"] == -1]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(plot_df.index, plot_df["Close"], color="gray", lw=1.2, label=f"{symbol} close")
    ax.scatter(up.index, up["SuperTrend"], s=4, color="green", label="SuperTrend (support / bullish)")
    ax.scatter(dn.index, dn["SuperTrend"], s=4, color="red", label="SuperTrend (resistance / bearish)")
    ax.set_title(f"{symbol} daily — SuperTrend (10, 3)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()