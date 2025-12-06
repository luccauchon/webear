import pandas as pd
import numpy as np
from constants import FYAHOO__OUTPUTFILENAME_DAY
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compute_skew_energy(df, skew_col, median_window=200, energy_window=90, q=0.5):
    """
    Computes rolling 200-day median, zero-out below median, and 90-day energy.
    """
    data = df.copy()
    median_col = f"median_{median_window}"
    # 1. Rolling quantile
    data[median_col] = data[skew_col].rolling(median_window).quantile(q)
    # 2. Zero-out values below median
    data["above_median"] = np.where(
        data[skew_col] >= data[median_col],
        data[skew_col],
        0
    )

    # 3. Rolling integral (energy)
    data["energy"] = data["above_median"].rolling(energy_window).sum()

    return data, median_col


def detect_energy_spikes(data, energy_col="energy", threshold=None, percentile=90):
    """
    Detects spike events in the energy signal.
    threshold: fixed optional value.
    If None → use percentile (default 90th)
    """
    if threshold is None:
        threshold = np.nanpercentile(data[energy_col], percentile)

    spikes = data[data[energy_col] >= threshold]
    return spikes, threshold


def plot_skew_energy(data, skew_col, median_col, spx_df, monthly_drop_drawn, energy_percentile_threshold):
    """
    Plots SKEW, above-median SKEW, and energy with spike markers.
    Overlays monthly SPX candles on the energy plot.
    """

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # ---------------------- Plot SKEW ----------------------
    axes[0].plot(data[skew_col], label="SKEW")
    axes[0].plot(data[median_col], label="Median 200d", linewidth=1)
    axes[0].set_title("SKEW & 200-day Median")
    axes[0].legend()
    axes[0].grid(True)

    # ---------------------- Plot above-median SKEW ----------------------
    axes[1].plot(data["above_median"], label="SKEW Above Median (Zeroed)", color="darkorange")
    axes[1].set_title("Transformed SKEW Signal")
    axes[1].legend()
    axes[1].grid(True)

    # ---------------------- Plot Energy ----------------------
    ax_energy = axes[2]
    ax_energy.plot(data["energy"], label="Energy (90-day Integral)", color="green")

    spikes, threshold = detect_energy_spikes(data=data, percentile=energy_percentile_threshold*100)
    ax_energy.axhline(threshold, color="red", linestyle="--", label=f"Spike Threshold ({threshold:.1f})")
    ax_energy.scatter(spikes.index, spikes["energy"], color="red", marker="o", label="Spikes")
    ax_energy.set_title("Energy Signal with Spike Detection")
    ax_energy.legend()
    ax_energy.grid(True)

    # ---------------------- Overlay Monthly SPX Candles (Log Scale) ----------------------
    # Resample SPX to monthly OHLC
    spx_monthly = pd.DataFrame({
        'Open': spx_df[('Open', '^GSPC')].resample('ME').first(),
        'High': spx_df[('High', '^GSPC')].resample('ME').max(),
        'Low': spx_df[('Low', '^GSPC')].resample('ME').min(),
        'Close': spx_df[('Close', '^GSPC')].resample('ME').last()
    }).dropna()

    # Secondary y-axis for SPX
    ax_spx = ax_energy.twinx()
    ax_spx.set_ylabel('SPX (Monthly, Log Scale)', color='steelblue')
    ax_spx.set_yscale('log')  # ←←← LOG SCALE HERE

    # Candle width
    width = pd.Timedelta(days=8)

    for date, row in spx_monthly.iterrows():
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        color = 'green' if c >= o else 'red'

        # Wick
        ax_spx.plot([date, date], [l, h], color='black', linewidth=0.8)

        # Body
        body_height = abs(c - o)
        body_bottom = min(o, c)
        rect = patches.Rectangle(
            (date - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        ax_spx.add_patch(rect)

    # Optional: improve log ticks (optional but cleaner)
    from matplotlib.ticker import LogLocator, LogFormatter
    ax_spx.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax_spx.yaxis.set_major_formatter(LogFormatter())

    ax_spx.tick_params(axis='y', labelcolor='steelblue')

    # ---------------------- Add vertical lines for SPX monthly drops >= 3% ----------------------
    # Compute monthly returns from SPX close
    spx_monthly_close = spx_df[('Close', '^GSPC')].resample('ME').last().dropna()
    monthly_returns = spx_monthly_close.pct_change()

    # Find months with drop >= 3%
    big_drop_months = monthly_returns[monthly_returns <= -monthly_drop_drawn].index

    # Draw gray vertical lines on the energy axis for those months
    for drop_date in big_drop_months:
        ax_energy.axvline(x=drop_date, color='gray', linestyle=':', alpha=0.7, linewidth=5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
        master_data_cache = pickle.load(f)
    setup = 'test'
    monthly_drop_drawn = 0.03
    median_window = 200
    energy_window = 90
    if setup == 'rt':
        skew_df = master_data_cache["^SKEW"].copy()
        spx_df = master_data_cache["^GSPC"].copy()
        skew_df = skew_df[-2250:].copy()
        spx_df  = spx_df[-2250:].copy()
        data,median_col = compute_skew_energy(df=skew_df, skew_col=("Close","^SKEW"))
        plot_skew_energy(data=data, skew_col=("Close","^SKEW"), spx_df=spx_df, median_col=median_col, monthly_drop_drawn=0.03, energy_percentile_threshold=0.8)
    if setup == 'test':
        skew_df = master_data_cache["^SKEW"].copy()
        spx_df = master_data_cache["^GSPC"].copy()
        idx1, idx2 = -1250, -1
        monthly_drop_drawn = 0.03
        median_window = 200
        energy_window = 10
        skew_df = skew_df[idx1:idx2].copy()
        spx_df  = spx_df[idx1:idx2].copy()
        data, median_col = compute_skew_energy(df=skew_df, skew_col=("Close", "^SKEW"), median_window=median_window, energy_window=energy_window)
        plot_skew_energy(data=data, skew_col=("Close", "^SKEW"), spx_df=spx_df, median_col=median_col, monthly_drop_drawn=monthly_drop_drawn, energy_percentile_threshold=0.8)

