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
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import pickle
import argparse
from utils import get_filename_for_dataset
import os
# ============================================
# 1. ONE-EURO FILTER (Adaptive Smoothing)
# ============================================
from algorithms.one_euro_filter import one_euro_filter
from utils import DATASET_AVAILABLE


# ============================================
# MODEL LOADING & REAL-TIME EXECUTION
# ============================================
def load_model(model_path: str) -> dict:
    """Load a saved model file and return its parameters."""
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def run_real_time(model_path: str, output_signal_only):
    """Run forecast in real-time mode using saved model parameters."""
    # Load model
    model_data = load_model(model_path)
    params = model_data['best_params']
    metadata = model_data.get('metadata', {})
    dataset_id = metadata.get('dataset_id')
    ticker = metadata.get('ticker')
    # Load data cache
    cache_filename = get_filename_for_dataset(dataset_id, older_dataset=None)
    with open(cache_filename, 'rb') as f:
        master_data_cache = pickle.load(f)

    df = master_data_cache[ticker].sort_index()
    price_col = ('Close', ticker)

    # Determine minimum history needed for indicators
    assert 'rsi_period' in params and 'macd_slow' in params
    min_history = max(
        params.get('rsi_period', 14),
        params.get('macd_slow', 26),
        params.get('rsi_period', 14) + params.get('macd_slow', 26),  # safety margin
        100
    )

    # Get latest window of data (needed for indicator calculations)
    latest_df = df.iloc[-min_history:].copy()

    # Extract parameters for run_forecast (filter only valid ones)
    valid_params = {k: v for k, v in params.items() if k in [
        'rsi_period', 'rsi_oversold', 'rsi_overbought', 'rsi_mode',
        'macd_fast', 'macd_slow', 'macd_signal',
        'one_euro_min', 'one_euro_factor',
        'lookahead_bars', 'threshold_pct'
    ]}
    assert 'target_type' in metadata
    target_type = metadata['target_type']
    valid_params.update({'target_type': target_type})
    # Run forecast on the window
    df_results, _ = run_forecast(
        df=latest_df,
        price_col=price_col,
        ticker=str(ticker),
        **valid_params
    )

    # Get the very latest signal
    latest_row = df_results.iloc[-1]
    signal = latest_row['Signal']
    current_price = latest_row[price_col]
    current_date = latest_row.name  # Index is datetime

    # Get config values (prefer params, fallback to metadata)
    assert 'metric' in metadata
    the_metric = params.get('metric', metadata.get('metric', 'long_accuracy'))
    assert 'lookahead_bars' in metadata
    lookahead_bars = params.get('lookahead_bars', metadata.get('lookahead_bars', 5))
    assert 'threshold_pct' in metadata
    threshold_pct = params.get('threshold_pct', metadata.get('threshold_pct', 0.))

    assert 1 == len(signal)
    signal = signal.values[0]
    # Calculate target price and date
    if signal == 1:  # Buy signal
        target_price = current_price * (1 + threshold_pct)
        direction = "BuySignal"
        operator = ">"
    elif signal == -1:  # Sell signal
        target_price = current_price * (1 - threshold_pct)
        direction = "SellSignal"
        operator = "<"
    else:
        target_price = None
        direction = "NoSignal"
        operator = ""

    # Estimate target date (assumes daily data; adjust if needed)
    try:
        # Try to infer frequency from index
        freq = pd.infer_freq(df.index[-10:]) or 'D'
        target_date = current_date + pd.tseries.frequencies.to_offset(freq) * lookahead_bars
    except:
        # Fallback: assume daily
        target_date = current_date + pd.Timedelta(days=lookahead_bars)

    # Format output
    ticker_display = ticker if ticker != "^GSPC" else "SPX500"

    if output_signal_only:
        # Minimal output for automation/scripts
        if signal == 1:
            print(f"BUY|{current_date.strftime('%Y-%m-%d')}|{current_price:.2f}|+{lookahead_bars}bars|{target_price:.2f}")
        elif signal == -1:
            print(f"SELL|{current_date.strftime('%Y-%m-%d')}|{current_price:.2f}|+{lookahead_bars}bars|{target_price:.2f}")
        else:
            print(f"HOLD|{current_date.strftime('%Y-%m-%d')}|{current_price:.2f}")
    else:
        # Human-readable output
        if signal != 0:
            target_mode_desc = "any point in window" if target_type == "any" else "exact future point"
            print(f"\n🎯 Real-Time Signal Detected:")
            print(f"On {current_date.strftime('%Y-%m-%d')} {ticker_display} is at {current_price:.2f} and {direction} activated for +{lookahead_bars}Bars , {ticker_display} {operator} {target_price:.2f} on {target_date.strftime('%Y-%m-%d')}")
            print(f"   Threshold: {threshold_pct * 100:.2f}% | Target Mode: '{target_type}' ({target_mode_desc}) | Model: {os.path.basename(model_path)}\n")
        else:
            print(f"\n⏸️  No signal on {current_date.strftime('%Y-%m-%d')}: {ticker_display} at {current_price:.2f}")
            print(f"   (Threshold: {threshold_pct * 100:.2f}%, Lookahead: {lookahead_bars} bars, Target Mode: '{target_type}', Metric: {the_metric})\n")

    return signal, current_price, target_price if signal != 0 else None


# ============================================
# 2. RSI (Relative Strength Index)
# ============================================
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing method."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============================================
# 3. MACD (Fixed: preserves index)
# ============================================
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram. Preserves pandas index."""
    if not isinstance(close, pd.Series):
        close = pd.Series(close.iloc[:, 0])
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'MACD': macd_line, 'Signal': signal_line, 'Histogram': histogram
    }, index=close.index)


# ============================================
# 4. LOOK-AHEAD LABELING (Forecast Target)
# ============================================
def create_future_labels(close: pd.Series, lookahead_bars: int, threshold_pct: float = 0.0) -> pd.Series:
    """Exact: Checks price exactly at t + lookahead_bars"""
    future_close = close.shift(-lookahead_bars)
    labels = (future_close > close * (1 + threshold_pct)).astype(float)
    labels.iloc[-lookahead_bars:] = np.nan
    return labels


def create_target(close: pd.Series, lookahead_bars: int, threshold_pct: float = 0.0) -> pd.Series:
    """
    Any: Checks if price EXCEEDS threshold at ANY point from t to t+lookahead_bars.
    Uses a forward-looking rolling maximum for efficient vectorized computation.
    """
    # Version optimisée (Exclut le prix actuel t, regarde uniquement de t+1 à t+lookahead_bars)
    future_max = close.iloc[::-1].rolling(window=lookahead_bars, min_periods=lookahead_bars).max().iloc[::-1].shift(-1)

    # True if max in window > threshold
    labels = (future_max > close * (1 + threshold_pct)).astype(float)

    # Mask out last lookahead_bars where future data is incomplete
    labels.iloc[-lookahead_bars:] = np.nan

    return labels


# ============================================
# 5. RULE-BASED FORECASTING SYSTEM
# ============================================
class ForecastSystem:
    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 30, rsi_overbought: float = 70,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 one_euro_min: float = 10, one_euro_factor: float = 0.2,
                 lookahead_bars: int = 5, threshold_pct: float = 0.01,
                 rsi_mode: str = "momentum", target_type: str = "any"):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.one_euro_min = one_euro_min
        self.one_euro_factor = one_euro_factor
        self.lookahead_bars = lookahead_bars
        self.threshold_pct = threshold_pct
        self.rsi_mode = rsi_mode
        self.target_type = target_type  # 'exact' or 'any'

    def generate_signals(self, df: pd.DataFrame, price_col, ticker) -> pd.DataFrame:
        df = df.copy()
        close = df[price_col]
        df['OneEuro'] = one_euro_filter(close.values, self.one_euro_min, self.one_euro_factor)
        df['RSI'] = calculate_rsi(close, self.rsi_period)
        macd_df = calculate_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        df['MACD'] = macd_df['MACD']
        df['MACD_Signal'] = macd_df['Signal']
        df['Histogram'] = macd_df['Histogram']

        # ✅ DYNAMIC TARGET SELECTION
        if self.target_type == "any":
            df['FutureLabel'] = create_target(close, self.lookahead_bars, self.threshold_pct)
        else:
            df['FutureLabel'] = create_future_labels(close, self.lookahead_bars, self.threshold_pct)

        df['Signal'] = 0

        rsi_shift = df['RSI'].shift(1).fillna(df['RSI'])
        hist_shift = df['Histogram'].shift(1).fillna(df['Histogram'])

        if self.rsi_mode == "reversion":
            long_rsi_cond = df['RSI'] < self.rsi_oversold
            short_rsi_cond = df['RSI'] > self.rsi_overbought
        else:  # momentum
            long_rsi_cond = df['RSI'] > self.rsi_overbought
            short_rsi_cond = df['RSI'] < self.rsi_oversold

        long_cond = ((close > df['OneEuro']) & long_rsi_cond & (df['RSI'] > rsi_shift) &
                     (df['Histogram'] > 0) & (df['Histogram'] > hist_shift))
        short_cond = ((close < df['OneEuro']) & short_rsi_cond & (df['RSI'] < rsi_shift) &
                      (df['Histogram'] < 0) & (df['Histogram'] < hist_shift))

        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1
        df['Signal'] = df['Signal'].fillna(0)
        return df

    def evaluate_signals(self, df: pd.DataFrame, price_col) -> dict:
        valid = df[['Signal', 'FutureLabel']].dropna()
        if len(valid) == 0: return {}
        valid = valid[valid['Signal'] != 0].copy()
        if len(valid) == 0: return {'total_signals': 0}

        valid['PredUp'] = (valid['Signal'] == 1).astype(int)
        valid['ActualUp'] = valid['FutureLabel'].astype(int)
        total = len(valid)
        correct = (valid['PredUp'] == valid['ActualUp']).sum()
        long_signals = valid[valid['Signal'] == 1]
        short_signals = valid[valid['Signal'] == -1]

        long_correct = (long_signals['ActualUp'] == 1).sum() if len(long_signals) > 0 else 0
        short_correct = (short_signals['ActualUp'] == 0).sum() if len(short_signals) > 0 else 0

        return {
            'total_signals': total,
            'total_bars': len(df),
            'accuracy': correct / total if total > 0 else 0,
            'long_signals': len(long_signals),
            'long_accuracy': long_correct / len(long_signals) if len(long_signals) > 0 else 0,
            'short_signals': len(short_signals),
            'short_accuracy': short_correct / len(short_signals) if len(short_signals) > 0 else 0,
        }


# ============================================
# 6. RUN FORECAST (Updated to accept parameters)
# ============================================
def run_forecast(df: pd.DataFrame, price_col, ticker: str = "^GSPC",
                 rsi_period: int = 14, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0, rsi_mode: str = "momentum",
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 one_euro_min: float = 10.0, one_euro_factor: float = 0.2,
                 lookahead_bars: int = 5, threshold_pct: float = 0.01,
                 target_type: str = "any") -> Tuple[pd.DataFrame, dict]:
    if not pd.api.types.is_numeric_dtype(df[price_col].to_numpy().dtype if isinstance(df[price_col], pd.DataFrame) else df[price_col].dtype):
        df[price_col] = df[price_col].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[price_col])

    system = ForecastSystem(
        rsi_period=rsi_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought, rsi_mode=rsi_mode,
        macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
        one_euro_min=one_euro_min, one_euro_factor=one_euro_factor,
        lookahead_bars=lookahead_bars, threshold_pct=threshold_pct,
        target_type=target_type
    )
    df_signals = system.generate_signals(df=df, price_col=price_col, ticker=ticker)
    metrics = system.evaluate_signals(df=df_signals, price_col=price_col)
    return df_signals, metrics


# ============================================
# 7. PLOTTING HELPER (Unchanged)
# ============================================
def plot_forecast_results(df: pd.DataFrame, price_col, sample: int = 200, start_idx: int = -1,
                          highlight_signals: bool = True, zoom_region: Optional[Tuple[int, int]] = None):
    if start_idx == -1:
        start_idx = max(0, len(df) - sample)
    plot_df = df.iloc[start_idx:start_idx + sample].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(plot_df.index, plot_df[price_col], label='Close', alpha=0.7, linewidth=1, color='black')
    ax1.plot(plot_df.index, plot_df['OneEuro'], label='One-Euro Filter', color='blue', linewidth=2)
    longs = plot_df[plot_df['Signal'] == 1]
    shorts = plot_df[plot_df['Signal'] == -1]
    ax1.scatter(longs.index, longs[price_col], marker='^', color='green', s=100, label='Long Signal', zorder=6, edgecolors='darkgreen', linewidth=1.5)
    ax1.scatter(shorts.index, shorts[price_col], marker='v', color='red', s=100, label='Short Signal', zorder=6, edgecolors='darkred', linewidth=1.5)
    if highlight_signals:
        for idx in longs.index: ax1.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax1.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax1.set_title('Price + One-Euro Filter + Trading Signals', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10);
    ax1.legend(loc='upper left', fontsize=9);
    ax1.grid(True, alpha=0.3, linestyle='--');
    ax1.set_axisbelow(True)

    ax2.plot(plot_df.index, plot_df['RSI'], label='RSI', color='purple', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Overbought/Oversold')
    ax2.axhline(30, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
    ax2.fill_between(plot_df.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(plot_df.index, 0, 30, color='green', alpha=0.1)
    if highlight_signals:
        for idx in longs.index: ax2.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax2.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax2.set_ylabel('RSI', fontsize=10);
    ax2.set_ylim(0, 100);
    ax2.legend(loc='lower right', fontsize=9);
    ax2.grid(True, alpha=0.3, linestyle='--');
    ax2.set_axisbelow(True)

    colors = ['green' if val >= 0 else 'red' for val in plot_df['Histogram']]
    ax3.bar(plot_df.index, plot_df['Histogram'], color=colors, alpha=0.6, label='Histogram', width=1)
    ax3.plot(plot_df.index, plot_df['MACD'], label='MACD', color='blue', linewidth=1.2)
    ax3.plot(plot_df.index, plot_df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1.2)
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.4, linewidth=0.8)
    if highlight_signals:
        for idx in longs.index: ax3.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax3.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax3.set_ylabel('MACD', fontsize=10);
    ax3.set_xlabel('Date', fontsize=10);
    ax3.legend(loc='lower right', fontsize=9);
    ax3.grid(True, alpha=0.3, linestyle='--');
    ax3.set_axisbelow(True)

    if zoom_region is not None:
        zoom_start, zoom_end = zoom_region
        if zoom_start < len(plot_df) and zoom_end <= len(plot_df) and zoom_start < zoom_end:
            zoom_df = plot_df.iloc[zoom_start:zoom_end]
            from matplotlib.patches import Rectangle
            ax1_inset = ax1.inset_axes([0.62, 0.55, 0.35, 0.35])
            ax1_inset.plot(zoom_df.index, zoom_df[price_col], color='black', linewidth=1.5)
            ax1_inset.plot(zoom_df.index, zoom_df['OneEuro'], color='blue', linewidth=2)
            ax1_inset.scatter(zoom_df[zoom_df['Signal'] == 1].index, zoom_df[zoom_df['Signal'] == 1][price_col], marker='^', color='green', s=50, zorder=5)
            ax1_inset.scatter(zoom_df[zoom_df['Signal'] == -1].index, zoom_df[zoom_df['Signal'] == -1][price_col], marker='v', color='red', s=50, zorder=5)
            ax1_inset.set_xticks([]);
            ax1_inset.set_yticks([]);
            ax1_inset.set_title('Zoom', fontsize=8, fontweight='bold');
            ax1_inset.grid(True, alpha=0.3)
            rect = Rectangle((zoom_df.index[0], plot_df[price_col].min()), zoom_df.index[-1] - zoom_df.index[0], plot_df[price_col].max() - plot_df[price_col].min(), linewidth=1.5, edgecolor='gold', facecolor='none', linestyle='--')
            ax1.add_patch(rect)
            ax1.indicate_inset_zoom(ax1_inset, edgecolor="gold", alpha=0.7)

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for ax in [ax1, ax2]: ax.tick_params(labelbottom=False)
    plt.suptitle('🔗 Linked Technical Analysis Dashboard (Zoom: sharex enabled)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.text(0.5, 0.01, "💡 Tip: Use mouse wheel to zoom, drag to pan — all panels stay synchronized!", ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.show()


# ============================================
# 8. ARGPARSE & ENTRY POINT
# ============================================
def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based Technical Forecasting System (One-Euro + RSI + MACD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    data_grp = parser.add_argument_group("Data & Environment")
    data_grp.add_argument("--dataset-id", type=str, default="day", help="Dataset identifier",choices=DATASET_AVAILABLE)
    data_grp.add_argument("--ticker", type=str, default="^GSPC", help="Ticker symbol to analyze")
    data_grp.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    data_grp.add_argument("--disable-print", action="store_true", help="Skip prints")

    algo_grp = parser.add_argument_group("Algorithm Parameters")
    algo_grp.add_argument("--rsi-period", type=int, default=14, help="RSI calculation window")
    algo_grp.add_argument("--rsi-oversold", type=float, default=30.0, help="RSI oversold threshold")
    algo_grp.add_argument("--rsi-overbought", type=float, default=70.0, help="RSI overbought threshold")
    algo_grp.add_argument("--rsi-mode", type=str, choices=["momentum", "reversion"], default="momentum", help="RSI signal logic")
    algo_grp.add_argument("--macd-fast", type=int, default=12, help="MACD fast EMA period")
    algo_grp.add_argument("--macd-slow", type=int, default=26, help="MACD slow EMA period")
    algo_grp.add_argument("--macd-signal", type=int, default=9, help="MACD signal line period")
    algo_grp.add_argument("--one-euro-min", type=float, default=10.0, help="One-Euro filter min cutoff")
    algo_grp.add_argument("--one-euro-factor", type=float, default=0.2, help="One-Euro filter beta factor")
    algo_grp.add_argument("--lookahead-bars", type=int, default=5, help="Future bars to forecast")
    algo_grp.add_argument("--threshold-pct", type=float, default=0., help="Min %% move to create the target")

    # Target labeling mode
    algo_grp.add_argument(
        "--target-type",
        type=str,
        choices=["exact", "any"],
        default="any",
        help="Target labeling method: 'exact' (price at t+lookahead) or 'any' (price > threshold anywhere in window)"
    )

    viz_grp = parser.add_argument_group("Visualization")
    viz_grp.add_argument("--plot-sample", type=int, default=100, help="Number of recent bars to plot")
    viz_grp.add_argument("--disable-plot-sample", action="store_true", help="Skip plotting section")

    realtime_grp = parser.add_argument_group("Real-Time Mode")
    realtime_grp.add_argument(
        "--real-time",
        action="store_true",
        help="Run in real-time mode: evaluate only the latest data point using a saved model"
    )
    realtime_grp.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model file (.pkl) to load optimized parameters"
    )
    realtime_grp.add_argument(
        "--output-signal-only",
        action="store_true",
        help="In real-time mode, output only the signal decision (for automation)"
    )

    return parser


def entry(args):
    # ✅ REAL-TIME MODE: Load model and evaluate latest point only
    if args.real_time:
        if not args.model_path:
            raise ValueError("--model-path is required when using --real-time")
        return run_real_time(output_signal_only=args.output_signal_only, model_path=args.model_path)

    # ✅ BATCH MODE: Original behavior
    np.random.seed(args.seed)
    cache_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(cache_filename, 'rb') as f:
        master_data_cache = pickle.load(f)

    df_spx500 = master_data_cache[args.ticker].sort_index()
    price_col = ('Close', args.ticker) if isinstance(df_spx500.columns, pd.MultiIndex) else 'Close'

    df_results, metrics = run_forecast(
        df=df_spx500, price_col=price_col, ticker=args.ticker,
        rsi_period=args.rsi_period, rsi_oversold=args.rsi_oversold, rsi_overbought=args.rsi_overbought, rsi_mode=args.rsi_mode,
        macd_fast=args.macd_fast, macd_slow=args.macd_slow, macd_signal=args.macd_signal,
        one_euro_min=args.one_euro_min, one_euro_factor=args.one_euro_factor,
        lookahead_bars=args.lookahead_bars, threshold_pct=args.threshold_pct,
        target_type=args.target_type
    )

    if not args.disable_print:
        print("\n📊 Forecast System Evaluation")
        print(f"   Ticker: {args.ticker}")
        print(f"   Total Signals: {metrics.get('total_signals', 0)}")
        print(f"   Overall Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
        print(f"   Long Signals: {metrics.get('long_signals', 0)} | Accuracy: {metrics.get('long_accuracy', 0) * 100:.2f}%")
        print(f"   Short Signals: {metrics.get('short_signals', 0)} | Accuracy: {metrics.get('short_accuracy', 0) * 100:.2f}%")
        print(f"   Look-ahead Horizon: {args.lookahead_bars} bars")
        print(f"   Target Mode: '{args.target_type}'")
        print(f"   Threshold For Creating Target: {args.threshold_pct * 100:.2f}%\n")

    if not args.disable_plot_sample:
        plot_forecast_results(df=df_results, price_col=price_col, sample=args.plot_sample)

    return metrics


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)