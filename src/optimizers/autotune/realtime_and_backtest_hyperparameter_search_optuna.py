"""
AutoTune Cycle-Aware Trading Strategy
======================================
Overview:
    A quantitative trading algorithm that leverages digital signal processing (DSP)
    techniques to dynamically adapt to market cycles. By isolating cyclical price
    movements from long-term trends and noise, the strategy generates momentum-based
    trading signals that are optimized for specific risk/reward profiles using Optuna.

Core Algorithm:
    1. Dominant Cycle Estimation (`_auto_tune`):
       Applies a 3rd-order high-pass filter to remove low-frequency trends, then
       computes rolling autocorrelation to dynamically estimate the market's current
       dominant cycle period (in bars).
    2. Adaptive Bandpass Filtering (`_bandpass2`):
       Constructs a 2nd-order bandpass filter dynamically tuned to the estimated
       cycle period. This extracts cyclical momentum while suppressing non-cyclical
       drift and high-frequency noise.
    3. Signal Generation (`_run_backtest`):
       Triggers LONG/SHORT signals on 2-bar Rate-of-Change (ROC) crossovers of the
       bandpass output. Signals are gated by:
         - Cycle Strength: Minimum correlation threshold to ensure the detected cycle
           is statistically robust.
         - Baseline Alignment: Optional high-pass trend filter (above/below zero) to
           align signals with the underlying macro trend.
    4. Forward Evaluation:
       For every generated signal, the algorithm simulates forward performance over
       a configurable `--lookahead-bars` window, recording path-dependent min/max
       prices, terminal prices, and maximum achievable returns.
    5. Hyperparameter Optimization:
       Uses Optuna (TPE sampler + Median pruner) to search for optimal filter
       parameters (`window`, `bandwidth`, `threshold`, baseline toggles) that
       maximize a user-defined performance metric on a training set, with final
       validation on a held-out dataset.

The --optimize Flag: Metric Comparison
---------------------------------------
This flag defines the primary objective function for Optuna hyperparameter tuning.
Each metric evaluates trade success differently over the configured lookahead window,
using `--win-threshold` (`wt`) as the boundary/reference value:

  • profit_target (Direction-Agnostic)
    Measures the % of signals where the MAXIMUM forward return exceeds `wt`.
    Focus: Absolute profit magnitude. Ignores intra-trade drawdown and terminal price.
    NOTE: With negative wt, this becomes a lenient target (e.g., wt=-0.04 means
    "max return > -4%", which is easy to achieve).

  • range_bound (Direction-Agnostic)
    Measures the % of signals where price stays strictly within `[min(1-wt,1+wt), max(1-wt,1+wt)]`
    throughout the ENTIRE lookahead window.
    Focus: Volatility suppression & mean-reversion. Penalizes any breakout.
    NOTE: Negative wt values are now supported - the band bounds are automatically
    ordered to ensure lower <= upper.

  • hold_floor (Long-Focused)
    Measures the % of LONG signals where price NEVER drops below `1-wt` at any
    point during the lookahead window.
    Focus: Downside protection & stop-loss integrity. Upside is uncapped.
    NOTE: With negative wt, this becomes MORE aggressive (e.g., wt=-0.04 requires
    price to stay above 104% of entry - maintaining a minimum 4% gain).

  • hold_ceiling (Short-Focused)
    Measures the % of SHORT signals where price NEVER rises above `1+wt` during
    the lookahead window.
    Focus: Short-side risk containment & trend continuation safety.
    NOTE: With negative wt, this becomes MORE aggressive (e.g., wt=-0.04 requires
    price to stay below 96% of entry - maintaining a minimum 4% drop).

  • finish_above (Long-Focused)
    Measures the % of LONG signals where the price at the EXACT FINAL BAR of the
    lookahead window is above `1+wt`.
    Focus: Terminal momentum & breakout conviction. Allows intra-trade volatility.
    NOTE: With negative wt, this becomes a lenient target.

  • finish_below (Short-Focused)
    Measures the % of SHORT signals where the price at the EXACT FINAL BAR of the
    lookahead window is below `1-wt`.
    Focus: Terminal downside momentum & trend resolution.
    NOTE: With negative wt, this becomes a lenient target.

Key Distinctions:
  - Path-Dependent vs. Endpoint-Dependent: `hold_*`, `range_bound` evaluate price
    behavior across every bar. `finish_*` only evaluate the terminal price.
  - Directional Bias: The optimizer automatically sets `signal_type` to 1 (long),
    -1 (short), or 0 (both) based on your chosen metric.
  - Risk vs. Reward: `hold_*` and `range_bound` prioritize capital preservation.
    `profit_target` and `finish_*` prioritize directional conviction.
  - Negative Threshold Support: All metrics now properly handle negative win_threshold
    values, with semantics adjusted as noted above.

Usage Example:
    python autotune.py --dataset-id day --ticker ^GSPC --optimize hold_floor \\
                       --lookahead-bars 10 --win-threshold -0.04 --n-trials 500 \\
                       --storage sqlite:///optuna.db
"""
try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from optuna.pruners import MedianPruner
import argparse
import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Dict
import warnings
import optuna
from optuna.samplers import TPESampler
from utils import get_filename_for_dataset, get_next_step
import pickle
import os
import glob
import sys

# =============================================================================
# 📊 CENTRAL METRIC MAPPING (Single Source of Truth)
# =============================================================================
METRIC_MAP = {
    'profit_target': {'col': 'forward_return_{}b', 'calc': lambda s, wt: (s > wt).mean()},
    'range_bound': {'col': 'stay_in_band_{}b', 'calc': lambda s, wt: s.mean()},
    'hold_floor': {'col': 'stay_above_lower_{}b', 'calc': lambda s, wt: s.mean()},
    'hold_ceiling': {'col': 'stay_below_upper_{}b', 'calc': lambda s, wt: s.mean()},
    'finish_above': {'col': 'above_upper_last_{}b', 'calc': lambda s, wt: s.mean()},
    'finish_below': {'col': 'below_lower_last_{}b', 'calc': lambda s, wt: s.mean()},
}


# =============================================================================
# 1. JIT-COMPILED CORE FUNCTIONS
# =============================================================================
@njit(cache=True)
def _highpass3(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    hp = np.zeros(n)
    if n < 4: return hp
    alpha = (1 - np.sin(2 * np.pi / period)) / (1 + np.sin(2 * np.pi / period))
    a1 = (1 - alpha / 2) ** 2
    b1 = 2 * (1 - alpha)
    b2 = -(1 - alpha) ** 2
    for i in range(3, n):
        hp[i] = a1 * (data[i] - 2 * data[i - 1] + data[i - 2]) + b1 * hp[i - 1] + b2 * hp[i - 2]
    return hp


@njit(cache=True)
def _bandpass2(data: np.ndarray, period: np.ndarray, bandwidth: float) -> np.ndarray:
    n = len(data)
    bp = np.zeros(n)
    bp_1, bp_2 = 0.0, 0.0
    for i in range(2, n):
        p = period[i]
        p = 4.0 if p < 4.0 else (100.0 if p > 100.0 else p)
        L1 = np.cos(2 * np.pi / p)
        G1 = np.cos(bandwidth * 2 * np.pi / p)
        if abs(G1) < 1e-10:
            G1 = 1e-10 if G1 >= 0 else -1e-10
        G1 = -0.99 if G1 < -0.99 else (0.99 if G1 > 0.99 else G1)
        disc = 1.0 / (G1 ** 2) - 1.0
        disc = 0.0 if disc < 0.0 else disc
        S1 = 1.0 / G1 - np.sqrt(disc)
        bp[i] = (0.5 * (1 - S1) * (data[i] - data[i - 2]) + L1 * (1 + S1) * bp_1 - S1 * bp_2)
        bp_2, bp_1 = bp_1, bp[i]
    return bp


@njit(cache=True)
def _auto_tune(data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data)
    dc_series = np.zeros(n)
    min_corr_series = np.zeros(n)
    hp = _highpass3(data, window)
    dc_smooth = float(window)
    corr = np.zeros(window + 1)
    for i in range(window + 5, n):
        for lag in range(1, window + 1):
            Sx, Sy, Sxx, Syy, Sxy = 0.0, 0.0, 0.0, 0.0, 0.0
            seg_len = window - lag
            base = i - window
            for k in range(seg_len):
                x = hp[base + k]; y = hp[base + k + lag]
                Sx += x; Sy += y; Sxx += x * x; Syy += y * y; Sxy += x * y
            den1 = seg_len * Sxx - Sx * Sx
            den2 = seg_len * Syy - Sy * Sy
            if den1 > 1e-10 and den2 > 1e-10:
                corr[lag] = (seg_len * Sxy - Sx * Sy) / np.sqrt(den1 * den2)
            else:
                corr[lag] = 0.0
        min_idx, min_val = 1, corr[1]
        for lag in range(2, window + 1):
            if corr[lag] < min_val: min_val = corr[lag]; min_idx = lag
        dc_raw = 2.0 * min_idx
        if dc_raw > dc_smooth + 2.0:
            dc_smooth += 2.0
        elif dc_raw < dc_smooth - 2.0:
            dc_smooth -= 2.0
        else:
            dc_smooth = dc_raw
        dc_series[i] = dc_smooth
        min_corr_series[i] = min_val
    return dc_series, min_corr_series, hp


@njit(cache=True)
def _run_backtest(prices: np.ndarray, dc_series: np.ndarray, min_corr_series: np.ndarray,
                  hp: np.ndarray, window: int, bandwidth: float, threshold: float,
                  lookahead_bars: int, win_threshold: float, signal_type: int,
                  enable_above_baseline: int, enable_below_baseline: int) -> Tuple:
    n = len(prices)
    bp = _bandpass2(prices, dc_series, bandwidth)
    roc = np.zeros(n)
    for i in range(2, n): roc[i] = bp[i] - bp[i - 2]
    signals = np.zeros(n)
    start_i = max(window + 10, 5)

    for i in range(start_i, n):
        long_cross = roc[i] > 0 >= roc[i - 1]
        short_cross = roc[i] < 0 <= roc[i - 1]
        strong_cycle = min_corr_series[i] < threshold
        above_baseline = hp[i] > 0 if enable_above_baseline else True
        below_baseline = hp[i] < 0 if enable_below_baseline else True

        if signal_type == 1:
            if long_cross and strong_cycle and below_baseline: signals[i] = 1.0
        elif signal_type == -1:
            if short_cross and strong_cycle and above_baseline: signals[i] = -1.0
        else:
            if long_cross and strong_cycle and below_baseline:
                signals[i] = 1.0
            elif short_cross and strong_cycle and above_baseline:
                signals[i] = -1.0

    forward_returns = np.zeros(n)
    within_bound = np.zeros(n)
    within_lower = np.zeros(n)
    within_upper = np.zeros(n)
    above_upper_last = np.zeros(n)
    below_lower_last = np.zeros(n)

    for i in range(n):
        if signals[i] != 0.0:
            if i + lookahead_bars >= n: continue
            end = i + lookahead_bars + 1
            last_idx = end - 1

            best_ret = -np.inf
            has_valid_bar = False
            for j in range(i + 1, end):
                ret = (prices[j] / prices[i] - 1.0) * signals[i]
                if ret > best_ret: best_ret = ret
                has_valid_bar = True
            forward_returns[i] = best_ret if has_valid_bar else 0.0

            min_p = prices[i]; max_p = prices[i]
            for j in range(i + 1, end):
                if prices[j] < min_p: min_p = prices[j]
                if prices[j] > max_p: max_p = prices[j]

            # FIXED: Handle negative win_threshold by ensuring bounds are properly ordered
            lower_bound = prices[i] * (1.0 - win_threshold)
            upper_bound = prices[i] * (1.0 + win_threshold)
            if lower_bound > upper_bound:
                lower_bound, upper_bound = upper_bound, lower_bound
            within_bound[i] = 1.0 if (min_p >= lower_bound and max_p <= upper_bound) else 0.0

            min_p = prices[i]
            for j in range(i + 1, end):
                if prices[j] < min_p: min_p = prices[j]
            within_lower[i] = 1.0 if min_p >= prices[i] * (1.0 - win_threshold) else 0.0

            max_p = prices[i]
            for j in range(i + 1, end):
                if prices[j] > max_p: max_p = prices[j]
            within_upper[i] = 1.0 if max_p <= prices[i] * (1.0 + win_threshold) else 0.0

            above_upper_last[i] = 1.0 if prices[last_idx] > prices[i] * (1.0 + win_threshold) else 0.0
            below_lower_last[i] = 1.0 if prices[last_idx] < prices[i] * (1.0 - win_threshold) else 0.0

    return signals, None, None, forward_returns, within_bound, within_lower, within_upper, bp, roc, above_upper_last, below_lower_last


# =============================================================================
# 2. CLASS WRAPPER
# =============================================================================
class AutoTuneStrategy:
    def __init__(self, window: int = 26, bandwidth: float = 0.22, threshold: float = -0.22,
                 lookahead_bars: int = 5, win_threshold: float = 0.01, signal_type: int = 0,
                 enable_above_baseline: int = 1, enable_below_baseline: int = 1):
        self.window = window
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.lookahead_bars = lookahead_bars
        self.win_threshold = win_threshold
        self.signal_type = signal_type
        self.enable_above_baseline = enable_above_baseline
        self.enable_below_baseline = enable_below_baseline

    def generate_signals(self, prices) -> pd.DataFrame:
        prices = np.asarray(prices, dtype=np.float64)
        dc_series, min_corr_series, hp = _auto_tune(data=prices, window=self.window)
        sig, _, _, fwd, wr2, wr3, wr4, bp, roc, wr5, wr6 = _run_backtest(
            prices=prices, dc_series=dc_series, min_corr_series=min_corr_series, hp=hp,
            window=self.window, bandwidth=self.bandwidth, threshold=self.threshold,
            win_threshold=self.win_threshold, lookahead_bars=self.lookahead_bars,
            signal_type=self.signal_type, enable_below_baseline=self.enable_below_baseline,
            enable_above_baseline=self.enable_above_baseline
        )
        LA = self.lookahead_bars
        return pd.DataFrame({
            'price': prices, 'dominant_cycle': dc_series, 'bandpass': bp, 'roc': roc,
            'min_correlation': min_corr_series, 'highpass': hp, 'signal': sig,
            f'forward_return_{LA}b': fwd,
            f'stay_in_band_{LA}b': wr2,
            f'stay_above_lower_{LA}b': wr3,
            f'stay_below_upper_{LA}b': wr4,
            f'above_upper_last_{LA}b': wr5,
            f'below_lower_last_{LA}b': wr6
        })

    def evaluate(self, results: pd.DataFrame) -> Dict:
        df = results.copy()
        LA = self.lookahead_bars
        fwd_col = f'forward_return_{LA}b'
        wr2_col = f'stay_in_band_{LA}b'
        wr3_col = f'stay_above_lower_{LA}b'
        wr4_col = f'stay_below_upper_{LA}b'
        wr5_col = f'above_upper_last_{LA}b'
        wr6_col = f'below_lower_last_{LA}b'

        signals = df[df['signal'] != 0].copy()
        label = 'long' if self.signal_type == 1 else ('short' if self.signal_type == -1 else 'both')

        total_signals = len(signals)
        if total_signals == 0:
            return {'status': f'No {label} signals generated in evaluation period'}

        profit_target = (signals[fwd_col].values > self.win_threshold).mean()
        range_bound = signals[wr2_col].values.mean()
        hold_floor = signals[wr3_col].values.mean()
        hold_ceiling = signals[wr4_col].values.mean()
        finish_above = signals[wr5_col].values.mean()
        finish_below = signals[wr6_col].values.mean()

        return {
            'total_signals': total_signals,
            'signal_type': label,
            f'profit_target_{label}_{LA}b': f"{profit_target * 100:.1f}%",
            f'range_bound_{label}_{LA}b': f"{range_bound * 100:.1f}%",
            f'hold_floor_{label}_{LA}b': f"{hold_floor * 100:.1f}%",
            f'hold_ceiling_{label}_{LA}b': f"{hold_ceiling * 100:.1f}%",
            f'finish_above_{label}_{LA}b': f"{finish_above * 100:.1f}%",
            f'finish_below_{label}_{LA}b': f"{finish_below * 100:.1f}%",
            f'expectancy_{label}_{LA}b': f"{signals[fwd_col].mean() * 100:.3f}%",
        }


# =============================================================================
# 3. ARGUMENT PARSER
# =============================================================================
def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoTune Strategy: Cycle-aware trading algorithm with Optuna hyperparameter optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    data_group = parser.add_argument_group('Data & Symbol')
    data_group.add_argument('--dataset-id', type=str, default='day')
    data_group.add_argument('--ticker', type=str, default='^GSPC')
    data_group.add_argument('--length-dataset', type=int, default=999999)
    data_group.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")

    strat_group = parser.add_argument_group('Strategy Parameters')
    strat_group.add_argument('--lookahead-bars', type=int, default=10)
    strat_group.add_argument('--win-threshold', type=float, default=0.04,
                             help="Win threshold for optimization metrics. Can be negative for aggressive targets.")
    strat_group.add_argument('--min-signal-density', type=float, default=0.04)

    opt_group = parser.add_argument_group('Optimization & Execution')
    opt_group.add_argument('--train-ratio', type=float, default=0.7)
    opt_group.add_argument('--optimize', type=str, choices=list(METRIC_MAP.keys()), default='hold_floor')
    opt_group.add_argument('--n-trials', type=int, default=999999)
    opt_group.add_argument('--timeout', type=int, default=86400)
    opt_group.add_argument('--output-dir', type=str, default='models')
    opt_group.add_argument('--storage', type=str, default=None,
                           help='Optuna storage URL (e.g., sqlite:///optuna.db). Default: in-memory.')
    opt_group.add_argument('--study-name', type=str, default=None,
                           help='Optuna study name. Auto-generated if --storage is provided but name is omitted.')

    flag_group = parser.add_argument_group('Execution Flags')
    flag_group.add_argument('--real-time', action=argparse.BooleanOptionalAction, default=False)
    flag_group.add_argument('--model-path', type=str, default=None)
    flag_group.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)
    flag_group.add_argument('--verbose-short', action=argparse.BooleanOptionalAction, default=False)
    flag_group.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False)
    return parser


def perfect_score_callback(study, trial):
    if study.best_value is not None and study.best_value >= 0.9999:
        print("\n🎯 Perfect score reached (≥ 0.9999). Stopping optimization early.")
        study.stop()


# =============================================================================
# 4. ENTRY POINT
# =============================================================================
def entry(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_id = args.dataset_id
    ticker = args.ticker
    verbose = args.verbose
    verbose_short = args.verbose_short
    real_time = args.real_time
    output_dir = args.output_dir
    length_dataset = args.length_dataset
    optimize = args.optimize
    clip = args.clip
    np.random.seed(42)
    filename = get_filename_for_dataset(dataset_choice=dataset_id, older_dataset=None)
    if verbose: print(f"📂 Loading dataset from: {filename}")
    with open(filename, "rb") as f:
        cache = pickle.load(f)
    spx = cache[ticker].copy()
    spx = spx.iloc[-length_dataset:].copy()

    if verbose:
        first_date = spx.index[0]
        last_date = spx.index[-1]
        num_bars = len(spx)
        print(f"\n📊 Dataset Loaded: {ticker} ({dataset_id})")
        print(f"   Bars: {num_bars:,} | Range: {first_date.strftime('%Y%m%d')}  ->  {last_date.strftime('%Y%m%d')}\n")

    closes = spx['Close'].squeeze().dropna().copy()

    # =============================================================================
    # 🔄 REAL-TIME MODE
    # =============================================================================
    if real_time:
        os.makedirs(output_dir, exist_ok=True)
        if args.model_path:
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Specified model not found: {args.model_path}")
            model_path = args.model_path
            if verbose and not verbose_short: print(f"📥 Loading specified real-time model: {model_path}")
        else:
            model_files = glob.glob(os.path.join(output_dir, "autotune_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in '{output_dir}'. Run with --no-real-time first.")
            model_path = max(model_files, key=os.path.getmtime)
            if verbose and not verbose_short: print(f"📥 Loading most recent real-time model: {model_path}")
        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)
        rt_params = saved_model['params']
        assert 'win_threshold' in saved_model
        rt_win_threshold = saved_model['win_threshold']
        assert 'signal_type' in saved_model
        rt_signal_type = saved_model.get('signal_type', 0)
        strat_rt = AutoTuneStrategy(**rt_params, win_threshold=rt_win_threshold, signal_type=rt_signal_type)
        assert dataset_id == saved_model['dataset_id'], f"{dataset_id} == {saved_model['dataset_id']}"
        assert ticker == saved_model['ticker'], f"{ticker} == {saved_model['ticker']}"
        if clip:
            fd1 = closes.index[-1].strftime('%Y-%m-%d')
            closes = closes.iloc[:-1].copy()
            fd2 = closes.index[-1].strftime('%Y-%m-%d')
            if verbose:
                print(f"Clipping :: {fd1} to {fd2}")
        results_rt = strat_rt.generate_signals(closes)
        last_row = results_rt.iloc[-1]
        last_signal = last_row['signal']
        if 'ticker' in saved_model and ticker != saved_model['ticker']:
            raise ValueError(f"Ticker mismatch: CLI={ticker}, Model={saved_model['ticker']}")
        total_rt_signals = (results_rt['signal'] != 0).sum()
        long_rt = (results_rt['signal'] == 1.0).sum()
        short_rt = (results_rt['signal'] == -1.0).sum()
        signal_density = total_rt_signals / len(closes)
        last_price = last_row['price']
        last_date = closes.index[-1].strftime('%Y-%m-%d')
        la_date = get_next_step(the_date=closes.index[-1], dataset_id=saved_model['dataset_id'], nn=saved_model['params']['lookahead_bars']).strftime('%Y-%m-%d')
        signal = 1 if (last_signal == 1.0 and rt_signal_type in ('long', 'both')) else (-1 if (last_signal == -1.0 and rt_signal_type in ('short', 'both')) else 0)
        signal_str = "🟢 LONG" if signal == 1.0 else ("🔴 SHORT" if last_signal == -1.0 else "⚪ NONE")
        print(f"Dataset: {dataset_id} | Look Ahead: {rt_params['lookahead_bars']} bars")
        print(f"Training score: {saved_model['train_score']:.6%} | Validation score: {saved_model['val_score']:.6%}")
        print(f"Optimization metric: {saved_model['optimize_metric']} | Win Threshold: {rt_win_threshold:.2%} | Signal Type: {rt_signal_type}")
        print(f"Datapoint used: {last_date} | Signal computed: {last_signal} {signal_str}")
        target_price = 0.
        if last_signal != 0:
            training_score, validation_score = saved_model['train_score'], saved_model['val_score']
            if rt_signal_type in ('long', 'both'):
                floor_price = last_price * (1 - rt_win_threshold)
                print(f"TODO There's a {validation_score:.2%} historical probability that if you enter LONG now at price ${last_price:.0f}, the price will not fall below  ${floor_price:.0f} (${last_price:.0f} × {1 - rt_win_threshold}) at any point during the next {saved_model['params']['lookahead_bars']} bars.")
            if rt_signal_type in ('short', 'both'):
                ceiling_price = last_price * (1 + rt_win_threshold)
                print(f"TODO here's a {validation_score:.2%} historical probability that if you enter SHORT now at price ${last_price:.0f}, the price will not rise above ${ceiling_price:.0f} (${last_price:.0f} × {1 + rt_win_threshold}) at any point during the next {saved_model['params']['lookahead_bars']} bars.")
            if saved_model['optimize_metric'] == 'hold_floor':
                if last_signal == 1. and rt_signal_type == 'long':
                    floor_price = last_price * (1 - rt_win_threshold)
                    print(f"Last data point is {last_date} @{last_price:.0f}, {saved_model['score']:.2%} chance that price STAY ABOVE {floor_price:.0f} until {la_date} ({saved_model['params']['lookahead_bars']}B , {rt_win_threshold:.2%})")
            elif saved_model['optimize_metric'] == 'hold_ceiling':
                if last_signal == -1. and rt_signal_type == 'short':
                    ceiling_price = last_price * (1 + rt_win_threshold)
                    print(f"Last data point is {last_date} @{last_price:.0f}, {saved_model['score']:.2%} chance that price STAY BELOW {ceiling_price:.0f} until {la_date} ({saved_model['params']['lookahead_bars']}B , {rt_win_threshold:.2%})")
            elif saved_model['optimize_metric'] == 'finish_above':
                if last_signal == 1. and rt_signal_type == 'long':
                    target_price = last_price * (1 + rt_win_threshold)
                    print(f"Last data point is {last_date} @{last_price:.0f}, {saved_model['score']:.2%} chance that price CLOSES > {target_price:.0f} at last lookahead bar ({saved_model['params']['lookahead_bars']}B , {rt_win_threshold:.2%})")
            elif saved_model['optimize_metric'] == 'finish_below':
                if last_signal == -1. and rt_signal_type == 'short':
                    target_price = last_price * (1 - rt_win_threshold)
                    print(f"Last data point is {last_date} @{last_price:.0f}, {saved_model['score']:.2%} chance that price CLOSES < {target_price:.0f} at last lookahead bar ({saved_model['params']['lookahead_bars']}B , {rt_win_threshold:.2%})")
        return {'current_price': last_price, 'current_date': last_date, 'train_score': saved_model['train_score'], 'val_score': saved_model['val_score'],
                'threshold': rt_win_threshold, 'signal_type': rt_signal_type, 'dataset_id': dataset_id, 'ticker': ticker, 'optimization_metric': saved_model['optimize_metric'],
                'target_date': la_date, 'signal': last_signal, 'target_price': target_price, 'lookahead':saved_model['params']['lookahead_bars']}

    if verbose:
        print(__doc__)

    # =============================================================================
    # 🎯 OPTIMIZATION MODE
    # =============================================================================
    train_ratio = args.train_ratio
    assert 0.0 < train_ratio < 1.0, "--train-ratio must be between 0.0 and 1.0"
    split_idx = int(len(closes) * train_ratio)
    train_closes = closes.iloc[:split_idx]
    valid_closes = closes.iloc[split_idx:]

    if verbose:
        print(f"📊 Train/Validation Split: {train_ratio:.0%} / {1 - train_ratio:.0%}")
        print(f"   Train: {len(train_closes)} bars ({train_closes.index[0].strftime('%Y%m%d')}::{train_closes.index[-1].strftime('%Y%m%d')}) | "
              f"Validation: {len(valid_closes)} bars ({valid_closes.index[0].strftime('%Y%m%d')}::{valid_closes.index[-1].strftime('%Y%m%d')})\n")

    lookahead_bars = args.lookahead_bars
    win_threshold = args.win_threshold
    # FIXED: Allow negative win_threshold values
    assert -1 < win_threshold < 1, "--win-threshold must be between -1.0 and 1.0 (exclusive)"
    n_trials = args.n_trials
    timeout = args.timeout
    required_signal_density = args.min_signal_density

    METRIC_TO_SIGNAL_TYPE = {
        'hold_floor': 1, 'finish_above': 1,
        'hold_ceiling': -1, 'finish_below': -1,
        'profit_target': 0, 'range_bound': 0
    }
    signal_type = METRIC_TO_SIGNAL_TYPE.get(optimize, 0)
    signal_label = {1: 'long', -1: 'short', 0: 'both'}[signal_type]

    # 🔧 OPTIMIZATION SETUP
    window_min, window_max, window_step          = 2, 160, 1
    bandwidth_min, bandwidth_max, bandwidth_step = 0.00015, 0.99015, 0.01
    threshold_min, threshold_max, threshold_step = -0.9996, -0.0096, 0.01

    if verbose:
        print(f"\n🔧 Running AutoTune Strategy with Optuna Optimization  |  Optimize: {optimize} ({signal_label})  |  Look Ahead: {lookahead_bars}b  |  Dataset id: {dataset_id}  |  "
              f"Win Threshold: {win_threshold:.4%}  |  Optuna: {n_trials}/{timeout}  |  Dataset Length: {len(closes)}")

    def objective(trial):
        try:
            window                = trial.suggest_int('window', window_min, window_max, step=window_step)
            bandwidth             = trial.suggest_float('bandwidth', bandwidth_min, bandwidth_max, step=bandwidth_step)
            threshold             = trial.suggest_float('threshold', threshold_min, threshold_max, step=threshold_step)
            _lookahead_bars       = trial.suggest_int('lookahead_bars', lookahead_bars, lookahead_bars)
            enable_above_baseline = trial.suggest_int('enable_above_baseline', 0, 1)
            enable_below_baseline = trial.suggest_int('enable_below_baseline', 0, 1)

            strat = AutoTuneStrategy(
                window=window, bandwidth=bandwidth, threshold=threshold,
                lookahead_bars=_lookahead_bars, win_threshold=win_threshold,
                signal_type=signal_type, enable_above_baseline=enable_above_baseline, enable_below_baseline=enable_below_baseline,
            )
            results = strat.generate_signals(train_closes)
            df = results.copy()
            signals = df[df['signal'] != 0].copy()

            n_signals = len(signals)
            if n_signals == 0: return 0.0

            # ✅ DYNAMICALLY SELECT CORRECT COLUMN & METRIC BASED ON OPTIMIZE
            col_fmt = METRIC_MAP[optimize]['col'].format(_lookahead_bars)
            obj = METRIC_MAP[optimize]['calc'](signals[col_fmt].values, win_threshold)

            signal_density = n_signals / len(df)
            if signal_density < required_signal_density:
                gap = required_signal_density - signal_density
                penalty = max(0.0, 1.0 - gap * 5.0)
                return obj * penalty
            return obj
        except Exception as e:
            print(f"[WARNING] Trial failed: {e}")
            return 0.0

    # 🗄️ OPTUNA PERSISTENCE SETUP
    storage = args.storage
    study_name = args.study_name
    load_if_exists = False

    if storage:
        load_if_exists = True
        if not study_name:
            study_name = f"autotune_{ticker}_{dataset_id}_{optimize}"

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists
    )

    # 📊 OUTPUT BEST PARAMS IF AVAILABLE BEFORE OPTIMIZATION
    if len(study.trials) > 0 and study.best_value is not None:
        print(f"\n📂 Resuming existing study '{study_name}' ({len(study.trials)} trials found).")
        print(f"   🎯 Best Objective So Far: {study.best_value * 100:.2f}%")
        print(f"   ⚙️ Best Parameters    : {study.best_params}")
    elif len(study.trials) > 0:
        print(f"\n📂 Resuming existing study '{study_name}' ({len(study.trials)} trials found, but no successful trials yet).")
    else:
        if verbose: print(f"\n📂 Starting new Optuna study '{study_name or 'in-memory'}'.")

    if verbose: print(f"\n⚙️ Running Optuna search ({n_trials} trials, {timeout}s timeout) on TRAINING SET...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose, gc_after_trial=True, timeout=timeout, callbacks=[perfect_score_callback])

    if verbose:
        print(f"\n✅ Best Trial Completed (Training Set):")
        print(f"   🎯 Objective Value: {study.best_trial.value * 100:.2f}%")
        print(f"   ⚙️ Parameters     : {study.best_trial.params}")

    score_of_best_trial = study.best_trial.value
    best_params = study.best_trial.params

    # 🟢 EVALUATE BEST PARAMS ON VALIDATION SET
    strat_best = AutoTuneStrategy(**best_params, win_threshold=win_threshold, signal_type=signal_type)
    results_valid = strat_best.generate_signals(valid_closes)
    validation_score = 0.
    sig_valid = results_valid[results_valid['signal'] != 0]
    la = strat_best.lookahead_bars
    if len(sig_valid) > 0:
        # ✅ DYNAMICALLY SELECT CORRECT COLUMN FOR VALIDATION PRINT
        col_fmt = METRIC_MAP[optimize]['col'].format(la)
        win_rate = METRIC_MAP[optimize]['calc'](sig_valid[col_fmt].values, strat_best.win_threshold)
        validation_score = win_rate
        if verbose:
            print("\n📊 Final Performance (VALIDATION SET):")
            print(f"   🎯 Win Rate ({optimize}): {win_rate * 100:.2f}%")
    else:
        if verbose:
            print("\n📊 Final Performance (VALIDATION SET):")
            print("   🎯 Win Rate: N/A (0 signals generated)")

    # 💾 SAVE MODEL
    os.makedirs(output_dir, exist_ok=True)
    w, bw, th, la, op = best_params['window'], best_params['bandwidth'], best_params['threshold'], best_params['lookahead_bars'], optimize
    model_name = f"autotune_model_w{w}_bw{bw:.3f}_th{th:.3f}_la{la}_{op}____{score_of_best_trial}.pkl"
    model_path = os.path.join(output_dir, model_name)

    model_data = {
        'params': best_params, 'win_threshold': win_threshold, 'signal_type': signal_label,
        'signal_type_code': signal_type, 'optimize_metric': optimize, 'train_score': score_of_best_trial, 'val_score': validation_score,
        'dataset_id': dataset_id, 'ticker': ticker,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if verbose: print(f"\n💾 Model saved to: {model_path}")

    if args.plot:
        final_metrics = strat_best.evaluate(results_valid)
        plot_strategy_results(
            results=results_valid, params=best_params, metrics=final_metrics,
            ticker=ticker, dataset_id=dataset_id, output_dir=output_dir,
            verbose=verbose, signal_type=signal_label, optimize_metric=optimize,
            win_threshold=win_threshold,
        )


# =============================================================================
# 🆕 PLOTTING FUNCTION
# =============================================================================
def plot_strategy_results(results: pd.DataFrame, params: dict, metrics: dict,
                          ticker: str, dataset_id: str, output_dir: str = '.',
                          verbose: bool = True, signal_type: str = 'both',
                          optimize_metric: str = 'profit_target', win_threshold: float = 0.04) -> str:
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except: pass

    df = results.copy()
    dates = df.index
    LA = params.get('lookahead_bars', 10)

    signals = df[df['signal'] != 0]
    long_signals = signals[signals['signal'] > 0]
    short_signals = signals[signals['signal'] < 0]

    def classify(df_sig, metric, la, thresh):
        if len(df_sig) == 0: return pd.DataFrame(), pd.DataFrame()
        # ✅ USES CENTRAL METRIC_MAP FOR CONSISTENCY
        col = METRIC_MAP[metric]['col'].format(la)
        if metric == 'profit_target':
            return df_sig[df_sig[col] > thresh], df_sig[df_sig[col] <= thresh]
        return df_sig[df_sig[col] >= 0.99], df_sig[df_sig[col] < 0.99]

    long_success, long_fail = classify(long_signals, optimize_metric, LA, win_threshold)
    short_success, short_fail = classify(short_signals, optimize_metric, LA, win_threshold)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'AutoTune Strategy Results - {ticker} ({dataset_id}) | Signal Type: {signal_type.upper()}\n'
                 f'Optimized for: {optimize_metric.upper()} @ {win_threshold:.4%} threshold',
                 fontsize=15, fontweight='bold', y=0.995)

    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], hspace=0.3, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, df['price'], label='Price', linewidth=1.5, color='steelblue', zorder=1)

    marker_kwargs = dict(s=110, zorder=5, edgecolors='black', linewidths=0.8, alpha=0.85)
    if len(long_success) > 0: ax1.scatter(long_success.index, long_success['price'], marker='o', color='limegreen', label=f'Long ({optimize_metric} Success)', **marker_kwargs)
    if len(long_fail) > 0: ax1.scatter(long_fail.index, long_fail['price'], marker='s', color='crimson', label=f'Long ({optimize_metric} Fail)', **marker_kwargs)
    if len(short_success) > 0: ax1.scatter(short_success.index, short_success['price'], marker='o', color='limegreen', label=f'Short ({optimize_metric} Success)', **marker_kwargs)
    if len(short_fail) > 0: ax1.scatter(short_fail.index, short_fail['price'], marker='s', color='crimson', label=f'Short ({optimize_metric} Fail)', **marker_kwargs)

    ax1.set_ylabel('Price', fontweight='bold')
    ax1.set_title('Price Action with Trading Signals', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    if hasattr(dates[0], 'year') or hasattr(dates, 'strftime'):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(dates, df['bandpass'], label='Bandpass', linewidth=1, color='purple')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_ylabel('Bandpass', fontweight='bold')
    ax2.set_title('Bandpass Filter Output', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3.plot(dates, df['roc'], label='ROC (2-bar)', linewidth=1, color='orange')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax3.set_ylabel('ROC', fontweight='bold')
    ax3.set_title('Rate of Change (Momentum)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    n_long_win = len(long_success); n_long_fail = len(long_fail)
    n_short_win = len(short_success); n_short_fail = len(short_fail)
    total_signals = len(signals)
    plot_success_rate = (n_long_win + n_short_win) / total_signals if total_signals > 0 else 0.0

    metrics_text = (f"Optimized Metric: {optimize_metric.upper()}\n"
                    f"Win Threshold: {win_threshold:.2%} | Lookahead: {LA} bars\n\n"
                    f"Performance:\n")
    metrics_text += (f"\n📊 Visual Classification (per {optimize_metric}):\n"
                     f"  Success: {n_long_win + n_short_win} | Fail: {n_long_fail + n_short_fail}\n"
                     f"  Plotted Rate: {plot_success_rate:.2%} (should match best trial score)")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    fig.text(0.985, 0.97, metrics_text, fontsize=9, verticalalignment='top',
             horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout(rect=[0, 0, 0.97, 0.96])

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"autotune_plot_{ticker}_{dataset_id}_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')

    if verbose:
        print(f"\n📊 Plot saved to: {plot_path}")
        print(f"   Visual success rate: {plot_success_rate:.2%} | Optimized score: {metrics.get(f'{optimize_metric}_{signal_type}_{LA}b', 'N/A')}")
        print("   Showing plot window... (close it to continue)")
    plt.show()
    return plot_path


# =============================================================================
# 5. MAIN
# =============================================================================
if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)