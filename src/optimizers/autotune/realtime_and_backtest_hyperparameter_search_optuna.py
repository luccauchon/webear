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
                x = hp[base + k];
                y = hp[base + k + lag]
                Sx += x;
                Sy += y;
                Sxx += x * x;
                Syy += y * y;
                Sxy += x * y
            den1 = seg_len * Sxx - Sx * Sx
            den2 = seg_len * Syy - Sy * Sy
            if den1 > 1e-10 and den2 > 1e-10:
                corr[lag] = (seg_len * Sxy - Sx * Sy) / np.sqrt(den1 * den2)
            else:
                corr[lag] = 0.0
        min_idx, min_val = 1, corr[1]
        for lag in range(2, window + 1):
            if corr[lag] < min_val:
                min_val = corr[lag];
                min_idx = lag
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
                  capital: float, margin_cost: float, commission: float,
                  lookahead_bars: int, win_threshold: float) -> Tuple:
    n = len(prices)
    bp = _bandpass2(prices, dc_series, bandwidth)
    roc = np.zeros(n)
    for i in range(2, n): roc[i] = bp[i] - bp[i - 2]
    signals = np.zeros(n)
    positions = np.zeros(n)
    equity = np.zeros(n)
    equity[0] = capital
    profit_total = 0.0
    current_pos = 0.0
    entry_price = 0.0
    start_i = max(window + 10, 5)
    for i in range(start_i, n):
        lots = 0.5 * (capital + np.sqrt(1 + profit_total / capital)) / margin_cost
        lots = 0.1 if lots < 0.1 else (10.0 if lots > 10.0 else lots)
        if roc[i] > 0 and roc[i - 1] <= 0 and min_corr_series[i] < threshold and current_pos <= 0:
            signals[i] = 1.0
            if current_pos != 0: profit_total += current_pos * (prices[i] - entry_price)
            entry_price = prices[i];
            current_pos = lots
            profit_total -= lots * entry_price * commission
        elif roc[i] < 0 and roc[i - 1] >= 0 and min_corr_series[i] < threshold and hp[i] > 0 and current_pos >= 0:
            signals[i] = -1.0
            if current_pos != 0: profit_total += current_pos * (prices[i] - entry_price)
            entry_price = prices[i];
            current_pos = -lots
            profit_total -= lots * entry_price * commission
        if current_pos != 0:
            positions[i] = current_pos
            profit_total = current_pos * (prices[i] - entry_price)
        equity[i] = capital + profit_total

    # 📊 Forward Metrics Arrays
    forward_returns = np.zeros(n)
    within_bound = np.zeros(n)  # win_rate_2
    within_lower = np.zeros(n)  # win_rate_3
    within_upper = np.zeros(n)  # win_rate_4

    for i in range(n):
        if signals[i] != 0:
            end = min(n, i + lookahead_bars + 1)

            # 1️⃣ Forward Return (Best signed return)
            best_ret = 0.0
            for j in range(i + 1, end):
                ret = (prices[j] / prices[i] - 1.0) * signals[i]
                if ret > best_ret: best_ret = ret
            forward_returns[i] = best_ret

            # 2️⃣ win_rate_2: Stayed inside ±win_threshold band
            if signals[i] > 0.0:
                max_p = prices[i]
                for j in range(i + 1, end):
                    if prices[j] > max_p: max_p = prices[j]
                within_bound[i] = 1.0 if max_p <= prices[i] * (1.0 + win_threshold) else 0.0
            else:
                min_p = prices[i]
                for j in range(i + 1, end):
                    if prices[j] < min_p: min_p = prices[j]
                within_bound[i] = 1.0 if min_p >= prices[i] * (1.0 - win_threshold) else 0.0

            # 3️⃣ win_rate_3: Stayed ABOVE entry*(1 - win_threshold)
            min_p = prices[i]
            for j in range(i + 1, end):
                if prices[j] < min_p: min_p = prices[j]
            within_lower[i] = 1.0 if min_p >= prices[i] * (1.0 - win_threshold) else 0.0

            # 4️⃣ win_rate_4: Stayed BELOW entry*(1 + win_threshold)
            max_p = prices[i]
            for j in range(i + 1, end):
                if prices[j] > max_p: max_p = prices[j]
            within_upper[i] = 1.0 if max_p <= prices[i] * (1.0 + win_threshold) else 0.0

    return signals, positions, equity, forward_returns, within_bound, within_lower, within_upper, bp, roc


# =============================================================================
# 2. CLASS WRAPPER
# =============================================================================
class AutoTuneStrategy:
    def __init__(self, window: int = 26, bandwidth: float = 0.22, threshold: float = -0.22,
                 capital: float = 100_000, margin_cost: float = 5000, commission: float = 0.0005,
                 lookahead_bars: int = 5, win_threshold: float = 0.01):
        self.window = window
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.capital = capital
        self.margin_cost = margin_cost
        self.commission = commission
        self.lookahead_bars = lookahead_bars
        self.win_threshold = win_threshold

    def generate_signals(self, prices) -> pd.DataFrame:
        prices = np.asarray(prices, dtype=np.float64)
        dc_series, min_corr_series, hp = _auto_tune(prices, self.window)
        sig, pos, eq, fwd, wr2, wr3, wr4, bp, roc = _run_backtest(
            prices, dc_series, min_corr_series, hp,
            self.window, self.bandwidth, self.threshold,
            self.capital, self.margin_cost, self.commission,
            self.lookahead_bars, self.win_threshold
        )
        LA = self.lookahead_bars
        return pd.DataFrame({
            'price': prices, 'dominant_cycle': dc_series, 'bandpass': bp, 'roc': roc,
            'min_correlation': min_corr_series, 'highpass': hp, 'signal': sig,
            'position': pos, 'equity': eq,
            f'forward_return_{LA}b': fwd,
            f'stay_in_band_{LA}b': wr2,
            f'stay_above_lower_{LA}b': wr3,
            f'stay_below_upper_{LA}b': wr4
        })

    def evaluate(self, results: pd.DataFrame, min_bars: int = 100, signal_type: str = 'both') -> Dict:
        if signal_type not in ('long', 'short', 'both'):
            raise ValueError("signal_type must be 'long', 'short', or 'both'")

        df = results.iloc[min_bars:].copy()
        LA = self.lookahead_bars
        fwd_col = f'forward_return_{LA}b'
        wr2_col = f'stay_in_band_{LA}b'
        wr3_col = f'stay_above_lower_{LA}b'
        wr4_col = f'stay_below_upper_{LA}b'

        if signal_type == 'long':
            signals = df[df['signal'] == 1.0].copy();
            label = 'long'
        elif signal_type == 'short':
            signals = df[df['signal'] == -1.0].copy();
            label = 'short'
        else:
            signals = df[df['signal'] != 0].copy();
            label = 'both'

        total_signals = len(signals)
        if total_signals == 0:
            return {'status': f'No {label} signals generated in evaluation period'}

        win_rate = (signals[fwd_col].values > self.win_threshold).mean()
        win_rate_2 = signals[wr2_col].values.mean()
        win_rate_3 = signals[wr3_col].values.mean()
        win_rate_4 = signals[wr4_col].values.mean()

        equity = df['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown, max_dd, total_return, sharpe_approx = 0, 0, 0, 0
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                drawdown = (equity - peak) / peak
                max_dd = drawdown.min()
            except Exception:
                drawdown = 0;
                max_dd = 0
            try:
                total_return = (equity[-1] / equity[0]) - 1 if len(equity) > 0 else 0.0
            except Exception:
                total_return = 0
            try:
                std_dd = np.std(drawdown)
                sharpe_approx = total_return / (std_dd + 1e-10) if std_dd > 0 else 0
            except Exception:
                sharpe_approx = 0

        return {
            'total_signals': total_signals,
            'signal_type': label,
            f'win_rate_{label}_{LA}b': f"{win_rate * 100:.1f}%",
            f'win_rate_2_{label}_{LA}b': f"{win_rate_2 * 100:.1f}%",
            f'win_rate_3_{label}_{LA}b': f"{win_rate_3 * 100:.1f}%",
            f'win_rate_4_{label}_{LA}b': f"{win_rate_4 * 100:.1f}%",
            f'expectancy_{label}_{LA}b': f"{signals[fwd_col].mean() * 100:.3f}%",
            'total_return': f"{total_return * 100:.2f}%",
            'max_drawdown': f"{max_dd * 100:.2f}%",
            'sharpe_approx': f"{sharpe_approx:.2f}",
            'final_equity': f"${equity[-1]:,.0f}" if len(equity) > 0 else "$0"
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
    data_group.add_argument('--dataset-id', type=str, default='day',
                            help='Dataset identifier passed to get_filename_for_dataset()')
    data_group.add_argument('--ticker', type=str, default='^GSPC',
                            help='Ticker symbol to analyze (e.g., ^GSPC, AAPL, ES)')
    data_group.add_argument('--length-dataset', type=int, default=999999,
                            help='Number of trailing data points to load from cache')

    strat_group = parser.add_argument_group('Strategy Parameters')
    strat_group.add_argument('--capital', type=float, default=100_000,
                             help='Initial account capital for position sizing')
    strat_group.add_argument('--margin-cost', type=float, default=5_000,
                             help='Cost per contract/margin unit')
    strat_group.add_argument('--commission', type=float, default=0.0005,
                             help='Commission rate per trade (e.g., 0.0005 = 0.05%%)')
    strat_group.add_argument('--lookahead-bars', type=int, default=10,
                             help='Forward-looking window (in bars) for signal evaluation')
    strat_group.add_argument('--win-threshold', type=float, default=0.04,
                             help='Price movement threshold for win-rate calculation (0.04 = 4%%)')
    strat_group.add_argument('--min-bars', type=int, default=100,
                             help='Warm-up period (bars) before starting signal evaluation')
    strat_group.add_argument('--min-signal-density', type=float, default=0.04,
                             help='Minimum number of signal detected w.r.t. the total length of df (n-signals / total-length-df) to be considered as a valid optimization pass (0.04 = 4%%)')

    opt_group = parser.add_argument_group('Optimization & Execution')
    opt_group.add_argument('--signal-type', type=str, choices=['long', 'short', 'both'], default='long',
                           help='Direction of signals to analyze/optimize')
    opt_group.add_argument('--optimize', type=str, choices=['win_rate', 'win_rate_2', 'win_rate_3', 'win_rate_4'], default='win_rate_3',
                           help='Objective metric for Optuna to maximize')
    opt_group.add_argument('--n-trials', type=int, default=999999,
                           help='Number of Optuna optimization trials')
    opt_group.add_argument('--timeout', type=int, default=86400,
                           help='Maximum runtime in seconds for the optimization phase')
    opt_group.add_argument('--output-dir', type=str, default='.',
                           help='Directory to save and load optimized model files')

    flag_group = parser.add_argument_group('Execution Flags')
    flag_group.add_argument('--real-time', action=argparse.BooleanOptionalAction, default=False,
                            help='Run in real-time signal checking mode (loads latest saved model)')
    flag_group.add_argument('--model-path', type=str, default=None,
                            help='Specific path to a saved .pkl model for real-time mode. Overrides auto-loading the newest model.')
    flag_group.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True,
                            help='Print detailed progress, metrics, and explanations')
    flag_group.add_argument('--verbose-short', action=argparse.BooleanOptionalAction, default=False,
                            help='For Real-Time Only, Print very short progress, metrics, and explanations')
    return parser


# =============================================================================
# 4. ENTRY POINT
# =============================================================================
def entry(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dataset_id = args.dataset_id
    ticker = args.ticker
    verbose = args.verbose
    verbose_short = args.verbose_short
    real_time = args.real_time
    output_dir = args.output_dir
    length_dataset = args.length_dataset

    np.random.seed(42)

    filename = get_filename_for_dataset(dataset_choice=dataset_id, older_dataset=None)
    with open(filename, "rb") as f:
        cache = pickle.load(f)
    spx = cache[ticker].copy()
    spx = spx.iloc[-length_dataset:].copy()
    closes = spx['Close'].squeeze().dropna().copy()

    # =============================================================================
    # 🔄 REAL-TIME MODE
    # =============================================================================
    if real_time:
        os.makedirs(output_dir, exist_ok=True)

        # 🆕 SPECIFIED MODEL PATH LOGIC
        if args.model_path:
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Specified model not found: {args.model_path}")
            model_path = args.model_path
            if verbose and not verbose_short: print(f"📥 Loading specified real-time model: {model_path}")
        else:
            model_files = glob.glob(os.path.join(output_dir, "autotune_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in '{output_dir}'. Run with --no-real-time first to optimize & save.")
            model_path = max(model_files, key=os.path.getmtime)
            if verbose and not verbose_short: print(f"📥 Loading most recent real-time model (out of {len(model_files)}): {model_path}")

        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)

        rt_params = saved_model['params']
        rt_win_threshold = saved_model['win_threshold']
        strat_rt = AutoTuneStrategy(**rt_params, win_threshold=rt_win_threshold)
        assert dataset_id == saved_model['dataset_id'], f"{dataset_id} == {saved_model['dataset_id']}"
        results_rt = strat_rt.generate_signals(closes)
        last_row = results_rt.iloc[-1]
        last_signal = last_row['signal']
        assert ticker == saved_model['ticker'] if 'ticker' in saved_model else True
        # 📊 SIGNAL COUNTING
        total_rt_signals = (results_rt['signal'] != 0).sum()
        long_rt = (results_rt['signal'] == 1.0).sum()
        short_rt = (results_rt['signal'] == -1.0).sum()
        if saved_model['signal_type'] == 'both':
            signal_density = total_rt_signals / len(closes)
        elif saved_model['signal_type'] == 'long':
            signal_density = long_rt / len(closes)
        elif saved_model['signal_type'] == 'short':
            signal_density = short_rt / len(closes)
        last_price = last_row['price']
        last_date = spx.index[-1].strftime('%Y-%m-%d')
        la_date = get_next_step(the_date=spx.index[-1], dataset_id=saved_model['dataset_id'], nn=saved_model['params']['lookahead_bars']).strftime('%Y-%m-%d')
        signal_str = "🟢 LONG" if last_signal == 1.0 else ("🔴 SHORT" if last_signal == -1.0 else "⚪ NONE")
        if verbose_short:
            if last_signal != 0:
                if saved_model['optimize_metric'] == 'win_rate_3':
                    if last_signal == 1. and saved_model['signal_type'] == 'long':
                        print(f"Last data point is {last_date} @{last_price:.0f}, {saved_model['score']:.2%} chance that price STAY ABOVE {last_price*(1-rt_win_threshold):.0f} until {la_date} ({saved_model['params']['lookahead_bars']}B , {rt_win_threshold:.2%})")
                    else:
                        assert last_signal == -1.
                else:
                    if saved_model['optimize_metric'] == 'win_rate_4':
                        if last_signal == -1. and saved_model['signal_type'] == 'short':
                            print(f"Last data point is {last_date} @{last_price:.0f}, {saved_model['score']:.2%} chance that price STAY BELOW {last_price * (1 + rt_win_threshold):.0f} until {la_date} ({saved_model['params']['lookahead_bars']}B , {rt_win_threshold:.2%})")
                        else:
                            print(f"{saved_model}")
                            print("TODO_1")
                    else:
                        print(f"{saved_model}")
                        print("TODO_3")
        if verbose and not verbose_short:
            win_rate_str   = f"Price reaches ≥ +{rt_win_threshold:.0%} during lookahead      → % of signals that achieve a forward return ≥ +{rt_win_threshold:.0%}"
            win_rate_2_str = f"Price stays INSIDE ±{rt_win_threshold:.0%} (consolidation)    → % of signals where price stays inside a ±{rt_win_threshold:.0%} range (mean-reversion/consolidation)"
            win_rate_3_str = f"Price stays ABOVE {last_price*(1-rt_win_threshold):.0f})      → % of signals where price stays above the lower bound (useful for credit spreads or trailing stops)"
            win_rate_4_str = f"Price stays BELOW entry*(1+{rt_win_threshold:.0%})            → % of signals where price stays below the upper bound"
            _win_rate_tmp_str = win_rate_str
            _win_rate_tmp_str = win_rate_2_str if saved_model['optimize_metric'] == 'win_rate_2' else _win_rate_tmp_str
            _win_rate_tmp_str = win_rate_3_str if saved_model['optimize_metric'] == 'win_rate_3' else _win_rate_tmp_str
            _win_rate_tmp_str = win_rate_4_str if saved_model['optimize_metric'] == 'win_rate_4' else _win_rate_tmp_str
            print(f"\n🔍 Real-Time Signal Check (Last Bar={last_date}):")
            print(f"   📊 Total Signals (Dataset) : {total_rt_signals} (Long: {long_rt}, Short: {short_rt})")
            print(f"   📉  Signal Type            : {saved_model['signal_type']} (@{saved_model['win_threshold']:.2%} in {saved_model['params']['lookahead_bars']}B={la_date})")
            print(f"   📉  Optimized Metric       : {saved_model['optimize_metric']} ({_win_rate_tmp_str})")
            print(f"   📉 Signal Density ({saved_model['signal_type']})   : {signal_density:.2%} (over {len(closes)} bars)")
            print(f"   📉 Score                   : {saved_model['score']:.2%}")
            if total_rt_signals < 50:
                print(f"   ⚠️  LOW EVENT COUNT          : Statistical reliability may be compromised.")
            print(f"   Timestamp (last row)       : {spx.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Close Price                : {last_price:.2f}")
            print(f"   Signal                     : {signal_str}")
            print()
            print(f"   ROC (2-bar)                : {last_row['roc']:.4f}")
            print(f"   It measures the momentum of the bandpass-filtered price over the last 2 bars.")
            print(f"   Value                   Interpretation\n"
                  f"    ROC > 0 and rising      Bullish momentum building — potential long entry if other conditions align\n"
                  f"    ROC < 0 and falling     Bearish momentum building — potential short entry\n"
                  f"    ROC crossing 0 upward   Momentum shift from negative → positive (long trigger candidate)\n"
                  f"    ROC crossing 0 downward Momentum shift from positive → negative (short trigger candidate)\n"
                  f"    ROC near 0              Consolidation / indecision — avoid trading")
            print()
            print(f"   Min Corr    : {last_row['min_correlation']:.4f}")
            print(f"   During the _auto_tune process, the algorithm computes autocorrelation of the high-pass filtered data across lags 1 → window.\n"
                  f"    min_corr is the lowest (most negative) correlation value found — and the lag where it occurs helps estimate the dominant cycle.")
            print(f"   Value Range            Interpretation\n"
                  f"    < -0.5                 Strong, well-defined cycle — high confidence in signal timing\n"
                  f"    -0.3 to -0.5           Moderate cycle definition — usable but verify with price action\n"
                  f"    > -0.3                 Weak/no clear cycle — market is noisy or trending; signals less reliable\n"
                  f"    Close to 0 or positive No detectable cycle — avoid trading; strategy likely whipsawing")
            print()
            print(f"   Dom Cycle   : {last_row['dominant_cycle']:.2f}")
            print(f"   It estimates the current dominant market cycle length (e.g., 20 bars = ~1 month on daily data). The value is smoothed to avoid jumping erratically.")
            print(f"   Value            Interpretation\n"
                  f"    10–25 bars       Short-term cycle — good for swing trades (2–5 day holds)\n"
                  f"    25–50 bars       Medium-term cycle — suitable for position trades (1–3 weeks)\n"
                  f"    >50 bars         Long-term cycle — signals may be infrequent but higher conviction\n"
                  f"    Rapidly changing Market regime shift — be cautious; parameters may need re-optimization")
        return
    lookahead_bars = args.lookahead_bars
    signal_type = args.signal_type
    required_signal_density = args.min_signal_density
    win_threshold = args.win_threshold
    optimize = args.optimize
    n_trials = args.n_trials
    timeout = args.timeout
    capital = args.capital
    margin_cost = args.margin_cost
    commission = args.commission
    min_bars = args.min_bars
    if verbose:
        win_rate_str   = f"Price reaches ≥ +{win_threshold:.0%} during lookahead      → % of signals that achieve a forward return ≥ +{win_threshold:.0%}"
        win_rate_2_str = f"Price stays INSIDE ±{win_threshold:.0%} (consolidation)    → % of signals where price stays inside a ±{win_threshold:.0%} range (mean-reversion/consolidation)"
        win_rate_3_str = f"Price stays ABOVE entry*(1-{win_threshold:.0%})            → % of signals where price stays above the lower bound (useful for credit spreads or trailing stops)"
        win_rate_4_str = f"Price stays BELOW entry*(1+{win_threshold:.0%})            → % of signals where price stays below the upper bound"
        print()
        print(f"win_rate    = {win_rate_str}")
        print(f"win_rate_2  = {win_rate_2_str}")
        print(f"win_rate_3  = {win_rate_3_str}")
        print(f"win_rate_4  = {win_rate_4_str}")
        print()
        print(f"signal_type = Which trades to look at (long/short/both)")
        print(f"optimize    = What metric to maximize for those trades")
        print(f"signal_type='long' + optimize='win_rate_2' → 'Find parameters where longs tend to stay tightly range-bound after entry (good for selling covered calls).'")
        print(f"signal_type='short' + optimize='win_rate'  → 'Find parameters where short signals reliably drop by at least {win_threshold:.0%}.'")
        print(f"signal_type='both' + optimize='win_rate_3' → 'Find parameters where all signals avoid breaking below the lower threshold (useful for risk-defined strategies).'")
        print(f"signal_type='long' + optimize='win_rate_3' → 'Find the parameter set that maximizes the percentage of long trades that never drop more than {win_threshold:.0%} below their entry price during the {lookahead_bars} lookahead window.'")
    # =============================================================================
    # 🎯 OPTIMIZATION MODE
    # =============================================================================
    window_min, window_max = 2, 160
    bandwith_min, bandwith_max = 0.00015, 0.9995
    threshold_min, threshold_max = -0.9996, -0.0001
    if verbose:
        print(f"🔧 Running AutoTune Strategy with Optuna Optimization  |  Signal Type: {signal_type}  |  Optimize: {optimize}   |  Look Ahead: {lookahead_bars}b  |  Dataset id: {dataset_id}  |  "
              f"Win Threshold: {win_threshold:.0%}  |  Optuna: {n_trials}/{timeout}  |  Dataset Length: {len(closes)}")
        print(f"Parameters search space\n"
              f"  Window    : {window_min}::{window_max}\n"
              f"  Bandwitdh : {bandwith_min}::{bandwith_max}\n"
              f"  Threshold : {threshold_min}::{threshold_max}\n")
    def objective(trial):
        try:
            # Parameter search space
            window = trial.suggest_int('window', window_min, window_max, step=1)
            bandwidth = trial.suggest_float('bandwidth', bandwith_min, bandwith_max)
            threshold = trial.suggest_float('threshold', threshold_min, threshold_max)
            lookahead = trial.suggest_int('lookahead_bars', lookahead_bars, lookahead_bars, step=1)

            strat = AutoTuneStrategy(
                window=window, bandwidth=bandwidth, threshold=threshold, lookahead_bars=lookahead,
                win_threshold=win_threshold, capital=capital, margin_cost=margin_cost, commission=commission
            )
            results = strat.generate_signals(closes)
            df = results.iloc[min_bars:].copy()

            fwd_col = f'forward_return_{lookahead}b'
            if signal_type == 'long':
                signals = df[df['signal'] == 1.0].copy()
            elif signal_type == 'short':
                signals = df[df['signal'] == -1.0].copy()
            else:
                signals = df[df['signal'] != 0].copy()

            if len(signals) == 0:
                return 0.0

            # 🆕 1. Calculate base objective first
            if optimize == 'win_rate':
                obj = (signals[fwd_col].values > win_threshold).mean()
            elif optimize == 'win_rate_2':
                obj = signals[f'stay_in_band_{lookahead}b'].mean()
            elif optimize == 'win_rate_3':
                obj = signals[f'stay_above_lower_{lookahead}b'].mean()
            elif optimize == 'win_rate_4':
                obj = signals[f'stay_below_upper_{lookahead}b'].mean()
            else:
                raise ValueError(f"Unknown optimize metric: {optimize}")

            # 🆕 2. Apply exponential penalty for low signal density
            signal_density = len(signals) / len(df)
            if signal_density < required_signal_density:
                gap = required_signal_density - signal_density
                # Exponential decay penalty.
                # Scale factor 100 means: 1% below threshold → ~37% penalty, 4% below → ~1.8% penalty
                penalty = np.exp(-100.0 * gap)
                return obj * penalty

            return obj
        except Exception:
            return 0.0

    if verbose:
        print(f"\n⚙️ Starting Optuna search ({n_trials} trials, {timeout}s timeout)")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose, gc_after_trial=True, timeout=timeout)

    if verbose:
        print(f"\n✅ Best Trial Completed:")
        print(f"   🎯 Objective Value: {study.best_trial.value * 100:.2f}%")
        print(f"   ⚙️ Parameters     : {study.best_trial.params}")

    # =====================================================================
    # 🏆 Print Top Results
    # =====================================================================
    n_top_results = 20
    if verbose:
        print(f"\n🏆 Top {n_top_results} Trials (by Win Rate):")
        valid_trials = [t for t in study.trials if t.value is not None and t.value > 0]
        valid_trials.sort(key=lambda t: t.value, reverse=True)
        top_k = min(n_top_results, len(valid_trials))

        print(f"{'Rank':<5} | {'Win Rate':<10} | {'Parameters'}")
        print("-" * 105)
        for rank, trial in enumerate(valid_trials[:top_k], 1):
            print(f"{rank:<5} | {trial.value * 100:>7.2f}%  | {trial.params}")
        print()

    score_of_best_trial = study.best_trial.value
    best_params = study.best_trial.params
    strat_best = AutoTuneStrategy(**best_params, win_threshold=win_threshold,
                                  capital=capital, margin_cost=margin_cost, commission=commission)
    results_best = strat_best.generate_signals(closes)

    if verbose:
        print("\n📊 Final Performance (Best Params):")
        final_metrics = strat_best.evaluate(results_best, min_bars=min_bars, signal_type=signal_type)

        if 'status' in final_metrics:
            print(f"   {final_metrics['status']}")
        else:
            for k, v in final_metrics.items():
                print(f"   {k.replace('_', ' ').title()}: {v}")

            total_signals = final_metrics.get('total_signals', 0)
            if total_signals > 0 and total_signals < 50:
                print(f"\n   ⚠️  STATISTICAL WARNING: Only {total_signals} signals generated in the evaluation period.")
                print(f"      Win rates, expectancy, and Sharpe approximations may lack statistical significance.")
    # 💾 SAVE MODEL WITH PARAMETERIZED NAME
    os.makedirs(output_dir, exist_ok=True)
    w = best_params['window']
    bw = best_params['bandwidth']
    th = best_params['threshold']
    la = best_params['lookahead_bars']
    model_name = f"autotune_model_w{w}_bw{bw:.3f}_th{th:.3f}_la{la}___{score_of_best_trial}.pkl"
    model_path = os.path.join(output_dir, model_name)

    model_data = {
        'params': best_params,
        'win_threshold': win_threshold,
        'signal_type': signal_type,
        'optimize_metric': optimize,
        'score': score_of_best_trial,
        'dataset_id': dataset_id,
        'ticker': ticker,
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    if verbose:
        print(f"\n💾 Model saved to: {model_path}")


# =============================================================================
# 5. MAIN
# =============================================================================
if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)