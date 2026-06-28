try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

import argparse
import glob
import json
import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import TimeSeriesSplit
from utils import get_filename_for_dataset, get_next_step

# Suppress Optuna & pandas_ta debug logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None


def prepare_plot_dataframe(df: pd.DataFrame, ticker: str, signals: list, price_col_name: str = 'Close') -> pd.DataFrame:
    """Prepares a DataFrame with the exact columns expected by plot_forecast_results."""
    df_plot = df.copy()
    orig_close = (price_col_name, ticker)

    # Flatten close price for the plotter
    df_plot[price_col_name] = df[orig_close]

    # Technical Indicators
    df_plot['RSI'] = ta.rsi(df[orig_close], length=14)
    macd_out = ta.macd(df[orig_close], fast=12, slow=26, signal=9)
    df_plot['MACD'] = macd_out[f'MACD_12_26_9']
    df_plot['MACD_Signal'] = macd_out[f'MACDs_12_26_9']
    df_plot['Histogram'] = macd_out[f'MACDh_12_26_9']

    # One-Euro Filter (EMA(20) proxy)
    df_plot['OneEuro'] = ta.ema(df[orig_close], length=20)

    # Map signals: 1 = Long, -1 = Short, 0 = Neutral
    df_plot['Signal'] = 0
    for sig in signals:
        df_plot.loc[sig['Index'], 'Signal'] = 1 if sig['Type'] == 'BUY' else -1

    return df_plot


def plot_forecast_results(df: pd.DataFrame, price_col, sample: int = 200, start_idx: int = -1,
                          highlight_signals: bool = True, zoom_region: Optional[Tuple[int, int]] = None):
    if start_idx == -1:
        start_idx = max(0, len(df) - sample)
    plot_df = df.iloc[start_idx:start_idx + sample].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(plot_df.index, plot_df[price_col], label='Close', alpha=0.7, linewidth=1, color='black')
    ax1.plot(plot_df.index, plot_df['OneEuro'], label='EMA-20', color='blue', linewidth=2)
    longs = plot_df[plot_df['Signal'] == 1]
    shorts = plot_df[plot_df['Signal'] == -1]
    ax1.scatter(longs.index, longs[price_col], marker='^', color='green', s=100, label='Long Signal', zorder=6, edgecolors='darkgreen', linewidth=1.5)
    ax1.scatter(shorts.index, shorts[price_col], marker='v', color='red', s=100, label='Short Signal', zorder=6, edgecolors='darkred', linewidth=1.5)
    if highlight_signals:
        for idx in longs.index: ax1.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax1.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax1.set_title('Price + EMA-20 + Trading Signals', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
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
    ax2.set_ylabel('RSI', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    colors = ['green' if val >= 0 else 'red' for val in plot_df['Histogram']]
    ax3.bar(plot_df.index, plot_df['Histogram'], color=colors, alpha=0.6, label='Histogram', width=1)
    ax3.plot(plot_df.index, plot_df['MACD'], label='MACD', color='blue', linewidth=1.2)
    ax3.plot(plot_df.index, plot_df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1.2)
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.4, linewidth=0.8)
    if highlight_signals:
        for idx in longs.index: ax3.axvline(x=idx, color='green', linestyle=':', alpha=0.4, linewidth=0.8)
        for idx in shorts.index: ax3.axvline(x=idx, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
    ax3.set_ylabel('MACD', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)

    if zoom_region is not None:
        zoom_start, zoom_end = zoom_region
        if zoom_start < len(plot_df) and zoom_end <= len(plot_df) and zoom_start < zoom_end:
            zoom_df = plot_df.iloc[zoom_start:zoom_end]
            ax1_inset = ax1.inset_axes([0.62, 0.55, 0.35, 0.35])
            ax1_inset.plot(zoom_df.index, zoom_df[price_col], color='black', linewidth=1.5)
            ax1_inset.plot(zoom_df.index, zoom_df['OneEuro'], color='blue', linewidth=2)
            ax1_inset.scatter(zoom_df[zoom_df['Signal'] == 1].index, zoom_df[zoom_df['Signal'] == 1][price_col], marker='^', color='green', s=50, zorder=5)
            ax1_inset.scatter(zoom_df[zoom_df['Signal'] == -1].index, zoom_df[zoom_df['Signal'] == -1][price_col], marker='v', color='red', s=50, zorder=5)
            ax1_inset.set_xticks([])
            ax1_inset.set_yticks([])
            ax1_inset.set_title('Zoom', fontsize=8, fontweight='bold')
            ax1_inset.grid(True, alpha=0.3)
            ax1.indicate_inset_zoom(ax1_inset, edgecolor="gold", alpha=0.7)

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    for ax in [ax1, ax2]: ax.tick_params(labelbottom=False)
    plt.suptitle('🔗 Linked Technical Analysis Dashboard (Zoom: sharex enabled)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.text(0.5, 0.01, "💡 Tip: Use mouse wheel to zoom, drag to pan — all panels stay synchronized!", ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.show()


def setup_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Double-Green / Double-Red Candle Stick Strategy Optimizer & Real-Time Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    data_group = parser.add_argument_group('Data & Symbol')
    data_group.add_argument('--dataset-id', type=str, default='day', help='Dataset identifier')
    data_group.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    data_group.add_argument('--length-dataset', type=int, default=999999, help='Trailing data points')

    strat_group = parser.add_argument_group('Strategy & P&L Parameters')
    strat_group.add_argument('--lookahead-bars', type=int, default=20, dest='lookahead_bars', help='Forward-looking window')
    strat_group.add_argument('--method', type=str, default='final_close', choices=['touched', 'final_close'], help='Strike evaluation method')
    strat_group.add_argument('--min-signal-density', type=float, default=0.04, help='Min signal frequency threshold')
    strat_group.add_argument('--put-strike-pct', type=float, default=0.96, help='Base put strike multiplier')
    strat_group.add_argument('--call-strike-pct', type=float, default=1.04, help='Base call strike multiplier')
    strat_group.add_argument('--wr-weight', type=float, default=0.9, help='Weight for Win-Rate')
    strat_group.add_argument('--td-weight', type=float, default=0.1, help='Weight for Trade-Density')
    strat_group.add_argument('--signal-type', type=str, default='buy', choices=['both', 'buy', 'sell'], help='Filter signals for optimization. Post-hoc breakdown always evaluates both.')

    opt_group = parser.add_argument_group('Optimization & Execution')
    opt_group.add_argument('--n-trials', type=int, default=100, help='Optuna trials')
    opt_group.add_argument('--timeout', type=int, default=3600, help='Max runtime (seconds)')
    opt_group.add_argument('--output-dir', type=str, default='models', help='Output directory')
    opt_group.add_argument('--train-ratio', type=float, default=0.7, help='Fraction of data used for training/optimization (rest for validation)')
    opt_group.add_argument('--n-splits', type=int, default=20, help='Number of splits for TimeSeriesSplit cross-validation')

    # 🆕 OPTUNA PERSISTENCE ARGUMENTS
    opt_group.add_argument('--optuna-storage', type=str, default=None,
                           help='Optuna storage URL (e.g., sqlite:///optuna.db, mysql://...). Defaults to in-memory.')
    opt_group.add_argument('--optuna-study-name', type=str, default=None,
                           help='Study name for persistence. Required if --optuna-storage is set.')

    flag_group = parser.add_argument_group('Execution Flags')
    flag_group.add_argument('--real-time', action=argparse.BooleanOptionalAction, default=False, help='Real-time mode')
    flag_group.add_argument('--model-path', type=str, default=None, help='Specific .pkl model path')
    flag_group.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')
    flag_group.add_argument('--verbose-study-progress-bar', action=argparse.BooleanOptionalAction, default=False, help='Verbose output')
    flag_group.add_argument('--verbose-short', action=argparse.BooleanOptionalAction, default=False, help='Short real-time output')
    flag_group.add_argument('--seed', type=int, default=123, help='Random seed')
    flag_group.add_argument('--plot', action='store_true', default=False, help='Plot results')

    return parser


def dgdr_strategy_vectorized(df, close_col, volume_col, open_col, high_col, low_col, ticker,
                             st_multipler=2, st_length=7, sup_wick_null_coef=0.1, inf_wick_null_coef=0.1, rsi_length=2,
                             buy_rsi_threshold=80, sell_rsi_threshold=20):
    """Vectorized implementation of the Sniper strategy."""
    rsi_col = ('RSI', ticker)
    df[rsi_col] = ta.rsi(df[close_col], length=rsi_length)

    vwap_col = ('VWAP', ticker)
    df[vwap_col] = ta.vwap(df[high_col], df[low_col], df[close_col], df[volume_col])

    st = ta.supertrend(df[high_col], df[low_col], df[close_col], multiplier=st_multipler, length=st_length)
    st_direction_col = ('ST_Direction', ticker)
    df[st_direction_col] = st.iloc[:, 1]

    C = df[close_col]
    O = df[open_col]
    H = df[high_col]
    L = df[low_col]

    C_prev = C.shift(1)
    O_prev = O.shift(1)
    H_prev = H.shift(1)
    L_prev = L.shift(1)

    body1 = (C_prev - O_prev).abs()
    body2 = (C - O).abs()
    c2_bigger = body2 > body1

    is_double_green = (
            (C > O) & (C_prev > O_prev) & c2_bigger &
            (H > H_prev) & (L > L_prev) & (C > H_prev) &
            ((H - C) <= body2 * sup_wick_null_coef)
    ).fillna(False)

    is_double_red = (
            (C < O) & (C_prev < O_prev) & c2_bigger &
            (H < H_prev) & (L < L_prev) & (C < L_prev) &
            ((C - L) <= body2 * inf_wick_null_coef)
    ).fillna(False)

    buy_mask = (
            (C > df[vwap_col]) & (df[st_direction_col] == 1) &
            is_double_green & (df[rsi_col] > buy_rsi_threshold)
    ).fillna(False)

    sell_mask = (
            (C < df[vwap_col]) & (df[st_direction_col] == -1) &
            is_double_red & (df[rsi_col] < sell_rsi_threshold)
    ).fillna(False)

    signals = []
    buy_idx = df.index[buy_mask]
    if len(buy_idx) > 0:
        prices = df.loc[buy_idx, close_col]
        sls = df.loc[buy_idx, low_col]
        tps = prices + (prices - sls) * 2
        signals.extend([{'Type': 'BUY', 'Index': idx, 'Price': p, 'SL': s, 'TP': t} for idx, p, s, t in zip(buy_idx, prices, sls, tps)])

    sell_idx = df.index[sell_mask]
    if len(sell_idx) > 0:
        prices = df.loc[sell_idx, close_col]
        sls = df.loc[sell_idx, high_col]
        tps = prices - (sls - prices) * 2
        signals.extend([{'Type': 'SELL', 'Index': idx, 'Price': p, 'SL': s, 'TP': t} for idx, p, s, t in zip(sell_idx, prices, sls, tps)])

    signals.sort(key=lambda x: x['Index'])
    return signals


def calculate_pnl_report(signals, df, close_col, high_col, low_col,
                         B, method, put__strike_pct, call__strike_pct, silent=False):
    """Evaluates credit spread signals and generates a P&L report."""
    results = []
    assert method in ["touched", "final_close"]
    assert 0 < put__strike_pct <= 1 <= call__strike_pct < 2

    for sig in signals:
        idx = sig['Index']
        price = sig['Price']
        sig_type = sig['Type']

        future_df = df.loc[df.index > idx].iloc[:B]
        if len(future_df) < B:
            continue

        success = False
        strike = None
        if sig_type == 'BUY':
            strike = price * put__strike_pct
            if method == "final_close":
                success = future_df[close_col].iloc[-1] > strike
            else:  # "touched" method
                # We win if the price NEVER drops below the strike
                success = not (future_df[low_col] < strike).any()

        elif sig_type == 'SELL':
            strike = price * call__strike_pct
            if method == "final_close":
                success = future_df[close_col].iloc[-1] < strike
            else:  # "touched" method
                # We win if the price NEVER goes above the strike
                success = not (future_df[high_col] > strike).any()

        results.append({'Signal_Index': idx, 'Type': sig_type, 'Entry_Price': price, 'Strike_Price': strike, 'Method': method, 'Success': success, 'PnL': 0.})

    pnl_df = pd.DataFrame(results)
    if pnl_df.empty:
        if not silent: print("⚠️  No valid signals to evaluate (insufficient lookahead data).")
        return pnl_df

    pnl_df['Cumulative_PnL'] = pnl_df['PnL'].cumsum()
    total_trades = len(pnl_df)
    wins = pnl_df['Success'].sum()
    win_rate = (wins / total_trades) * 100
    trade_density = total_trades / len(df)

    pnl_df['trade_density'] = trade_density
    pnl_df['dataset_length'] = len(df)
    pnl_df['win_rate'] = win_rate

    if not silent:
        print("\n" + "=" * 42)
        print(" 📈 CREDIT SPREAD REPORT")
        print("=" * 42)
        print(f" Dataset Length        : {len(df):,}")
        print(f" Trade Density         : {trade_density:.2%}")
        print(f" Method Used           : {method.upper()}")
        print(f" Lookahead Bars (B)    : {B}")
        print(f" Total Trades          : {total_trades}")
        print(f" Winning / Losing      : {wins} / {total_trades - wins}")
        print(f" Win Rate              : {win_rate:.2f}%")
        print("=" * 42 + "\n")

    return pnl_df


def compute_optimization_score(win_rate, trade_density, min_trade_density, wr_weight, td_weight):
    """Returns a score strictly between 0 and 1."""
    wr_norm = win_rate / 100.0
    td_norm = min(1.0, trade_density / min_trade_density)
    final_score = (wr_weight * wr_norm) + (td_weight * td_norm)
    return max(0.0, min(1.0, final_score))


def objective(trial, df, close_col, volume_col, open_col, high_col, low_col, ticker,
              B, method, min_trade_density, wr_weight, td_weight, put_base, call_base, signal_type, n_splits):
    put__strike_pct = trial.suggest_float("put__strike_pct", put_base, put_base)
    call__strike_pct = trial.suggest_float("call__strike_pct", call_base, call_base)
    st_multipler = trial.suggest_int("st_multipler", 1, 5, step=1)
    st_length = trial.suggest_int("st_length", 3, 20, step=1)
    rsi_length = trial.suggest_int("rsi_length", 2, 21, step=1)
    sup_wick_null_coef = trial.suggest_float("sup_wick_null_coef", 0.0, 0.95, step=0.01)
    inf_wick_null_coef = trial.suggest_float("inf_wick_null_coef", 0.0, 0.95, step=0.01)
    buy_rsi_threshold = trial.suggest_int("buy_rsi_threshold", 50, 95, step=1)
    sell_rsi_threshold = trial.suggest_int("sell_rsi_threshold", 5, 50, step=1)

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(df):
        if len(test_idx) == 0:
            continue

        end_pos = test_idx[-1]
        # Use data up to the end of the test fold to preserve indicator warmup (lookback)
        df_warmup = df.iloc[:end_pos + 1]

        signals = dgdr_strategy_vectorized(
            df=df_warmup.copy(), close_col=close_col, volume_col=volume_col,
            open_col=open_col, high_col=high_col, low_col=low_col, ticker=ticker,
            st_multipler=st_multipler, st_length=st_length,
            sup_wick_null_coef=sup_wick_null_coef, inf_wick_null_coef=inf_wick_null_coef,
            buy_rsi_threshold=buy_rsi_threshold, sell_rsi_threshold=sell_rsi_threshold,
            rsi_length=rsi_length
        )

        # Filter signals to only those generated strictly within the current test fold
        test_indices_set = set(df.index[test_idx])
        signals = [s for s in signals if s['Index'] in test_indices_set]

        if signal_type == 'buy':
            signals = [s for s in signals if s['Type'] == 'BUY']
        elif signal_type == 'sell':
            signals = [s for s in signals if s['Type'] == 'SELL']

        # Pass full df to calculate_pnl_report so it can look ahead B bars for PnL calculation
        pnl_df = calculate_pnl_report(
            signals=signals, df=df.copy(), close_col=close_col, high_col=high_col, low_col=low_col,
            B=B, method=method, put__strike_pct=put__strike_pct, call__strike_pct=call__strike_pct, silent=True
        )

        if pnl_df.empty:
            scores.append(0.0)
        else:
            # Recalculate trade density specifically for this fold's length to maintain accurate scaling
            fold_trade_density = len(pnl_df) / len(test_idx)
            score = compute_optimization_score(
                win_rate=pnl_df['win_rate'].iloc[0],
                trade_density=fold_trade_density,
                min_trade_density=min_trade_density,
                wr_weight=wr_weight,
                td_weight=td_weight
            )
            scores.append(score)

    return np.mean(scores) if scores else 0.0


def save_optimized_model(study, config, output_dir, ticker, dataset_id, train_metrics, val_metrics):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    p_tag = config.get('B', 'NA')
    m_tag = config.get('method', 'NA')
    md_tag = config.get('min_signal_density', 'NA')
    wr_tag = config.get('wr_weight', 'NA')
    td_tag = config.get('td_weight', 'NA')
    st_tag = config.get('signal_type', 'NA')
    train_win_rate = train_metrics.get("win_rate")
    val_win_rate = val_metrics.get("win_rate")
    best_score = getattr(study.best_trial, 'value', None)
    score_tag = f"score{best_score:.8f}".replace('.', 'p') if best_score is not None else "scoreNA"
    params_str = f"B{p_tag}_{m_tag}_md{md_tag}_wr{wr_tag}_td{td_tag}_{st_tag}_{score_tag}_{train_win_rate:.4f}_{val_win_rate:.4f}"

    safe_ticker = ticker.replace('^', '')
    safe_dataset = dataset_id.replace('/', '_').replace('\\', '_')
    base_name = f"{safe_ticker}_{safe_dataset}_{params_str}__{timestamp}"

    pkl_path = os.path.join(output_dir, f"{base_name}.pkl")

    meta = {'ticker': ticker, 'dataset_id': dataset_id, 'best_params': study.best_trial.params,
            'best_value': study.best_trial.value, 'n_trials': len(study.trials), 'timestamp': timestamp, 'filename_tag': params_str}
    with open(pkl_path, 'wb') as f:
        pickle.dump({'study_best_trial': study.best_trial, 'config': config, 'timestamp': timestamp, 'meta': meta, 'train_metrics': train_metrics, 'val_metrics': val_metrics}, f)

    print(f"✅ Model saved to: {pkl_path}")

    return pkl_path


def run_real_time_mode(args, df, config_cols):
    model_path = args.model_path
    if not model_path:
        search_pattern = os.path.join(args.output_dir, f"*{args.ticker.replace('^', '')}*{args.dataset_id}*.pkl")
        matches = glob.glob(search_pattern)
        if not matches:
            print("❌ No saved models found. Run optimization first or provide --model-path.")
            return
        model_path = max(matches, key=os.path.getmtime)
    if args.verbose:
        print(f"📦 Loading real-time model: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    assert model_data['config']['dataset_id'] == args.dataset_id
    assert model_data['config']['ticker'] == args.ticker
    best_params = model_data['study_best_trial'].params
    config = model_data['config']
    assert 'signal_type' in config
    signal_type = config.get('signal_type', 'both')
    print(f"📡 Real-time signal filter: {signal_type.upper()} (loaded from model config)")
    ticker = model_data['config']['ticker']
    assert ticker == args.ticker
    dataset_id = model_data['config']['dataset_id']
    assert dataset_id == args.dataset_id
    lookahead = model_data['config']['B']
    method = model_data['config']['method']
    min_signal_density = model_data['config']['min_signal_density']
    train_win_rate = model_data['train_metrics']['win_rate']
    train_score = model_data['train_metrics']['score']
    train_trade_density = model_data['train_metrics']['trade_density']
    val_win_rate = model_data['val_metrics']['win_rate']
    val_score = model_data['val_metrics']['score']
    val_trade_density = model_data['val_metrics']['trade_density']
    lookback_needed = 100
    df_tail = df.tail(lookback_needed).copy()
    close_col, volume_col, open_col, high_col, low_col, ticker = config_cols

    signals = dgdr_strategy_vectorized(df_tail, close_col, volume_col, open_col, high_col, low_col, ticker,
                                       **{k: best_params[k] for k in ['st_multipler', 'st_length', 'sup_wick_null_coef', 'inf_wick_null_coef', 'buy_rsi_threshold', 'sell_rsi_threshold']})

    latest_idx = df_tail.index[-1]
    prev_idx = df_tail.index[-2]
    latest_signals = [s for s in signals if s['Index'] in (latest_idx, prev_idx)]
    current_price, entry_date = df_tail.iloc[-1][close_col], df_tail.index[-1].strftime('%Y-%m-%d')
    target_date = get_next_step(the_date=entry_date, dataset_id=dataset_id, nn=lookahead).strftime('%Y-%m-%d')
    target_price = "N/A"
    if signal_type == 'buy':
        latest_signals = [s for s in latest_signals if s['Type'] == 'BUY']
    elif signal_type == 'sell':
        latest_signals = [s for s in latest_signals if s['Type'] == 'SELL']

    if args.verbose_short:
        if latest_signals:
            sig = latest_signals[-1]
            print(f"⚡ REAL-TIME: [{sig['Type']}] @ {sig['Price']:.2f} | SL: {sig['SL']:.2f} | TP: {sig['TP']:.2f}")
        else:
            print("⚪ REAL-TIME: No new signal on latest closed bar.")

    print("\n" + "─" * 40)
    print(" 🕒 REAL-TIME SIGNAL CHECK")
    print("─" * 40)
    print(f" Dataset Id: {dataset_id} | Lookahead: {lookahead} bars | Method: {method} | Minimum Signal Density: {min_signal_density:.2%} | Signal Type: {signal_type}")
    print(f" Train score : {train_score:.2%} | Train Win Rate: {train_win_rate:.2f}% | Train Density: {train_trade_density:.2%} | {config['train_range']}")
    print(f" Val score   : {val_score:.2%} | Val Win Rate  : {val_win_rate:.2f}% | Val Density  : {val_trade_density:.2%} | {config['val_range']}")
    print(f" Put Strike% : {model_data['meta']['best_params']['put__strike_pct']:.2%}")
    print(f" Call Strike%: {model_data['meta']['best_params']['call__strike_pct']:.2%}")
    print(f" Latest Bar Index : {latest_idx.strftime('%Y-%m-%d')} @ ${df_tail[close_col].iloc[-1]:.2f}")
    print(f" Previous Bar     : {prev_idx.strftime('%Y-%m-%d')}")
    buy_signal_detected, sell_signal_detected = False, False
    if latest_signals:
        sig = latest_signals[-1]
        print(f"TODO --> {sig}")
        print(f" 🟢 SIGNAL DETECTED: {sig['Type']}")
        print(f"    Entry Price : ${sig['Price']:.2f}")
        buy_signal_detected, sell_signal_detected = True, True
    else:
        print(" ⚪ NO SIGNAL on latest closed bar.")
    print("─" * 40 + "\n")
    result = {'train_score': train_score, 'train_trade_density': train_trade_density, 'val_score': val_score, 'val_trade_density': val_trade_density,
              'train_win_rate': train_win_rate / 100., 'val_win_rate': val_win_rate / 100.,
              'optimize_target': signal_type, 'current_price': current_price, 'current_date': entry_date, 'target_price': target_price, 'target_date': target_date,
              'dataset_id': dataset_id, 'ticker': args.ticker, 'lookahead': lookahead, 'method': method,
              'buy_signal_detected': buy_signal_detected, 'sell_signal_detected': sell_signal_detected,
              'put_strike_pct': model_data['meta']['best_params']['put__strike_pct'], 'call_strike_pct': model_data['meta']['best_params']['call__strike_pct']}
    return result


def perfect_score_callback(study, trial):
    if study.best_value is not None and study.best_value >= 0.9999:
        print("\n🎯 Perfect score reached (≥ 0.9999). Stopping optimization early.")
        study.stop()


def entry(args):
    print("\n" + "═" * 62)
    print(" 🎯 DGDR ALGORITHM INITIALIZED")
    print("    Double Green / Double Red Momentum")
    print("═" * 62)
    print(" 📖 CORE CONCEPT:")
    print("    A price-action momentum system that detects accelerating")
    print("    2-bar continuations. Filters market noise using dynamic")
    print("    trend alignment, VWAP positioning, and RSI(2) oscillators.")
    print("")
    print(" 🔍 SIGNAL LOGIC:")
    print("    🟢 DOUBLE GREEN  → Body expansion, structure breakout,")
    print("       minimal upper-wick rejection. Confirmed by: Price > VWAP")
    print("       & SuperTrend ↑ & RSI > Buy Threshold")
    print("    🔴 DOUBLE RED    → Body expansion, structure breakdown,")
    print("       minimal lower-wick rejection. Confirmed by: Price < VWAP")
    print("       & SuperTrend ↓ & RSI < Sell Threshold")
    print("")
    print(" ⚙️ OPTIMIZATION & EXECUTION:")
    print("    • Optuna auto-tunes SuperTrend, RSI & Wick thresholds")
    print("    • Composite scoring: Weighted Win-Rate + Trade-Density")
    print("    • Backtests credit-spread outcomes over lookahead window (B)")
    print("    • Supports 'touched' (price touch) or 'final_close' (close) strikes")
    print("═" * 62 + "\n")

    ticker = args.ticker
    dataset_id = args.dataset_id
    np.random.seed(args.seed)

    close_col = ('Close', ticker)
    volume_col = ('Volume', ticker)
    open_col = ('Open', ticker)
    high_col = ('High', ticker)
    low_col = ('Low', ticker)
    config_cols = (close_col, volume_col, open_col, high_col, low_col, ticker)

    cache_filename = get_filename_for_dataset(dataset_id, older_dataset=None)
    if args.verbose:
        print(f"📂 Loading dataset from: {cache_filename}")
    with open(cache_filename, 'rb') as f:
        master_data_cache = pickle.load(f)

    if ticker not in master_data_cache:
        raise KeyError(f"Ticker '{ticker}' not found in cache. Available: {list(master_data_cache.keys())}")

    df = master_data_cache[ticker].sort_index()
    if args.length_dataset and args.length_dataset < len(df):
        df = df.tail(args.length_dataset)

    first_date = df.index[0]
    last_date = df.index[-1]
    num_bars = len(df)
    print(f"\n📊 Dataset Loaded: {ticker} ({dataset_id})")
    print(f"   Bars: {num_bars:,} | Range: {first_date.strftime('%Y-%m-%d')}  ->  {last_date.strftime('%Y-%m-%d')}\n")

    if args.real_time:
        return run_real_time_mode(args, df, config_cols)

    # ✅ TRAIN / VALIDATION CHRONOLOGICAL SPLIT
    split_idx = int(len(df) * args.train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    print(f"📐 Data Split -> Train: {len(df_train):,} ({args.train_ratio:.0%}) | Test: {len(df_test):,} ({1 - args.train_ratio:.0%})")
    print(f"📐 Train from {df_train.index[0].strftime('%Y-%m-%d')} to {df_train.index[-1].strftime('%Y-%m-%d')} | Test from {df_test.index[0].strftime('%Y-%m-%d')} to {df_test.index[-1].strftime('%Y-%m-%d')}")
    if len(df_test) < 50:
        print(f"⚠️  Validation set is small ({len(df_test)} bars). Out-of-sample metrics may be noisy.\n")

    B = args.lookahead_bars
    method = args.method
    min_density = args.min_signal_density
    put_base, call_base = args.put_strike_pct, args.call_strike_pct
    wr_w, td_w = args.wr_weight, args.td_weight

    # ✅ OPTIMIZE ON TRAINING SET ONLY
    print(f"🔍 Starting Optuna optimization on TRAINING SET ({len(df_train):,} bars)...")
    print(f"📉 Min Trade Density: {min_density:.2%} | Look Ahead: {B} | Method: {method.upper()}")
    print(f"📡 Signal Filter (Optimization): {args.signal_type.upper()}")
    print(f"⚖️ Score Weights -> Win Rate: {wr_w}  Trade Density: {td_w} | Strike Range: [{put_base:.4f}, {call_base:.4f}]")
    print(f"🔄 Time Series Cross-Validation: {args.n_splits} splits\n")

    # 🆕 OPTUNA PERSISTENCE SETUP
    storage = args.optuna_storage
    study_name = args.optuna_study_name

    if storage:
        if not study_name:
            raise ValueError("❌ --optuna-study-name is required when --optuna-storage is specified.")
        print(f"💾 Optuna persistence enabled: storage='{storage}', study='{study_name}'")
        os.makedirs(args.output_dir, exist_ok=True)
        study = optuna.create_study(
            direction="maximize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(direction="maximize")

    # 🆕 LIST PREVIOUS BEST PARAMETERS IF STUDY ALREADY EXISTS
    if len(study.trials) > 0:
        print(f"\n📋 Resuming existing study with {len(study.trials)} completed trial(s).")
        print("🏆 Previous Best Parameters:")
        for k, v in study.best_trial.params.items():
            print(f"   {k:<25}: {v}")
        print(f"   {'Previous Best Score':<25}: {study.best_trial.value:.4f}\n")
    else:
        print(f"🆕 Created new {'ín-memory' if not storage else ''} study.\n")

    study.optimize(
        lambda trial: objective(trial, df_train.copy(), close_col, volume_col, open_col, high_col, low_col, ticker,
                                B, method, min_density, wr_w, td_w, put_base, call_base,
                                signal_type=args.signal_type, n_splits=args.n_splits),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=args.verbose_study_progress_bar,
        callbacks=[perfect_score_callback]
    )

    print("\n🏆 OPTIMIZATION COMPLETE")
    print("📊 Best Parameters:")
    for k, v in study.best_trial.params.items():
        print(f"   {k:<25}: {v}")
    print(f"   {'Objective Score':<25}: {study.best_trial.value:.4f} (max=1.0)\n")

    # ✅ FINAL TEST BACKTEST
    print("📉 Running final test backtest on TEST SET...")
    best = study.best_trial.params
    signals_val = dgdr_strategy_vectorized(df_test.copy(), close_col, volume_col, open_col, high_col, low_col, ticker,
                                           st_multipler=best['st_multipler'], st_length=best['st_length'],
                                           sup_wick_null_coef=best['sup_wick_null_coef'], inf_wick_null_coef=best['inf_wick_null_coef'],
                                           buy_rsi_threshold=best['buy_rsi_threshold'], sell_rsi_threshold=best['sell_rsi_threshold'])

    main_signals = signals_val.copy()
    if args.signal_type == 'buy':
        main_signals = [s for s in signals_val if s['Type'] == 'BUY']
    elif args.signal_type == 'sell':
        main_signals = [s for s in signals_val if s['Type'] == 'SELL']

    pnl_df = calculate_pnl_report(main_signals, df_test.copy(), close_col, high_col, low_col,
                                  B, method, best['put__strike_pct'], best['call__strike_pct'], silent=False)

    print("📊 DIRECTIONAL PERFORMANCE BREAKDOWN (Post-Hoc Test)")
    print("─" * 65)
    for dir_type in ['BUY', 'SELL']:
        dir_signals = [s for s in signals_val if s['Type'] == dir_type]
        if dir_signals:
            dir_pnl = calculate_pnl_report(dir_signals, df_test, close_col, high_col, low_col,
                                           B, method, best['put__strike_pct'], best['call__strike_pct'], silent=True)
            if not dir_pnl.empty:
                wr = dir_pnl['win_rate'].iloc[0]
                td = dir_pnl['trade_density'].iloc[0]
                desc = "PUT CREDIT SPREAD (bearish)" if dir_type == 'BUY' else "CALL CREDIT SPREAD (bullish)"
                trades = len(dir_pnl)
                print(f"  {dir_type:<6} | Trades: {trades:>4} | Win Rate: {wr:>5.2f}% | Density: {td:.4f} | {desc}")
            else:
                print(f"  {dir_type:<6} | {len(dir_signals):>4} signals generated (0 valid for PnL lookahead)")
        else:
            print(f"  {dir_type:<6} |    0 signals generated")
    print("─" * 65 + "\n")

    if args.plot and not pnl_df.empty:
        plot_signals = main_signals
        if plot_signals:
            df_plot = prepare_plot_dataframe(df_test, ticker, plot_signals, price_col_name="Close")
            plot_forecast_results(df_plot, price_col='Close', sample=2000, highlight_signals=True)
        else:
            print("⚠️ No signals of the selected type found to plot.")

    # ========================================================================
    # 📊 TRAIN vs VALIDATION COMPARISON - GENERALIZATION CHECK
    # ========================================================================
    print("\n" + "═" * 70)
    print(" 🔄 TRAIN vs VALIDATION PERFORMANCE COMPARISON")
    print("═" * 70)

    # Re-evaluate training set with BEST parameters for fair comparison
    signals_train_best = dgdr_strategy_vectorized(
        df_train.copy(), close_col, volume_col, open_col, high_col, low_col, ticker,
        st_multipler=best['st_multipler'], st_length=best['st_length'],
        sup_wick_null_coef=best['sup_wick_null_coef'],
        inf_wick_null_coef=best['inf_wick_null_coef'],
        buy_rsi_threshold=best['buy_rsi_threshold'],
        sell_rsi_threshold=best['sell_rsi_threshold']
    )

    # Filter signals if needed for fair comparison
    if args.signal_type == 'buy':
        signals_train_best = [s for s in signals_train_best if s['Type'] == 'BUY']
    elif args.signal_type == 'sell':
        signals_train_best = [s for s in signals_train_best if s['Type'] == 'SELL']

    pnl_train = calculate_pnl_report(
        signals_train_best, df_train.copy(), close_col, high_col, low_col,
        B, method, best['put__strike_pct'], best['call__strike_pct'], silent=True
    )

    # Extract comparable metrics
    def extract_metrics(pnl_df):
        if pnl_df.empty:
            return {'trades': 0, 'win_rate': 0.0, 'trade_density': 0.0, 'score': 0.0}
        wr = pnl_df['win_rate'].iloc[0]
        td = pnl_df['trade_density'].iloc[0]
        score = compute_optimization_score(wr, td, min_density, wr_w, td_w)
        return {
            'trades': len(pnl_df),
            'win_rate': wr,
            'trade_density': td,
            'score': score
        }

    train_metrics = extract_metrics(pnl_train)
    test_metrics = extract_metrics(pnl_df)

    # Display comparison table
    print(f"\n{'Metric':<20} {'Train':>12} {'Test':>12} {'Δ (Test-Train)':>15}")
    print("-" * 70)
    print(f"{'Trades':<20} {train_metrics['trades']:>12,} {test_metrics['trades']:>12,} {test_metrics['trades'] - train_metrics['trades']:>15,}")
    print(f"{'Win Rate (%)':<20} {train_metrics['win_rate']:>12.2f} {test_metrics['win_rate']:>12.2f} {test_metrics['win_rate'] - train_metrics['win_rate']:>15.2f}")
    print(f"{'Trade Density':<20} {train_metrics['trade_density']:>12.4f} {test_metrics['trade_density']:>12.4f} {test_metrics['trade_density'] - train_metrics['trade_density']:>15.4f}")
    print(f"{'Optimization Score':<20} {train_metrics['score']:>12.4f} {test_metrics['score']:>12.4f} {test_metrics['score'] - train_metrics['score']:>15.4f}")

    # Generalization assessment
    print("\n" + "🔍 GENERALIZATION ASSESSMENT:")
    print("-" * 70)

    wr_diff = abs(test_metrics['win_rate'] - train_metrics['win_rate'])
    score_diff = abs(test_metrics['score'] - train_metrics['score'])

    if test_metrics['trades'] < 10:
        print("⚠️  WARNING: Validation set has very few trades (<10). Metrics may be noisy.")
    elif wr_diff <= 5 and score_diff <= 0.05:
        print("✅ EXCELLENT: Validation performance closely matches training. Strong generalization!")
    elif wr_diff <= 10 and score_diff <= 0.10:
        print("✅ GOOD: Minor performance drop on validation. Acceptable generalization.")
    elif wr_diff <= 15 and score_diff <= 0.15:
        print("⚠️  MODERATE: Noticeable performance gap. Monitor for overfitting.")
    else:
        print("❌ POOR: Large performance drop on validation. Likely overfitting detected.")

    # Additional insights
    if test_metrics['win_rate'] > train_metrics['win_rate']:
        print("💡 Bonus: Validation win rate EXCEEDS training – model may be robust!")
    elif test_metrics['trade_density'] < train_metrics['trade_density'] * 0.5:
        print("💡 Note: Much lower trade density in validation – market regime may differ.")

    print("═" * 70 + "\n")
    # ========================================================================
    # END COMPARISON SECTION
    # ========================================================================

    config = {'ticker': ticker, 'dataset_id': dataset_id, 'B': B, 'method': method, 'train_ratio': args.train_ratio,
              'train_range': f"({df_train.index[0].strftime('%Y-%m-%d')}::{df_train.index[-1].strftime('%Y-%m-%d')})",
              'val_range': f"({df_test.index[0].strftime('%Y-%m-%d')}::{df_test.index[-1].strftime('%Y-%m-%d')})",
              'min_signal_density': min_density, 'wr_weight': wr_w, 'td_weight': td_w, 'signal_type': args.signal_type}
    save_optimized_model(study, config, args.output_dir, ticker, dataset_id, train_metrics, test_metrics)


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    entry(args)