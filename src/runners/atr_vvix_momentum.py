"""
Market Prediction Model with VIX-Based Volatility Regimes
+ VVIX (VIX-of-VIX) Dynamic Shock Absorption
+ ATR_5/ATR_14 Volatility Momentum Expansion/Contraction

This script implements a market range prediction algorithm that forecasts the expected High and Low
prices for a given financial instrument (e.g., S&P 500) over a specific timeframe (day, week, month).
It dynamically adjusts its predictions based on current market volatility (VIX), Volatility of Volatility (VVIX),
and short-term Volatility Momentum (ATR_5/ATR_14), optimizing its parameters using Optuna.
"""

try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from utils import get_filename_for_dataset, get_next_step
import pickle
import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import optuna
import joblib
from fetchers.serialize_fyahoo import realtime

# Suppress Optuna & pandas debug logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _extract_scalar(val):
    """Safely extracts a scalar float from pandas row extractions (handles tuples, arrays, scalars)."""
    try:
        if hasattr(val, 'item'):
            return float(val.item())
        elif hasattr(val, 'values'):
            return float(np.ravel(val)[0])
        elif hasattr(val, '__len__') and not isinstance(val, str) and len(val) > 0:
            return float(val[0])
        return float(val)
    except:
        return np.nan


def fetch_vvix_yfinance(start_date, end_date):
    """
    Fetches ^VVIX (VIX of VIX) data from yfinance as a fallback.
    Fills the missing last row up to end_date using forward fill (ffill).
    Returns a DataFrame with a tuple column ('Close', '^VVIX').
    """
    try:
        import yfinance as yf
    except ImportError:
        print("⚠️  yfinance not installed. Install with: pip install yfinance")
        return None

    try:
        vvix_raw = yf.download(
            "^VVIX", start=start_date, end=end_date, progress=False, auto_adjust=True
        )
        if vvix_raw.empty:
            return None

        # Handle yfinance multi-level columns
        if isinstance(vvix_raw.columns, pd.MultiIndex):
            close_series = vvix_raw['Close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
        else:
            close_series = vvix_raw['Close']

        # 1. Align baseline index to datetime
        vvix_index = pd.to_datetime(close_series.index)

        # 2. Reindex up to end_date to automatically create the missing empty row(s)
        full_range = pd.date_range(start=vvix_index.min(), end=pd.to_datetime(end_date), freq='B')  # 'B' excludes weekends
        close_series = close_series.reindex(full_range)

        # 3. Forward fill the missing values (propagate yesterday's price to today)
        close_series = close_series.ffill()

        # 4. Format according to your script's conventions
        df_vvix = pd.DataFrame({
            ('Close', '^VVIX'): close_series
        })
        df_vvix.index.name = 'Date'

        # Remove any rows at the end that could still be NaN if the dataset was completely empty
        return df_vvix.dropna()

    except Exception as e:
        print(f"⚠️  Failed to fetch ^VVIX via yfinance: {e}")
        return None


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Market Prediction Model with VIX Optimization + VVIX Shock Absorption + ATR Momentum"
    )

    parser.add_argument("--dataset-id", type=str, default="day", choices=["day", "week", "month"])
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    parser.add_argument('--dataframe', type=pd.DataFrame, default=None, help='Dataset supplied')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')
    parser.add_argument("--n-trials", type=int, default=500, help="Number of Optuna trials (default: 500).")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Timeout for Optuna optimization in seconds (default: None).")
    parser.add_argument("--use-realtime-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atr-window", type=int, default=14, help="Window size for ATR (default: 14).")
    parser.add_argument("--n-split", type=float, default=0.80, help="Train/Test split ratio (default: 0.80).")
    parser.add_argument('--clip-n', type=int, default=0,
                        help="Do a .iloc[:-n] on the loaded dataset. A value of greater than 0 enable it.")
    parser.add_argument("--tightness-weight", type=float, default=0.3, help="Weight for tightness penalty.")
    parser.add_argument("--use-close-for-range", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--use-vvix", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable VVIX dynamic shock absorption (default: True).")

    return parser


def calculate_atr(df, close_col, high_col, low_col, ticker, window):
    """Calculate Average True Range (ATR) with Wilder's smoothing."""
    df = df.copy()
    prev_close_col = ('Prev_Close', ticker)
    df[prev_close_col] = df[close_col].shift(1)
    hl_col, hpc_col, lpc_col = ('H-L', ticker), ('H-PC', ticker), ('L-PC', ticker)
    df[hl_col] = df[high_col] - df[low_col]
    df[hpc_col] = (df[high_col] - df[prev_close_col]).abs()
    df[lpc_col] = (df[low_col] - df[prev_close_col]).abs()
    tr_col = ('TR', ticker)
    df[tr_col] = df[[hl_col, hpc_col, lpc_col]].max(axis=1)

    atr_col = (f'ATR_{window}', ticker)
    # Shift by 1 to ensure the ATR used for today's prediction is based on YESTERDAY's close
    df[atr_col] = df[tr_col].ewm(alpha=1 / window, adjust=False).mean().shift(1)
    df = df.iloc[window:]
    cols_to_check = [close_col, high_col, low_col, atr_col, prev_close_col]
    return df.dropna(subset=cols_to_check).copy(), atr_col, prev_close_col


def run_backtest_with_vix(df_bt, open_col, close_col, atr_col, high_col, low_col,
                          optimized_params, use_close_for_range=False,
                          use_vvix=True, vvix_rank_col=None, vix_rank_col=None,
                          momentum_ratio_col=None):
    """Exécute le backtest avec la dynamique VVIX et Momentum ATR."""

    # Use the exact tuple column key to avoid MultiIndex KeyError issues
    df_bt['VIX_Regime'] = 'Normal'
    df_bt.loc[df_bt[vix_rank_col] < 0.30, 'VIX_Regime'] = 'Low'
    df_bt.loc[df_bt[vix_rank_col] > 0.70, 'VIX_Regime'] = 'High'
    assert optimized_params is not None

    k_params = {
        'Low': {'k_up': optimized_params['k_up_low'], 'k_down': optimized_params['k_down_low']},
        'Normal': {'k_up': optimized_params['k_up_normal'], 'k_down': optimized_params['k_down_normal']},
        'High': {'k_up': optimized_params['k_up_high'], 'k_down': optimized_params['k_down_high']}
    }

    vvix_shock_mult = optimized_params.get('vvix_shock_mult', 0.0) if use_vvix else 0.0
    atr_momentum_mult = optimized_params.get('atr_momentum_mult', 0.0)

    df_bt['k_up_val'] = df_bt['VIX_Regime'].map(lambda r: k_params[r]['k_up'])
    df_bt['k_down_val'] = df_bt['VIX_Regime'].map(lambda r: k_params[r]['k_down'])

    # 1. VVIX Shock Factor
    if use_vvix and vvix_rank_col is not None and vvix_shock_mult > 0.0:
        df_bt['Shock_Factor'] = 1.0 + (vvix_shock_mult * df_bt[vvix_rank_col].clip(0, 1))
    else:
        df_bt['Shock_Factor'] = 1.0

    # 2. ATR Momentum Factor (Expansion/Contraction)
    if momentum_ratio_col is not None and momentum_ratio_col in df_bt.columns:
        momentum_dev = df_bt[momentum_ratio_col] - 1.0  # Center around 0
        df_bt['Momentum_Factor'] = 1.0 + atr_momentum_mult * momentum_dev
        df_bt['Momentum_Factor'] = df_bt['Momentum_Factor'].clip(lower=0.1)  # Prevent zero/negative ATR
    else:
        df_bt['Momentum_Factor'] = 1.0

    # Apply both factors to base ATR
    df_bt['Effective_ATR'] = df_bt[atr_col] * df_bt['Shock_Factor'] * df_bt['Momentum_Factor']

    df_bt['High_Pred'] = df_bt[open_col] + (df_bt['Effective_ATR'] * df_bt['k_up_val'])
    df_bt['Low_Pred'] = df_bt[open_col] - (df_bt['Effective_ATR'] * df_bt['k_down_val'])

    if use_close_for_range:
        high_respected = df_bt[close_col] <= df_bt['High_Pred']
        low_respected = df_bt[close_col] >= df_bt['Low_Pred']
    else:
        high_respected = df_bt[high_col] <= df_bt['High_Pred']
        low_respected = df_bt[low_col] >= df_bt['Low_Pred']

    range_held = high_respected & low_respected

    global_metrics = {
        'Hit Rate Global (Range Tenu)': range_held.mean() * 100,
        'Borne Haute Respectée': high_respected.mean() * 100,
        'Borne Basse Respectée': low_respected.mean() * 100
    }

    regime_metrics = {}
    for regime in ['Low', 'Normal', 'High']:
        df_reg = df_bt[df_bt['VIX_Regime'] == regime]
        if len(df_reg) > 0:
            if use_close_for_range:
                hr = (df_reg[close_col] <= df_reg['High_Pred']) & (df_reg[close_col] >= df_reg['Low_Pred'])
                h_res = df_reg[close_col] <= df_reg['High_Pred']
                l_res = df_reg[close_col] >= df_reg['Low_Pred']
            else:
                hr = (df_reg[high_col] <= df_reg['High_Pred']) & (df_reg[low_col] >= df_reg['Low_Pred'])
                h_res = df_reg[high_col] <= df_reg['High_Pred']
                l_res = df_reg[low_col] >= df_reg['Low_Pred']

            regime_metrics[regime] = {
                'Hit Rate Global (Range Tenu)': hr.mean() * 100,
                'Borne Haute Respectée': h_res.mean() * 100,
                'Borne Basse Respectée': l_res.mean() * 100,
                'k_up': k_params[regime]['k_up'],
                'k_down': k_params[regime]['k_down'],
                'Count': len(df_reg)
            }

    return global_metrics, regime_metrics, len(df_bt), df_bt.copy()


def display_report_with_vix(global_metrics, regime_metrics, total_bars,
                            vvix_shock_mult=0.0, use_vvix=True, atr_momentum_mult=0.0):
    """Affiche un rapport propre des statistiques."""
    print("\n" + "=" * 60)
    print(f"[TEST] RAPPORT DE BACKTESTING  — ÉCHANTILLON : {total_bars} BARS")
    print("=" * 60)

    if use_vvix:
        print(f"\n⚡ VVIX Shock Absorption : ENABLED (vvix_shock_mult = {vvix_shock_mult:.3f})")
    else:
        print(f"\n⚡ VVIX Shock Absorption : DISABLED")

    print(f"⚡ ATR Momentum (5/14)  : ENABLED (atr_momentum_mult = {atr_momentum_mult:.3f})")

    print(f"\n📊 [TEST] MÉTRIQUES GLOBALES")
    print(f"  -> Taux de succès Global : {global_metrics['Hit Rate Global (Range Tenu)']:.2f}%")
    print(f"  -> Fiabilité Borne Haute : {global_metrics['Borne Haute Respectée']:.2f}%")
    print(f"  -> Fiabilité Borne Basse : {global_metrics['Borne Basse Respectée']:.2f}%")

    print("\n📈 [TEST] DÉTAIL PAR RÉGIME DE VIX")
    for regime, metrics in regime_metrics.items():
        print(f"\n[{regime.upper()} VIX] (k_up: {metrics['k_up']:.3f}, "
              f"k_down: {metrics['k_down']:.3f}) - {metrics['Count']} bars")
        print(f"  -> Taux de succès Global : {metrics['Hit Rate Global (Range Tenu)']:.2f}%")

    print("=" * 60)


def display_dataset_info(train_info, test_info, ticker, dataset_id, atr_window, use_vvix=True, use_momentum=True):
    """Displays dataset description."""
    print("\n" + "=" * 60)
    print(f"📂 DATASET SPLIT DESCRIPTION — {ticker} ({dataset_id.upper()})")
    print("=" * 60)
    print(f"  ATR Window          : {atr_window} periods")
    print(f"  VVIX Shock Absorber : {'✅ ENABLED' if use_vvix else '❌ DISABLED'}")
    print(f"  ATR Momentum (5/14) : {'✅ ENABLED' if use_momentum else '❌ DISABLED'}")
    print("-" * 60)
    print(f"  🟢 TRAIN SET")
    print(f"     Period     : {train_info['start_date']}  ➔  {train_info['end_date']}")
    print(f"     Data Points: {train_info['bars']} bars")
    print("-" * 60)
    print(f"  🔵 TEST SET")
    print(f"     Period     : {test_info['start_date']}  ➔  {test_info['end_date']}")
    print(f"     Data Points: {test_info['bars']} bars")
    print("=" * 60 + "\n")


def optimize_vix_multipliers(df_bt, vix_col, open_col, close_col, atr_col,
                             high_col, low_col, ticker, n_trials=200, timeout=None,
                             tightness_weight=0.3, use_close_for_range=False,
                             verbose=True, use_vvix=True,
                             vix_rank_col=None, vvix_rank_col=None,
                             momentum_ratio_col=None):
    """Optuna optimization with VVIX shock mult and ATR Momentum."""
    if verbose:
        vvix_msg = f" + VVIX Shock" if use_vvix else ""
        timeout_msg = f" | Timeout: {timeout}s" if timeout else ""
        print(f"\n🚀 Lancement de l'optimisation Optuna ({n_trials} essais){timeout_msg} | "
              f"Tightness Weight: {tightness_weight}{vvix_msg} + ATR Momentum...")

    # Use tuple key for VIX rank
    df_bt['VIX_Regime'] = 'Normal'
    df_bt.loc[df_bt[vix_rank_col] < 0.30, 'VIX_Regime'] = 'Low'
    df_bt.loc[df_bt[vix_rank_col] > 0.70, 'VIX_Regime'] = 'High'

    opens = df_bt[open_col].values
    closes = df_bt[close_col].values
    atrs = df_bt[atr_col].values
    highs = df_bt[high_col].values
    lows = df_bt[low_col].values
    regimes = df_bt['VIX_Regime'].values

    if use_vvix and vvix_rank_col is not None and vvix_rank_col in df_bt.columns:
        vvix_ranks = df_bt[vvix_rank_col].clip(0, 1).values
    else:
        vvix_ranks = np.zeros(len(df_bt))
        use_vvix = False

    if momentum_ratio_col is not None and momentum_ratio_col in df_bt.columns:
        momentum_ratios = df_bt[momentum_ratio_col].values
    else:
        momentum_ratios = np.ones(len(df_bt))

    idx_low = np.where(regimes == 'Low')[0]
    idx_normal = np.where(regimes == 'Normal')[0]
    idx_high = np.where(regimes == 'High')[0]

    def objective(trial):
        k_up_low = trial.suggest_float('k_up_low', low=0.025, high=2.5)
        k_down_low = trial.suggest_float('k_down_low', low=0.025, high=2.5)
        k_up_normal = trial.suggest_float('k_up_normal', low=0.025, high=2.5)
        k_down_normal = trial.suggest_float('k_down_normal', low=0.025, high=2.5)
        k_up_high = trial.suggest_float('k_up_high', low=0.025, high=2.5)
        k_down_high = trial.suggest_float('k_down_high', low=0.025, high=2.5)

        if use_vvix:
            vvix_shock_mult = trial.suggest_float('vvix_shock_mult', low=0.0, high=3.0)
        else:
            vvix_shock_mult = 0.0

        # Suggest ATR Momentum multiplier (-2.0 to +2.0 allows both trend-following and mean-reverting adjustments)
        atr_momentum_mult = trial.suggest_float('atr_momentum_mult', low=-2.0, high=2.0)

        params = {
            'Low': {'up': k_up_low, 'down': k_down_low},
            'Normal': {'up': k_up_normal, 'down': k_down_normal},
            'High': {'up': k_up_high, 'down': k_down_high}
        }

        hits = np.zeros(len(df_bt), dtype=bool)

        def _compute_hits(idx, p):
            if len(idx) == 0:
                return
            shock_factor = 1.0 + (vvix_shock_mult * vvix_ranks[idx])

            # Momentum logic: if ratio > 1, expand. if ratio < 1, contract.
            momentum_dev = momentum_ratios[idx] - 1.0
            momentum_factor = 1.0 + atr_momentum_mult * momentum_dev
            momentum_factor = np.maximum(momentum_factor, 0.1)  # Prevent negative/zero ATR

            effective_atr = atrs[idx] * shock_factor * momentum_factor
            h_pred = opens[idx] + (effective_atr * p['up'])
            l_pred = opens[idx] - (effective_atr * p['down'])
            if use_close_for_range:
                hits[idx] = (closes[idx] <= h_pred) & (closes[idx] >= l_pred)
            else:
                hits[idx] = (highs[idx] <= h_pred) & (lows[idx] >= l_pred)

        _compute_hits(idx_low, params['Low'])
        _compute_hits(idx_normal, params['Normal'])
        _compute_hits(idx_high, params['High'])

        hit_rate = hits.mean()
        avg_k_sum = (k_up_low + k_down_low + k_up_normal +
                     k_down_normal + k_up_high + k_down_high) / 6.0
        score = hit_rate - (tightness_weight * avg_k_sum)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=True if verbose else False, n_jobs=1)

    best_params = study.best_params
    best_value = study.best_value

    if verbose:
        print(f"✅ Meilleur Score Composite trouvé : {best_value:.4f}")
        print(f"   Paramètres optimaux : {best_params}")

    final_hr = run_backtest_with_vix(
        df_bt=df_bt, open_col=open_col, close_col=close_col,
        atr_col=atr_col, high_col=high_col, low_col=low_col,
        optimized_params={
            'k_up_low': best_params['k_up_low'], 'k_down_low': best_params['k_down_low'],
            'k_up_normal': best_params['k_up_normal'], 'k_down_normal': best_params['k_down_normal'],
            'k_up_high': best_params['k_up_high'], 'k_down_high': best_params['k_down_high'],
            'vvix_shock_mult': best_params.get('vvix_shock_mult', 0.0),
            'atr_momentum_mult': best_params.get('atr_momentum_mult', 0.0)
        },
        use_close_for_range=use_close_for_range,
        use_vvix=use_vvix,
        vvix_rank_col=vvix_rank_col,
        vix_rank_col=vix_rank_col,
        momentum_ratio_col=momentum_ratio_col
    )[0]['Hit Rate Global (Range Tenu)']

    avg_k = sum([best_params[k] for k in best_params if 'k_' in k]) / 6.0
    vvix_m = best_params.get('vvix_shock_mult', 0.0)
    mom_m = best_params.get('atr_momentum_mult', 0.0)

    if verbose:
        print(f"   [TRAIN] → Hit Rate Réel : {final_hr:.2f}% | Avg K : {avg_k:.3f} | VVIX Mult : {vvix_m:.3f} | Mom Mult : {mom_m:.3f}")

    return best_params, best_value


def display_realtime_prediction(df_bt, vix_col, open_col, close_col, atr_col,
                                high_col, low_col, ticker, optimized_params,
                                use_close_for_range=False, verbose=True,
                                use_vvix=True, vvix_rank_col=None, vix_rank_col=None,
                                momentum_ratio_col=None):
    """Displays a formatted real-time prediction."""
    if df_bt.empty:
        print("\n⚠️  No data available for real-time prediction.")
        return

    last_row = df_bt.iloc[-1]
    last_date = df_bt.index[-1].strftime('%Y-%m-%d')

    vix_rank = _extract_scalar(last_row[vix_rank_col])

    if vix_rank < 0.30:
        regime = 'Low'
    elif vix_rank > 0.70:
        regime = 'High'
    else:
        regime = 'Normal'

    k_up = optimized_params[f'k_up_{regime.lower()}']
    k_down = optimized_params[f'k_down_{regime.lower()}']

    current_open = last_row[open_col]
    current_atr = last_row[atr_col]

    vvix_shock_mult = optimized_params.get('vvix_shock_mult', 0.0) if use_vvix else 0.0
    atr_momentum_mult = optimized_params.get('atr_momentum_mult', 0.0)

    if use_vvix and vvix_rank_col is not None and vvix_rank_col in df_bt.columns and vvix_shock_mult > 0.0:
        vvix_rank_val = _extract_scalar(last_row[vvix_rank_col])
        vvix_rank_val = max(0.0, min(1.0, vvix_rank_val))
        shock_factor = 1.0 + (vvix_shock_mult * vvix_rank_val)
    else:
        vvix_rank_val = 0.0
        shock_factor = 1.0

    if momentum_ratio_col is not None and momentum_ratio_col in df_bt.columns:
        current_momentum_ratio = _extract_scalar(last_row[momentum_ratio_col])
        momentum_dev = current_momentum_ratio - 1.0
        momentum_factor = 1.0 + atr_momentum_mult * momentum_dev
        momentum_factor = max(0.1, momentum_factor)
    else:
        current_momentum_ratio = 1.0
        momentum_factor = 1.0

    effective_atr = current_atr * shock_factor * momentum_factor
    predicted_high = current_open + (effective_atr * k_up)
    predicted_low = current_open - (effective_atr * k_down)

    actual_high = last_row[high_col]
    actual_low = last_row[low_col]
    actual_close = last_row[close_col]

    if use_close_for_range:
        high_status = ("✅" if pd.notna(actual_close) and actual_close <= predicted_high
                       else ("❌" if pd.notna(actual_close) else "⏳"))
        low_status = ("✅" if pd.notna(actual_close) and actual_close >= predicted_low
                      else ("❌" if pd.notna(actual_close) else "⏳"))
    else:
        high_status = ("✅" if pd.notna(actual_high) and actual_high <= predicted_high
                       else ("❌" if pd.notna(actual_high) else "⏳"))
        low_status = ("✅" if pd.notna(actual_low) and actual_low >= predicted_low
                      else ("❌" if pd.notna(actual_low) else "⏳"))

    range_status = ("✅" if high_status == "✅" and low_status == "✅"
                    else ("❌" if high_status == "❌" or low_status == "❌" else "⏳"))

    if verbose:
        print("\n" + "=" * 60)
        print(f"🔮 REAL-TIME PREDICTION — {ticker} | {last_date}")
        print("=" * 60)
        print(f"  VIX Regime       : {regime.upper()} (Rank: {vix_rank:.2%})")
        print(f"  Multipliers      : k_up={k_up:.3f} | k_down={k_down:.3f}")
        print(f"  Current Open     : {current_open:,.2f}")
        print(f"  Base ATR         : {current_atr:,.2f}")
        if use_vvix and vvix_rank_val > 0:
            print(f"  VVIX Rank        : {vvix_rank_val:.2%}")
            print(f"  Shock Factor     : {shock_factor:.3f}x")
        if current_momentum_ratio != 1.0 or atr_momentum_mult != 0.0:
            print(f"  ATR Mom. Ratio   : {current_momentum_ratio:.3f} (5/14)")
            print(f"  Momentum Factor  : {momentum_factor:.3f}x")
        print(f"  Effective ATR    : {effective_atr:,.2f}")
        print("-" * 60)
        if use_close_for_range:
            print(f"  📈 Predicted High : {predicted_high:.2f}   Actual Close: {actual_close:.2f}  {high_status}")
            print(f"  📉 Predicted Low  : {predicted_low:.2f}   Actual Close: {actual_close:.2f}  {low_status}")
        else:
            print(f"  📈 Predicted High : {predicted_high:.2f}   Actual: {actual_high:.2f}  {high_status}")
            print(f"  📉 Predicted Low  : {predicted_low:.2f}   Actual: {actual_low:.2f}  {low_status}")
        print("-" * 60)
        print(f"  🎯 Range Held     : {range_status}")
        print("=" * 60)

    return {
        'realtime': {
            'predicted_high': predicted_high, 'predicted_low': predicted_low,
            'actual_high': actual_close if use_close_for_range else actual_high,
            'actual_low': actual_close if use_close_for_range else actual_low,
            'actual_close': actual_close, 'vix_regime': regime.upper(),
            'vix_rank': vix_rank, 'vvix_rank': vvix_rank_val if use_vvix else None,
            'shock_factor': shock_factor if use_vvix else 1.0,
            'momentum_factor': momentum_factor,
            'effective_atr': effective_atr, 'ticker': ticker, 'last_date': last_date
        }
    }


def entry(args=None):
    random.seed(42)
    np.random.seed(42)

    total_start = time.time()
    timings = {}

    _master_data_cache = {}
    open_col = ('Open', args.ticker)
    close_col = ('Close', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)

    use_vvix = args.use_vvix

    # Explicit tuple keys to avoid MultiIndex KeyError clashes
    vix_rank_col = ('VIX_Rolling_Rank', args.ticker)
    vvix_rank_col = ('VVIX_Rolling_Rank', '^VVIX')
    momentum_ratio_col = ('ATR_Momentum_Ratio', args.ticker)

    t0 = time.time()
    if args.dataframe is None:
        if args.use_realtime_data:
            assert args.ticker in ["^GSPC"]
            (daily_data_cache, weekly_data_cache, monthly_data_cache,
             quaterly_data_cache, yearly_data_cache) = realtime()

            if args.dataset_id == "day":
                _master_data_cache[args.ticker] = daily_data_cache[args.ticker]
                _master_data_cache["^VIX"] = daily_data_cache["^VIX"]
            elif args.dataset_id == "week":
                _master_data_cache[args.ticker] = weekly_data_cache[args.ticker]
                _master_data_cache["^VIX_MEAN"] = weekly_data_cache["^VIX_MEAN"]
            elif args.dataset_id == "month":
                _master_data_cache[args.ticker] = monthly_data_cache[args.ticker]
                _master_data_cache["^VIX_MEAN"] = monthly_data_cache["^VIX_MEAN"]
            elif args.dataset_id == "quarter":
                _master_data_cache[args.ticker] = quaterly_data_cache[args.ticker]
                _master_data_cache["^VIX_MEAN"] = quaterly_data_cache["^VIX_MEAN"]
            elif args.dataset_id == "year":
                _master_data_cache[args.ticker] = yearly_data_cache[args.ticker]
                _master_data_cache["^VIX_MEAN"] = yearly_data_cache["^VIX_MEAN"]

            df_ticker = _master_data_cache[args.ticker].sort_index()
        else:
            with open(get_filename_for_dataset(args.dataset_id, older_dataset=None), 'rb') as f:
                _master_data_cache = pickle.load(f)
            df_ticker = _master_data_cache[args.ticker].sort_index()

        try:
            df_vix = _master_data_cache["^VIX_MEAN"].sort_index()
        except:
            df_vix = _master_data_cache["^VIX"].sort_index()

        df_vvix = None
        if use_vvix:
            vvix_candidates = ["^VVIX", "^VVIX_MEAN"]
            for vvix_key in vvix_candidates:
                if vvix_key in _master_data_cache:
                    df_vvix = _master_data_cache[vvix_key].sort_index()
                    if args.verbose: print(f"   ✅ Found {vvix_key} in data cache.")
                    break

            if df_vvix is None:
                if args.verbose: print("   🔄 ^VVIX not in cache, fetching via yfinance...")
                start_date = df_ticker.index[0].strftime('%Y-%m-%d')
                end_date = df_ticker.index[-1].strftime('%Y-%m-%d')
                df_vvix = fetch_vvix_yfinance(start_date, end_date)

            if df_vvix is not None and not df_vvix.empty:
                if args.verbose: print(f"   ✅ VVIX data loaded: {len(df_vvix)} bars")
            else:
                if args.verbose: print("   ⚠️  VVIX data unavailable. Disabling VVIX.")
                use_vvix = False
                df_vvix = None

        timings['data_loading'] = time.time() - t0
        if args.verbose: print(f"\n✨ Loaded {args.ticker} | SPX: {len(df_ticker)} | VIX: {len(df_vix)} | VVIX: {len(df_vvix) if df_vvix is not None else 0}")

        t0 = time.time()
        # Calculate ATR_5 first
        df_ticker_5, atr_5_col, _ = calculate_atr(
            df=df_ticker, close_col=close_col, high_col=high_col, low_col=low_col,
            ticker=args.ticker, window=5
        )

        # Calculate ATR_14
        df_ticker, atr_col, prev_close_col = calculate_atr(
            df=df_ticker, close_col=close_col, high_col=high_col, low_col=low_col,
            ticker=args.ticker, window=args.atr_window
        )
        timings['atr_calculation'] = time.time() - t0

        t0 = time.time()
        vix_col = next((col for col in df_vix.columns if isinstance(col, tuple) and 'Close' in col), None)
        df_bt = df_ticker.join(df_vix[[vix_col]], how='inner').dropna().copy()

        df_bt[vix_col] = df_bt[vix_col].shift(1)
        df_bt = df_bt.dropna()

        # Join ATR_5 and compute Momentum Ratio
        df_bt = df_bt.join(df_ticker_5[[atr_5_col]], how='inner')

        # Prevent division by zero
        atr_14_safe = df_bt[atr_col].replace(0, np.nan)
        df_bt[momentum_ratio_col] = df_bt[atr_5_col] / atr_14_safe
        df_bt = df_bt.dropna()

        # Assign VIX Rank using the explicit tuple key
        df_bt[vix_rank_col] = df_bt[vix_col].rolling(window=252, min_periods=20).rank(pct=True)

        if use_vvix and df_vvix is not None:
            vvix_close_col = next((col for col in df_vvix.columns if isinstance(col, tuple) and 'Close' in col), None)
            if vvix_close_col is None:
                vvix_close_col = df_vvix.columns[0]
                df_vvix = df_vvix.rename(columns={vvix_close_col: ('Close', '^VVIX')})
                vvix_close_col = ('Close', '^VVIX')

            df_bt = df_bt.join(df_vvix[[vvix_close_col]], how='inner').dropna().copy()
            df_bt[vvix_close_col] = df_bt[vvix_close_col].shift(1)
            df_bt = df_bt.dropna()

            # Assign VVIX Rank using the explicit tuple key
            df_bt[vvix_rank_col] = df_bt[vvix_close_col].rolling(window=252, min_periods=20).rank(pct=True)

            # Global dropna() safely removes the first ~252 rows where ranks are NaN
            # This entirely avoids pandas MultiIndex KeyError issues on subset lists
            df_bt = df_bt.dropna().copy()
        else:
            df_bt = df_bt.dropna().copy()
            use_vvix = False
            vvix_rank_col = None

    else:
        df_bt = args.dataframe
        vix_col = next((col for col in df_bt.columns if isinstance(col, tuple) and 'Close' in col and 'VIX' in col[1]), None)
        atr_col = (f'ATR_{args.atr_window}', args.ticker)
        atr_5_col = ('ATR_5', args.ticker)

        if atr_5_col not in df_bt.columns:
            if args.verbose: print("   ⚠️  Provided dataframe missing 'ATR_5'. Assuming ratio = 1.0.")
            df_bt[momentum_ratio_col] = 1.0
        elif momentum_ratio_col not in df_bt.columns:
            atr_14_safe = df_bt[atr_col].replace(0, np.nan)
            df_bt[momentum_ratio_col] = df_bt[atr_5_col] / atr_14_safe

        if use_vvix and vvix_rank_col not in df_bt.columns:
            if args.verbose: print("   ⚠️  Provided dataframe missing 'VVIX_Rolling_Rank'. Disabling.")
            use_vvix = False
            vvix_rank_col = None
    if args.clip_n > 0:
        df_bt = df_bt.iloc[:-args.clip_n].copy()
    if args.verbose: print(f"WORKING DATASET : {df_bt.index[0].strftime('%Y-%m-%d')}::{df_bt.index[-1].strftime('%Y-%m-%d')}")
    _n = int(args.n_split * len(df_bt))
    df_bt_train = df_bt.iloc[:_n].copy()
    df_bt_test = df_bt.iloc[_n:].copy()
    timings['merge_split'] = time.time() - t0

    train_info = {'start_date': df_bt_train.index[0].strftime('%Y-%m-%d'), 'end_date': df_bt_train.index[-1].strftime('%Y-%m-%d'), 'bars': len(df_bt_train), 'split_ratio': args.n_split}
    test_info = {'start_date': df_bt_test.index[0].strftime('%Y-%m-%d'), 'end_date': df_bt_test.index[-1].strftime('%Y-%m-%d'), 'bars': len(df_bt_test), 'split_ratio': 1.0 - args.n_split}

    if args.verbose: display_dataset_info(train_info, test_info, args.ticker, args.dataset_id, args.atr_window, use_vvix=use_vvix, use_momentum=True)

    t0 = time.time()
    optimized_params, best_score = optimize_vix_multipliers(
        df_bt=df_bt_train.copy(), vix_col=vix_col, open_col=open_col, close_col=close_col,
        atr_col=atr_col, high_col=high_col, low_col=low_col, ticker=args.ticker,
        n_trials=args.n_trials, timeout=args.timeout, tightness_weight=args.tightness_weight,
        use_close_for_range=args.use_close_for_range, verbose=args.verbose,
        use_vvix=use_vvix, vix_rank_col=vix_rank_col, vvix_rank_col=vvix_rank_col,
        momentum_ratio_col=momentum_ratio_col
    )
    timings['optuna_optimization'] = time.time() - t0

    t0 = time.time()
    global_stats, regime_stats, bars_analyzed, df_bt_test = run_backtest_with_vix(
        df_bt=df_bt_test.copy(), open_col=open_col, close_col=close_col,
        atr_col=atr_col, high_col=high_col, low_col=low_col,
        optimized_params=optimized_params, use_close_for_range=args.use_close_for_range,
        use_vvix=use_vvix, vvix_rank_col=vvix_rank_col, vix_rank_col=vix_rank_col,
        momentum_ratio_col=momentum_ratio_col
    )
    timings['backtest'] = time.time() - t0

    if args.verbose:
        display_report_with_vix(
            global_stats, regime_stats, bars_analyzed,
            vvix_shock_mult=optimized_params.get('vvix_shock_mult', 0.0),
            use_vvix=use_vvix,
            atr_momentum_mult=optimized_params.get('atr_momentum_mult', 0.0)
        )

    t0 = time.time()
    realtime_results = display_realtime_prediction(
        df_bt=df_bt_test, vix_col=vix_col, open_col=open_col, close_col=close_col,
        atr_col=atr_col, high_col=high_col, low_col=low_col, ticker=args.ticker,
        optimized_params=optimized_params, use_close_for_range=args.use_close_for_range,
        verbose=args.verbose, use_vvix=use_vvix, vvix_rank_col=vvix_rank_col, vix_rank_col=vix_rank_col,
        momentum_ratio_col=momentum_ratio_col
    )
    timings['realtime_prediction'] = time.time() - t0

    total_elapsed = time.time() - total_start
    if args.verbose:
        print("\n" + "=" * 60)
        print("⏱️  EXECUTION TIME SUMMARY")
        print("=" * 60)
        for phase, duration in timings.items():
            label = phase.replace('_', ' ').title()
            print(f"  {label:<25}: {duration:>8.3f}s")
        print("-" * 60)
        print(f"  {'TOTAL':<25}: {total_elapsed:>8.3f}s")
        print("=" * 60)

    return realtime_results


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    entry(args)