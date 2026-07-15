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
import numpy as np
import pandas as pd
import optuna
import joblib
from fetchers.serialize_fyahoo import realtime

# Suppress Optuna & pandas debug logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.options.mode.chained_assignment = None


def get_parser():
    """Creates and configures the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Market Prediction Model with VIX Optimization")

    parser.add_argument("--dataset-id", type=str, default="day", choices=["day", "week", "month"])
    parser.add_argument('--ticker', type=str, default='^GSPC', help='Ticker symbol')
    parser.add_argument('--dataframe', type=pd.DataFrame, default=None, help='Dataset supplied')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True, help='Verbose output')
    # OPTUNA PERSISTENCE AND SAMPLER ARGUMENTS
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Number of Optuna trials for VIX optimization (default: 200).")

    # USER-CONFIGURABLE PARAMETERS
    parser.add_argument("--use-realtime-data", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable real-time data fetching (default: True).")
    parser.add_argument("--atr-window", type=int, default=14,
                        help="Window size for ATR calculation (default: 14).")
    parser.add_argument("--n-split", type=float, default=0.80,
                        help="Train/Test split ratio (default: 0.80).")
    parser.add_argument("--tightness-weight", type=float, default=0.3,
                        help="Weight for tightness penalty in Optuna objective (default: 0.3).")
    parser.add_argument("--use-close-for-range", action=argparse.BooleanOptionalAction, default=False,
                        help="If True, consider range Held if close is within [Predicted Low, Predicted High], ignoring intraday High/Low breaches.")

    return parser


def calculate_atr(df, close_col, high_col, low_col, ticker, window):
    """
    Calculate Average True Range (ATR) with Wilder's smoothing.
    """
    df = df.copy()  # Isoler les modifications
    prev_close_col = ('Prev_Close', ticker)
    df[prev_close_col] = df[close_col].shift(1)
    hl_col, hpc_col, lpc_col = ('H-L', ticker), ('H-PC', ticker), ('L-PC', ticker)
    df[hl_col] = df[high_col] - df[low_col]
    df[hpc_col] = (df[high_col] - df[prev_close_col]).abs()
    df[lpc_col] = (df[low_col] - df[prev_close_col]).abs()
    tr_col = ('TR', ticker)
    df[tr_col] = df[[hl_col, hpc_col, lpc_col]].max(axis=1)

    # Calcul de l'ATR (Moyenne mobile exponentielle de Wilder, standard pour l'ATR)
    atr_col = (f'ATR_{window}', ticker)
    df[atr_col] = df[tr_col].ewm(alpha=1 / window, adjust=False).mean().shift(1)
    df = df.iloc[window:]  # On coupe les premières lignes faussées
    cols_to_check = [close_col, high_col, low_col, atr_col, prev_close_col]
    return df.dropna(subset=cols_to_check).copy(), atr_col, prev_close_col


def run_backtest_with_vix(df_bt, open_col, close_col, atr_col, high_col, low_col, optimized_params, use_close_for_range=False):
    """
    Exécute le backtest en utilisant soit les params optimisés par Optuna,
    soit les heuristiques par défaut.
    """

    df_bt['VIX_Regime'] = 'Normal'
    df_bt.loc[df_bt['VIX_Rolling_Rank'] < 0.30, 'VIX_Regime'] = 'Low'
    df_bt.loc[df_bt['VIX_Rolling_Rank'] > 0.70, 'VIX_Regime'] = 'High'
    assert optimized_params is not None
    k_params = {
        'Low': {'k_up': optimized_params['k_up_low'], 'k_down': optimized_params['k_down_low']},
        'Normal': {'k_up': optimized_params['k_up_normal'], 'k_down': optimized_params['k_down_normal']},
        'High': {'k_up': optimized_params['k_up_high'], 'k_down': optimized_params['k_down_high']}
    }

    df_bt['k_up_val'] = df_bt['VIX_Regime'].map(lambda r: k_params[r]['k_up'])
    df_bt['k_down_val'] = df_bt['VIX_Regime'].map(lambda r: k_params[r]['k_down'])

    df_bt['High_Pred'] = df_bt[open_col] + (df_bt[atr_col] * df_bt['k_up_val'])
    df_bt['Low_Pred'] = df_bt[open_col] - (df_bt[atr_col] * df_bt['k_down_val'])

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


def display_report_with_vix(global_metrics, regime_metrics, total_bars):
    """Affiche un rapport propre des statistiques avec VIX."""
    print("\n" + "=" * 60)
    print(f"[TEST] RAPPORT DE BACKTESTING  — ÉCHANTILLON : {total_bars} BARS")
    print("=" * 60)

    print("\n📊 [TEST] MÉTRIQUES GLOBALES")
    print(f"  -> Taux de succès Global (Le prix reste DANS le range) : {global_metrics['Hit Rate Global (Range Tenu)']:.2f}%")
    print(f"  -> Fiabilité Borne Haute (Pas de cassure par le haut)   : {global_metrics['Borne Haute Respectée']:.2f}%")
    print(f"  -> Fiabilité Borne Basse (Pas de cassure par le bas)   : {global_metrics['Borne Basse Respectée']:.2f}%")

    print("\n📈 [TEST] DÉTAIL PAR RÉGIME DE VIX")
    for regime, metrics in regime_metrics.items():
        print(f"\n[{regime.upper()} VIX] (k_up: {metrics['k_up']}, k_down: {metrics['k_down']}) - {metrics['Count']} bars")
        print(f"  -> Taux de succès Global : {metrics['Hit Rate Global (Range Tenu)']:.2f}%")
        print(f"  -> Fiabilité Borne Haute : {metrics['Borne Haute Respectée']:.2f}%")
        print(f"  -> Fiabilité Borne Basse : {metrics['Borne Basse Respectée']:.2f}%")

    print("=" * 60)


def display_dataset_info(train_info, test_info, ticker, dataset_id, atr_window):
    """Displays a nicely formatted description of the Train and Test sets."""
    print("\n" + "=" * 60)
    print(f"📂 DATASET SPLIT DESCRIPTION — {ticker} ({dataset_id.upper()})")
    print("=" * 60)
    print(f"  ATR Window : {atr_window} periods")
    print("-" * 60)
    print(f"  🟢 TRAIN SET")
    print(f"     Period     : {train_info['start_date']}  ➔  {train_info['end_date']}")
    print(f"     Data Points: {train_info['bars']} bars")
    print(f"     Proportion : {train_info['split_ratio']:.0%} of total data")
    print("-" * 60)
    print(f"  🔵 TEST SET")
    print(f"     Period     : {test_info['start_date']}  ➔  {test_info['end_date']}")
    print(f"     Data Points: {test_info['bars']} bars")
    print(f"     Proportion : {test_info['split_ratio']:.0%} of total data")
    print("=" * 60 + "\n")


def optimize_vix_multipliers(df_bt, vix_col, open_col, close_col, atr_col, high_col, low_col, ticker, n_trials=200, tightness_weight=0.3, use_close_for_range=False, verbose=True):
    """
    Utilise Optuna pour trouver les meilleurs k_up/k_down pour chaque régime de VIX.
    Objectif : Maximiser le Hit Rate tout en MINIMISANT la largeur du range (k values).
    """
    if verbose: print(f"\n🚀 Lancement de l'optimisation Optuna ({n_trials} essais) | Tightness Weight: {tightness_weight}...")

    # Définition des régimes fixes
    df_bt['VIX_Regime'] = 'Normal'
    df_bt.loc[df_bt['VIX_Rolling_Rank'] < 0.30, 'VIX_Regime'] = 'Low'
    df_bt.loc[df_bt['VIX_Rolling_Rank'] > 0.70, 'VIX_Regime'] = 'High'

    # Extraire les séries nécessaires pour la performance
    opens = df_bt[open_col].values
    closes = df_bt[close_col].values
    atrs = df_bt[atr_col].values
    highs = df_bt[high_col].values
    lows = df_bt[low_col].values
    regimes = df_bt['VIX_Regime'].values

    # Mapping rapide des indices par régime
    idx_low = np.where(regimes == 'Low')[0]
    idx_normal = np.where(regimes == 'Normal')[0]
    idx_high = np.where(regimes == 'High')[0]

    def objective(trial):
        # Suggestion des paramètres avec bornes resserrées pour encourager la tightness
        k_up_low = trial.suggest_float('k_up_low', 0.5, 1.5)
        k_down_low = trial.suggest_float('k_down_low', 0.5, 1.5)

        k_up_normal = trial.suggest_float('k_up_normal', 0.5, 1.5)
        k_down_normal = trial.suggest_float('k_down_normal', 0.5, 1.5)

        k_up_high = trial.suggest_float('k_up_high', 0.5, 1.5)
        k_down_high = trial.suggest_float('k_down_high', 0.5, 1.5)

        params = {
            'Low': {'up': k_up_low, 'down': k_down_low},
            'Normal': {'up': k_up_normal, 'down': k_down_normal},
            'High': {'up': k_up_high, 'down': k_down_high}
        }

        hits = np.zeros(len(df_bt), dtype=bool)

        # Régime Low
        if len(idx_low) > 0:
            p = params['Low']
            h_pred = opens[idx_low] + (atrs[idx_low] * p['up'])
            l_pred = opens[idx_low] - (atrs[idx_low] * p['down'])
            if use_close_for_range:
                hits[idx_low] = (closes[idx_low] <= h_pred) & (closes[idx_low] >= l_pred)
            else:
                hits[idx_low] = (highs[idx_low] <= h_pred) & (lows[idx_low] >= l_pred)

        # Régime Normal
        if len(idx_normal) > 0:
            p = params['Normal']
            h_pred = opens[idx_normal] + (atrs[idx_normal] * p['up'])
            l_pred = opens[idx_normal] - (atrs[idx_normal] * p['down'])
            if use_close_for_range:
                hits[idx_normal] = (closes[idx_normal] <= h_pred) & (closes[idx_normal] >= l_pred)
            else:
                hits[idx_normal] = (highs[idx_normal] <= h_pred) & (lows[idx_normal] >= l_pred)

        # Régime High
        if len(idx_high) > 0:
            p = params['High']
            h_pred = opens[idx_high] + (atrs[idx_high] * p['up'])
            l_pred = opens[idx_high] - (atrs[idx_high] * p['down'])
            if use_close_for_range:
                hits[idx_high] = (closes[idx_high] <= h_pred) & (closes[idx_high] >= l_pred)
            else:
                hits[idx_high] = (highs[idx_high] <= h_pred) & (lows[idx_high] >= l_pred)

        hit_rate = hits.mean()
        avg_k_sum = (k_up_low + k_down_low + k_up_normal + k_down_normal + k_up_high + k_down_high) / 6.0
        score = hit_rate - (tightness_weight * avg_k_sum)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value

    if verbose: print(f"✅ Meilleur Score Composite trouvé : {best_value:.4f}")
    if verbose: print(f"   Paramètres optimaux : {best_params}")

    # Afficher le hit_rate réel et la tightness séparément pour transparence
    final_hr = run_backtest_with_vix(
        df_bt=df_bt, open_col=open_col, close_col=close_col,
        atr_col=atr_col, high_col=high_col, low_col=low_col,
        optimized_params={
            'k_up_low': best_params['k_up_low'], 'k_down_low': best_params['k_down_low'],
            'k_up_normal': best_params['k_up_normal'], 'k_down_normal': best_params['k_down_normal'],
            'k_up_high': best_params['k_up_high'], 'k_down_high': best_params['k_down_high']
        },
        use_close_for_range=use_close_for_range
    )[0]['Hit Rate Global (Range Tenu)']

    avg_k = sum(best_params.values()) / 6.0
    if verbose: print(f"   [TRAIN] → Hit Rate Réel : {final_hr:.2f}% | Avg K (Tightness) : {avg_k:.3f}")

    return best_params, best_value


def display_realtime_prediction(df_bt, vix_col, open_col, close_col, atr_col, high_col, low_col, ticker, optimized_params, use_close_for_range=False, verbose=True):
    """Displays a formatted real-time prediction for the last available bar."""
    if df_bt.empty:
        print("\n⚠️  No data available for real-time prediction.")
        return

    last_row = df_bt.iloc[-1]
    last_date = df_bt.index[-1].strftime('%Y-%m-%d')

    # Determine current VIX regime
    vix_rank = last_row['VIX_Rolling_Rank']
    assert 1 == len(vix_rank)
    vix_rank = vix_rank.values[0]
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
    predicted_high = current_open + (current_atr * k_up)
    predicted_low = current_open - (current_atr * k_down)

    # Actual values (may be NaN if bar is incomplete / real-time)
    actual_high = last_row[high_col]
    actual_low = last_row[low_col]
    actual_close = last_row[close_col]

    if use_close_for_range:
        high_status = "✅" if pd.notna(actual_close) and actual_close <= predicted_high else ("❌" if pd.notna(actual_close) else "⏳")
        low_status = "✅" if pd.notna(actual_close) and actual_close >= predicted_low else ("❌" if pd.notna(actual_close) else "⏳")
    else:
        high_status = "✅" if pd.notna(actual_high) and actual_high <= predicted_high else ("❌" if pd.notna(actual_high) else "⏳")
        low_status = "✅" if pd.notna(actual_low) and actual_low >= predicted_low else ("❌" if pd.notna(actual_low) else "⏳")

    range_status = "✅" if high_status == "✅" and low_status == "✅" else ("❌" if high_status == "❌" or low_status == "❌" else "⏳")
    if verbose:
        print("\n" + "=" * 60)
        print(f"🔮 REAL-TIME PREDICTION — {ticker} | {last_date}")
        print("=" * 60)
        print(f"  VIX Regime      : {regime.upper()} (Rolling Rank: {vix_rank:.2%})")
        print(f"  Multipliers     : k_up={k_up:.3f} | k_down={k_down:.3f}")
        print(f"  Current Open    : {current_open:,.2f}")
        print(f"  Current ATR     : {current_atr:,.2f}")
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

    return {'realtime': {'predicted_high': predicted_high, 'predicted_low': predicted_low,
                         'actual_high': actual_close if use_close_for_range else actual_high,
                         'actual_low': actual_close if use_close_for_range else actual_low,
                         'vix_regime': regime.upper(), 'vix_rank': vix_rank, 'ticker': ticker, 'last_date': last_date}}


def entry(args=None):
    total_start = time.time()
    timings = {}

    _master_data_cache = {}
    open_col = ('Open', args.ticker)
    close_col = ('Close', args.ticker)
    high_col = ('High', args.ticker)
    low_col = ('Low', args.ticker)

    # --- DATA LOADING ---
    t0 = time.time()
    if args.dataframe is None:
        if args.use_realtime_data:
            assert args.ticker in ["^GSPC"]
            daily_data_cache, weekly_data_cache, monthly_data_cache, quaterly_data_cache, yearly_data_cache = realtime()
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
            else:
                assert False, f"{args.dataset_id} is not a valid dataset id"
            df_ticker = _master_data_cache[args.ticker].sort_index()
        else:
            with open(get_filename_for_dataset(args.dataset_id, older_dataset=None), 'rb') as f:
                _master_data_cache = pickle.load(f)
            assert _master_data_cache is not None
            df_ticker = _master_data_cache[args.ticker].sort_index()

        try:
            df_vix = _master_data_cache["^VIX_MEAN"].sort_index()
        except:
            # DAY data has not the VIX_MEAN dataset.
            df_vix = _master_data_cache["^VIX"].sort_index()

        timings['data_loading'] = time.time() - t0

        print(f"\n✨ Loaded {args.ticker} | Dataset: {args.dataset_id} | SPX Bars: {len(df_ticker)} | VIX Bars: {len(df_vix)} | ATR: {args.atr_window}")

        # --- ATR CALCULATION ---
        t0 = time.time()
        df_ticker, atr_col, prev_close_col = calculate_atr(
            df=df_ticker, close_col=close_col, high_col=high_col, low_col=low_col,
            ticker=args.ticker, window=args.atr_window
        )
        timings['atr_calculation'] = time.time() - t0

        print(f"{args.ticker} : {df_ticker.index[0].strftime('%Y-%m-%d')}::{df_ticker.index[-1].strftime('%Y-%m-%d')}    VIX: {df_vix.index[0].strftime('%Y-%m-%d')}::{df_vix.index[-1].strftime('%Y-%m-%d')}")

        # --- MERGE & SPLIT ---
        t0 = time.time()
        vix_col = next((col for col in df_vix.columns if isinstance(col, tuple) and 'Close' in col), None)
        df_bt = df_ticker.join(df_vix[[vix_col]], how='inner').dropna().copy()

        # Before joining or calculating rolling rank, shift VIX by 1
        df_bt[vix_col] = df_bt[vix_col].shift(1)
        df_bt = df_bt.dropna()  # Drop the first row where VIX is now NaN
        # Calculates the rank of the current element within its rolling window automatically
        df_bt['VIX_Rolling_Rank'] = df_bt[vix_col].rolling(window=252, min_periods=20).rank(pct=True)
    else:
        df_bt = args.dataframe
        vix_col = next((col for col in df_bt.columns if isinstance(col, tuple) and 'Close' in col and 'VIX' in col[1]), None)
        atr_col = (f'ATR_{args.atr_window}', args.ticker)

    _n = int(args.n_split * len(df_bt))
    df_bt_train = df_bt.iloc[:_n].copy()
    df_bt_test = df_bt.iloc[_n:].copy()
    timings['merge_split'] = time.time() - t0

    train_info = {
        'start_date': df_bt_train.index[0].strftime('%Y-%m-%d'),
        'end_date': df_bt_train.index[-1].strftime('%Y-%m-%d'),
        'bars': len(df_bt_train),
        'split_ratio': args.n_split
    }
    test_info = {
        'start_date': df_bt_test.index[0].strftime('%Y-%m-%d'),
        'end_date': df_bt_test.index[-1].strftime('%Y-%m-%d'),
        'bars': len(df_bt_test),
        'split_ratio': 1.0 - args.n_split
    }

    if args.verbose: display_dataset_info(train_info, test_info, args.ticker, args.dataset_id, args.atr_window)

    # --- OPTUNA OPTIMIZATION ---
    t0 = time.time()
    optimized_params, best_score = optimize_vix_multipliers(
        df_bt=df_bt_train,
        vix_col=vix_col,
        open_col=open_col,
        close_col=close_col,
        atr_col=atr_col,
        high_col=high_col,
        low_col=low_col,
        ticker=args.ticker,
        n_trials=args.n_trials,
        tightness_weight=args.tightness_weight,
        use_close_for_range=args.use_close_for_range,
        verbose=args.verbose
    )
    timings['optuna_optimization'] = time.time() - t0

    # --- BACKTEST ---
    t0 = time.time()
    global_stats, regime_stats, bars_analyzed, df_bt_test = run_backtest_with_vix(
        df_bt=df_bt_test,
        open_col=open_col,
        close_col=close_col,
        atr_col=atr_col,
        high_col=high_col,
        low_col=low_col,
        optimized_params=optimized_params,
        use_close_for_range=args.use_close_for_range
    )
    timings['backtest'] = time.time() - t0

    if args.verbose: display_report_with_vix(global_stats, regime_stats, bars_analyzed)

    # --- REALTIME PREDICTION ---
    t0 = time.time()
    realtime_results = display_realtime_prediction(
        df_bt=df_bt_test,
        vix_col=vix_col,
        open_col=open_col,
        close_col=close_col,
        atr_col=atr_col,
        high_col=high_col,
        low_col=low_col,
        ticker=args.ticker,
        optimized_params=optimized_params,
        use_close_for_range=args.use_close_for_range,
        verbose=args.verbose,
    )
    timings['realtime_prediction'] = time.time() - t0

    # --- TIMING SUMMARY ---
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