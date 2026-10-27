"""
Implémentation v6.2 de la Stratégie Triple EMA Scalping / Reversal avec Optimisation Optuna
==================================================================
Améliorations v6 :
- Ajout de l'évaluation Out-of-Sample sur le jeu de test (df_test) avec les meilleurs paramètres trouvés.
- Ajout d'un comparatif Train vs Test pour détecter automatiquement le sur-apprentissage (overfitting).
- Ajout d'une contrainte de densité de signaux (cible par défaut 10%) avec une pénalité forte dans l'objectif.
- Utilisation de Walk-Forward Validation (TimeSeriesSplit) pour garantir la robustesse des paramètres sur plusieurs régimes de marché.

Améliorations Anti-Overfitting (v6.1) :
- Restriction de l'espace de recherche (bornes minimales augmentées) pour éviter le "curve-fitting" sur le bruit.
- Ajout d'une "Pénalité de Complexité" dans l'objectif pour favoriser les canaux plus larges (moins de faux signaux).
- Modification de la fonction objectif pour pénaliser l'instabilité (Maximin) : on pénalise le "pire" fold de marché.

Améliorations v6.2 :
- Sauvegarde automatique des résultats train/test dans un fichier JSON.
- Nom de fichier extrêmement explicite, contenant les informations de l'expérience :
  lookahead, train_ratio, target_density, deadbands, splits, meilleur canal, etc.
- [NOUVEAU] Tous les paramètres de configuration (ticker, lookahead, deadbands, opt_kwargs, etc.)
  sont maintenant gérés via argparse pour une flexibilité totale en ligne de commande.

Améliorations de Performance (Vectorisation) :
- Vectorisation complète du backtest (suppression de la boucle for Python).
- Optimisation du calcul du True Range (ATR).
- Vectorisation de la boucle Walk-Forward dans la fonction objectif Optuna.
"""
import os
import sys
import json
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import argparse
from datetime import datetime
from pathlib import Path
from multiprocessing import freeze_support
from pprint import pprint

# Réduit le verbosity d'Optuna pour ne pas polluer la console
optuna.logging.set_verbosity(optuna.logging.WARNING)

from fetchers.data_factory import factory_load_data
from utils import (
    round_price_for_call_credit_spread,
    round_price_for_put_credit_spread,
    get_next_step, format_duration
)

GLOBAL__default_opt_kwargs = {
        "channel_type": ["keltner", "envelope", "original"], "strict_patterns": [False, True],
        "p_center": [5, 45], "p_high": [5, 45], "p_low": [5, 45],
        "atr_mult_low": [1.0, 4.0], "atr_mult_high": [1.0, 4.0],
        "env_pct_high": [0.01, 0.05], "env_pct_low": [0.01, 0.05],
    }

def calculate_channel_indicators(
        df: pd.DataFrame,
        period: int = 20,
        period_low: int = 20,
        period_high: int = 20,
        channel_type: str = 'keltner',
        atr_period: int = 14,
        atr_multiplier_high: float = 2.0,
        atr_multiplier_low: float = 2.0,
        envelope_pct_high: float = 0.015,
        envelope_pct_low: float = 0.015,
) -> pd.DataFrame:
    df = df.copy()
    df['EMA_Center'] = df['Close'].ewm(span=period, adjust=False).mean()

    if channel_type == 'original':
        df['EMA_Upper'] = df['High'].ewm(span=period_high, adjust=False).mean()
        df['EMA_Lower'] = df['Low'].ewm(span=period_low, adjust=False).mean()

    elif channel_type == 'keltner':
        # Optimisation vectorielle du True Range (évite pd.concat)
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(np.maximum(df['High'] - df['Low'], high_close), low_close)

        df['ATR'] = true_range.ewm(span=atr_period, adjust=False).mean()
        df['EMA_Upper'] = df['EMA_Center'] + (atr_multiplier_high * df['ATR'])
        df['EMA_Lower'] = df['EMA_Center'] - (atr_multiplier_low * df['ATR'])

    elif channel_type == 'envelope':
        df['EMA_Upper'] = df['EMA_Center'] * (1 + envelope_pct_high)
        df['EMA_Lower'] = df['EMA_Center'] * (1 - envelope_pct_low)

    else:
        raise ValueError(f"Type de canal inconnu : {channel_type}.")

    return df


def detect_candlestick_patterns(df: pd.DataFrame, strict_patterns: bool = False) -> pd.DataFrame:
    df = df.copy()

    df['completely_below'] = df['Close'] < df['EMA_Lower']
    df['completely_above'] = df['Close'] > df['EMA_Upper']

    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)

    curr_high = df['High']
    curr_low = df['Low']
    curr_close = df['Close']
    curr_open = df['Open']

    if strict_patterns:
        bullish_inside = (prev_close < prev_open) & (curr_high < prev_high) & (curr_low > prev_low)
    else:
        bullish_inside = (curr_high < prev_high) & (curr_low > prev_low)

    bullish_engulfing = (
            (prev_close < prev_open) & (curr_close > curr_open) &
            (curr_high > prev_high) & (curr_low < prev_low)
    )

    if strict_patterns:
        bearish_inside = (prev_close > prev_open) & (curr_high < prev_high) & (curr_low > prev_low)
    else:
        bearish_inside = (curr_high < prev_high) & (curr_low > prev_low)

    bearish_engulfing = (
            (prev_close > prev_open) & (curr_close < curr_open) &
            (curr_high > prev_high) & (curr_low < prev_low)
    )

    df['buy_signal'] = df['completely_below'] & (bullish_inside | bullish_engulfing)
    df['sell_signal'] = df['completely_above'] & (bearish_inside | bearish_engulfing)

    conditions = [
        df['completely_below'] & bullish_inside,
        df['completely_below'] & bullish_engulfing,
        df['completely_above'] & bearish_inside,
        df['completely_above'] & bearish_engulfing
    ]
    choices = ['Bullish Inside Bar', 'Bullish Engulfing', 'Bearish Inside Bar', 'Bearish Engulfing']
    df['pattern'] = np.select(conditions, choices, default='None')

    return df


def backtest_triple_ema_strategy_for_credit_spread(
        df: pd.DataFrame,
        lookahead_bar_1: int = 1,
        lookahead_bar_2: int = 3,
        put_dead_band: float = 0.,
        call_dead_band: float = 0.
) -> pd.DataFrame:
    """
    VERSION VECTORIZÉE : Élimine la boucle for Python pour un gain de vitesse massif.
    """
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'index' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'index': 'Date'})
    else:
        df = df.reset_index(drop=True)

    df = df.dropna(subset=['EMA_Center', 'buy_signal', 'sell_signal']).copy()

    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'date' in df.columns:
        date_col = 'date'
    elif 'index' in df.columns:
        date_col = 'index'
    else:
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

    assert 0 < lookahead_bar_1 < lookahead_bar_2

    n = len(df)
    valid_n = n - lookahead_bar_2
    if valid_n <= 0:
        return pd.DataFrame()

    close_vals = df['Close'].values
    dates = df[date_col].values
    orig_idx = df['orig_idx'].values
    buy_signals = df['buy_signal'].values
    sell_signals = df['sell_signal'].values

    # Vectorisation des fonctions de rounding si elles ne le sont pas nativement
    v_round_put = np.vectorize(round_price_for_put_credit_spread)
    v_round_call = np.vectorize(round_price_for_call_credit_spread)

    # Création de la matrice des prix futurs (shape: valid_n x nombre_de_shifts)
    shifts = np.arange(lookahead_bar_1, lookahead_bar_2 + 1)
    future_closes = np.zeros((valid_n, len(shifts)))
    future_dates = np.zeros((valid_n, len(shifts)), dtype=dates.dtype)

    for i, shift in enumerate(shifts):
        future_closes[:, i] = close_vals[shift:shift + valid_n]
        future_dates[:, i] = dates[shift:shift + valid_n]

    trades_list = []

    # --- TRAITEMENT DES SIGNAUX D'ACHAT (PUT CREDIT SPREAD) ---
    buy_idx = np.where(buy_signals[:valid_n])[0]
    if len(buy_idx) > 0:
        buy_close = close_vals[buy_idx]
        buy_strikes = v_round_put(buy_close * (1. - put_dead_band))
        buy_future = future_closes[buy_idx]
        buy_success = np.any(buy_future > buy_strikes[:, None], axis=1)

        buy_start_dates = pd.to_datetime(future_dates[buy_idx, 0]).strftime('%Y-%m-%d')
        buy_end_dates = pd.to_datetime(future_dates[buy_idx, -1]).strftime('%Y-%m-%d')

        buy_min_close = np.min(buy_future, axis=1).astype(int)
        buy_max_close = np.max(buy_future, axis=1).astype(int)

        trades_list.append(pd.DataFrame({
            'orig_idx': orig_idx[buy_idx],
            'type': 'PUT_CREDIT_SPREAD',
            'signal_date': dates[buy_idx],
            'exit_date': future_dates[buy_idx, -1],
            'exit_date_slice': buy_start_dates + "::" + buy_end_dates,
            'entry_price': buy_strikes,
            'strike_price': buy_strikes,
            'exit_price': buy_future[:, -1],
            'exit_price_slice': buy_min_close.astype(str) + "::" + buy_max_close.astype(str),
            'result': np.where(buy_success, 'WIN', 'LOSS'),
            'pnl': np.where(buy_success, 1, -1)
        }))

    # --- TRAITEMENT DES SIGNAUX DE VENTE (CALL CREDIT SPREAD) ---
    sell_idx = np.where(sell_signals[:valid_n])[0]
    if len(sell_idx) > 0:
        sell_close = close_vals[sell_idx]
        sell_strikes = v_round_call(sell_close * (1. + call_dead_band))
        sell_future = future_closes[sell_idx]
        sell_success = np.any(sell_future < sell_strikes[:, None], axis=1)

        sell_start_dates = pd.to_datetime(future_dates[sell_idx, 0]).strftime('%Y-%m-%d')
        sell_end_dates = pd.to_datetime(future_dates[sell_idx, -1]).strftime('%Y-%m-%d')

        sell_min_close = np.min(sell_future, axis=1).astype(int)
        sell_max_close = np.max(sell_future, axis=1).astype(int)

        trades_list.append(pd.DataFrame({
            'orig_idx': orig_idx[sell_idx],
            'type': 'CALL_CREDIT_SPREAD',
            'signal_date': dates[sell_idx],
            'exit_date': future_dates[sell_idx, -1],
            'exit_date_slice': sell_start_dates + "::" + sell_end_dates,
            'entry_price': sell_strikes,
            'strike_price': sell_strikes,
            'exit_price': sell_future[:, -1],
            'exit_price_slice': sell_min_close.astype(str) + "::" + sell_max_close.astype(str),
            'result': np.where(sell_success, 'WIN', 'LOSS'),
            'pnl': np.where(sell_success, 1, -1)
        }))

    if not trades_list:
        return pd.DataFrame()

    return pd.concat(trades_list, ignore_index=True)


def calculate_credit_spread_win_rates(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {k: 0 for k in [
            'put_win_rate', 'put_trades', 'put_wins',
            'call_win_rate', 'call_trades', 'call_wins',
            'total_win_rate', 'total_trades', 'total_wins'
        ]}

    types = trades_df['type'].values
    pnls = trades_df['pnl'].values

    put_mask = types == 'PUT_CREDIT_SPREAD'
    call_mask = types == 'CALL_CREDIT_SPREAD'

    put_trades = put_mask.sum()
    call_trades = call_mask.sum()
    total_trades = len(trades_df)

    put_wins = int((pnls[put_mask] > 0).sum())
    call_wins = int((pnls[call_mask] > 0).sum())
    total_wins = int((pnls > 0).sum())

    return {
        'put_win_rate': round((put_wins / put_trades * 100.0) if put_trades > 0 else 0.0, 2),
        'put_trades': int(put_trades),
        'put_wins': put_wins,
        'call_win_rate': round((call_wins / call_trades * 100.0) if call_trades > 0 else 0.0, 2),
        'call_trades': int(call_trades),
        'call_wins': call_wins,
        'total_win_rate': round((total_wins / total_trades * 100.0) if total_trades > 0 else 0.0, 2),
        'total_trades': int(total_trades),
        'total_wins': total_wins
    }


def sanitize_for_filename(value) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in str(value)).strip("-")


def compact_float_for_filename(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", "-")


def to_serializable_params(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        if isinstance(value, (bool, np.bool_)):
            clean[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            clean[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            clean[key] = float(value)
        else:
            clean[key] = value
    return clean


def compute_signal_counts(df: pd.DataFrame) -> dict:
    df_clean = df.dropna(subset=['EMA_Center', 'buy_signal', 'sell_signal'])
    if df_clean.empty:
        return {"bars_evaluated": 0, "buy_signals": 0, "sell_signals": 0, "total_signals": 0}

    buy_signals = int(df_clean['buy_signal'].sum())
    sell_signals = int(df_clean['sell_signal'].sum())

    return {
        "bars_evaluated": int(len(df_clean)),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "total_signals": buy_signals + sell_signals,
    }


def build_results_filename(
        results_dir: str, experiment_name: str, ticker: str, optimize_metric: str,
        lookahead_bar_1: int, lookahead_bar_2: int, train_ratio: float, target_signal_density: float,
        put_dead_band: float, call_dead_band: float, n_splits: int, best_params: dict, test_win_rate: float,
) -> Path:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticker_clean = sanitize_for_filename(ticker.replace("^", ""))

    density_label = compact_float_for_filename(target_signal_density * 100.0, 2)
    put_db_label = compact_float_for_filename(put_dead_band, 3)
    call_db_label = compact_float_for_filename(call_dead_band, 3)

    channel_type = str(best_params.get("channel_type", "unknown"))
    p_center = best_params.get("p_center", "NA")
    extra_params = f"{sanitize_for_filename(channel_type)}_pcenter{sanitize_for_filename(p_center)}"

    if channel_type == "envelope":
        env_pct_high = float(best_params.get("env_pct_high", 0.0))
        env_pct_low = float(best_params.get("env_pct_low", 0.0))
        extra_params += f"_envH{int(round(env_pct_high * 1000))}_envL{int(round(env_pct_low * 1000))}"
    elif channel_type == "keltner":
        atr_mult_high = float(best_params.get("atr_mult_high", 0.0))
        atr_mult_low = float(best_params.get("atr_mult_low", 0.0))
        extra_params += f"_atrH{int(round(atr_mult_high * 10))}_atrL{int(round(atr_mult_low * 10))}"
    elif channel_type == "original":
        p_high = best_params.get("p_high", "NA")
        p_low = best_params.get("p_low", "NA")
        extra_params += f"_pH{sanitize_for_filename(p_high)}_pL{sanitize_for_filename(p_low)}"

    strict_label = "strict" if bool(best_params.get("strict_patterns", False)) else "flex"

    parts = [run_timestamp]
    if experiment_name: parts.append(sanitize_for_filename(experiment_name))

    parts.extend([
        ticker_clean, sanitize_for_filename(optimize_metric),
        f"lb{lookahead_bar_1}-{lookahead_bar_2}", f"train{int(round(train_ratio * 100))}pct",
        f"dens{density_label}pct", f"putDB{put_db_label}", f"callDB{call_db_label}",
        f"splits{n_splits}", extra_params, strict_label, f"__twr_{test_win_rate / 100.:0.6f}__"
    ])

    filename = "_".join(parts) + ".json"
    return output_dir / filename


# ==========================================================
# OPTUNA OBJECTIVE FUNCTION (OPTIMISÉE)
# ==========================================================
def create_objective(df_data, lb1, lb2, metric, n_splits, target_density, put_dead_band, call_dead_band, opt_kwargs):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # Pré-calcul des splits pour éviter de les refaire à chaque trial
    split_indices = list(tscv.split(df_data))
    assert "p_center" in opt_kwargs
    assert "channel_type" in opt_kwargs
    assert 'atr_mult_low' in opt_kwargs and 'atr_mult_high' in opt_kwargs
    assert 'env_pct_high' in opt_kwargs and 'env_pct_low' in opt_kwargs
    assert 'p_high' in opt_kwargs and 'p_low' in opt_kwargs
    def objective(trial):
        channel_type = trial.suggest_categorical('channel_type', opt_kwargs.get('channel_type'))
        strict_patterns = trial.suggest_categorical('strict_patterns', opt_kwargs.get('strict_patterns', [False, True]))
        p_center = trial.suggest_int('p_center', opt_kwargs.get('p_center')[0], opt_kwargs.get('p_center')[1])

        atr_mult_low = atr_mult_high = env_pct_high = env_pct_low = p_high = p_low = 0

        if channel_type == 'keltner':
            atr_mult_low = trial.suggest_float('atr_mult_low', opt_kwargs.get('atr_mult_low')[0], opt_kwargs.get('atr_mult_low')[1], step=0.1)
            atr_mult_high = trial.suggest_float('atr_mult_high', opt_kwargs.get('atr_mult_high')[0], opt_kwargs.get('atr_mult_high')[1], step=0.1)
        if channel_type == 'envelope':
            env_pct_high = trial.suggest_float('env_pct_high', opt_kwargs.get('env_pct_high')[0], opt_kwargs.get('env_pct_high')[1], step=0.001)
            env_pct_low = trial.suggest_float('env_pct_low', opt_kwargs.get('env_pct_low')[0], opt_kwargs.get('env_pct_low')[1], step=0.001)
        if channel_type == 'original':
            p_high = trial.suggest_int('p_high', opt_kwargs.get("p_high")[0], opt_kwargs.get("p_high")[1])
            p_low = trial.suggest_int('p_low', opt_kwargs.get("p_low")[0], opt_kwargs.get("p_low")[1])

        df_temp = calculate_channel_indicators(
            df=df_data, period=p_center, period_low=p_low, period_high=p_high, channel_type=channel_type,
            atr_multiplier_high=atr_mult_high, atr_multiplier_low=atr_mult_low,
            envelope_pct_high=env_pct_high, envelope_pct_low=env_pct_low,
        )
        df_temp = detect_candlestick_patterns(df_temp, strict_patterns=strict_patterns)
        df_temp = df_temp.dropna(subset=['EMA_Center', 'buy_signal', 'sell_signal']).copy()

        total_bars = len(df_temp)
        if total_bars == 0: return -100.0

        buy_signals = df_temp['buy_signal'].sum()
        sell_signals = df_temp['sell_signal'].sum()

        if metric == 'put_win_rate':
            current_density = buy_signals / total_bars
        elif metric == 'call_win_rate':
            current_density = sell_signals / total_bars
        else:
            current_density = (buy_signals + sell_signals) / total_bars

        density_penalty = abs(current_density - target_density) * 100.
        assert 0 <= density_penalty <= 100

        trades_df = backtest_triple_ema_strategy_for_credit_spread(
            df_temp, lookahead_bar_1=lb1, lookahead_bar_2=lb2,
            put_dead_band=put_dead_band, call_dead_band=call_dead_band,
        )

        if trades_df.empty: return -100.0

        stats = calculate_credit_spread_win_rates(trades_df)
        if stats['total_trades'] < 10: return -100.0
        if metric == 'put_win_rate' and stats['put_trades'] < 5: return -100.0
        if metric == 'call_win_rate' and stats['call_trades'] < 5: return -100.0

        # Walk-Forward Validation Vectorisée
        fold_scores = []
        orig_idx_vals = trades_df['orig_idx'].values
        pnl_vals = trades_df['pnl'].values
        type_vals = trades_df['type'].values

        for fold, (train_index, test_index) in enumerate(split_indices):
            mask = np.isin(orig_idx_vals, test_index)
            if not np.any(mask):
                fold_scores.append(0.0)
                continue

            fold_pnl = pnl_vals[mask]
            fold_type = type_vals[mask]

            if metric == 'put_win_rate':
                put_mask = fold_type == 'PUT_CREDIT_SPREAD'
                f_trades = put_mask.sum()
                fold_scores.append((fold_pnl[put_mask] > 0).sum() / f_trades * 100.0 if f_trades >= 1 else 0.0)
            elif metric == 'call_win_rate':
                call_mask = fold_type == 'CALL_CREDIT_SPREAD'
                f_trades = call_mask.sum()
                fold_scores.append((fold_pnl[call_mask] > 0).sum() / f_trades * 100.0 if f_trades >= 1 else 0.0)
            else:
                f_trades = len(fold_pnl)
                fold_scores.append((fold_pnl > 0).sum() / f_trades * 100.0 if f_trades >= 2 else 0.0)

        mean_score = np.mean(fold_scores)
        assert 0 <= mean_score <= 100
        std_score = np.mean(fold_scores)
        min_score = np.min(fold_scores)
        max_score = np.max(fold_scores)
        instability_penalty = (max_score - min_score)
        assert 0 <= instability_penalty <= 100
        worst_case_penalty = max(0, 0. - min_score)  # Disabled.
        assert 0 <= worst_case_penalty <= 100
        return mean_score - std_score - instability_penalty - worst_case_penalty - density_penalty

    return objective


# --- CONFIGURATION DU PARSER D'ARGUMENTS ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration de l'optimisation du modèle.")
    parser.add_argument('--ticker', type=str, default="^GSPC")
    parser.add_argument('--dataset-id', type=str, default="day")
    parser.add_argument('--lookahead-bar-1', type=int, default=16)
    parser.add_argument('--lookahead-bar-2', type=int, default=20)
    parser.add_argument('--n-splits', type=int, default=12)
    parser.add_argument('--train-ratio', type=float, default=0.90)
    parser.add_argument('--put-dead-band', type=float, default=0.03)
    parser.add_argument('--call-dead-band', type=float, default=0.03)
    parser.add_argument('--optimize-metric', type=str, default='total_win_rate', choices=['total_win_rate', 'put_win_rate', 'call_win_rate'])
    parser.add_argument('--n-trials', type=int, default=999999)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--target-signal-density', type=float, default=0.01)
    parser.add_argument('--opt-kwargs', type=json.loads, default=json.dumps(GLOBAL__default_opt_kwargs))
    parser.add_argument('--results-dir', type=str, default="triple_ema")
    parser.add_argument('--experiment-name', type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    freeze_support()
    args = parse_arguments()
    # 1. Capturer les variables d'environnement commençant par Q__
    quant_env_vars = {k: v for k, v in os.environ.items() if k.startswith("Q__")}
    # 2. Construire la string des variables au format PowerShell ($env:VAR=valeur;)
    env_string = "".join([f"$env:{k}={v}; " for k, v in quant_env_vars.items()])
    # 3. Combiner le tout avec la commande Python
    command_line = f"{env_string}python " + " ".join(sys.argv)

    ticker = args.ticker
    dataset_id = args.dataset_id
    lookahead_bar_1 = int(args.lookahead_bar_1)
    lookahead_bar_2 = int(args.lookahead_bar_2)
    assert lookahead_bar_2 > lookahead_bar_1
    n_splits = args.n_splits
    train_ratio = args.train_ratio
    put_dead_band = args.put_dead_band
    call_dead_band = args.call_dead_band

    opt_kwargs_raw = args.opt_kwargs
    opt_kwargs = {
        "channel_type": opt_kwargs_raw.get("channel_type", GLOBAL__default_opt_kwargs["channel_type"]),
        "strict_patterns": opt_kwargs_raw.get("strict_patterns", GLOBAL__default_opt_kwargs["strict_patterns"]),
        "p_center": tuple(opt_kwargs_raw.get("p_center", GLOBAL__default_opt_kwargs["p_center"])),
        "p_high": tuple(opt_kwargs_raw.get("p_high", GLOBAL__default_opt_kwargs["p_high"])),
        "p_low": tuple(opt_kwargs_raw.get("p_low", GLOBAL__default_opt_kwargs["p_low"])),
        "atr_mult_low": tuple(opt_kwargs_raw.get("atr_mult_low", GLOBAL__default_opt_kwargs["atr_mult_low"])),
        "atr_mult_high": tuple(opt_kwargs_raw.get("atr_mult_high", GLOBAL__default_opt_kwargs["atr_mult_high"])),
        "env_pct_high": tuple(opt_kwargs_raw.get("env_pct_high", GLOBAL__default_opt_kwargs["env_pct_high"])),
        "env_pct_low": tuple(opt_kwargs_raw.get("env_pct_low", GLOBAL__default_opt_kwargs["env_pct_low"])),
    }

    optimize_metric = args.optimize_metric
    n_trials = args.n_trials
    timeout = args.timeout
    target_signal_density = args.target_signal_density
    results_dir = args.results_dir
    experiment_name = args.experiment_name

    df_mom = factory_load_data(_dataset_id=dataset_id, _ticker=ticker, _args={})
    if isinstance(df_mom.columns, pd.MultiIndex):
        df_mom.columns = df_mom.columns.get_level_values(0)

    df_mom['orig_idx'] = np.arange(len(df_mom))
    is_datetime_index = isinstance(df_mom.index, pd.DatetimeIndex)
    assert is_datetime_index
    split_idx = int(len(df_mom) * train_ratio)
    df_train_and_val = df_mom.iloc[:split_idx].copy()
    df_test = df_mom.iloc[split_idx:].copy()

    print(f"🚀 Lancement de l'optimisation Optuna pour maximiser : {optimize_metric}")
    print(f"🎯 Cible de densité de signaux : {target_signal_density * 100}%   Range du lookahead: {lookahead_bar_1}::{lookahead_bar_2}")
    print(f"🔧 Nombre d'essais (trials) : {n_trials}   Délai maximal : {format_duration(timeout)}")

    train_start_date = df_train_and_val.index[0].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_train_and_val.index[0])
    train_end_date = df_train_and_val.index[-1].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_train_and_val.index[-1])
    print(f"📊 Train Set ({len(df_train_and_val)} bars) - {train_start_date}::{train_end_date}\n")

    test_start_date = df_test.index[0].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_test.index[0])
    test_end_date = df_test.index[-1].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_test.index[-1])
    print(f"📊 Test Set ({len(df_test)} bars) - {test_start_date}::{test_end_date}\n")
    for key, val in opt_kwargs.items(): print(f"  :: {key:<16} -> {val}")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        create_objective(
            df_data=df_train_and_val, lb1=lookahead_bar_1, lb2=lookahead_bar_2, metric=optimize_metric,
            n_splits=n_splits, target_density=target_signal_density,
            put_dead_band=put_dead_band, call_dead_band=call_dead_band, opt_kwargs=opt_kwargs,
        ),
        n_trials=n_trials, timeout=timeout, show_progress_bar=True,
    )

    best_params = study.best_params
    best_value = study.best_value
    best_trial_number = study.best_trial.number if study.best_trial is not None else None

    print(f"\n✅ Optimisation terminée !")
    print(f"🏆 Meilleur score robuste : {best_value:.8f}")
    print(f"🔧 Meilleurs paramètres trouvés : {best_params}\n")

    # ==========================================================
    # BACKTEST FINAL AVEC LES MEILLEURS PARAMÈTRES (TRAIN SET)
    # ==========================================================
    df_train_and_val = calculate_channel_indicators(
        df=df_train_and_val, period=best_params["p_center"],
        period_low=best_params.get("p_low", 0), period_high=best_params.get("p_high", 0),
        channel_type=best_params['channel_type'],
        atr_multiplier_high=best_params.get("atr_mult_high", 0), atr_multiplier_low=best_params.get("atr_mult_low", 0),
        envelope_pct_high=best_params.get("env_pct_high", 0), envelope_pct_low=best_params.get("env_pct_low", 0),
    )
    df_train_and_val = detect_candlestick_patterns(df_train_and_val, strict_patterns=best_params['strict_patterns'])

    train_signal_counts = compute_signal_counts(df_train_and_val)
    credit_trades_df_train_and_val = backtest_triple_ema_strategy_for_credit_spread(
        df_train_and_val, lookahead_bar_1=lookahead_bar_1, lookahead_bar_2=lookahead_bar_2,
        put_dead_band=put_dead_band, call_dead_band=call_dead_band,
    )
    train_overall_stats = calculate_credit_spread_win_rates(credit_trades_df_train_and_val)

    if optimize_metric == 'put_win_rate':
        display_trades_train = credit_trades_df_train_and_val[credit_trades_df_train_and_val['type'] == 'PUT_CREDIT_SPREAD']
    elif optimize_metric == 'call_win_rate':
        display_trades_train = credit_trades_df_train_and_val[credit_trades_df_train_and_val['type'] == 'CALL_CREDIT_SPREAD']
    else:
        display_trades_train = credit_trades_df_train_and_val

    if credit_trades_df_train_and_val.empty:
        print("⚠️ Aucun trade généré sur l'ensemble du dataset d'entraînement avec ces paramètres.")
        train_density_obtained = 0
    else:
        print(f"--- [Train] Résumé des 10 Derniers Credit Spreads (Lookahead: [{lookahead_bar_1}, {lookahead_bar_2}]) ---")
        if not display_trades_train.empty:
            print(display_trades_train[['type', 'signal_date', 'strike_price', 'exit_price_slice', 'exit_date_slice', 'result']].tail(10))
        else:
            print("Aucun trade de ce type spécifique à afficher.")

        print(f"\n{'=' * 25} RÉSUMÉ GLOBAL (TRAIN) {'=' * 25}")

        if optimize_metric == 'put_win_rate':
            print(f"🔵 PUT Credit Spread (BUY Setup)  : {train_overall_stats['put_wins']}/{train_overall_stats['put_trades']} gagnants -> Win Rate = {train_overall_stats['put_win_rate']}%")
            train_density_obtained = train_overall_stats['put_trades'] / len(df_train_and_val)
        elif optimize_metric == 'call_win_rate':
            print(f"🔴 CALL Credit Spread (SELL Setup) : {train_overall_stats['call_wins']}/{train_overall_stats['call_trades']} gagnants -> Win Rate = {train_overall_stats['call_win_rate']}%")
            train_density_obtained = train_overall_stats['call_trades'] / len(df_train_and_val)
        elif optimize_metric == 'total_win_rate':
            print(f"📊 GLOBAL                         : {train_overall_stats['total_wins']}/{train_overall_stats['total_trades']} gagnants -> Win Rate Total = {train_overall_stats['total_win_rate']}%")
            train_density_obtained = train_overall_stats['total_trades'] / len(df_train_and_val)

    # ==========================================================
    # ÉVALUATION OUT-OF-SAMPLE (TEST SET)
    # ==========================================================
    print(f"\n{'=' * 25} ÉVALUATION SUR LE JEU DE TEST (OUT-OF-SAMPLE) {'=' * 25}")
    print(f"📊 Test Set ({len(df_test)} bars) - {test_start_date}::{test_end_date}\n")
    df_test = calculate_channel_indicators(
        df=df_test, period=best_params['p_center'],
        period_low=best_params.get("p_low", 0), period_high=best_params.get("p_high", 0),
        channel_type=best_params['channel_type'],
        atr_multiplier_high=best_params.get("atr_mult_high", 0), atr_multiplier_low=best_params.get("atr_mult_low", 0),
        envelope_pct_high=best_params.get("env_pct_high", 0), envelope_pct_low=best_params.get("env_pct_low", 0),
    )
    df_test = detect_candlestick_patterns(df_test, strict_patterns=best_params['strict_patterns'])
    test_signal_counts = compute_signal_counts(df_test)

    if optimize_metric == 'total_win_rate':
        print(f"🔍 Signaux BUY/SELL détectés sur TEST -> BUY: {test_signal_counts['buy_signals']} | SELL: {test_signal_counts['sell_signals']}")
    elif optimize_metric == 'call_win_rate':
        print(f"🔍 Signaux SELL détectés sur TEST -> {test_signal_counts['sell_signals']}")
    elif optimize_metric == 'put_win_rate':
        print(f"🔍 Signaux BUY détectés sur TEST -> {test_signal_counts['buy_signals']}")

    credit_trades_df_test = backtest_triple_ema_strategy_for_credit_spread(
        df_test, lookahead_bar_1=lookahead_bar_1, lookahead_bar_2=lookahead_bar_2,
        put_dead_band=put_dead_band, call_dead_band=call_dead_band,
    )
    test_stats = calculate_credit_spread_win_rates(credit_trades_df_test)

    if optimize_metric == 'put_win_rate':
        display_trades_test = credit_trades_df_test[credit_trades_df_test['type'] == 'PUT_CREDIT_SPREAD']
    elif optimize_metric == 'call_win_rate':
        display_trades_test = credit_trades_df_test[credit_trades_df_test['type'] == 'CALL_CREDIT_SPREAD']
    else:
        display_trades_test = credit_trades_df_test

    if credit_trades_df_test.empty:
        print("⚠️ Aucun trade généré sur le jeu de test.")
    else:
        if optimize_metric == 'put_win_rate':
            test_density_obtained = test_stats['put_trades'] / len(df_test)
            print(f"🔵 PUT Credit Spread (BUY Setup) , (density of {test_density_obtained:.2%})  : {test_stats['put_wins']}/{test_stats['put_trades']} gagnants -> Win Rate = {test_stats['put_win_rate']}%")
        elif optimize_metric == 'call_win_rate':
            test_density_obtained = test_stats['call_trades'] / len(df_test)
            print(f"🔴 CALL Credit Spread (SELL Setup) , (density of {test_density_obtained:.2%})  : {test_stats['call_wins']}/{test_stats['call_trades']} gagnants -> Win Rate = {test_stats['call_win_rate']}%")
        elif optimize_metric == 'total_win_rate':
            test_density_obtained = test_stats['total_trades'] / len(df_test)
            print(f"📊 GLOBAL (CALL+PUT) , (density of {test_density_obtained:.2%})  : {test_stats['total_wins']}/{test_stats['total_trades']} gagnants -> Win Rate Total = {test_stats['total_win_rate']}%")

        print(f"\n--- Derniers trades sur le jeu de TEST ---")
        if not display_trades_test.empty:
            print(display_trades_test[['type', 'signal_date', 'strike_price', 'exit_price_slice', 'result']].tail(5))
        else:
            print("Aucun trade de ce type spécifique à afficher.")

    # ==========================================================
    # COMPARAISON FINALE (TRAIN vs TEST)
    # ==========================================================
    if optimize_metric == 'put_win_rate':
        train_wr, test_wr = train_overall_stats['put_win_rate'], test_stats['put_win_rate']
        train_trades_metric, test_trades_metric = train_overall_stats['put_trades'], test_stats['put_trades']
        train_wins_metric, test_wins_metric = train_overall_stats['put_wins'], test_stats['put_wins']
        metric_label = "PUT Win Rate"
    elif optimize_metric == 'call_win_rate':
        train_wr, test_wr = train_overall_stats['call_win_rate'], test_stats['call_win_rate']
        train_trades_metric, test_trades_metric = train_overall_stats['call_trades'], test_stats['call_trades']
        train_wins_metric, test_wins_metric = train_overall_stats['call_wins'], test_stats['call_wins']
        metric_label = "CALL Win Rate"
    else:
        train_wr, test_wr = train_overall_stats['total_win_rate'], test_stats['total_win_rate']
        train_trades_metric, test_trades_metric = train_overall_stats['total_trades'], test_stats['total_trades']
        train_wins_metric, test_wins_metric = train_overall_stats['total_wins'], test_stats['total_wins']
        metric_label = "GLOBAL (PUT+CALL) Win Rate"

    print(f"   🏋️ Train {metric_label} : {train_wr}%   (density {train_density_obtained:.2%})")
    print(f"   🚀 Test {metric_label}  : {test_wr}%   (density {test_density_obtained:.2%})")

    diff = train_wr - test_wr

    if diff > 15:
        comparison_status = "WARNING_OVERFITTING"
        print(f"   ⚠️ Écart de {diff:.1f}% : Attention, probable sur-apprentissage (overfitting) sur les données d'entraînement !")
        print(f"      Le modèle a mémorisé le passé mais peine à généraliser sur des données récentes.")
    elif diff < -5:
        comparison_status = "TEST_BETTER"
        print(f"   🌟 Écart de {abs(diff):.1f}% : La stratégie performe étonnamment mieux sur les données récentes (Test) !")
    else:
        comparison_status = "ROBUST"
        print(f"   ✅ Écart raisonnable de {diff:.1f}% : Le modèle généralise bien aux nouvelles données. La stratégie est robuste.")

    # ==========================================================
    # SAUVEGARDE DES RÉSULTATS
    # ==========================================================
    results_filename = build_results_filename(
        results_dir=results_dir, experiment_name=experiment_name, ticker=ticker, optimize_metric=optimize_metric,
        lookahead_bar_1=lookahead_bar_1, lookahead_bar_2=lookahead_bar_2, train_ratio=train_ratio,
        target_signal_density=target_signal_density, put_dead_band=put_dead_band, call_dead_band=call_dead_band,
        n_splits=n_splits, best_params=best_params, test_win_rate=test_wr,
    )

    results_payload = {
        "command_line": command_line, "generated_at": datetime.now().isoformat(),
        "results_file": str(results_filename), "experiment_name": experiment_name,
        "dataset_id": dataset_id, "ticker": ticker, "cli_args": vars(args),
        "experience": {
            "optimize_metric": optimize_metric, "lookahead_bar_1": lookahead_bar_1, "lookahead_bar_2": lookahead_bar_2,
            "train_ratio": train_ratio, "n_splits": n_splits, "put_dead_band": put_dead_band, "call_dead_band": call_dead_band,
            "target_signal_density": target_signal_density, "n_trials": n_trials, "timeout": timeout, "opt_kwargs": opt_kwargs,
        },
        "data_split": {
            "train_bars": int(len(df_train_and_val)), "test_bars": int(len(df_test)),
            "train_start_date": train_start_date, "train_end_date": train_end_date,
            "test_start_date": test_start_date, "test_end_date": test_end_date,
        },
        "optuna": {
            "best_score": float(best_value), "best_trial_number": best_trial_number,
            "best_params": to_serializable_params(best_params),
        },
        "train": {
            "signal_counts": train_signal_counts, "stats": train_overall_stats,
            "trades_count": int(len(credit_trades_df_train_and_val)),
            "last_trades": display_trades_train.tail(10).to_dict(orient="records") if not display_trades_train.empty else [],
        },
        "test": {
            "signal_counts": compute_signal_counts(df_test), "stats": test_stats,
            "trades_count": int(len(credit_trades_df_test)),
            "last_trades": display_trades_test.tail(10).to_dict(orient="records") if not display_trades_test.empty else [],
        },
        "comparison": {
            "metric_label": metric_label, "train_win_rate": train_wr, "test_win_rate": test_wr,
            "train_trades": train_trades_metric, "test_trades": test_trades_metric,
            "train_wins": train_wins_metric, "test_wins": test_wins_metric,
            "difference_train_minus_test": round(diff, 2), "status": comparison_status,
        },
    }

    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4, ensure_ascii=False, default=str)

    print(f"\n💾 Résultats sauvegardés dans : {results_filename}")