"""
Implémentation v6 de la Stratégie Triple EMA Scalping / Reversal avec Optimisation Optuna
==================================================================
Améliorations v6 :
- Ajout de l'évaluation Out-of-Sample sur le jeu de test (df_test) avec les meilleurs paramètres trouvés.
- Ajout d'un comparatif Train vs Test pour détecter automatiquement le sur-apprentissage (overfitting).
- Ajout d'une contrainte de densité de signaux (cible par défaut 10%) avec une pénalité forte dans l'objectif.
- Utilisation de Walk-Forward Validation (TimeSeriesSplit) pour garantir la robustesse des paramètres sur plusieurs régimes de marché.
"""
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Réduit le verbosity d'Optuna pour ne pas polluer la console
optuna.logging.set_verbosity(optuna.logging.WARNING)

from fetchers.data_factory import factory_load_data
from utils import round_price_for_call_credit_spread, round_price_for_put_credit_spread, get_next_step
from multiprocessing import freeze_support

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
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
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


def backtest_triple_ema_strategy_for_credit_spread(df: pd.DataFrame, lookahead_bar_1: int = 1, lookahead_bar_2: int = 3) -> pd.DataFrame:
    # CORRECTION : Conserver l'index temporel s'il existe pour ne pas perdre les dates
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'index' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'index': 'Date'})
    else:
        df = df.reset_index(drop=True)

    df = df.dropna(subset=['EMA_Center', 'buy_signal', 'sell_signal']).copy()

    # CORRECTION : Meilleure détection de la colonne de date
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'date' in df.columns:
        date_col = 'date'
    elif 'index' in df.columns:
        date_col = 'index'
    else:
        # Recherche d'une colonne datetime par défaut avant de prendre la première colonne
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

    trades = []

    for i in range(len(df) - lookahead_bar_2):
        row = df.iloc[i]
        future_slice = df.iloc[i + lookahead_bar_1: i + lookahead_bar_2 + 1]
        last_future_row = future_slice.iloc[-1]

        if row['buy_signal']:
            strike_price = round_price_for_put_credit_spread(row['Close'])
            success = (future_slice['Close'] > strike_price).any()

            trades.append({
                'orig_idx': row['orig_idx'],
                'type': 'PUT_CREDIT_SPREAD',
                'signal_date': row[date_col],
                'exit_date': last_future_row[date_col],
               'exit_date_slice': f"{pd.Timestamp(future_slice[date_col].values.min()).strftime('%Y-%m-%d')}::{pd.Timestamp(future_slice[date_col].values.max()).strftime('%Y-%m-%d')}",
                'entry_price': strike_price,
                'strike_price': strike_price,
                'exit_price': last_future_row['Close'],
                'exit_price_slice': f"{future_slice['Close'].values.min().astype(int)}::{future_slice['Close'].values.max().astype(int)}",
                'result': 'WIN' if success else 'LOSS',
                'pnl': 1 if success else -1
            })

        elif row['sell_signal']:
            strike_price = round_price_for_call_credit_spread(row['Close'])
            success = (future_slice['Close'] < strike_price).any()

            trades.append({
                'orig_idx': row['orig_idx'],
                'type': 'CALL_CREDIT_SPREAD',
                'signal_date': row[date_col],
                'exit_date': last_future_row[date_col],
                'exit_date_slice': f"{pd.Timestamp(future_slice[date_col].values.min()).strftime('%Y-%m-%d')}::{pd.Timestamp(future_slice[date_col].values.max()).strftime('%Y-%m-%d')}",
                'entry_price': strike_price,
                'strike_price': strike_price,
                'exit_price': last_future_row['Close'],
                'exit_price_slice': f"{future_slice['Close'].values.min().astype(int)}::{future_slice['Close'].values.max().astype(int)}",
                'result': 'WIN' if success else 'LOSS',
                'pnl': 1 if success else -1
            })

    return pd.DataFrame(trades)


def calculate_credit_spread_win_rates(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {k: 0 for k in ['put_win_rate', 'put_trades', 'put_wins', 'call_win_rate',
                               'call_trades', 'call_wins', 'total_win_rate', 'total_trades', 'total_wins']}

    put_df = trades_df[trades_df['type'] == 'PUT_CREDIT_SPREAD']
    call_df = trades_df[trades_df['type'] == 'CALL_CREDIT_SPREAD']

    put_wins = int((put_df['pnl'] > 0).sum())
    call_wins = int((call_df['pnl'] > 0).sum())
    total_wins = int((trades_df['pnl'] > 0).sum())

    return {
        'put_win_rate': round((put_wins / len(put_df) * 100.0) if len(put_df) > 0 else 0.0, 2),
        'put_trades': len(put_df), 'put_wins': put_wins,
        'call_win_rate': round((call_wins / len(call_df) * 100.0) if len(call_df) > 0 else 0.0, 2),
        'call_trades': len(call_df), 'call_wins': call_wins,
        'total_win_rate': round((total_wins / len(trades_df) * 100.0) if len(trades_df) > 0 else 0.0, 2),
        'total_trades': len(trades_df), 'total_wins': total_wins
    }


# ==========================================================
# OPTUNA OBJECTIVE FUNCTION
# ==========================================================
def create_objective(df_data, lb1, lb2, metric, n_splits, target_density=0.1):
    """
    Crée la fonction objectif pour Optuna.
    Intègre une Walk-Forward Validation (TimeSeriesSplit) pour garantir la robustesse.
    Ajoute une contrainte de densité de signaux avec une pénalité forte.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        # Suggestion des paramètres à optimiser
        channel_type = trial.suggest_categorical('channel_type', ['original', 'keltner', 'envelope']) # ['original', 'keltner', 'envelope']
        strict_patterns = trial.suggest_categorical('strict_patterns', [False, True])  # , [False, True])
        p_center = trial.suggest_int('p_center', 15, 25)

        atr_mult_low = atr_mult_high = env_pct_high = env_pct_low = p_high = p_low = 0
        if channel_type == 'keltner':
            atr_mult_low = trial.suggest_float('atr_mult_low', 0.5, 4.0, step=0.1)
            atr_mult_high = trial.suggest_float('atr_mult_high', 0.5, 4.0, step=0.1)
        if channel_type == 'envelope':
            env_pct_high = trial.suggest_float('env_pct_high', 0.005, 0.05)
            env_pct_low = trial.suggest_float('env_pct_low', 0.005, 0.05)
        if channel_type == 'original':
            p_high = trial.suggest_int('p_high', 15, 25)
            p_low = trial.suggest_int('p_low', 15, 25)

        # Calcul des indicateurs avec les paramètres du trial (L'algo est causal, pas de fuite de données)
        df_temp = calculate_channel_indicators(
            df=df_data,
            period=p_center, period_low=p_low, period_high=p_high,
            channel_type=channel_type,
            atr_multiplier_high=atr_mult_high, atr_multiplier_low=atr_mult_low,
            envelope_pct_high=env_pct_high, envelope_pct_low=env_pct_low,
        )

        df_temp = detect_candlestick_patterns(df_temp, strict_patterns=strict_patterns)

        # Nettoyage des NaN avant le calcul de la densité et du backtest
        df_temp = df_temp.dropna(subset=['EMA_Center', 'buy_signal', 'sell_signal']).copy()

        # --- CALCUL ET PÉNALITÉ DE DENSITÉ ---
        total_bars = len(df_temp)
        total_signals = df_temp['buy_signal'].sum() + df_temp['sell_signal'].sum()
        current_density = total_signals / total_bars if total_bars > 0 else 0.0

        if metric == 'put_win_rate':
            total_signals = df_temp['buy_signal'].sum()
            current_density = total_signals / total_bars if total_bars > 0 else 0.0
        elif metric == 'call_win_rate':
            total_signals = df_temp['sell_signal'].sum()
            current_density = total_signals / total_bars if total_bars > 0 else 0.0

        # Pénalité forte pour s'écarter de la densité cible
        density_penalty = abs(current_density - target_density) * 250.0
        # ---------------------------------------

        # Génération de tous les trades sur l'ensemble du dataset d'entraînement
        trades_df = backtest_triple_ema_strategy_for_credit_spread(
            df_temp,
            lookahead_bar_1=lb1,
            lookahead_bar_2=lb2
        )

        # Pénalité globale si pas assez de trades
        total_stats = calculate_credit_spread_win_rates(trades_df)
        if total_stats['total_trades'] < 10:
            return 0.0

        if metric == 'put_win_rate' and total_stats['put_trades'] < 5:
            return 0.0
        if metric == 'call_win_rate' and total_stats['call_trades'] < 5:
            return 0.0

        # ==========================================================
        # WALK-FORWARD VALIDATION (TimeSeriesSplit)
        # ==========================================================
        # Au lieu d'évaluer sur tout le bloc d'un coup (ce qui pourrait favoriser
        # un seul régime de marché), on évalue uniquement sur les parties "TEST"
        # de chaque fold. Cela force la stratégie à être robuste dans le temps.
        fold_scores = []
        for fold, (train_index, test_index) in enumerate(tscv.split(df_data)):
            # On filtre les trades générés pour ne garder que ceux de la période de test du fold
            # Note: orig_idx correspond bien aux index générés par tscv.split car df_data
            # a été découpé avec iloc et orig_idx a été créé avec np.arange.
            fold_trades = trades_df[trades_df['orig_idx'].isin(test_index)]

            stats = calculate_credit_spread_win_rates(fold_trades)

            # Pénalité si pas assez de trades dans ce fold spécifique
            if metric == 'put_win_rate' and stats['put_trades'] < 1:
                fold_scores.append(0.0)
            elif metric == 'call_win_rate' and stats['call_trades'] < 1:
                fold_scores.append(0.0)
            elif stats['total_trades'] < 2:
                fold_scores.append(0.0)
            else:
                fold_scores.append(stats[metric])

        # Retourne la moyenne des scores sur tous les folds (Walk-Forward),
        # ajustée par l'écart-type (pour pénaliser l'instabilité) et la densité.
        alpha = 0.80
        final_score = np.mean(fold_scores) - (alpha * np.std(fold_scores)) - density_penalty
        return final_score

    return objective


if __name__ == "__main__":
    freeze_support()
    ticker = "^GSPC"
    df_mom = factory_load_data(_dataset_id="day", _ticker=ticker, _args={})

    lookahead_bar_1 = 6
    lookahead_bar_2 = 10
    n_splits = 12
    train_ratio = 0.95

    # --- CONFIGURATION DE L'OPTIMISATION ---
    optimize_metric = 'total_win_rate'  # Choisir entre 'total_win_rate', 'put_win_rate', 'call_win_rate'
    n_trials = 99999  # Nombre d'essais pour Optuna
    timeout = int(86400 *2.5)
    target_signal_density = 0.101575  # Cible de densité des signaux
    # ---------------------------------------

    if isinstance(df_mom.columns, pd.MultiIndex):
        df_mom.columns = df_mom.columns.get_level_values(0)

    # Création d'un index original pour pouvoir mapper les trades aux folds du TimeSeriesSplit
    df_mom['orig_idx'] = np.arange(len(df_mom))

    is_datetime_index = isinstance(df_mom.index, pd.DatetimeIndex)

    # CORRECTION : Meilleure détection de la colonne de date dans le script principal
    if not is_datetime_index:
        date_col = None
        for col in df_mom.columns:
            if pd.api.types.is_datetime64_any_dtype(df_mom[col]):
                date_col = col
                break

        if date_col is None:
            if 'Date' in df_mom.columns:
                date_col = 'Date'
            elif 'date' in df_mom.columns:
                date_col = 'date'
            elif 'index' in df_mom.columns:
                date_col = 'index'
            else:
                date_col = df_mom.columns[0]
    else:
        date_col = None

    split_idx = int(len(df_mom) * train_ratio)
    df_train_and_val = df_mom.iloc[:split_idx].copy()
    df_test = df_mom.iloc[split_idx:].copy()

    print(f"🚀 Lancement de l'optimisation Optuna pour maximiser : {optimize_metric}")
    print(f"🎯 Cible de densité de signaux : {target_signal_density * 100}%")
    print(f"🔧 Nombre d'essais (trials) : {n_trials}\n")
    train_start_date = df_test.index[0].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_test.index[0])
    train_end_date = df_test.index[-1].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_test.index[-1])
    print(f"📊 Train Set ({len(df_train_and_val)} bars) - {train_start_date}::{train_end_date}\n")
    test_start_date = df_test.index[0].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_test.index[0])
    test_end_date = df_test.index[-1].strftime("%Y%m%d_%H%M") if is_datetime_index else str(df_test.index[-1])
    print(f"📊 Test Set ({len(df_test)} bars) - Données jamais vues par Optuna - {test_start_date}::{test_end_date}\n")

    # Création et lancement de l'étude Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(
        create_objective(
            df_data=df_train_and_val,
            lb1=lookahead_bar_1,
            lb2=lookahead_bar_2,
            metric=optimize_metric,
            n_splits=n_splits,
            target_density=target_signal_density
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value = study.best_value

    print(f"\n✅ Optimisation terminée !")
    print(f"🏆 Meilleur {optimize_metric} (moyenne des folds Walk-Forward ajustée) : {best_value:.2f}%")
    print(f"🔧 Meilleurs paramètres trouvés : {best_params}\n")
    print(f"🚀 Lancement du backtest final de validation avec les meilleurs paramètres...\n")

    # ==========================================================
    # BACKTEST FINAL AVEC LES MEILLEURS PARAMÈTRES (TRAIN SET)
    # ==========================================================
    df_train_and_val = calculate_channel_indicators(
        df=df_train_and_val,
        period=best_params["p_center"],
        period_low=best_params.get("p_low", 0),
        period_high=best_params.get("p_high", 0),
        channel_type=best_params['channel_type'],
        atr_multiplier_high=best_params.get("atr_mult_high", 0),
        atr_multiplier_low=best_params.get("atr_mult_low", 0),
        envelope_pct_high=best_params.get("env_pct_high", 0),
        envelope_pct_low=best_params.get("env_pct_low", 0),
    )

    df_train_and_val = detect_candlestick_patterns(df_train_and_val, strict_patterns=best_params['strict_patterns'])

    if optimize_metric == "total_win_rate":
        print(f"🔍 Signaux bruts détectés -> BUY: {df_train_and_val['buy_signal'].sum()} | SELL: {df_train_and_val['sell_signal'].sum()}")
    elif optimize_metric == "put_win_rate":
        print(f"🔍 Signaux bruts détectés -> BUY: {df_train_and_val['buy_signal'].sum()}")
    elif optimize_metric == "call_win_rate":
        print(f"🔍 Signaux bruts détectés -> SELL: {df_train_and_val['sell_signal'].sum()}")

    credit_trades_df_train_and_val = backtest_triple_ema_strategy_for_credit_spread(
        df_train_and_val,
        lookahead_bar_1=lookahead_bar_1,
        lookahead_bar_2=lookahead_bar_2
    )

    if credit_trades_df_train_and_val.empty:
        print("⚠️ Aucun trade généré sur l'ensemble du dataset avec ces paramètres.")
    else:
        print(f"✅ Total de trades générés sur le dataset complet : {len(credit_trades_df_train_and_val)}\n")

        # Filtrage des trades affichés selon la métrique optimisée
        if optimize_metric == 'put_win_rate':
            display_trades_train = credit_trades_df_train_and_val[credit_trades_df_train_and_val['type'] == 'PUT_CREDIT_SPREAD']
        elif optimize_metric == 'call_win_rate':
            display_trades_train = credit_trades_df_train_and_val[credit_trades_df_train_and_val['type'] == 'CALL_CREDIT_SPREAD']
        else:
            display_trades_train = credit_trades_df_train_and_val

        print(f"--- [Train] Résumé des 10 Derniers Credit Spreads (Lookahead: [{lookahead_bar_1}, {lookahead_bar_2}]) ---")
        if not display_trades_train.empty:
            print(display_trades_train[['type', 'signal_date', 'strike_price', 'exit_price_slice', 'exit_date_slice', 'result']].tail(10))
        else:
            print("Aucun trade de ce type spécifique à afficher.")

        print(f"\n{'=' * 25} RÉSUMÉ GLOBAL (TRAIN) {'=' * 25}")
        train_overall_stats = calculate_credit_spread_win_rates(credit_trades_df_train_and_val)

        if optimize_metric == 'put_win_rate':
            print(f"🔵 PUT Credit Spread (BUY Setup)  : {train_overall_stats['put_wins']}/{train_overall_stats['put_trades']} gagnants -> Win Rate = {train_overall_stats['put_win_rate']}%")
        elif optimize_metric == 'call_win_rate':
            print(f"🔴 CALL Credit Spread (SELL Setup) : {train_overall_stats['call_wins']}/{train_overall_stats['call_trades']} gagnants -> Win Rate = {train_overall_stats['call_win_rate']}%")
        elif optimize_metric == 'total_win_rate':
            print(f"📊 GLOBAL                         : {train_overall_stats['total_wins']}/{train_overall_stats['total_trades']} gagnants -> Win Rate Total = {train_overall_stats['total_win_rate']}%")

    # ==========================================================
    # ÉVALUATION OUT-OF-SAMPLE (TEST SET)
    # ==========================================================
    print(f"\n{'=' * 25} ÉVALUATION SUR LE JEU DE TEST (OUT-OF-SAMPLE) {'=' * 25}")

    # Gestion de l'affichage des dates pour le set de test


    df_test = calculate_channel_indicators(
        df=df_test,
        period=best_params['p_center'],
        period_low=best_params.get("p_low", 0),
        period_high=best_params.get("p_high", 0),
        channel_type=best_params['channel_type'],
        atr_multiplier_high=best_params.get("atr_mult_high", 0),
        atr_multiplier_low=best_params.get("atr_mult_low", 0),
        envelope_pct_high=best_params.get("env_pct_high", 0),
        envelope_pct_low=best_params.get("env_pct_low", 0),
    )

    df_test = detect_candlestick_patterns(df_test, strict_patterns=best_params['strict_patterns'])

    if optimize_metric == 'total_win_rate':
        print(f"🔍 Signaux bruts détectés sur TEST -> BUY: {df_test['buy_signal'].sum()} | SELL: {df_test['sell_signal'].sum()}")
    elif optimize_metric == 'call_win_rate':
        print(f"🔍 Signaux bruts détectés sur TEST -> SELL: {df_test['sell_signal'].sum()}")
    elif optimize_metric == 'put_win_rate':
        print(f"🔍 Signaux bruts détectés sur TEST -> BUY: {df_test['buy_signal'].sum()}")

    credit_trades_df_test = backtest_triple_ema_strategy_for_credit_spread(
        df_test,
        lookahead_bar_1=lookahead_bar_1,
        lookahead_bar_2=lookahead_bar_2
    )

    if credit_trades_df_test.empty:
        print("⚠️ Aucun trade généré sur le jeu de test.")
    else:
        print(f"✅ Total de trades générés sur le jeu de test : {len(credit_trades_df_test)}\n")

        test_stats = calculate_credit_spread_win_rates(credit_trades_df_test)

        if optimize_metric == 'put_win_rate':
            print(f"🔵 PUT Credit Spread (BUY Setup)  : {test_stats['put_wins']}/{test_stats['put_trades']} gagnants -> Win Rate = {test_stats['put_win_rate']}%")
        elif optimize_metric == 'call_win_rate':
            print(f"🔴 CALL Credit Spread (SELL Setup) : {test_stats['call_wins']}/{test_stats['call_trades']} gagnants -> Win Rate = {test_stats['call_win_rate']}%")
        elif optimize_metric == 'total_win_rate':
            print(f"📊 GLOBAL (TEST)                  : {test_stats['total_wins']}/{test_stats['total_trades']} gagnants -> Win Rate Total = {test_stats['total_win_rate']}%")

        if optimize_metric == 'put_win_rate':
            display_trades_test = credit_trades_df_test[credit_trades_df_test['type'] == 'PUT_CREDIT_SPREAD']
        elif optimize_metric == 'call_win_rate':
            display_trades_test = credit_trades_df_test[credit_trades_df_test['type'] == 'CALL_CREDIT_SPREAD']
        else:
            display_trades_test = credit_trades_df_test

        print(f"\n--- Derniers trades sur le jeu de TEST ---")
        if not display_trades_test.empty:
            print(display_trades_test[['type', 'signal_date', 'strike_price', 'exit_price_slice', 'result']].tail(5))
        else:
            print("Aucun trade de ce type spécifique à afficher.")

        # ==========================================================
        # COMPARAISON FINALE (TRAIN vs TEST)
        # ==========================================================
        print(f"\n{'=' * 25} COMPARAISON FINALE {'=' * 25}")

        if optimize_metric == 'put_win_rate':
            train_wr = train_overall_stats['put_win_rate'] if not credit_trades_df_train_and_val.empty else 0
            test_wr = test_stats['put_win_rate']
            metric_label = "PUT Win Rate"
        elif optimize_metric == 'call_win_rate':
            train_wr = train_overall_stats['call_win_rate'] if not credit_trades_df_train_and_val.empty else 0
            test_wr = test_stats['call_win_rate']
            metric_label = "CALL Win Rate"
        else:
            train_wr = train_overall_stats['total_win_rate'] if not credit_trades_df_train_and_val.empty else 0
            test_wr = test_stats['total_win_rate']
            metric_label = "GLOBAL (PUT+CALL) Win Rate"

        print(f"   🏋️ Train {metric_label} : {train_wr}%")
        print(f"   🚀 Test {metric_label}  : {test_wr}%")

        diff = train_wr - test_wr
        if diff > 15:
            print(f"   ⚠️ Écart de {diff:.1f}% : Attention, probable sur-apprentissage (overfitting) sur les données d'entraînement !")
            print(f"      Le modèle a mémorisé le passé mais peine à généraliser sur des données récentes.")
        elif diff < -5:
            print(f"   🌟 Écart de {abs(diff):.1f}% : La stratégie performe étonnamment mieux sur les données récentes (Test) !")
        else:
            print(f"   ✅ Écart raisonnable de {diff:.1f}% : Le modèle généralise bien aux nouvelles données. La stratégie est robuste.")