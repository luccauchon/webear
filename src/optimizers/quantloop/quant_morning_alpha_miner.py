try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import time
from utils import str2bool, format_duration
from fetchers.data_factory import factory_load_data
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from itertools import combinations
from datetime import datetime
import pickle
import io
from multiprocessing import Process, Queue, freeze_support, Value
import warnings
import argparse
from argparse import Namespace
# Ignore cet avertissement spécifique lié à sklearn parallel
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")


class TopNEnsembleModel:
    """
    Agrège les meilleurs modèles sélectionnés, par exemple le top 50.

    Logique:
    - Chaque modèle du top est réentraîné sur les données fournies dans fit().
    - La prédiction agrégée est la médiane des probabilités de la classe positive.
    - La marge d'erreur est l'écart-type des probabilités des modèles membres.

    Remarque:
    - Ce modèle est prévu pour une classification binaire.
    - Pour les SVM, il est conseillé d'utiliser une factory avec probability=True.
    """

    def __init__(self, threshold=0.5, command_line="", args=Namespace()):
        self.threshold = threshold
        self.is_fitted_ = False
        self.models_ = []
        self.classes_ = None
        self.positive_class_ = None
        self.n_models_ = 0
        self.metadata_ = {}
        self.command_line = command_line
        self.args = args

    def __repr__(self):
        return (
            f"TopNEnsembleModel(n_models={self.n_models_}, "
            f"fitted={self.is_fitted_}, threshold={self.threshold})"
            f"  Command Line: {self.command_line}"
        )

    def fit(self, X, y, top_models_df, model_factory):
        """
        Entraîne tous les modèles du top sur les données fournies.

        Parameters
        ----------
        X : pd.DataFrame
            Données complètes contenant toutes les features possibles.
        y : array-like
            Target binaire.
        top_models_df : pd.DataFrame
            DataFrame contenant au minimum:
            - 'Model'
            - 'Features_str' ou 'Features'
            - éventuellement 'CV_F1_Score'
        model_factory : callable ou dict
            Une fonction qui retourne un dict comme get_models(), ou directement un dict.
            Ce dict doit mapper nom de modèle -> (type, estimateur).
        """
        if len(top_models_df) == 0:
            raise ValueError("top_models_df est vide, impossible de créer un agrégat.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if len(self.classes_) != 2:
            raise ValueError("TopNEnsembleModel est conçu pour une classification binaire.")

        # Par convention, la classe positive est la dernière classe (ex: 1 si classes=[0, 1])
        self.positive_class_ = self.classes_[-1]

        self.models_ = []
        self.n_models_ = int(len(top_models_df))

        for _, row in top_models_df.iterrows():
            if isinstance(row.get('Features_str', None), str):
                features = row['Features_str'].split('|')
            elif 'Features' in row:
                features = list(row['Features'])
            else:
                raise KeyError("top_models_df doit contenir 'Features_str' ou 'Features'.")

            model_name = row['Model']

            missing_features = [f for f in features if f not in X.columns]
            if missing_features:
                raise KeyError(f"Features manquantes dans X: {missing_features}")

            factory_output = model_factory() if callable(model_factory) else model_factory
            data_type, estimator = factory_output[model_name]

            if data_type == 'scaled':
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', estimator)
                ])
            else:
                pipe = Pipeline([
                    ('model', estimator)
                ])

            pipe.fit(X[features], y_arr)

            self.models_.append({
                'model_name': model_name,
                'features': features,
                'cv_f1_score': row.get('CV_F1_Score', np.nan),
                'pipeline': pipe
            })

        self.is_fitted_ = True
        self.metadata_ = {
            'n_models': self.n_models_,
            'threshold': self.threshold,
            'classes': self.classes_.tolist(),
            'positive_class': self.positive_class_.item() if hasattr(self.positive_class_, 'item') else self.positive_class_,
            'description': "Median of member probabilities; std as margin of error."
        }

        return self

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Le modèle n'est pas entraîné. Appelle fit(...) avant.")

    def _member_positive_probabilities(self, X):
        """
        Retourne une matrice (n_samples, n_models) contenant la probabilité
        de la classe positive pour chaque modèle membre.
        """
        self._check_is_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_samples = X.shape[0]
        member_probas = []

        for info in self.models_:
            features = info['features']
            X_sub = X[features]
            pipe = info['pipeline']

            if hasattr(pipe, 'predict_proba'):
                proba = pipe.predict_proba(X_sub)

                if hasattr(pipe, 'classes_'):
                    model_classes = np.asarray(pipe.classes_)
                else:
                    final_estimator = pipe.steps[-1][1]
                    model_classes = np.asarray(getattr(final_estimator, 'classes_', []))

                if len(model_classes) == 0:
                    raise ValueError("Impossible de récupérer les classes d'un modèle membre.")

                if self.positive_class_ in model_classes:
                    pos_idx = int(np.where(model_classes == self.positive_class_)[0][0])
                    member_probas.append(np.asarray(proba[:, pos_idx], dtype=float))
                else:
                    # Cas très improbable si tous les modèles sont entraînés sur les mêmes classes.
                    member_probas.append(np.zeros(n_samples, dtype=float))
            else:
                # Fallback si un modèle n'a pas predict_proba.
                # Dans ce cas, on transforme la prédiction en probabilité 0/1.
                preds = np.asarray(pipe.predict(X_sub))
                member_probas.append((preds == self.positive_class_).astype(float))

        if len(member_probas) == 0:
            return np.empty((n_samples, 0), dtype=float)

        return np.column_stack(member_probas)

    def predict_aggregated(self, X):
        """
        Retourne un DataFrame contenant:
        - median_probability : médiane des probabilités de la classe positive
        - std_probability : écart-type des probabilités de la classe positive
        - margin_error : copie de std_probability
        - lower / upper : médiane +/- std, borné dans [0, 1]
        - predicted_class : classe prédite avec seuil
        - n_member_models : nombre de modèles membres utilisés
        """
        self._check_is_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        member_probas = self._member_positive_probabilities(X)

        if member_probas.shape[1] == 0:
            raise RuntimeError("Aucun modèle membre disponible pour prédire.")

        median_probability = np.median(member_probas, axis=1)
        std_probability = np.std(member_probas, axis=1, ddof=0)

        lower = np.clip(median_probability - std_probability, 0.0, 1.0)
        upper = np.clip(median_probability + std_probability, 0.0, 1.0)

        predicted_class = np.where(
            median_probability >= self.threshold,
            self.positive_class_,
            self.classes_[0]
        )

        return pd.DataFrame(
            {
                'median_probability': median_probability,
                'std_probability': std_probability,
                'margin_error': std_probability,
                'lower_median_minus_std': lower,
                'upper_median_plus_std': upper,
                'predicted_class': predicted_class,
                'n_member_models': member_probas.shape[1]
            },
            index=X.index
        )

    def predict(self, X):
        """
        Retourne la classe prédite via la seuil sur la médiane des probabilités.
        """
        return self.predict_aggregated(X)['predicted_class'].to_numpy()

    def predict_proba(self, X):
        """
        Retourne une approximation binaire de predict_proba:
        colonne 0 = 1 - médiane positive
        colonne 1 = médiane positive
        """
        self._check_is_fitted()

        if len(self.classes_) != 2:
            raise NotImplementedError("predict_proba n'est disponible que pour le cas binaire.")

        p_positive = self.predict_aggregated(X)['median_probability'].to_numpy()
        return np.column_stack([1.0 - p_positive, p_positive])

    def save(self, filepath):
        """
        Sauvegarde le modèle agrégé sur disque avec pickle.
        """
        directory = os.path.dirname(os.path.abspath(filepath))
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath):
        """
        Charge un modèle agrégé depuis le disque.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared, k):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)
    # for cle, valeur in os.environ.items(): print(f"{cle}: {valeur}")
    # Traitement des requêtes
    all_results_computed, best_mean_cv_f1, times = [], 0., [0]
    while True:
        t1 = time.time()
        try:
            tmp = use_cases__shared.get(timeout=0.033)
        except:
            time.sleep(3.033)
            try:
                tmp = use_cases__shared.get(timeout=0.033)
            except:
                time.sleep(9.033)
                try:
                    tmp = use_cases__shared.get(timeout=0.033)
                except:
                    break
        try:
            with io.BytesIO(tmp) as memoire:
                # On recharge le tuple et on réaffecte chaque variable dans le même ordre
                pipe, X_train_sub, y_train, tscv, scoring, features_str, model_name, combo_list = pickle.load(memoire)

            cv_scores = cross_val_score(pipe, X_train_sub, y_train, cv=tscv, scoring='f1_weighted', n_jobs=1)
            mean_cv_f1 = cv_scores.mean()

            all_results_computed.append({
                'Model': model_name,
                'Num_Features': len(combo_list),
                'Features_str': features_str,
                'CV_F1_Score': mean_cv_f1
            })

            if mean_cv_f1 > best_mean_cv_f1:
                best_mean_cv_f1 = mean_cv_f1

            if 0 == len(all_results_computed) % int(os.environ.get("Q__DISPLAY_RATE", 100000)):
                print(
                    f"[{str(os.getpid()):<6}:{str(k):<4}]  {best_mean_cv_f1:8.8f} :: {mean_cv_f1:8.8f}  "
                    f"| {len(all_results_computed):6d} cases computed  "
                    f"| Mean time per pass: {format_duration(np.mean(times))}"
                )
        except Exception as e:
            print(e)

        t2 = time.time()
        times.append(t2 - t1)

    out__shared.put(all_results_computed)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Entraînement et évaluation de modèles de classification sur données intraday.")

    # Ticker de l'actif financier (ex: ^GSPC)
    parser.add_argument('--ticker', type=str, default="^GSPC", help="Ticker de l'actif à analyser.")

    # Identifiant du dataset à charger
    parser.add_argument('--dataset-id', type=str, default="intraday_1min", help="ID du dataset à charger.")

    # Nombre de workers pour le multiprocessing
    parser.add_argument('--nb-worker', type=int, default=12, help="Nombre de workers pour l'évaluation parallèle.")

    # Périodes des indicateurs techniques
    parser.add_argument('--ema9', type=int, default=9, help="Période de la première EMA.")
    parser.add_argument('--ema21', type=int, default=21, help="Période de la deuxième EMA.")
    parser.add_argument('--rsi14', type=int, default=14, help="Période du RSI.")
    parser.add_argument('--macd5', type=int, default=5, help="Période du signal MACD.")

    # Fenêtre d'analyse matinale (calcul des features)
    parser.add_argument('--st1', type=str, default="09:30", help="Heure de début de la fenêtre d'analyse (HH:MM).")
    parser.add_argument('--st2', type=str, default="10:30", help="Heure de fin de la fenêtre d'analyse (HH:MM).")

    # Fenêtre cible après-midi (calcul du High pour la target)
    parser.add_argument('--et1', type=str, default="15:00", help="Heure de début de la fenêtre cible (HH:MM).")
    parser.add_argument('--et2', type=str, default="16:00", help="Heure de fin de la fenêtre cible (HH:MM).")

    # Définition de la cible (target)
    parser.add_argument('--seuil-pos', type=float, default=1., help="Seuil multiplicateur du Daily Open pour la cible.")
    parser.add_argument('--et-gt-open', type=int, default=1, help="1 si la cible est High > (Open * seuil), 0 si High < (Open * seuil).")

    # Sélection des features
    parser.add_argument('--reduce-feature', type=int, default=-1, help="Nombre max de features à utiliser (-1 pour toutes les garder).")

    parser.add_argument("--realtime", action="store_true", default=False, help="Run in real-time mode to check for live signals using a saved model.")
    parser.add_argument("--modelfile", type=str, default=None, help="Specific model filename to load in real-time mode. If not provided, loads the latest model.")
    parser.add_argument("--output-dir", type=str, default="quant", help="Model save directory")
    return parser.parse_args()


def run_realtime(modelfile, verbose):
    # Load model
    model = TopNEnsembleModel.load(modelfile)

    # Load data
    daily_df = factory_load_data(_dataset_id=model.args.dataset_id, _ticker=model.args.ticker, _args={"realtime": True})
    if isinstance(daily_df.columns, pd.MultiIndex): daily_df.columns = daily_df.columns.get_level_values(0)
    daily_df = daily_df.sort_index()
    if verbose: print(f"{len(daily_df)} :: {daily_df.index[0]} --> {daily_df.index[-1]}")

    # Build dataset
    X_new = build_df(df=daily_df, args=Namespace(st1=model.args.st1, st2=model.args.st2, ema9=model.args.ema9, ema21=model.args.ema21, rsi14=model.args.rsi14,
                                                 macd5=model.args.macd5, seuil_pos=model.args.seuil_pos, et1=model.args.et1, et2=model.args.et2, reduce_feature=-1, verbose=verbose,),
                     generate_for_realtime=True)
    agg = model.predict_aggregated(X_new.iloc[-1:])
    if verbose: print(agg[['median_probability', 'std_probability', 'predicted_class']])
    return agg[['median_probability', 'std_probability', 'predicted_class']]


def build_df(df, args, generate_for_realtime):
    # Récupérer les variables du shell
    heure_debut_str = str(args.st1)
    heure_fin_str = str(args.st2)

    time_debut = datetime.strptime(heure_debut_str, "%H:%M").time()
    time_fin = datetime.strptime(heure_fin_str, "%H:%M").time()

    assert time_debut < time_fin and time_debut >= datetime.strptime("09:30", "%H:%M").time()

    ema_9 = int(args.ema9)
    ema_21 = int(args.ema21)
    rsi_14 = int(args.rsi14)
    macd_5 = int(args.macd5)
    fenetre_analysee = (time_debut, time_fin)
    seuil_pos = float(args.seuil_pos)

    # ---------------------------------------------------------
    # 1.5 NOUVELLE LOGIQUE DE CIBLE (TARGET)
    # ---------------------------------------------------------
    # 1. Récupérer le prix d'ouverture quotidien (Daily Open)
    daily_open = df.groupby(df.index.date)['Open'].first()

    # 2. Filtrer les données pour la fenêtre cible
    time_15ish = datetime.strptime(str(args.et1), "%H:%M").time()
    time_16ish = datetime.strptime(str(args.et2), "%H:%M").time()

    afternoon_mask = (df.index.time >= time_15ish) & (df.index.time <= time_16ish)
    df_afternoon = df[afternoon_mask]

    assert time_15ish < time_16ish <= datetime.strptime("16:00", "%H:%M").time()

    # 3. Récupérer le "High" (plus haut) de cette fenêtre.
    max_high_15ish_16ish = df_afternoon.groupby(df_afternoon.index.date)['High'].max()

    # ---------------------------------------------------------
    # 2. Analyse de la fenêtre et Feature Engineering
    # ---------------------------------------------------------
    morning_mask = (df.index.time >= fenetre_analysee[0]) & (df.index.time <= fenetre_analysee[1])
    df_morning = df[morning_mask].copy()

    df_morning['typical_price'] = (df_morning['High'] + df_morning['Low'] + df_morning['Close']) / 3.0
    df_morning['tp_vol'] = df_morning['typical_price'] * df_morning['Volume']
    df_morning['up_volume'] = np.where(df_morning['Close'] > df_morning['Open'], df_morning['Volume'], 0)

    daily_morning_groups = df_morning.groupby(df_morning.index.date)

    df_morning['cum_tp_vol'] = daily_morning_groups['tp_vol'].cumsum()
    df_morning['cum_vol'] = daily_morning_groups['Volume'].cumsum()
    df_morning['cum_up_vol'] = daily_morning_groups['up_volume'].cumsum()
    df_morning['vwap'] = np.where(df_morning['cum_vol'] > 0, df_morning['cum_tp_vol'] / df_morning['cum_vol'], np.nan)
    df_morning['vwap_slope'] = daily_morning_groups['vwap'].diff()
    df_morning['vwap_deriv'] = daily_morning_groups['vwap_slope'].diff()

    morning_groups = df_morning.groupby(df_morning.index.date)

    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df_morning['ema_9'] = morning_groups['Close'].transform(lambda x: x.ewm(span=ema_9, adjust=False).mean())
    df_morning['ema_21'] = morning_groups['Close'].transform(lambda x: x.ewm(span=ema_21, adjust=False).mean())
    df_morning['rsi_14'] = morning_groups['Close'].transform(lambda x: calc_rsi(x, period=rsi_14))
    df_morning['macd_line'] = df_morning['ema_9'] - df_morning['ema_21']
    df_morning['macd_signal'] = morning_groups['macd_line'].transform(lambda x: x.ewm(span=macd_5, adjust=False).mean())
    df_morning['macd_hist'] = df_morning['macd_line'] - df_morning['macd_signal']
    df_morning['ema_diff'] = df_morning['ema_9'] - df_morning['ema_21']

    # Agrégation journalière
    df_morning['is_above_vwap'] = df_morning['Close'] > df_morning['vwap']
    pct_above_vwap_series = morning_groups['is_above_vwap'].mean()
    last_slope_series = morning_groups['vwap_slope'].last()
    last_deriv_series = morning_groups['vwap_deriv'].last()

    morning_high = morning_groups['High'].max()
    morning_low = morning_groups['Low'].min()
    morning_open = morning_groups['Open'].first()
    morning_close_hhmm = morning_groups['Close'].last()
    morning_range_pct = ((morning_high - morning_low) / morning_open) * 100

    morning_high_ref = morning_groups.apply(lambda g: g['High'].iloc[:-1].max(), include_groups=False)
    morning_low_ref = morning_groups.apply(lambda g: g['Low'].iloc[:-1].min(), include_groups=False)

    breakout_signal = pd.Series(0, index=morning_close_hhmm.index, dtype=int)
    breakout_signal[morning_close_hhmm > morning_high_ref] = 1
    breakout_signal[morning_close_hhmm < morning_low_ref] = -1

    total_morning_vol = morning_groups['Volume'].sum()
    total_morning_up_vol = morning_groups['up_volume'].sum()
    up_volume_ratio = total_morning_up_vol / total_morning_vol

    vwap_1030 = morning_groups['vwap'].last()
    vwap_distance_pct = ((morning_close_hhmm - vwap_1030) / vwap_1030) * 100

    df_morning['cross_vwap'] = (df_morning['Close'] > df_morning['vwap']).astype(int)
    vwap_crossings = morning_groups['cross_vwap'].apply(lambda x: x.diff().abs().sum() / 2)

    rsi_last = morning_groups['rsi_14'].last()
    rsi_first = morning_groups['rsi_14'].first()
    rsi_delta = rsi_last - rsi_first

    ema9_last = morning_groups['ema_9'].last()
    close_ema9_dist_pct = ((morning_close_hhmm - ema9_last) / ema9_last) * 100
    ema_cross_direction = np.sign(morning_groups['ema_diff'].last())

    macd_hist_last = morning_groups['macd_hist'].last()
    macd_hist_first = morning_groups['macd_hist'].first()
    macd_hist_delta = macd_hist_last - macd_hist_first

    daily_morning_open = daily_morning_groups['Open'].first()
    daily_morning_close = daily_morning_groups['Close'].last()
    prev_daily_morning_close = daily_morning_close.shift(1)

    daily_df = pd.DataFrame({
        'pct_above_vwap_morning': pct_above_vwap_series,
        'last_vwap_slope': last_slope_series,
        'last_vwap_deriv': last_deriv_series,
        'morning_range_pct': morning_range_pct,
        'breakout_signal': breakout_signal,
        'up_volume_ratio': up_volume_ratio,
        'vwap_distance_pct': vwap_distance_pct,
        'vwap_crossings': vwap_crossings,
        'rsi_last': rsi_last,
        'rsi_delta': rsi_delta,
        'close_ema9_dist_pct': close_ema9_dist_pct,
        'ema_cross_direction': ema_cross_direction,
        'macd_hist_last': macd_hist_last,
        'macd_hist_delta': macd_hist_delta,
        'Morning_Open': daily_morning_open,
        'Morning_Close': daily_morning_close,
        'gap_pct': ((daily_morning_open - prev_daily_morning_close) / prev_daily_morning_close) * 100,
        'prev_day_return_t1': ((daily_morning_close - prev_daily_morning_close) / prev_daily_morning_close) * 100,
        'prev_day_return_t2': ((daily_morning_close - prev_daily_morning_close.shift(1)) / prev_daily_morning_close.shift(1)) * 100,
        'prev_day_return_t3': ((daily_morning_close - prev_daily_morning_close.shift(2)) / prev_daily_morning_close.shift(2)) * 100,
        'prev_day_return_t4': ((daily_morning_close - prev_daily_morning_close.shift(3)) / prev_daily_morning_close.shift(3)) * 100,
    })

    feature_cols = [
        'pct_above_vwap_morning',
        'last_vwap_slope',
        'last_vwap_deriv',
        'morning_range_pct',
        'breakout_signal',
        'up_volume_ratio',
        'vwap_distance_pct',
        'vwap_crossings',
        'rsi_last',
        'rsi_delta',
        'close_ema9_dist_pct',
        'ema_cross_direction',
        'macd_hist_last',
        'macd_hist_delta',
        'gap_pct',
        'prev_day_return_t1',
        'prev_day_return_t2',
        # 'prev_day_return_t3',
        # 'prev_day_return_t4',
        # 'Morning_Open',
        # 'Morning_Close',
    ]
    reduce_feature = int(args.reduce_feature)
    if -1 != reduce_feature:
        feature_cols = feature_cols[:reduce_feature]

    if getattr(args, "verbose", True): print(f"X is: Features computed from {heure_debut_str} and {heure_fin_str} and from previous days")
    if generate_for_realtime:
        X = daily_df[feature_cols].copy()
        if getattr(args, "verbose", True): print(f"[REALTIME] [{len(X):5d}]  {X.index[0].strftime('%Y-%m-%d')} --> {X.index[-1].strftime('%Y-%m-%d')}")
        return X

    # ---------------------------------------------------------
    # 2.5 APPLICATION DE LA NOUVELLE CIBLE
    # ---------------------------------------------------------
    strict_sup = int(args.et_gt_open)

    if 1 == strict_sup:
        daily_df['target'] = (max_high_15ish_16ish.reindex(daily_df.index) > (daily_open.reindex(daily_df.index) * seuil_pos)).astype(int)
    else:
        daily_df['target'] = (max_high_15ish_16ish.reindex(daily_df.index) < (daily_open.reindex(daily_df.index) * seuil_pos)).astype(int)

    daily_df = daily_df.dropna()

    if getattr(args, "verbose", True): print(f"Y is: There is a value between {time_15ish} and {time_16ish} that is {'>' if 1 == strict_sup else '<'} Daily Open * {seuil_pos}")

    experience_str_desc = (f"{seuil_pos}__{heure_debut_str}-{heure_fin_str}__{time_15ish}-{time_16ish}_High_{strict_sup}_Open__"
        f"{ema_9}{ema_21}{rsi_14}{macd_5}__").replace('datetime.time', '').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_').replace(':', '')

    # ---------------------------------------------------------
    # 3. Préparation des données et Split Temporel
    # ---------------------------------------------------------
    X = daily_df[feature_cols].copy()
    y = daily_df['target'].copy()

    # Split temporel strict (90% Train pour CV / 10% Test pour évaluation finale)
    split_idx = int(len(daily_df) * 0.90)

    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    if getattr(args, "verbose", True):
        print(f"[TRAIN] [{len(X_train):5d}]  {X_train.index[0].strftime('%Y-%m-%d')} --> {X_train.index[-1].strftime('%Y-%m-%d')}")
        print(f"[TEST]  [{len(X_test):5d}]  {X_test.index[0].strftime('%Y-%m-%d')} --> {X_test.index[-1].strftime('%Y-%m-%d')}\n")
    return X, y, X_train, X_test, y_train, y_test, feature_cols, experience_str_desc


def entry(args):
    # ---------------------------------------------------------
    # 1. Chargement et préparation des données
    # ---------------------------------------------------------
    # 1. Capturer les variables d'environnement commençant par Q__
    quant_env_vars = {k: v for k, v in os.environ.items() if k.startswith("Q__")}
    # 2. Construire la string des variables au format PowerShell ($env:VAR=valeur;)
    env_string = "".join([f"$env:{k}={v}; " for k, v in quant_env_vars.items()])
    # 3. Combiner le tout avec la commande Python
    command_line = f"{env_string}python " + " ".join(sys.argv)

    # ---------------------------------------------------------
    #
    # ---------------------------------------------------------
    if args.realtime:
        return run_realtime(modelfile=args.modelfile, verbose=getattr(args, "verbose", True))

    print(f"Command Line: {command_line}")
    df = factory_load_data(_dataset_id=args.dataset_id, _ticker=args.ticker, _args={"realtime": args.realtime})

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    print(f"{len(df)} :: {df.index[0]} --> {df.index[-1]}")

    nb_worker = int(args.nb_worker)
    # ---------------------------------------------------------
    #
    # ---------------------------------------------------------
    X, y, X_train, X_test, y_train, y_test, feature_cols, experience_str_desc = build_df(df=df, args=args, generate_for_realtime=False)

    # ---------------------------------------------------------
    # 4. Configuration de la Cross-Validation et des Modèles
    # ---------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=12)

    def get_models(probability=False):
        """
        Retourne une instance fraîche de chaque modèle et son type de prétraitement.

        probability=False est utilisé pour la CV afin de ne pas ralentir les SVM.
        probability=True est utilisé pour l'agrégat final afin de pouvoir
        récupérer les probabilités et calculer médiane / std.
        """
        return {
            'LogisticRegression': ('scaled', LogisticRegression(random_state=42, max_iter=1000)),
            'RandomForest': ('raw', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
            'SVM': ('scaled', SVC(kernel='rbf', C=1.0, random_state=42, class_weight="balanced", probability=probability)),
            'GradientBoosting': ('raw', GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42))
        }

    def get_models_for_ensemble():
        """
        Factory utilisée pour l'agrégat final.
        On active probability=True pour les SVM afin de pouvoir calculer
        la médiane et l'écart-type des probabilités.
        """
        return get_models(probability=True)

    # ---------------------------------------------------------
    # 5. Évaluation de toutes les combinaisons via Cross-Validation
    # ---------------------------------------------------------
    data_from_workers = []
    combo_to_be_processed = []
    for r in range(1, len(feature_cols) + 1):
        for combo in combinations(feature_cols, r):
            combo_to_be_processed.append(list(combo))

    zzz = len(combo_to_be_processed) * len(get_models())

    use_cases__shared, master_cmd__shared = Queue(), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, nb_worker)]

    print(f"Lancement des evaluations en CV")
    for k in range(0, nb_worker):
        p = Process(
            target=_worker_processor,
            args=(use_cases__shared, master_cmd__shared, out__shared[k], k,)
        )
        p.start()

    print(f"Evaluation CV de {len(combo_to_be_processed) * len(get_models())} combinaisons avec {nb_worker} workers...")
    print(f"Starting training now...")
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1

    pbar = tqdm(combo_to_be_processed, desc="Submitting training with CV")
    uuu = 0

    for combo_list in pbar:
        X_train_sub = X_train[combo_list]
        features_str = "|".join(combo_list)

        for model_name, (data_type, model) in get_models().items():
            if data_type == 'scaled':
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            else:
                pipe = Pipeline([('model', model)])

            with io.BytesIO() as memoire:
                pickle.dump(
                    (
                        pipe,
                        X_train_sub,
                        y_train,
                        tscv,
                        'f1_weighted',
                        features_str,
                        model_name,
                        combo_list
                    ),
                    memoire
                )
                bytes_donnees = memoire.getvalue()

            use_cases__shared.put(bytes_donnees)
            uuu += 1

    for k in range(0, nb_worker):
        data_from_workers.extend(out__shared[k].get())

    results = data_from_workers

    assert len(results) == uuu == zzz, f"{len(results)=} == {uuu=} == {zzz=}"
    print("Évaluation CV terminée")

    # ---------------------------------------------------------
    # 6. Sélection des meilleurs et pires modèles
    # ---------------------------------------------------------
    df_sorted_results = pd.DataFrame(results).sort_values(by='CV_F1_Score', ascending=False).reset_index(drop=True)

    top_50 = df_sorted_results.head(50).copy()
    bottom_10 = df_sorted_results.tail(10).copy()
    models_to_test = pd.concat([top_50, bottom_10]).reset_index(drop=True)

    print("=" * 90)
    print(f"SÉLECTION : {len(top_50)} meilleurs modèles et {len(bottom_10)} pires modèles pour le test final.")
    print("=" * 90 + "\n")

    # ---------------------------------------------------------
    # 7. Évaluation finale sur le jeu de Test (Hold-out set)
    # ---------------------------------------------------------
    test_results = []
    print("Évaluation des modèles sélectionnés sur le jeu de TEST (données jamais utilisées)\n")

    for idx, row in tqdm(models_to_test.iterrows(), total=len(models_to_test), desc="Test Final"):
        combo_list = row['Features_str'].split("|")
        model_name = row['Model']

        X_train_sub = X_train[combo_list]
        X_test_sub = X_test[combo_list]

        data_type, model = get_models()[model_name]

        if data_type == 'scaled':
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_sub)
            X_te = scaler.transform(X_test_sub)
        else:
            X_tr = X_train_sub
            X_te = X_test_sub

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        test_f1 = f1_score(y_test, preds, average='weighted')

        test_results.append({
            'Model': model_name,
            'Num_Features': len(combo_list),
            'Features': combo_list,
            'CV_F1_Score': row['CV_F1_Score'],
            'Test_F1_Score': test_f1
        })

    df_test_results = pd.DataFrame(test_results)

    # ---------------------------------------------------------
    # 8. Affichage des résultats finaux
    # ---------------------------------------------------------
    df_test_top = df_test_results.head(10).sort_values(by='CV_F1_Score', ascending=False)
    df_test_bottom = df_test_results.tail(3).sort_values(by='CV_F1_Score', ascending=True)

    str_info_dataset = "Classés par F1-Score :: Entraînement sur TRAIN+VAL et Évaluation sur TEST"

    print("=" * 90)
    print(f"TOP 10 MODÈLES ({str_info_dataset})")
    print("=" * 90)

    for i, row in df_test_top.iterrows():
        print(
            f"[{row['Model']:<24}] CV: {row['CV_F1_Score']:.8f} | "
            f"TEST: {row['Test_F1_Score']:.8f} | Features: {', '.join(row['Features'])}"
        )

    print("\n" + "=" * 90)
    print(f"BOTTOM 3 MODÈLES ({str_info_dataset})")
    print("=" * 90)

    for i, row in df_test_bottom.iterrows():
        print(
            f"[{row['Model']:<24}] CV: {row['CV_F1_Score']:.8f} | "
            f"TEST: {row['Test_F1_Score']:.8f} | Features: {', '.join(row['Features'])}"
        )

    best_model = df_test_top.iloc[0]

    print("\n" + "=" * 90)
    print(f"🏆 MEILLEUR MODÈLE ABSOLU ({str_info_dataset}) 🏆")
    print("=" * 90)
    print(f"Modèle       : {best_model['Model']}")
    print(f"Nb Features  : {best_model['Num_Features']}")
    print(f"CV F1-Score  : {best_model['CV_F1_Score']:.8%}")
    print(f"TEST F1-Score: {best_model['Test_F1_Score']:.8%}")
    print(f"Features     : {', '.join(best_model['Features'])}")

    # ---------------------------------------------------------
    # 9. Sauvegarde des résultats finaux en CSV
    # ---------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    output_filename_csv = os.path.join(args.output_dir, f'_cv_all_combinations__{experience_str_desc}.csv')
    df_sorted_results.to_csv(output_filename_csv, index=False)

    df_test_results_sorted = df_test_results.sort_values(by='Test_F1_Score', ascending=False).reset_index(drop=True)
    df_test_results_sorted.to_csv(
        os.path.join(args.output_dir, f'_test_results__{experience_str_desc}.csv'),
        index=False
    )

    top_50.to_csv(
        os.path.join(args.output_dir, f'_top50_selection__{experience_str_desc}.csv'),
        index=False
    )

    # ---------------------------------------------------------
    # 10. Création du modèle agrégé TOP 10, 50, 250
    # ---------------------------------------------------------
    top_10  = df_sorted_results.head(10).copy()
    top_250 = df_sorted_results.head(250).copy()
    for name_top, top_n in zip(["top_10", "top_50", "top_250"], [top_10, top_50, top_250]):
        if len(top_n) == 0:
            print(f"[WARN] Aucun modèle disponible pour construire l'agrégat {name_top}.")
            continue

        print("=" * 90)
        print(f"CRÉATION DU MODÈLE AGRÉGÉ {name_top}")
        print("=" * 90)
        agg_test_f1 = None
        # -----------------------------------------------------
        # 10.1 Optionnel: évaluation honnête de l'agrégat sur TEST
        #      avec un modèle agrégé entraîné uniquement sur TRAIN.
        #      Cela donne une idée de la performance avant le fit final
        #      sur toutes les données.
        # -----------------------------------------------------
        try:
            eval_top_n_model = TopNEnsembleModel(threshold=0.5)
            eval_top_n_model.fit(
                X=X_train,
                y=y_train,
                top_models_df=top_n,
                model_factory=get_models_for_ensemble
            )

            agg_test = eval_top_n_model.predict_aggregated(X_test)
            agg_test_preds = agg_test['predicted_class'].to_numpy()
            agg_test_f1 = f1_score(y_test, agg_test_preds, average='weighted')

            print(f"[INFO] Agrégat {name_top} (entraîné sur TRAIN seulement) | TEST F1: {agg_test_f1:.8f}")
            print(
                f"[INFO] Moyenne médiane proba test: {agg_test['median_probability'].mean():.6f} | "
                f"Moyenne std proba test: {agg_test['std_probability'].mean():.6f}"
            )

            del eval_top_n_model
        except Exception as e:
            print(f"[WARN] Impossible d'évaluer l'agrégat {name_top} sur le test: {e}")

        # -----------------------------------------------------
        # 10.2 Modèle agrégé final entraîné sur TOUTES les données
        # -----------------------------------------------------
        final_top_n_model = TopNEnsembleModel(threshold=0.5, command_line=command_line, args=args)

        final_top_n_model.fit(
            X=X,
            y=y,
            top_models_df=top_n,
            model_factory=get_models_for_ensemble
        )

        ensemble_filename = os.path.join(
            args.output_dir,
            f'_{name_top}_ensemble_model__{experience_str_desc}__tf1_{agg_test_f1:.8f}.pkl'
        )

        final_top_n_model.save(ensemble_filename)

        print(f"✅ Modèle agrégé {name_top} sauvegardé: {ensemble_filename}")
        print("Réutilisation:")
        print(f"    model = TopNEnsembleModel.load({ensemble_filename!r})")
        print("    agg = model.predict_aggregated(X_new)")
        print("    # agg[['median_probability', 'std_probability', 'predicted_class']]")


if __name__ == "__main__":
    freeze_support()
    args = parse_arguments()
    entry(args=args)