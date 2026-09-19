import time
import matplotlib.pyplot as plt
from fetchers.data_factory import factory_load_data
import seaborn as sns
import os
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
import time
from datetime import datetime
import pickle
import io
from multiprocessing import Process, Queue, freeze_support, Value
import warnings
# Ignore cet avertissement spécifique lié à sklearn parallel
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return ''.join(parts)


def _worker_processor(use_cases__shared, master_cmd__shared, out__shared, k):
    # Attendre le Go du master
    while True:
        with master_cmd__shared.get_lock():
            if 0 != master_cmd__shared.value:
                break
        time.sleep(0.333)

    # Traitement des requêtes
    all_results_computed, best_mean_cv_f1, times = [], 0., [0]
    while True:
        t1 = time.time()
        try:
            tmp = use_cases__shared.get(timeout=0.033)
        except:
            time.sleep(1.033)
            try:
                tmp = use_cases__shared.get(timeout=0.033)
            except:
                time.sleep(3.033)
                try:
                    tmp = use_cases__shared.get(timeout=0.033)
                except:
                    break
        try:
            with io.BytesIO(tmp) as memoire:
                # On recharge le tuple et on réaffecte chaque variable dans le même ordre
                pipe, X_train_sub, y_train, tscv, scoring, features_str, model_name, combo_list = pickle.load(memoire)
            cv_scores = cross_val_score(pipe, X_train_sub, y_train, cv=tscv, scoring='f1_weighted', n_jobs=None)
            mean_cv_f1 = cv_scores.mean()
            all_results_computed.append({
                'Model': model_name,
                'Num_Features': len(combo_list),
                'Features_str': features_str,
                'CV_F1_Score': mean_cv_f1
            })
            if mean_cv_f1 > best_mean_cv_f1: best_mean_cv_f1 = mean_cv_f1
            if 0 == len(all_results_computed)%500:
                print(f"[{str(os.getpid()):<6}:{str(k):<4}]  {best_mean_cv_f1:8.8f} :: {mean_cv_f1:8.8f}  | {len(all_results_computed):6d} cases computed  | Mean time per pass: {format_duration(int(np.mean(times)))}")
        except Exception as e:
            print(e)
        t2 = time.time()
        times.append(t2-t1)
    out__shared.put(all_results_computed)
    print(f"[{os.getpid()}:{k}]  Quitting")


def entry():
    # ---------------------------------------------------------
    # 1. Chargement et préparation des données
    # ---------------------------------------------------------
    nb_worker = int(os.environ.get("Q__N_CORE", 12))
    ticker = "^GSPC"
    df = factory_load_data(_dataset_id="intraday_1min", _ticker=ticker, _args={})

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    print(f"{len(df)} :: {df.index[0]} --> {df.index[-1]}")
    daily_close_1600 = df.groupby(df.index.date)['Close'].last()

    # Récupérer les variables du shell
    heure_debut_str = os.environ.get("Q__HEURE_DEBUT", "09:30")
    heure_fin_str = os.environ.get("Q__HEURE_FIN", "10:30")

    time_debut = datetime.strptime(heure_debut_str, "%H:%M").time()
    time_fin = datetime.strptime(heure_fin_str, "%H:%M").time()

    t_target = os.environ.get("Q__TARGET", "Morning_Open")
    print(f"Target is: {t_target}")
    ema_9  = int(os.environ.get("Q__EMA9", 9))
    ema_21 = int(os.environ.get("Q__EMA21", 21))
    rsi_14 = int(os.environ.get("Q__RSI14", 14))
    macd_5 = int(os.environ.get("Q__MACD5", 5))
    fenetre_analysee = (time_debut, time_fin)
    seuil_pos = float(os.environ.get("Q__SEUIL_POS", 1.))
    experience_str_desc = f"{seuil_pos}"+f"__{t_target}__"+f"__{ema_9}{ema_21}{rsi_14}{macd_5}__" + f"__{fenetre_analysee}".replace('datetime.time', '').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
    print(f"{fenetre_analysee=} | {seuil_pos=} | {experience_str_desc=}")

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
        'pct_above_vwap_morning': pct_above_vwap_series, 'last_vwap_slope': last_slope_series, 'last_vwap_deriv': last_deriv_series,
        'morning_range_pct': morning_range_pct, 'breakout_signal': breakout_signal, 'up_volume_ratio': up_volume_ratio,
        'vwap_distance_pct': vwap_distance_pct, 'vwap_crossings': vwap_crossings,
        'rsi_last': rsi_last, 'rsi_delta': rsi_delta, 'close_ema9_dist_pct': close_ema9_dist_pct, 'ema_cross_direction': ema_cross_direction,
        'macd_hist_last': macd_hist_last, 'macd_hist_delta': macd_hist_delta,
        'Morning_Open': daily_morning_open, 'Morning_Close': daily_morning_close,
        'gap_pct': ((daily_morning_open - prev_daily_morning_close) / prev_daily_morning_close) * 100,
        'prev_day_return_t1': ((daily_morning_close - prev_daily_morning_close) / prev_daily_morning_close) * 100,
        'prev_day_return_t2': ((daily_morning_close - prev_daily_morning_close.shift(1)) / prev_daily_morning_close.shift(1)) * 100,
        'prev_day_return_t3': ((daily_morning_close - prev_daily_morning_close.shift(2)) / prev_daily_morning_close.shift(2)) * 100,
        'prev_day_return_t4': ((daily_morning_close - prev_daily_morning_close.shift(3)) / prev_daily_morning_close.shift(3)) * 100,
    })
    daily_df['target'] = (daily_close_1600.reindex(daily_df.index) > daily_df[t_target] * seuil_pos).astype(int)
    daily_df = daily_df.dropna()

    # ---------------------------------------------------------
    # 3. Préparation des données et Split Temporel
    # ---------------------------------------------------------
    feature_cols = ['pct_above_vwap_morning', 'last_vwap_slope', 'last_vwap_deriv',
                    'morning_range_pct', 'breakout_signal', 'up_volume_ratio',
                    'vwap_distance_pct', 'vwap_crossings',
                    'rsi_last', 'rsi_delta', 'close_ema9_dist_pct', 'ema_cross_direction',
                    'macd_hist_last', 'macd_hist_delta',
                    'gap_pct', 'prev_day_return_t1', 'prev_day_return_t2',
                    # 'prev_day_return_t3', 'prev_day_return_t4',
                    # 'Morning_Open', 'Morning_Close',
    ]

    X = daily_df[feature_cols].copy()
    y = daily_df['target'].copy()

    # Split temporel strict (90% Train pour CV / 10% Test pour évaluation finale)
    split_idx = int(len(daily_df) * 0.90)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[TRAIN] [{len(X_train)}]  {X_train.index[0].strftime('%Y-%m-%d')} --> {X_train.index[-1].strftime('%Y-%m-%d')}")
    print(f"[TEST]  [{len(X_test)}]   {X_test.index[0].strftime('%Y-%m-%d')} --> {X_test.index[-1].strftime('%Y-%m-%d')}\n")

    # ---------------------------------------------------------
    # 4. Configuration de la Cross-Validation et des Modèles
    # ---------------------------------------------------------
    # TimeSeriesSplit est crucial pour les séries temporelles afin d'éviter le look-ahead bias
    tscv = TimeSeriesSplit(n_splits=12)

    def get_models():
        """Retourne une instance fraîche de chaque modèle et son type de prétraitement."""
        return {
            'LogisticRegression': ('scaled', LogisticRegression(random_state=42, max_iter=1000)),
            'RandomForest': ('raw', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
            'SVM': ('scaled', SVC(kernel='rbf', C=1.0, random_state=42, class_weight="balanced")),
            'GradientBoosting': ('raw', GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42))
        }


    # ---------------------------------------------------------
    # 5. Évaluation de toutes les combinaisons via Cross-Validation
    # ---------------------------------------------------------
    data_from_workers = []
    # Construction des cas à traiter
    combo_to_be_processed, required_set = [], ("pct_above_vwap_morning", "last_vwap_slope", "vwap_distance_pct", "up_volume_ratio","prev_day_return_t1", "last_vwap_deriv", "rsi_last", "macd_hist_delta", "macd_hist_last", "ema_cross_direction")
    required_set = set(required_set)
    for r in range(1, len(feature_cols) + 1):
        for combo in combinations(feature_cols, r):
            #if required_set.issubset(combo):
            combo_to_be_processed.append(list(combo))
    zzz = len(combo_to_be_processed) * len(get_models())
    # Variables partagées
    use_cases__shared, master_cmd__shared = Queue(2 * zzz), Value("i", 0)
    out__shared = [Queue(1) for k in range(0, nb_worker)]
    print(f"Lancement des evaluations en CV | Queue de dimension {2*zzz}")
    # Lancement des workers
    for k in range(0, nb_worker):
        p = Process(target=_worker_processor, args=(use_cases__shared, master_cmd__shared, out__shared[k], k,))
        p.start()
    # Envoie les informations aux workers pour traitement
    print(f"Evaluation CV de {len(combo_to_be_processed)} combinaisons sur {len(get_models())} modeles avec {nb_worker} workers...")
    pbar = tqdm(combo_to_be_processed, desc="Submitting training with CV")
    uuu = 0
    for combo_list in pbar:
        X_train_sub = X_train[combo_list]
        # On joint les features en string pour éviter les problèmes de sérialisation dans Pandas
        features_str = "|".join(combo_list)
        for model_name, (data_type, model) in get_models().items():
            # Création d'un Pipeline pour éviter les fuites de données lors du scaling dans la CV
            if data_type == 'scaled':
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            else:
                pipe = Pipeline([('model', model)])
            with io.BytesIO() as memoire:
                pickle.dump((pipe, X_train_sub, y_train, tscv, 'f1_weighted', features_str, model_name, combo_list), memoire)
                bytes_donnees = memoire.getvalue()
            use_cases__shared.put(bytes_donnees)
            uuu += 1
    # Autoriser les workers à traiter
    with master_cmd__shared.get_lock():
        master_cmd__shared.value = 1
    # Récupération des résultats
    for k in range(0, nb_worker):
        data_from_workers.extend(out__shared[k].get())
    results = data_from_workers
    print(f"{len(results)=} == {uuu=} == {zzz=}")
    assert len(results) == uuu == zzz
    print("Évaluation CV terminée")

    # ---------------------------------------------------------
    # 6. Sélection des meilleurs et pires modèles
    # ---------------------------------------------------------
    df_sorted_results = pd.DataFrame(results).sort_values(by='CV_F1_Score', ascending=False).reset_index(drop=True)

    # Sélection du Top 50 et Bottom 10
    top_50 = df_sorted_results.head(50)
    bottom_10 = df_sorted_results.tail(10)
    models_to_test = pd.concat([top_50, bottom_10]).reset_index(drop=True)

    print("=" * 90)
    print(f"SÉLECTION : {len(top_50)} meilleurs modèles et {len(bottom_10)} pires modèles pour le test final.")
    print("=" * 90 + "\n")

    # ---------------------------------------------------------
    # 7. Évaluation finale sur le jeu de Test (Hold-out set)
    # ---------------------------------------------------------
    test_results = []
    print("Évaluation des modèles sélectionnés sur le jeu de TEST (données jamais vues)...\n")

    for idx, row in tqdm(models_to_test.iterrows(), total=len(models_to_test), desc="Test Final"):
        combo_list = row['Features_str'].split("|")
        model_name = row['Model']

        X_train_sub = X_train[combo_list]
        X_test_sub = X_test[combo_list]

        data_type, model = get_models()[model_name]  # Récupère une instance fraîche

        # Scaling sur le train complet, puis transformation du test
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
    # On sépare à nouveau pour l'affichage
    df_test_top = df_test_results.head(50).sort_values(by='CV_F1_Score', ascending=False)
    df_test_bottom = df_test_results.tail(10).sort_values(by='CV_F1_Score', ascending=True)

    print("=" * 90)
    print(f"TOP 50 MODÈLES (Classés par F1-Score sur le TRAIN/VAL)")
    print("=" * 90)
    for i, row in df_test_top.iterrows():
        print(f"[{row['Model']:<24}] CV: {row['CV_F1_Score']:.8f} | TEST: {row['Test_F1_Score']:.8f} | Features: {', '.join(row['Features'])}")

    print("\n" + "=" * 90)
    print(f"BOTTOM 10 MODÈLES (Classés par F1-Score sur le TRAIN/VAL)")
    print("=" * 90)
    for i, row in df_test_bottom.iterrows():
        print(f"[{row['Model']:<24}] CV: {row['CV_F1_Score']:.8f} | TEST: {row['Test_F1_Score']:.8f} | Features: {', '.join(row['Features'])}")

    # Affichage du champion absolu
    best_model = df_test_top.iloc[0]
    print("\n" + "=" * 90)
    print("🏆 MEILLEUR MODÈLE ABSOLU (Sur le jeu de TEST) 🏆")
    print("=" * 90)
    print(f"Modèle       : {best_model['Model']}")
    print(f"Nb Features  : {best_model['Num_Features']}")
    print(f"CV F1-Score  : {best_model['CV_F1_Score']:.8%}")
    print(f"TEST F1-Score: {best_model['Test_F1_Score']:.8%}")
    print(f"Features     : {', '.join(best_model['Features'])}")

    # ---------------------------------------------------------
    # 9. Sauvegarde des résultats finaux en CSV
    # ---------------------------------------------------------
    # df_results (CV) — déjà trié par CV_F1_Score dans le code existant
    output_filename_csv = f'model_ranking_cv_all_combinations__{experience_str_desc}.csv'
    df_sorted_results.to_csv(output_filename_csv,index=False)
    # df_test_results (Test) — trié par Test_F1_Score décroissant
    df_test_results_sorted = df_test_results.sort_values(by='Test_F1_Score', ascending=False).reset_index(drop=True)
    df_test_results_sorted.to_csv(f'model_ranking_test_results__{experience_str_desc}.csv',index=False)
    print(f"\n✅ CSV sauvegardés :")
    print(f"   → model_ranking__cv_all_combinations__{experience_str_desc}.csv  ({len(df_sorted_results)} lignes)")
    print(f"   → model_ranking__test_results__{experience_str_desc}.csv          ({len(df_test_results_sorted)} lignes)")

    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    # Pour ton vrai fichier, décommente la ligne suivante :
    df = pd.read_csv(output_filename_csv)

    # Transformation de la chaîne de caractères en liste pour faciliter l'analyse
    df['Features_List'] = df['Features_str'].str.split('|')

    # ==========================================
    # 2. ANALYSE 1 : Le Top des combinaisons
    # ==========================================
    print("🏆 TOP 5 DES MEILLEURES COMBINAISONS (Meilleur F1 Score) :\n")
    top_models = df.sort_values(by='CV_F1_Score', ascending=False).head(5)
    for i, row in top_models.iterrows():
        print(f"Score: {row['CV_F1_Score']:.5f} | Modèle: {row['Model']} | Nb Features: {row['Num_Features']}")
        print(f"  -> Features: {row['Features_str']}\n")

    # ==========================================
    # 3. ANALYSE 2 : Les features "Must-Have" (Fréquence dans le Top 20%)
    # ==========================================
    # On garde les 20% meilleurs modèles pour voir quelles features reviennent le plus souvent
    top_20_percent = df.sort_values(by='CV_F1_Score', ascending=False).head(int(len(df) * 0.20))
    exploded_top = top_20_percent.explode('Features_List')
    feature_freq_top = exploded_top['Features_List'].value_counts()

    print("\n🔥 FEATURES LES PLUS FRÉQUENTES DANS LE TOP 20% DES MODÈLES :\n")
    print(feature_freq_top.head(10))

    # ==========================================
    # 4. ANALYSE 3 : Impact moyen de chaque feature
    # ==========================================
    # Quelle est la moyenne des scores quand une feature est présente vs absente ?
    all_features = set(exploded_top['Features_List'])
    feature_impact = []

    for feat in all_features:
        # Modèles qui contiennent la feature
        mask_with = df['Features_List'].apply(lambda x: feat in x)
        mean_with = df[mask_with]['CV_F1_Score'].mean()

        feature_impact.append({
            'Feature': feat,
            'Avg_Score_When_Present': mean_with,
            'Count_In_Top20': feature_freq_top.get(feat, 0)
        })

    impact_df = pd.DataFrame(feature_impact).sort_values(by='Avg_Score_When_Present', ascending=False)

    print("\n📈 IMPACT MOYEN DES FEATURES (Score moyen quand la feature est utilisée) :\n")
    print(impact_df.head(10))

    # ==========================================
    # 5. ANALYSE 4 : Parcimonie (Nombre de features vs Performance)
    # ==========================================
    # Est-ce que 12 features sont vraiment mieux que 6 ?
    perf_by_num = df.groupby('Num_Features')['CV_F1_Score'].agg(['mean', 'max', 'count']).reset_index()

    print("\n📉 PERFORMANCE SELON LE NOMBRE DE FEATURES :\n")
    print(perf_by_num)

    # ==========================================
    # 6. VISUALISATIONS
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Graphique 1: Fréquence dans le Top 20%
    sns.barplot(x=feature_freq_top.head(10).values, y=feature_freq_top.head(10).index, ax=axes[0], palette='viridis')
    axes[0].set_title('Top 10 Features (Présentes dans le Top 20% des modèles)', fontsize=12)
    axes[0].set_xlabel('Nombre d\'apparitions')
    axes[0].set_ylabel('Feature')

    # Graphique 2: Score max et moyen par nombre de features
    sns.lineplot(data=perf_by_num, x='Num_Features', y='max', marker='o', label='Score Max', ax=axes[1], color='red')
    sns.lineplot(data=perf_by_num, x='Num_Features', y='mean', marker='s', label='Score Moyen', ax=axes[1], color='blue')
    axes[1].set_title('Impact du nombre de features sur le F1-Score', fontsize=12)
    axes[1].set_xlabel('Nombre de Features')
    axes[1].set_ylabel('F1-Score')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    freeze_support()
    entry()