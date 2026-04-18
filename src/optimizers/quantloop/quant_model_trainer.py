try:
    from version import sys__name, sys__version
except ImportError:
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
import os
import argparse
import itertools
import pickle
import random
from datetime import datetime
import time

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning, EfficiencyWarning
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, make_scorer)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from tqdm import tqdm


def parse_arguments():
    """
    Parse command-line arguments for the market/macro prediction script.
    All defaults match the original hardcoded values to ensure backward compatibility.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate time-series classifiers on combined market & macro data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        Examples:
          # Run with default settings
          python train_model.py

          # Run with a 10-minute time limit and save to custom directory
          python train_model.py --time-limit 600 --output-dir ./my_models

          # Change target type, use random search, and limit iterations
          python train_model.py --target-type higher --training-mode random --max-random-iterations 500
        """
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DATA & TARGET CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data & Target Configuration")
    data_grp.add_argument(
        "--dataset", "-d", type=str, default="combined_monthly_macro.data",
        help="Path to the pickled dataset containing 'market_data' and 'macro_data'."
    )
    data_grp.add_argument(
        "--n-test", "-nt", type=int, default=24,
        help="Number of final time steps (months) to reserve for out-of-sample testing."
    )
    data_grp.add_argument(
        "--target-type", "-tt", type=str, default="soft_higher",
        choices=["higher", "soft_higher", "lower", "soft_lower", "in_between"],
        help="Logic used to generate the binary prediction target."
    )
    data_grp.add_argument(
        "--target-percentage", "-tp", type=float, default=0.01,
        help="Threshold fraction for 'soft' or 'in_between' target types (e.g., 0.01 = 1pct)."
    )
    data_grp.add_argument(
        "--look-ahead", "-la", type=float, default=1.0,
        help="Number of periods to shift forward for target generation. Must be > 0."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURES & TECHNICAL INDICATORS
    # ─────────────────────────────────────────────────────────────────────────
    feat_grp = parser.add_argument_group("Features & Technical Indicators")
    feat_grp.add_argument(
        "--features", "-f", nargs="+", default=[
            "VIX_Ratio", "RSI", "Price_to_MA", "VIX", "MA_Short", "MA_Long",
            "Shifted_MA_Short", "Shifted_MA_Long", "Shifted_Price_to_MA",
            "VIX_Lag1", "RSI_Lag1", "Fed_Rate_Diff", "Unrate_Diff",
            "Inflation_Rate", "Spread_10Y2Y", "Log_Close", "Dist_from_ATH"
        ],
        help="List of column names to use as model features."
    )
    feat_grp.add_argument(
        "--vix-lag", "-vl", type=int, default=1,
        help="Lag period applied to the VIX feature."
    )
    feat_grp.add_argument(
        "--rsi-lag", "-rl", type=int, default=1,
        help="Lag period applied to the RSI feature."
    )
    feat_grp.add_argument(
        "--rsi-window", "-rw", type=int, default=14,
        help="Rolling window size for RSI calculation."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL & TRAINING CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model & Training Configuration")
    model_grp.add_argument(
        "--scaler", "-s", type=str, default="FunctionTransformer",
        choices=["RobustScaler", "StandardScaler", "FunctionTransformer"],
        help="Feature scaling method applied before model training."
    )
    model_grp.add_argument(
        "--estimator", "-e", type=str, default="RandomForestClassifier",
        choices=["RandomForestClassifier", "LinearSVC", "SVC", "SGDClassifier"],
        help="Machine learning classifier to use for prediction."
    )
    model_grp.add_argument(
        "--scorer", "-sc", type=str, default="F0.5",
        choices=["F0.5"],
        help="Metric used for hyperparameter optimization and evaluation."
    )
    model_grp.add_argument(
        "--training-mode", "-tm", type=str, default="random",
        choices=["exhaustive", "random"],
        help="Strategy for iterating through feature combinations."
    )
    model_grp.add_argument(
        "--max-random-iterations", "-mri", type=int, default=99999,
        help="Maximum number of feature sets to evaluate when --training-mode is 'random'."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTION & OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    exec_grp = parser.add_argument_group("Execution & Output")
    exec_grp.add_argument(
        "--time-limit", "-tl", type=float, default=None,
        help="Maximum time in seconds for the search process. None = no limit."
    )
    exec_grp.add_argument(
        "--output-dir", "-o", type=str, default="./output_models",
        help="Directory to save the best model and training metadata."
    )
    exec_grp.add_argument(
        "--no-verbose", action="store_false", dest="verbose", default=True,
        help="Disable detailed console output and progress reports."
    )

    args = parser.parse_args()

    # Map string choices to actual Python classes for downstream use
    scaler_map = {
        "RobustScaler": RobustScaler,
        "StandardScaler": StandardScaler,
        "FunctionTransformer": FunctionTransformer,
    }
    estimator_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "LinearSVC": svm.LinearSVC,
        "SVC": svm.SVC,
        "SGDClassifier": SGDClassifier
    }
    args.my_scaler = scaler_map[args.scaler]
    args.the_estimator = estimator_map[args.estimator]

    return args


def build_features_and_target(_df_market, _df_macro, _rsi_window, _vix_lag, _rsi_lag, _look_head_for_prediction, _percentage_of_type_target, _type_of_target, _verbose):
    # 3. Création de Features Macro (Lags et Variations)
    _df_macro['Spread_10Y2Y'] = _df_macro['T10Y2Y'].shift(1)
    _df_macro['Fed_Rate_Diff'] = _df_macro['FEDFUNDS'].diff().shift(1)
    _df_macro['Unrate_Diff'] = _df_macro['UNRATE'].diff().shift(1)
    _df_macro['Inflation_Rate'] = _df_macro['CPIAUCSL'].pct_change(12, fill_method=None).shift(1)

    # 4. Fusion avec tes données S&P 500
    monthly = _df_market.join(_df_macro[['Fed_Rate_Diff', 'Unrate_Diff', 'Inflation_Rate', 'Spread_10Y2Y']]).dropna()

    # 3. Ajout de nouvelles "Features"
    monthly['VIX_Ratio'] = monthly['VIX'] / monthly['VIX'].rolling(12).mean()

    delta = monthly['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=_rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=_rsi_window).mean()
    rs = gain / loss
    monthly['RSI'] = 100 - (100 / (1 + rs))

    monthly['MA_Short'] = monthly['Close'].rolling(window=3).mean()
    monthly['MA_Long'] = monthly['Close'].rolling(window=12).mean()
    monthly['Price_to_MA'] = monthly['Close'] / monthly['MA_Long']

    monthly['Shifted_MA_Short'] = monthly['MA_Short'].diff()
    monthly['Shifted_MA_Long'] = monthly['MA_Long'].pct_change()
    monthly['Shifted_Price_to_MA'] = monthly['Price_to_MA'].diff()

    monthly['VIX_Lag1'] = monthly['VIX'].shift(_vix_lag)
    monthly['RSI_Lag1'] = monthly['RSI'].shift(_rsi_lag)

    monthly['Log_Close'] = np.log(monthly['Close'])
    monthly['Dist_from_ATH'] = (monthly['Close'] / monthly['Close'].cummax()) - 1
    if _verbose:
        print(f"Dates in DF (before target):  {monthly.dropna().index[0].strftime('%Y-%m-%d')} :: {monthly.dropna().index[-1].strftime('%Y-%m-%d')}")

    # --- Cible (Target) ---
    # S'assurer que le décalage est un entier positif
    _look_head_for_prediction = int(_look_head_for_prediction)
    assert _look_head_for_prediction > 0
    assert 0 < _percentage_of_type_target < 1

    # 1. On récupère le prix futur (M + N) pour toutes les comparaisons
    future_close = monthly["Close"].shift(-_look_head_for_prediction)

    # 2. Calcul des Targets selon le type
    if _type_of_target == 'higher':
        # Est-ce que le prix futur est strictement supérieur au prix actuel ?
        monthly["Target"] = (future_close > monthly["Close"]).astype(int)

    elif _type_of_target == 'lower':
        # Est-ce que le prix futur est strictement inférieur au prix actuel ?
        monthly["Target"] = (future_close < monthly["Close"]).astype(int)

    elif _type_of_target == 'soft_higher':
        # Rendement cumulé entre maintenant et le futur
        # Formule : (Prix_Futur / Prix_Actuel) - 1
        monthly["Return"] = (future_close / monthly["Close"]) - 1
        monthly["Target"] = (monthly["Return"] > _percentage_of_type_target).astype(int)

    elif _type_of_target == 'soft_lower':
        # Rendement cumulé entre maintenant et le futur
        monthly["Return"] = (future_close / monthly["Close"]) - 1
        # On compare à l'opposé du pourcentage (ex: < -0.05 pour une baisse de 5%)
        monthly["Target"] = (monthly["Return"] < -_percentage_of_type_target).astype(int)

    elif _type_of_target == 'in_between':
        # Tunnel de prix autour du prix actuel
        lower_bound = monthly["Close"] * (1 - _percentage_of_type_target)
        upper_bound = monthly["Close"] * (1 + _percentage_of_type_target)
        monthly["Target"] = ((future_close >= lower_bound) & (future_close <= upper_bound)).astype(int)

    else:
        raise ValueError(f"Type de target inconnu : {_type_of_target}")

    # 3. Nettoyage (Optionnel mais recommandé)
    # Les dernières lignes seront des NaN à cause du shift() vers le futur.
    # Il faut les supprimer pour ne pas fausser l'entraînement.
    monthly.dropna(subset=["Target"], inplace=True)
    if _verbose:
        print(f"Dates in DF (after target):  {monthly.dropna().index[0].strftime('%Y-%m-%d')} :: {monthly.dropna().index[-1].strftime('%Y-%m-%d')}")
    return monthly


def save_best_model(best_setup, args, output_dir, verbose=True):
    """Saves the best model configuration to a pickle file in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    est_name = args.the_estimator.__name__
    target_name = args.target_type
    filename = f"best_model_{est_name}_{target_name}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)

    # Keep only inference-ready & metadata objects to avoid bloating the file
    save_data = {
        'best_setup': best_setup,
        'saved_at': datetime.now().isoformat()
    }

    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)

    if verbose:
        print(f"\n💾 Best model successfully saved to: {filepath}")


def entry_point(args):
    n_test = args.n_test
    type_of_target = args.target_type
    percentage_of_type_target = args.target_percentage
    look_head_for_prediction = args.look_ahead
    my_scaler = args.my_scaler
    features = args.features
    vix_lag = args.vix_lag
    rsi_lag = args.rsi_lag
    rsi_window = args.rsi_window
    verbose = args.verbose
    the_estimator = args.the_estimator
    the_scorer = args.scorer
    training_mode = args.training_mode
    max_random_iterations = args.max_random_iterations
    final_dataset_filename = args.dataset
    time_limit = args.time_limit
    output_dir = args.output_dir

    with open(final_dataset_filename, 'rb') as f:
        loaded_data = pickle.load(f)
    df_market = loaded_data["market_data"]
    df_macro = loaded_data["macro_data"]

    df = build_features_and_target(
        _df_market=df_market, _df_macro=df_macro,
        _rsi_window=rsi_window, _vix_lag=vix_lag, _rsi_lag=rsi_lag,
        _look_head_for_prediction=look_head_for_prediction,
        _percentage_of_type_target=percentage_of_type_target,
        _type_of_target=type_of_target, _verbose = args.verbose,
    )

    total_combinations = (2 ** len(features)) - 1
    if verbose:
        print(f"Nombre de features disponibles : {len(features)}")
        print(f"Total des combinaisons de features possibles : {total_combinations:,}  \n"
              f"Features: {features}")
        print(f"N test: {n_test}   Target: {type_of_target} , {percentage_of_type_target}   Look Ahead: {look_head_for_prediction}   Scaler: {my_scaler.__name__}   "
              f"Estimator: {the_estimator.__name__}   Scorer: {the_scorer}   Training: {training_mode}")

    if time_limit and verbose:
        print(f"⏱️ Time limit set to: {time_limit} seconds")

    def get_feature_combinations(_features, _mode):
        if _mode == "exhaustive":
            for r in range(1, len(_features) + 1):
                for combo in itertools.combinations(_features, r):
                    yield list(combo)
        elif _mode == "random":
            history = set()
            for _ in range(max_random_iterations):
                k = random.randint(1, len(_features))
                combo = tuple(sorted(set(random.sample(_features, k))))
                while combo in history:
                    k = random.randint(1, len(_features))
                    combo = tuple(sorted(set(random.sample(_features, k))))
                history.add(combo)
                yield list(combo)

    feature_iterator = get_feature_combinations(_features=features, _mode=training_mode)
    total_to_process = total_combinations if training_mode == "exhaustive" else max_random_iterations

    best_setup_found = {'train': {'train_score':0, 'test_score':0}, 'test': {'train_score':0, 'test_score':0}}
    displayed_output_once = False
    start_time = time.time()
    for selected_features in tqdm(feature_iterator, total=total_to_process, disable=not verbose):
        # ─────────────────────────────────────────────────────────────────────
        # ⏱️ Time limit check
        # ─────────────────────────────────────────────────────────────────────
        if time_limit and (time.time() - start_time) >= time_limit:
            if verbose:
                print(f"\n⏱️ Time limit of {time_limit} seconds reached. Stopping search gracefully...")
            break
        assert len(selected_features) == len(list(set(selected_features)))
        X = df[selected_features].copy()
        y = df["Target"].copy()

        assert n_test > 0
        assert n_test < len(X)
        X_train_full = X.iloc[:-n_test].copy()
        y_train_full = y.iloc[:-n_test].copy()
        X_test_final = X.iloc[-n_test:].copy()
        y_test_final = y.iloc[-n_test:].copy()

        if not displayed_output_once and verbose:
            print(f"\n[STATIC] {len(X_train_full)=}  {len(X_test_final)=}")
            print(f"[TRAIN]  {X_train_full.index[0].strftime('%Y-%m-%d')} :: {X_train_full.index[-1].strftime('%Y-%m-%d')}")
            print(f"[TEST]   {X_test_final.index[0].strftime('%Y-%m-%d')} :: {X_test_final.index[-1].strftime('%Y-%m-%d')}")

        scaler = my_scaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_final)

        # Param grids based on estimator
        if the_estimator == svm.SVC:
            param_dist = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'probability': [True]
            }
        elif the_estimator == svm.LinearSVC:
            param_dist = {
                'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000],
                'penalty': ['l2'],
                'loss': ['hinge', 'squared_hinge'],
                'tol': [1e-4, 1e-3, 1e-2],
                'max_iter': [2000, 4000]
            }
        elif the_estimator == RandomForestClassifier:
            param_dist = {
                'n_estimators': [200, 500],
                'max_depth': [1, 3, 5, 8, 10, None],
                'min_samples_split': [2, 10, 20, 50],
                'min_samples_leaf': [1, 5, 10, 20],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True]
            }
        elif the_estimator == SGDClassifier:
            param_dist = {
                'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': [1e-4, 1e-3, 1e-2, 0.1],
                'max_iter': [8000, 16000],
                'tol': [1e-3, 1e-2],
                'class_weight': [None, 'balanced'],
            }
        else:
            assert False, f"{the_estimator}"

        tscv = TimeSeriesSplit(n_splits=5)
        scoring_sl1, scoring_sl2 = None, None
        if the_scorer == 'F0.5':
            scoring_sl1, scoring_sl2 = make_scorer(fbeta_score, beta=0.5, zero_division=0), fbeta_score
        assert scoring_sl1 is not None and scoring_sl2 is not None
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        random_search = RandomizedSearchCV(
            estimator=the_estimator(random_state=1),
            param_distributions=param_dist,
            scoring=scoring_sl1,
            n_iter=100,
            cv=tscv,
            n_jobs=-1,
            random_state=1,
            verbose=0
        )

        random_search.fit(X_train_scaled, y_train_full)
        best_model = random_search.best_estimator_
        train_score = random_search.best_score_
        test_preds = best_model.predict(X_test_scaled)
        test_score = scoring_sl2(y_test_final, test_preds, beta=0.5, zero_division=0)
        _tmp_update_snapshot, do_display = {'test_score': test_score, 'train_score': train_score,
                                'model': best_model, 'estimator': the_estimator, 'type_of_target': type_of_target,
                                'features': selected_features, 'scaler': scaler, 'scorer': the_scorer,
                                'X_train_full__before_scaled': X_train_full, 'X_train_scaled': X_train_scaled,
                                'y_train_full': y_train_full, 'X_test_final__before_scaled': X_test_final,
                                'X_test_scaled': X_test_scaled, 'y_test_final': y_test_final,
                                'X': X, 'y': y, 'n_test': n_test}, False
        if train_score >= best_setup_found['train']['train_score']:
            cd1 = train_score > best_setup_found['train']['train_score']
            cd2 = train_score == best_setup_found['train']['train_score'] and test_score > best_setup_found['train']['test_score']
            if cd1 or cd2:
                best_setup_found['train'], do_display = _tmp_update_snapshot, True
        if test_score >= best_setup_found['test']['test_score']:
            cd1 = test_score > best_setup_found['test']['test_score']
            cd2 = test_score == best_setup_found['test']['test_score'] and train_score > best_setup_found['test']['train_score']
            if cd1 or cd2:
                best_setup_found['test'], do_display = _tmp_update_snapshot, True
        if verbose and (not displayed_output_once or do_display):
            for category in ['train', 'test']:
                data = best_setup_found[category]
                _type_target_str = f"TARGET:{type_of_target}" if type_of_target in ["higher","lower"] else f"TARGET:{type_of_target} @{percentage_of_type_target*100:.0f}%"
                print("\n" + "═" * 70)
                print(f"⭐ BEST SETUP RECORDED FOR: {category.upper()}  |  TARGET:{type_of_target}")
                print("═" * 70)

                # Accessing keys directly from the dictionary
                print(f"Features         : {data['features']}")
                print(f"Scaler           : {data['scaler']}")
                print(f"Scorer           : {data['scorer']}")

                # Displaying the scores saved in this specific snapshot
                print(f"\nScores for this snapshot:")
                print(f"  - Train {data['scorer']}: {data['train_score']:.2%}")
                print(f"  - Test  {data['scorer']}: {data['test_score']:.2%}")

                # We need to generate predictions using the saved model and scaled data
                # to show the classification report for this specific 'best' setup
                saved_preds = data['model'].predict(data['X_test_scaled'])

                print(f"\nClassification Report ({category} snapshot):")
                print(classification_report(data['y_test_final'], saved_preds, zero_division=0))
                print(f"\n"
                      f"y    : {data['y_test_final'].values}\n"
                      f"y_hat: {saved_preds}\n")
                print("Confusion Matrix:")
                print(confusion_matrix(data['y_test_final'], saved_preds))
                print("═" * 70)
        displayed_output_once = True

    # ─────────────────────────────────────────────────────────────────────────
    # 💾 Save the best model after loop finishes or breaks
    # ─────────────────────────────────────────────────────────────────────────
    if best_setup_found is not None:
        save_best_model(best_setup_found, args, output_dir, verbose)
    else:
        if verbose:
            print("\n⚠️ No valid model was trained. Nothing to save.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    entry_point(parsed_args)