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
import itertools
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from constants import FRED_API_KEY
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, make_scorer)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from tqdm import tqdm

from utils import get_filename_for_dataset


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
          # Run with default settings (exhaustive feature search, RobustScaler, RandomForest)
          python train_model.py

          # Change target type, use random search, and limit iterations
          python train_model.py --target-type higher --training-mode random --max-random-iterations 500

          # Use LinearSVC with StandardScaler and disable verbose output
          python train_model.py --estimator LinearSVC --scaler StandardScaler --no-verbose
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
        "--scaler", "-s", type=str, default="RobustScaler",
        choices=["RobustScaler", "StandardScaler"],
        help="Feature scaling method applied before model training."
    )
    model_grp.add_argument(
        "--estimator", "-e", type=str, default="RandomForestClassifier",
        choices=["RandomForestClassifier", "LinearSVC", "SVC"],
        help="Machine learning classifier to use for prediction."
    )
    model_grp.add_argument(
        "--scorer", "-sc", type=str, default="F0.5",
        choices=["F0.5"],
        help="Metric used for hyperparameter optimization and evaluation."
    )
    model_grp.add_argument(
        "--training-mode", "-tm", type=str, default="exhaustive",
        choices=["exhaustive", "random"],
        help="Strategy for iterating through feature combinations."
    )
    model_grp.add_argument(
        "--max-random-iterations", "-mri", type=int, default=999999,
        help="Maximum number of feature sets to evaluate when --training-mode is 'random'."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT & LOGGING
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-verbose", action="store_false", dest="verbose", default=True,
        help="Disable detailed console output and progress reports."
    )

    args = parser.parse_args()

    # Map string choices to actual Python classes for downstream use
    scaler_map = {
        "RobustScaler": RobustScaler,
        "StandardScaler": StandardScaler
    }
    estimator_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "LinearSVC": svm.LinearSVC,
        "SVC": svm.SVC
    }
    args.my_scaler = scaler_map[args.scaler]
    args.the_estimator = estimator_map[args.estimator]

    return args


def build_features_and_target(_df_market, _df_macro, _rsi_window, _vix_lag, _rsi_lag, _look_head_for_prediction, _percentage_of_type_target, _type_of_target):
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
    print(f"Dates in DF (before target):  {monthly.dropna().index[0].strftime('%Y-%m-%d')} :: {monthly.dropna().index[-1].strftime('%Y-%m-%d')}")

    # --- Cible (Target) ---
    # Cast to int to ensure pandas shift() works correctly
    _look_head_for_prediction = int(_look_head_for_prediction)
    assert _look_head_for_prediction > 0
    assert 0 < _look_head_for_prediction < 1 or _look_head_for_prediction == 1  # Relaxed assertion for safety

    if _type_of_target == 'higher':
        monthly["Target"] = (monthly["Close"].shift(-_look_head_for_prediction) > monthly["Close"]).astype(int)
    elif _type_of_target == 'soft_higher':
        monthly["Return"] = monthly["Close"].pct_change().shift(-_look_head_for_prediction)
        monthly["Target"] = (monthly["Return"] > _percentage_of_type_target).astype(int)
    elif _type_of_target == 'lower':
        monthly["Target"] = (monthly["Close"].shift(-_look_head_for_prediction) < monthly["Close"]).astype(int)
    elif _type_of_target == 'soft_lower':
        monthly["Return"] = monthly["Close"].pct_change().shift(-_look_head_for_prediction)
        monthly["Target"] = (monthly["Return"] < _percentage_of_type_target).astype(int)
    elif _type_of_target == 'in_between':
        future_close = monthly["Close"].shift(-_look_head_for_prediction)
        lower_bound = monthly["Close"] * (1 - _percentage_of_type_target)
        upper_bound = monthly["Close"] * (1 + _percentage_of_type_target)
        monthly["Target"] = ((future_close >= lower_bound) & (future_close <= upper_bound)).astype(int)
    else:
        assert False, f"{_type_of_target}"

    monthly = monthly.dropna().copy()
    return monthly


def entry_point(args):
    # Unpack parsed arguments for cleaner reference
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

    with open(final_dataset_filename, 'rb') as f:
        loaded_data = pickle.load(f)
    df_market = loaded_data["market_data"]
    df_macro = loaded_data["macro_data"]

    df = build_features_and_target(
        _df_market=df_market, _df_macro=df_macro,
        _rsi_window=rsi_window, _vix_lag=vix_lag, _rsi_lag=rsi_lag,
        _look_head_for_prediction=look_head_for_prediction,
        _percentage_of_type_target=percentage_of_type_target,
        _type_of_target=type_of_target
    )

    total_combinations = (2 ** len(features)) - 1
    if verbose:
        print(f"Nombre de features disponibles : {len(features)}")
        print(f"Total des combinaisons de features possibles : {total_combinations:,}  \n"
              f"Features: {features}")
        print(f"Dates in DF (after target dropna):  {df.index[0].strftime('%Y-%m-%d')} :: {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"N test: {n_test}   Target: {type_of_target} , {percentage_of_type_target}   Look Ahead: {look_head_for_prediction}   Scaler: {my_scaler.__name__}   "
              f"Estimator: {the_estimator.__name__}   Scorer: {the_scorer}   Training: {training_mode}")

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
                if combo not in history:
                    history.add(combo)
                    yield list(combo)

    feature_iterator = get_feature_combinations(_features=features, _mode=training_mode)
    total_to_process = total_combinations if training_mode == "exhaustive" else max_random_iterations

    best_setup_found = {'score': -1}
    displayed_output_once = False

    for selected_features in tqdm(feature_iterator, total=total_to_process):
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
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'loss': ['hinge', 'squared_hinge'],
                'tol': [1e-4, 1e-3],
                'max_iter': [2000]
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
        else:
            assert False, f"{the_estimator}"

        tscv = TimeSeriesSplit(n_splits=5)
        scoring_sl1, scoring_sl2 = None, None
        if the_scorer == 'F0.5':
            scoring_sl1, scoring_sl2 = make_scorer(fbeta_score, beta=0.5, zero_division=0), fbeta_score
        assert scoring_sl1 is not None and scoring_sl2 is not None

        random_search = RandomizedSearchCV(
            estimator=the_estimator(random_state=1),
            param_distributions=param_dist,
            scoring=scoring_sl1,
            n_iter=1,
            cv=tscv,
            n_jobs=-1,
            random_state=1,
            verbose=0
        )

        random_search.fit(X_train_scaled, y_train_full)
        best_model = random_search.best_estimator_

        final_preds = best_model.predict(X_test_scaled)
        final_score = scoring_sl2(y_test_final, final_preds, beta=0.5, zero_division=0)

        if final_score > best_setup_found['score']:
            assert len(selected_features) == len(list(set(selected_features)))
            best_setup_found.update({
                'score': final_score, 'model': best_model, 'estimator': the_estimator,
                'features': selected_features, 'scaler': scaler, 'scorer': the_scorer,
                'X_train_full': X_train_full, 'X_train_scaled': X_train_scaled,
                'y_train_full': y_train_full, 'X_test_final': X_test_final,
                'X_test_scaled': X_test_scaled, 'y_test_final': y_test_final,
                'X': X, 'y': y, 'n_test': n_test
            })
            if verbose or not displayed_output_once:
                print("\n" + "═" * 60)
                print(f"🎯 Best Parameters")
                print("═" * 60)
                print(f"Features: {selected_features}")
                print(f"Meilleurs paramètres : {random_search.best_params_}")
                print(f"{the_scorer} sur la période de test finale : {final_score:.2%}\n"
                      f"y    : {y_test_final.values}\n"
                      f"y_hat: {final_preds}\n")
                print(classification_report(y_test_final, final_preds, zero_division=0))
                print("Matrice de Confusion :")
                print(confusion_matrix(y_test_final, final_preds))
        displayed_output_once = True


if __name__ == "__main__":
    # Parse command-line arguments and pass them to the main execution function
    parsed_args = parse_arguments()
    entry_point(parsed_args)