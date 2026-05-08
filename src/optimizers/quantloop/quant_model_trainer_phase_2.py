import os
import random
import numpy as np
import optuna

# ─────────────────────────────────────────────────────────────────────────
# 🔒 DETERMINISM SETUP
# ─────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# Force single-threaded BLAS/OMP to guarantee identical floating-point accumulation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
import pickle
from datetime import datetime
import time
from pathlib import Path
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning, EfficiencyWarning
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, f1_score, make_scorer)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from tqdm import tqdm
from utils import get_next_step
from optimizers.quantloop.quant_model_trainer import save_best_model, build_features_and_target


def parse_arguments():
    """
    Parse command-line arguments for the market/macro prediction script.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate time-series classifiers on combined market & macro data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        Examples:
          # Train normally (random search)
          python train_model.py

          # Real-time prediction (uses latest model in ./output_models)
          python train_model.py --real-time

          # 🔹 Use Optuna to optimize feature engineering params with a SPECIFIC feature list
          python train_model.py --training-mode fixed --features VIX_Ratio RSI Log_Close Spread_10Y2Y --optuna-trials 50
        """
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DATA & TARGET CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────
    data_grp = parser.add_argument_group("Data & Target Configuration")
    data_grp.add_argument(
        "--dataset", "-d", type=str, default="combined_month_macro.data",
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
        choices=["F0.5", "F", "F2"],
        help="Metric used for hyperparameter optimization and evaluation."
    )
    model_grp.add_argument(
        "--training-mode", "-tm", type=str, default="fixed",
        choices=["fixed"],
        help="Strategy for iterating through feature combinations. 'fixed' uses exactly the features provided via --features without searching."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # OPTUNA CONFIGURATION (Fixed Mode Only)
    # ─────────────────────────────────────────────────────────────────────────
    optuna_grp = parser.add_argument_group("Optuna Configuration (Fixed Mode Only)")
    optuna_grp.add_argument(
        "--optuna-trials", type=int, default=50,
        help="Number of Optuna trials for feature engineering parameter optimization."
    )
    optuna_grp.add_argument(
        "--optuna-rsi-window-min", type=int, default=5,
        help="Lower bound for Optuna RSI window search."
    )
    optuna_grp.add_argument(
        "--optuna-rsi-window-max", type=int, default=30,
        help="Upper bound for Optuna RSI window search."
    )
    optuna_grp.add_argument(
        "--optuna-lag-min", type=int, default=1,
        help="Lower bound for Optuna VIX/RSI lag search."
    )
    optuna_grp.add_argument(
        "--optuna-lag-max", type=int, default=6,
        help="Upper bound for Optuna VIX/RSI lag search."
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


def run_training_pipeline(_df, _selected_features, _args):
    """Reusable function to scale data, run CV hyperparameter search, and evaluate."""
    n_test = _args.n_test
    my_scaler = _args.my_scaler
    the_estimator = _args.the_estimator
    the_scorer = _args.scorer
    verbose = _args.verbose

    X = _df[_selected_features].copy()
    y = _df["Target"].copy()

    assert n_test > 0
    assert n_test < len(X)
    X_train_full = X.iloc[:-n_test].copy()
    y_train_full = y.iloc[:-n_test].copy()
    X_test_final = X.iloc[-n_test:].copy()
    y_test_final = y.iloc[-n_test:].copy()

    scaler = my_scaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_final)

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
    elif the_scorer == 'F2':
        scoring_sl1, scoring_sl2 = make_scorer(fbeta_score, beta=2, zero_division=0), fbeta_score
    elif the_scorer == 'F':
        scoring_sl1, scoring_sl2 = make_scorer(f1_score, zero_division=0), f1_score

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=EfficiencyWarning)

    random_search = RandomizedSearchCV(
        estimator=the_estimator(random_state=SEED),
        param_distributions=param_dist,
        scoring=scoring_sl1,
        n_iter=100,
        cv=tscv,
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
        refit=True
    )

    random_search.fit(X_train_scaled, y_train_full)
    best_model = random_search.best_estimator_

    # ✅ CHANGED: best_score_ is the mean cross-validated score. Explicitly named cv_score.
    cv_score = random_search.best_score_

    test_preds = best_model.predict(X_test_scaled)
    assert the_scorer in ["F0.5", "F", "F2"]
    if the_scorer == 'F0.5':
        test_score = scoring_sl2(y_test_final, test_preds, beta=0.5, zero_division=0)
    elif the_scorer == 'F2':
        test_score = scoring_sl2(y_test_final, test_preds, beta=2, zero_division=0)
    elif the_scorer == 'F':
        test_score = scoring_sl2(y_test_final, test_preds, beta=1.0, zero_division=0)
    else:
        test_score = 0.

    snapshot = {
        'cv_score': cv_score,  # ✅ CHANGED
        'test_score': test_score,  # Kept for final out-of-sample reporting
        'model': best_model,
        'estimator': the_estimator,
        'type_of_target': _args.target_type,
        'features': _selected_features,
        'scaler': scaler,
        'scorer': the_scorer,
        'X_train_full__before_scaled': X_train_full,
        'X_train_scaled': X_train_scaled,
        'y_train_full': y_train_full,
        'X_test_final__before_scaled': X_test_final,
        'X_test_scaled': X_test_scaled,
        'y_test_final': y_test_final,
        'y_hat_test_final': test_preds,
        'X': X, 'y': y, 'n_test': n_test,
        'train_t1': X_train_full.index[0].strftime('%Y-%m-%d'),
        'train_t2': X_train_full.index[-1].strftime('%Y-%m-%d'),
        'test_t1': X_test_final.index[0].strftime('%Y-%m-%d'),
        'test_t2': X_test_final.index[-1].strftime('%Y-%m-%d')
    }
    # ✅ CHANGED: Return cv_score instead of test_score
    return snapshot, cv_score


def entry_point(args):
    type_of_target = args.target_type
    percentage_of_type_target = args.target_percentage
    look_head_for_prediction = args.look_ahead
    features = args.features
    verbose = args.verbose
    the_scorer = args.scorer
    training_mode = args.training_mode
    output_dir = args.output_dir
    final_dataset_filename = args.dataset
    np.set_printoptions(linewidth=np.inf)

    if verbose:
        print(f"Beta    Priorite    Philosophie\n"
              f"0.5     Precision   'Je ne veux pas me tromper quand j\'investis.'\n"
              f"1.0     Equilibre   'Je veux un bon melange de fiabilite et d\'opportunites.'\n"
              f"2.0     Rappel      'Je ne veux surtout pas rater une hausse du marche.")
        print(f"Scorer: {the_scorer}")
        print(f"🔄 Loading data from <<{final_dataset_filename}>>")
        print(f"     Type of Target   : {type_of_target}")
        print(f"     % of Type Target : {percentage_of_type_target}")
        print(f"     Look Ahead       : {look_head_for_prediction}")

    with open(final_dataset_filename, 'rb') as f:
        loaded_data = pickle.load(f)
    df_market = loaded_data["market_data"]
    df_macro = loaded_data["macro_data"]

    # ─────────────────────────────────────────────────────────────────────────
    # 📚 STANDARD TRAINING / OPTUNA LOGIC
    # ─────────────────────────────────────────────────────────────────────────
    best_setup_found = {'train': {'cv_score': 0, 'test_score': 0}, 'test': {'cv_score': 0, 'test_score': 0}}
    start_time = time.time()

    assert training_mode == "fixed"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        _rsi_w = trial.suggest_int("rsi_window", args.optuna_rsi_window_min, args.optuna_rsi_window_max)
        _vix_l = trial.suggest_int("vix_lag", args.optuna_lag_min, args.optuna_lag_max)
        _rsi_l = trial.suggest_int("rsi_lag", args.optuna_lag_min, args.optuna_lag_max)

        _df = build_features_and_target(
            _df_market=df_market, _df_macro=df_macro, _rsi_window=_rsi_w, _vix_lag=_vix_l, _rsi_lag=_rsi_l,
            _look_head_for_prediction=look_head_for_prediction, _percentage_of_type_target=percentage_of_type_target, _type_of_target=type_of_target,
            _verbose=False, create_target=True
        )

        # ✅ CHANGED: Optimize on CV score, not test score (prevents data leakage)
        _, cv_sc = run_training_pipeline(_df=_df, _selected_features=features, _args=args)
        return cv_sc

    if verbose:
        print(f"\n🔍 Running Optuna optimization for fixed feature set: {features}")
        print(f"   Trials: {args.optuna_trials} | RSI Window: [{args.optuna_rsi_window_min}, {args.optuna_rsi_window_max}] | VIX/RSI Lag range: [{args.optuna_lag_min}, {args.optuna_lag_max}]")

    study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=verbose)

    print(f"\n✅ Optuna finished. Best parameters: {study.best_trial.params}")
    print(f"   Best CV score: {study.best_trial.value:.4f}")  # ✅ CHANGED

    # Final train with best parameters
    best_params = study.best_trial.params
    df_best = build_features_and_target(
        _df_market=df_market, _df_macro=df_macro,
        _rsi_window=best_params['rsi_window'], _vix_lag=best_params['vix_lag'], _rsi_lag=best_params['rsi_lag'],
        _look_head_for_prediction=look_head_for_prediction, _percentage_of_type_target=percentage_of_type_target, _type_of_target=type_of_target,
        _verbose=verbose, create_target=True
    )
    final_snapshot, _ = run_training_pipeline(_df=df_best, _selected_features=features, _args=args)

    # Match structure expected by save_best_model
    best_setup_found = {'test': final_snapshot, 'train': None}

    if verbose:
        data = final_snapshot
        _type_target_str = f"TARGET:{type_of_target}" if type_of_target in ["higher", "lower"] else f"TARGET:{type_of_target} @{percentage_of_type_target * 100:.2f}%"
        print("\n" + "═" * 128)
        print(f"⭐ OPTIMIZED FIXED SETUP  |  {_type_target_str}  |  LA:{look_head_for_prediction}  |  DF:{final_dataset_filename}")
        print("═" * 128)
        print(f"Features         : {data['features']}")
        print(f"Optimized Params : rsi_window={best_params['rsi_window']}, vix_lag={best_params['vix_lag']}, rsi_lag={best_params['rsi_lag']}")
        print(f"Scaler           : {data['scaler']}")
        print(f"Scorer           : {data['scorer']}")
        print(f"\nScores:")
        print(f"  - CV   {data['scorer']}: {data['cv_score']:.2%}")  # ✅ CHANGED
        print(f"  - Test {data['scorer']}: {data['test_score']:.2%}")  # Kept for OOS reporting
        print("═" * 128)

    # ─────────────────────────────────────────────────────────────────────────
    # 💾 Save the best model after loop finishes or breaks
    # ─────────────────────────────────────────────────────────────────────────
    if best_setup_found is not None:
        save_best_model(best_setup_found, args, output_dir, verbose)
    else:
        if verbose:
            print("\n⚠️ No valid model was trained. Nothing to save.")
    print(f"\nScript was executed in {Path.cwd()}")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    entry_point(parsed_args)