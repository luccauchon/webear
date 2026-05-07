import os
import random
import numpy as np
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
import itertools
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
from utils import get_next_step, WEBEARStyle


def parse_arguments():
    """
    Parse command-line arguments for the market/macro prediction script.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate time-series classifiers on combined market & macro data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        Examples:
          # Train normally
          python train_model.py

          # Real-time prediction (uses latest model in ./output_models)
          python train_model.py --real-time

          # Real-time prediction with a specific model file
          python train_model.py --real-time --model-path ./output_models/best_model_RandomForestClassifier_soft_higher_20260419_120000.pkl
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
        "--training-mode", "-tm", type=str, default="random",
        choices=["exhaustive", "random", "manual"],
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
    exec_grp.add_argument(
        "--real-time", action="store_true", default=False,
        help="Run in real-time prediction mode: loads data, builds features, and predicts the next month using the latest saved model."
    )
    exec_grp.add_argument(
        "--model-path", type=str, default=None,
        help="(Optional for --real-time) Path to a specific pickled model. Defaults to the latest model in --output-dir."
    )
    parser.add_argument("--clip", action="store_true", help="Exclude incomplete current bar in real-time")

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


def build_features_and_target(_df_market, _df_macro, _rsi_window, _vix_lag, _rsi_lag,
                              _look_head_for_prediction, _percentage_of_type_target,
                              _type_of_target, _verbose, create_target=True):
    # 1. Création de Features Macro (Lags et Variations)
    _df_macro['Spread_10Y2Y'] = _df_macro['T10Y2Y'].shift(1)
    _df_macro['Fed_Rate_Diff'] = _df_macro['FEDFUNDS'].diff().shift(1)
    _df_macro['Unrate_Diff'] = _df_macro['UNRATE'].diff().shift(1)
    _df_macro['Inflation_Rate'] = _df_macro['CPIAUCSL'].pct_change(12, fill_method=None).shift(1)

    # 2. Fusion avec tes données S&P 500
    _df_fusionned = _df_market.join(_df_macro[['Fed_Rate_Diff', 'Unrate_Diff', 'Inflation_Rate', 'Spread_10Y2Y']]).dropna()

    # 3. Ajout de nouvelles Features
    _df_fusionned['VIX_Ratio'] = _df_fusionned['VIX'] / _df_fusionned['VIX'].rolling(12).mean()

    delta = _df_fusionned['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=_rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=_rsi_window).mean()
    rs = gain / loss
    _df_fusionned['RSI'] = 100 - (100 / (1 + rs))

    _df_fusionned['MA_Short'] = _df_fusionned['Close'].rolling(window=3).mean()
    _df_fusionned['MA_Long'] = _df_fusionned['Close'].rolling(window=12).mean()
    _df_fusionned['Price_to_MA'] = _df_fusionned['Close'] / _df_fusionned['MA_Long']

    _df_fusionned['Shifted_MA_Short'] = _df_fusionned['MA_Short'].diff()
    _df_fusionned['Shifted_MA_Long'] = _df_fusionned['MA_Long'].pct_change()
    _df_fusionned['Shifted_Price_to_MA'] = _df_fusionned['Price_to_MA'].diff()

    _df_fusionned['VIX_Lag1'] = _df_fusionned['VIX'].shift(_vix_lag)
    _df_fusionned['RSI_Lag1'] = _df_fusionned['RSI'].shift(_rsi_lag)

    _df_fusionned['Log_Close'] = np.log(_df_fusionned['Close'])
    _df_fusionned['Dist_from_ATH'] = (_df_fusionned['Close'] / _df_fusionned['Close'].cummax()) - 1

    if _verbose:
        print(f"Dates in DF (before target):  {_df_fusionned.dropna().index[0].strftime('%Y-%m-%d')} :: {_df_fusionned.dropna().index[-1].strftime('%Y-%m-%d')}")

    # --- Cible (Target) ---
    if create_target:
        _look_head_for_prediction = int(_look_head_for_prediction)
        assert _look_head_for_prediction > 0
        assert 0 < _percentage_of_type_target < 1

        # 1. On récupère le prix futur (M + N) pour toutes les comparaisons
        future_close = _df_fusionned["Close"].shift(-_look_head_for_prediction)

        # 2. Calcul des Targets selon le type
        if _type_of_target == 'higher':
            _df_fusionned["Target"] = (future_close > _df_fusionned["Close"]).astype(int)
        elif _type_of_target == 'lower':
            _df_fusionned["Target"] = (future_close < _df_fusionned["Close"]).astype(int)
        elif _type_of_target == 'soft_higher':
            _df_fusionned["Return"] = (future_close / _df_fusionned["Close"]) - 1
            _df_fusionned["Target"] = (_df_fusionned["Return"] > _percentage_of_type_target).astype(int)
        elif _type_of_target == 'soft_lower':
            _df_fusionned["Return"] = (future_close / _df_fusionned["Close"]) - 1
            _df_fusionned["Target"] = (_df_fusionned["Return"] < -_percentage_of_type_target).astype(int)
        elif _type_of_target == 'in_between':
            lower_bound = _df_fusionned["Close"] * (1 - _percentage_of_type_target)
            upper_bound = _df_fusionned["Close"] * (1 + _percentage_of_type_target)
            _df_fusionned["Target"] = ((future_close >= lower_bound) & (future_close <= upper_bound)).astype(int)
        else:
            raise ValueError(f"Type de target inconnu : {_type_of_target}")

        # 3. Nettoyage
        # 🔧 Pandas boolean comparisons with NaN evaluate to False -> 0 after astype(int).
        # We explicitly restore NaNs where future_close is missing so dropna() works as expected.
        _df_fusionned.loc[future_close.isna(), "Target"] = np.nan

        # Drop only rows with invalid targets (preserves rows where other features might be NaN but you handle them elsewhere)
        _df_fusionned = _df_fusionned.dropna(subset=["Target"])
    else:
        # Real-time mode: only drop rows with missing features
        _df_fusionned = _df_fusionned.dropna()

    if _verbose:
        print(f"Dates in DF (features only): {_df_fusionned.index[0].strftime('%Y-%m-%d')} :: {_df_fusionned.index[-1].strftime('%Y-%m-%d')}")

    return _df_fusionned


def save_best_model(best_setup, args, output_dir, verbose=True):
    """Saves the best model configuration AND training parameters to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    est_name = args.the_estimator.__name__
    target_name = args.target_type
    filename = f"best_model_{est_name}_{target_name}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)

    save_data = {
        'best_setup': best_setup,
        'saved_at': datetime.now().isoformat(),
        # Save feature engineering & target parameters for real-time consistency
        'params': {
            'rsi_window': args.rsi_window,
            'vix_lag': args.vix_lag,
            'rsi_lag': args.rsi_lag,
            'look_ahead': args.look_ahead,
            'target_percentage': args.target_percentage,
            'target_type': args.target_type,
            'dataset_filename': args.dataset,
        }
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
    time_limit = args.time_limit
    output_dir = args.output_dir
    final_dataset_filename = args.dataset
    np.set_printoptions(linewidth=np.inf)
    saved_data = None

    # ─────────────────────────────────────────────────────────────────────────
    # 🔄 LOAD SAVED PARAMETERS FIRST IF IN REAL-TIME MODE
    # ─────────────────────────────────────────────────────────────────────────
    if args.real_time:
        model_path = args.model_path
        if model_path is None:
            os.makedirs(output_dir, exist_ok=True)
            pkl_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pkl')])
            if not pkl_files:
                print("❌ No saved models found. Please train a model first or specify --model-path.")
                return
            model_path = os.path.join(output_dir, pkl_files[-1])

        print(f"📦 Loading model and training parameters from: {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)

        # Override CLI args with saved training parameters
        saved_params = saved_data.get('params', {})
        assert 'rsi_window' in saved_params
        rsi_window = saved_params.get('rsi_window', rsi_window)
        assert 'vix_lag' in saved_params
        vix_lag = saved_params.get('vix_lag', vix_lag)
        assert 'rsi_lag' in saved_params
        rsi_lag = saved_params.get('rsi_lag', rsi_lag)
        assert 'look_ahead' in saved_params
        look_head_for_prediction = saved_params.get('look_ahead', look_head_for_prediction)
        assert 'target_percentage' in saved_params
        percentage_of_type_target = saved_params.get('target_percentage', percentage_of_type_target)
        assert 'target_type' in saved_params
        type_of_target = saved_params.get('target_type', type_of_target)
        _is_same_file_str = f"(overriding {saved_params['dataset_filename']})" if final_dataset_filename != saved_params['dataset_filename'] else ""
        print(f"📦 Loading dataset from : {final_dataset_filename}  {_is_same_file_str}")
        saved_params['dataset_filename'] = final_dataset_filename
        print("✅ Using saved training parameters for real-time feature engineering.")
    if verbose:
        print(f"🔄 Loading data from <<{final_dataset_filename}>>")
    with open(final_dataset_filename, 'rb') as f:
        loaded_data = pickle.load(f)
    df_market = loaded_data["market_data"]
    df_macro = loaded_data["macro_data"]

    # Build features & target (uses overridden params if real-time, else CLI args)
    df = build_features_and_target(
        _df_market=df_market, _df_macro=df_macro,
        _rsi_window=rsi_window, _vix_lag=vix_lag, _rsi_lag=rsi_lag,
        _look_head_for_prediction=look_head_for_prediction,
        _percentage_of_type_target=percentage_of_type_target,
        _type_of_target=type_of_target, _verbose=verbose,
        create_target=not args.real_time  # Skip target creation in real-time mode
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 🚀 REAL-TIME PREDICTION MODE
    # ─────────────────────────────────────────────────────────────────────────
    if args.real_time:
        if df.empty:
            print("❌ No valid data points available after feature engineering.")
            return
        if args.clip:
            df = df.iloc[:-1].copy()
        last_date  = df.index[-1].strftime('%Y-%m-%d')
        last_value = df.iloc[-1]['Close']

        best_setup = saved_data['best_setup']
        assert 'test' in best_setup
        setup_to_use             = best_setup.get('test', best_setup.get('train'))
        _target_type_used        = saved_data['params']['target_type']
        _target_pourcentage_used = saved_data['params']['target_percentage']
        _look_ahead_used         = saved_data['params']['look_ahead']
        _dataset_filename_used   = saved_data['params']['dataset_filename']
        assert final_dataset_filename == saved_data['params']['dataset_filename']
        scaler = setup_to_use['scaler']
        model = setup_to_use['model']
        feat_cols = setup_to_use['features']

        print(f"Beta    Priorite    Philosophie\n"
              f"0.5     Precision   'Je ne veux pas me tromper quand j\'investis.'\n"
              f"1.0     Equilibre   'Je veux un bon melange de fiabilite et d\'opportunites.'\n"
              f"2.0     Rappel      'Je ne veux surtout pas rater une hausse du marche.")
        _tmp_ttu_1 = f'{WEBEARStyle.BOLD}higher{WEBEARStyle.END}' if _target_type_used == 'higher' else f'higher'
        _tmp_ttu_2 = f'{WEBEARStyle.BOLD}lower{WEBEARStyle.END}'  if _target_type_used == 'lower'  else f'lower'
        _tmp_ttu_3 = f'{WEBEARStyle.BOLD}soft_higher{WEBEARStyle.END}' if _target_type_used == 'soft_higher' else f'soft_higher'
        _tmp_ttu_4 = f'{WEBEARStyle.BOLD}soft_lower{WEBEARStyle.END}' if _target_type_used == 'soft_lower' else f'soft_lower'
        _tmp_ttu_5 = f'{WEBEARStyle.BOLD}in_between{WEBEARStyle.END}' if _target_type_used == 'in_between' else f'in_between'
        print(f"🔹 {_tmp_ttu_1}\n"  # higher
              f"Logic: 1 if future_price > current_price, else 0\n"
              f"Meaning: Predicts any upward movement, no matter how small.\n"
              f"Use case: Pure directional bias. Highly sensitive to market noise.\n"
              f"\n"
              f"🔹 {_tmp_ttu_2}\n"  # lower
              f"Logic: 1 if future_price < current_price, else 0\n"
              f"Meaning: Predicts any downward movement.\n"
              f"Use case: Short/bearish directional bias.\n"
              f"\n"
              f"🔹 {_tmp_ttu_3} ⭐ (Recommended for trading)\n"  # soft_higher
              f"Logic: 1 if (future_price / current_price) - 1 > threshold, else 0\n"
              f"Meaning: Predicts an upward move greater than your threshold (e.g., > +1%). Moves between 0% and +1% are labeled 0.\n"
              f"Use case: Filters out noise and transaction costs. Only signals when there's meaningful upside potential.\n"
              f"\n"
              f"🔹 {_tmp_ttu_4}\n"  # soft_lower
              f"Logic: 1 if (future_price / current_price) - 1 < -threshold, else 0\n"
              f"Meaning: Predicts a downward move greater than your threshold (e.g., < -1%).\n"
              f"Use case: Filters out minor dips. Only signals meaningful downside risk.\n"
              f"\n"
              f"🔹 {_tmp_ttu_5}\n"  # in_between
              f"Logic: 1 if current_price * (1 - threshold) ≤ future_price ≤ current_price * (1 + threshold), else 0\n"
              f"Meaning: Predicts the market will stay sideways/consolidate within ± your threshold.\n"
              f"Use case: Useful for range-bound strategies, volatility selling, or avoiding false breakout signals.\n"
              f"\n")
        print(f"📅 Using last datapoint: {last_date} @ {last_value:.0f} {'(clipping activated)' if args.clip else ''}")

        # Prepare last row
        X_last = df[feat_cols].iloc[[-1]]
        X_last_scaled = scaler.transform(X_last)

        # Predict
        _realtime_prediction = model.predict(X_last_scaled)[0]
        proba = 0.
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_last_scaled)[0][1]
        # There is no dataset_id in the params.
        dataset_id = "day" if "_day_" in final_dataset_filename else None
        dataset_id = "month" if "_month_" in final_dataset_filename else dataset_id
        dataset_id = "week" if "_week_" in final_dataset_filename or "_week" in final_dataset_filename else dataset_id
        assert dataset_id is not None, f"{dataset_id=}   {final_dataset_filename=}"
        ff_date = get_next_step(the_date=last_date, dataset_id=dataset_id, nn=int(look_head_for_prediction))
        assert 0 < int(look_head_for_prediction)
        _now_timestamp = datetime.now().strftime("%Y-%m-%d")#_%H%M%S")
        print("\n" + "═" * 50)
        print(f"🚀 REAL-TIME PREDICTION FOR +{int(look_head_for_prediction)} BAR{'' if 1 == int(look_head_for_prediction) else 'S'}")
        print(f"📅 Date/Value used : {last_date} @ {last_value:.0f}")
        print(f"📅 Today's date    : {_now_timestamp}")
        if _target_type_used in ["lower", "soft_lower"]:
            print(f"📊 Prediction      : {'DOWN' if _realtime_prediction == 1 else '---'} @ {proba:.2%} : on {ff_date}, price {'<' if _realtime_prediction == 1 else '?'} {last_value*(1+_target_pourcentage_used):.0f}")
        if _target_type_used in ["higher", "soft_higher"]:
            print(f"📊 Prediction      : {'UP' if _realtime_prediction == 1 else '---'} @ {proba:.2%} : on {ff_date}, price {'>' if _realtime_prediction == 1 else '?'} {last_value*(1+_target_pourcentage_used):.0f}")
        if _target_type_used in ["in_between"]:
            _str_left_tmp  = f"{last_value*(1-_target_pourcentage_used):.0f}" if _realtime_prediction == 1 else '?'
            _str_right_tmp = f"{last_value*(1+_target_pourcentage_used):.0f}" if _realtime_prediction == 1 else '?'
            print(f"📊 Prediction      : {'RANGE' if _realtime_prediction == 1 else '---'} @ {proba:.2%} : on {ff_date}, {_str_left_tmp} < price < {_str_right_tmp}")
        print(f"📈 Parameters      : {_target_type_used} @{_target_pourcentage_used:.2%}  LA:{_look_ahead_used}  Dataset:{_dataset_filename_used}")
        print("═" * 50)
        print(f"🔑 Features Used   : {feat_cols}")
        print(f"   Scorer: {setup_to_use['scorer']}")
        print(f"   Train/Test scores: {setup_to_use['train_score']:.2%}/{setup_to_use['test_score']:.2%}")
        print(f"   Train : {setup_to_use['train_t1']}::{setup_to_use['train_t2']}")
        print(f"   Target: {type_of_target} @{percentage_of_type_target:.2%}")
        print(f"   Out of sample performance ({setup_to_use['test_t1']}::{setup_to_use['test_t2']}): \n"
              f"    y    : {setup_to_use['y_test_final'].values}\n"
              f"    y_hat: {setup_to_use['y_hat_test_final']}")
        print("Confusion Matrix:")
        print(confusion_matrix(setup_to_use['y_test_final'], setup_to_use['y_hat_test_final']))
        print("═" * 50)
        return _realtime_prediction

    # ─────────────────────────────────────────────────────────────────────────
    # 📚 STANDARD TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
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
        elif _mode == 'manual':
            provided_features = [
                ['Dist_from_ATH', 'Fed_Rate_Diff', 'Inflation_Rate', 'Log_Close', 'MA_Long', 'MA_Short', 'Price_to_MA', 'RSI_Lag1', 'Shifted_MA_Short', 'Spread_10Y2Y', 'Unrate_Diff', 'VIX', 'VIX_Lag1'],
                ['Fed_Rate_Diff', 'Inflation_Rate', 'MA_Long', 'MA_Short', 'RSI_Lag1', 'VIX_Lag1', 'VIX_Ratio'],
            ]
            for a_feartures_set in provided_features:
                yield a_feartures_set

    feature_iterator = get_feature_combinations(_features=features, _mode=training_mode)
    total_to_process = total_combinations if training_mode == "exhaustive" else max_random_iterations

    best_setup_found = {'train': {'train_score': 0, 'test_score': 0}, 'test': {'train_score': 0, 'test_score': 0}}
    displayed_output_once = False
    start_time = time.time()
    for selected_features in tqdm(feature_iterator, total=total_to_process, disable=not verbose):
        # ─────────────────────────────────────────────────────────────────────
        # ⏱️ Time limit check
        # ─────────────────────────────────────────────────────────────────────
        stop_search_time_limit_exceeded = False
        if time_limit and (time.time() - start_time) >= time_limit:
            if verbose:
                print(f"\n⏱️ Time limit of {time_limit} seconds reached. Stopping search gracefully...")
            stop_search_time_limit_exceeded = True
        assert len(selected_features) == len(list(set(selected_features)))
        X = df[selected_features].copy()
        y = df["Target"].copy()

        assert n_test > 0
        assert n_test < len(X)
        X_train_full = X.iloc[:-n_test].copy()
        y_train_full = y.iloc[:-n_test].copy()
        X_test_final = X.iloc[-n_test:].copy()
        y_test_final = y.iloc[-n_test:].copy()
        train_size, test_size = len(X_train_full), len(X_test_final)
        if not displayed_output_once and verbose:
            print(f"\n[STATIC] {train_size=}  {test_size=}")
            print(f"[TRAIN]  {X_train_full.index[0].strftime('%Y-%m-%d')} :: {X_train_full.index[-1].strftime('%Y-%m-%d')}")
            print(f"[TEST]   {X_test_final.index[0].strftime('%Y-%m-%d')} :: {X_test_final.index[-1].strftime('%Y-%m-%d')}")

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
        assert scoring_sl1 is not None and scoring_sl2 is not None
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        random_search = RandomizedSearchCV(
            estimator=the_estimator(random_state=1),
            param_distributions=param_dist,
            scoring=scoring_sl1,
            n_iter=100,
            cv=tscv,
            n_jobs=-1,
            random_state=SEED,
            verbose=0
        )

        random_search.fit(X_train_scaled, y_train_full)
        best_model = random_search.best_estimator_
        train_score = random_search.best_score_
        test_preds = best_model.predict(X_test_scaled)
        test_score = None
        if the_scorer == 'F0.5':
            test_score = scoring_sl2(y_test_final, test_preds, beta=0.5, zero_division=0)
        elif the_scorer == 'F2':
            test_score = scoring_sl2(y_test_final, test_preds, beta=2, zero_division=0)
        elif the_scorer == 'F':
            test_score = scoring_sl2(y_test_final, test_preds, beta=1.0, zero_division=0)
        assert test_score is not None
        _tmp_update_snapshot, do_display = {'test_score': test_score, 'train_score': train_score,
                                            'model': best_model, 'estimator': the_estimator, 'type_of_target': type_of_target,
                                            'features': selected_features, 'scaler': scaler, 'scorer': the_scorer,
                                            'X_train_full__before_scaled': X_train_full, 'X_train_scaled': X_train_scaled,
                                            'y_train_full': y_train_full, 'X_test_final__before_scaled': X_test_final,
                                            'X_test_scaled': X_test_scaled, 'y_test_final': y_test_final, 'y_hat_test_final': test_preds,
                                            'X': X, 'y': y, 'n_test': n_test, 'train_t1': X_train_full.index[0].strftime('%Y-%m-%d'), 'train_t2': X_train_full.index[-1].strftime('%Y-%m-%d'),
                                            'test_t1': X_test_final.index[0].strftime('%Y-%m-%d'), 'test_t2': X_test_final.index[-1].strftime('%Y-%m-%d')}, False
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
        if (verbose and (not displayed_output_once or do_display)) or stop_search_time_limit_exceeded:
            for category in ['train', 'test']:
                data = best_setup_found[category]
                _type_target_str = f"TARGET:{type_of_target}" if type_of_target in ["higher", "lower"] else f"TARGET:{type_of_target} @{percentage_of_type_target * 100:.2f}%"
                print("\n" + "═" * 128)
                print(f"⭐ BEST SETUP RECORDED FOR: {category.upper()}  |  {_type_target_str}  |  LA:{look_head_for_prediction}  |  DF:{final_dataset_filename} ({train_size}/{test_size}) ")
                print("═" * 128)

                print(f"Features         : {data['features']}")
                print(f"Scaler           : {data['scaler']}")
                print(f"Scorer           : {data['scorer']}")

                print(f"\nScores for this snapshot:")
                print(f"  - Train {data['scorer']}: {data['train_score']:.2%}")
                print(f"  - Test  {data['scorer']}: {data['test_score']:.2%}")

                saved_preds = data['model'].predict(data['X_test_scaled'])
                assert np.array_equal(saved_preds, data['y_hat_test_final']), "Sanity check failed!"
                print(f"\nClassification Report ({category} snapshot):")
                print(classification_report(data['y_test_final'], saved_preds, zero_division=0))
                print(f"\n"
                      f"y    : {data['y_test_final'].values}\n"
                      f"y_hat: {saved_preds}\n")
                print("Confusion Matrix:")
                print(confusion_matrix(data['y_test_final'], saved_preds))
                print("═" * 128)
        displayed_output_once = True
        if stop_search_time_limit_exceeded:
            break
    # ─────────────────────────────────────────────────────────────────────────
    # 💾 Save the best model after loop finishes or breaks
    # ─────────────────────────────────────────────────────────────────────────
    if best_setup_found is not None:
        save_best_model(best_setup_found, args, output_dir, verbose)
    else:
        if verbose:
            print("\n⚠️ No valid model was trained. Nothing to save.")
    print(f"\nScript was executed in {Path.cwd()}")
    return None


if __name__ == "__main__":
    parsed_args = parse_arguments()
    entry_point(parsed_args)