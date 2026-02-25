try:
    from version import sys__name, sys__version
except ImportError:
    # Fallback: dynamically add parent directory to path if 'version' module isn't found
    import sys
    import pathlib

    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version

import numpy as np
import pandas as pd
import pickle
from utils import DATASET_AVAILABLE, get_filename_for_dataset, str2bool
from runners.streak_probability import add_sequence_columns, add_sequence_columns_vectorized, new_main
from runners.streak_probability import new_main as streak_probability
from argparse import Namespace
from tqdm import tqdm
import copy
import argparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import xgboost as xgb
import warnings
import json

# Ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


# ---------------------------------------------------------
# HELPER FUNCTIONS FOR TECHNICAL INDICATORS
# ---------------------------------------------------------
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def main(args):
    # --- UPDATED: Load parameters from args instead of hardcoding ---
    macd_base_cols = ['MACD_Line', 'MACD_Signal', 'MACD_Hist']
    macd_params = json.loads(args.macd_params) if isinstance(args.macd_params, str) else args.macd_params
    def is_ema_enabled():
        return args.enable_ema
    def is_sma_enabled():
        return args.enable_sma
    def is_macd_enabled():
        return args.enable_macd
    def is_rsi_enabled():
        return args.enable_rsi
    # ---------------------------------------------------------
    use_vix = False
    # ------------------------------
    if args.verbose:
        # --- Nicely print the arguments ---
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            if 'master_data_cache' in arg:
                print(f"    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')} ({args.one_dataset_filename})")
                continue
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)
    assert args.look_ahead == 1.
    one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
    with open(one_dataset_filename, 'rb') as f:
        master_data_cache = pickle.load(f)
    close_col = (args.col, args.ticker)
    vix__master_data_cache = copy.deepcopy(master_data_cache['^VIX'].sort_index()[('Close', '^VIX')])
    vix1d__master_data_cache = copy.deepcopy(master_data_cache['^VIX1D'].sort_index()[('Close', '^VIX1D')])
    vix3m__master_data_cache = copy.deepcopy(master_data_cache['^VIX3M'].sort_index()[('Close', '^VIX3M')])
    #
    assert args.epsilon >= 0.
    master_data_cache = add_sequence_columns_vectorized(df=master_data_cache[args.ticker].sort_index(), col_name=close_col, ticker_name=args.ticker, epsilon=args.epsilon)
    # ---------------------------------------------------------
    # ADD MOVING AVERAGES
    # ---------------------------------------------------------
    if is_sma_enabled():
        for w in args.sma_windows:
            # Create column name tuple to match MultiIndex structure: ('MA_5', '^GSPC')
            ma_col_name = (f'SMA_{w}', args.ticker)
            # Calculate rolling mean on the close column
            master_data_cache[ma_col_name] = master_data_cache[close_col].rolling(window=w).mean()
        #
        for sw in args.shift_sma_col:
            for w in args.sma_windows:
                ma_col_name = (f'SMA_{w}', args.ticker)
                shift_ma_col_name = (f'SHIFTED_SMA{w}_{sw}', args.ticker)
                master_data_cache[shift_ma_col_name] = master_data_cache[ma_col_name].shift(sw)
    if is_ema_enabled():
        for w in args.ema_windows:
            ma_col_name = (f'EMA_{w}', args.ticker)
            master_data_cache[ma_col_name] = master_data_cache[close_col].ewm(span=w, adjust=False).mean()
        for sw in args.shift_sma_col:
            for w in args.ema_windows:
                ma_col_name = (f'EMA_{w}', args.ticker)
                shift_ma_col_name = (f'SHIFTED_EMA{w}_{sw}', args.ticker)
                master_data_cache[shift_ma_col_name] = master_data_cache[ma_col_name].shift(sw)

    # ---------------------------------------------------------
    # ADD RSI AND MACD (Like Moving Averages)
    # ---------------------------------------------------------
    # 1. Calculate Base RSI
    if is_rsi_enabled():
        for w in args.rsi_windows:
            rsi_col_name = (f'RSI_{w}', args.ticker)
            master_data_cache[rsi_col_name] = calculate_rsi(master_data_cache[close_col], window=w)

    # 2. Calculate Base MACD
    if is_macd_enabled():
        macd_line, signal_line, histogram = calculate_macd(master_data_cache[close_col],fast=macd_params['fast'],slow=macd_params['slow'],signal=macd_params['signal'])
        master_data_cache[('MACD_Line', args.ticker)] = macd_line
        master_data_cache[('MACD_Signal', args.ticker)] = signal_line
        master_data_cache[('MACD_Hist', args.ticker)] = histogram

    # 3. Add Shifted Versions for RSI
    if is_rsi_enabled():
        for sw in args.shift_rsi_col:  # Reusing shift_ma_col range (1-4)
            for w in args.rsi_windows:
                rsi_col_name = (f'RSI_{w}', args.ticker)
                shift_rsi_col_name = (f'SHIFTED_RSI{w}_{sw}', args.ticker)
                master_data_cache[shift_rsi_col_name] = master_data_cache[rsi_col_name].shift(sw)

    # 4. Add Shifted Versions for MACD
    if is_macd_enabled():
        for sw in args.shift_macd_col:
            for base_name in macd_base_cols:
                base_col_name = (base_name, args.ticker)
                shift_macd_col_name = (f'SHIFTED_{base_name}_{sw}', args.ticker)
                master_data_cache[shift_macd_col_name] = master_data_cache[base_col_name].shift(sw)

    # Adjust VIX
    if use_vix:
        if vix__master_data_cache.index[-1] != master_data_cache.index[-1]:
            if args.verbose:
                print(f"Removing last element of VIX DF.")
            vix__master_data_cache = vix__master_data_cache.iloc[:-1]
        assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix__master_data_cache.index[-1].strftime('%Y-%m-%d')
        assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix1d__master_data_cache.index[-1].strftime('%Y-%m-%d')
        assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix3m__master_data_cache.index[-1].strftime('%Y-%m-%d')

    step_back_range = args.step_back_range if args.step_back_range < len(master_data_cache) else len(master_data_cache)

    pos_proba = streak_probability(Namespace(frequency=args.dataset_id, direction='pos', max_n=15, min_n=0, delta=0., verbose=False, debug_verify_speeding=False, epsilon=args.epsilon))
    neg_proba = streak_probability(Namespace(frequency=args.dataset_id, direction='neg', max_n=13, min_n=0, delta=0., verbose=False, debug_verify_speeding=False, epsilon=-args.epsilon))
    # 1. Create lookup dictionaries mapping streak length (x) to probability
    # We assume pos_proba/neg_proba are lists where index = streak length
    pos_map, neg_map = {}, {}
    if args.dataset_id == 'day':
        pos_map = {x: pos_proba[x]['prob'] for x in range(0, np.max(list(pos_proba.keys()))+1)}
        neg_map = {x: neg_proba[x]['prob'] for x in range(0, np.max(list(neg_proba.keys()))+1)}
    elif args.dataset_id == 'week':
        pos_map = {x: pos_proba[x]['prob'] for x in range(0, 7)}
        neg_map = {x: neg_proba[x]['prob'] for x in range(0, 4)}
    elif args.dataset_id == 'month':
        pos_map = {x: pos_proba[x]['prob'] for x in range(0, 5)}
        neg_map = {x: neg_proba[x]['prob'] for x in range(0, 4)}
    elif args.dataset_id == 'quarter':
        pos_map = {x: pos_proba[x]['prob'] for x in range(0, 5)}
        neg_map = {x: neg_proba[x]['prob'] for x in range(0, 4)}
    elif args.dataset_id == 'year':
        pos_map = {x: pos_proba[x]['prob'] for x in range(0, 5)}
        neg_map = {x: neg_proba[x]['prob'] for x in range(0, 3)}
    else:
        assert False, f"TODO"
    if args.verbose:
        print(f"\n{pos_map=}\n{neg_map=}")
    # 2. Map the sequence columns to the new probability columns
    # We use Tuple column names ('name', TICKER) to match df2's existing MultiIndex structure
    master_data_cache[('positive_probability', args.ticker)] = master_data_cache[('POS_SEQ', args.ticker)].map(pos_map)
    master_data_cache[('negative_probability', args.ticker)] = master_data_cache[('NEG_SEQ', args.ticker)].map(neg_map)

    # ---------------------------------------------------------
    # BUILD FEATURE LIST (Xs)
    # ---------------------------------------------------------
    Xs = []
    if is_sma_enabled():
        if 0 != len(args.sma_windows):
            Xs = [(f'SMA_{w}', args.ticker) for w in args.sma_windows]
        if 0 != len(args.shift_sma_col):
            for w in args.sma_windows:
                Xs += [(f'SHIFTED_SMA{w}_{sw}', args.ticker) for sw in args.shift_sma_col]

    if is_ema_enabled():
        if 0 != len(args.ema_windows):
            Xs = [(f'EMA_{w}', args.ticker) for w in args.ema_windows]
        if 0 != len(args.shift_ema_col):
            for w in args.ema_windows:
                Xs += [(f'SHIFTED_EMA{w}_{sw}', args.ticker) for sw in args.shift_sma_col]

    # Add RSI Features
    if is_rsi_enabled():
        for w in args.rsi_windows:
            Xs += [(f'RSI_{w}', args.ticker)]
            if 0 != len(args.shift_rsi_col):
                for sw in args.shift_rsi_col:
                    Xs += [(f'SHIFTED_RSI{w}_{sw}', args.ticker)]

    # Add MACD Features
    if is_macd_enabled():
        for base_name in macd_base_cols:
            Xs += [(base_name, args.ticker)]
            for sw in args.shift_macd_col:
                Xs += [(f'SHIFTED_{base_name}_{sw}', args.ticker)]

    Xs += [("POS_SEQ", args.ticker), ("NEG_SEQ", args.ticker), close_col]
    Ys = [(args.target, args.ticker)]

    if args.verbose:
        print(f"Xs are:\n{Xs}\nYs are: {Ys}")
    only_print_once = False
    the_x_data, the_y_data, the_d_data = [], [], []
    for step_back in tqdm(range(0, step_back_range + 1)) if args.verbose else range(0, step_back_range + 1):
        if 0 == step_back:
            past_df = master_data_cache[close_col]
            future_df = master_data_cache[close_col]
            vix_df = vix__master_data_cache
            continue # skip real time
        else:
            past_df   = master_data_cache.iloc[:-step_back]
            future_df = master_data_cache.iloc[-step_back:]
            vix_df    = vix__master_data_cache.iloc[:-step_back]
        if args.look_ahead > len(future_df) or 0 == len(past_df):
            continue
        if use_vix and 0 == len(vix_df):
            continue
        # if step_back > 66666:
        #     break
        # print(f"")
        # print(f"MASTER:{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} --> {future_df.index[0].strftime('%Y-%m-%d')}/{future_df.index[-1].strftime('%Y-%m-%d')}")
        # print(f"MASTER:{past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} :: {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}")
        # if use_vix:
        #     print(f"VIX   : {vix_df.index[0].strftime('%Y-%m-%d')}/{vix_df.index[-1].strftime('%Y-%m-%d')}")
        # print(f"Using {past_df.index[0].strftime('%Y-%m-%d')}/{past_df.index[-1].strftime('%Y-%m-%d')} to predict {future_df.index[args.look_ahead - 1].strftime('%Y-%m-%d')}")
        # print(f"\n\n")
        assert past_df.index.intersection(future_df.index).empty, "Indices must be disjoint"
        the_X = past_df.iloc[-1][Xs].copy().sort_index()
        the_Y = future_df.iloc[args.look_ahead - 1][Ys].copy()
        # --- NORMALIZATION FOR STATIONARITY ---
        baseline_price = the_X[close_col]
        if isinstance(baseline_price, pd.Series):
            assert 1 == len(baseline_price)
            baseline_price = baseline_price.values[0]
        assert baseline_price > 0, f"{baseline_price=}   {close_col=}"
        price_cols = [col for col in Xs if ('_SMA' in col[0] or 'SMA_' in col[0] or '_EMA' in col[0] or 'EMA_' in col[0] or col == close_col) and col[1] == args.ticker]
        if args.verbose and not only_print_once:
            print(f"\nConvert price levels to returns relative to baseline: {price_cols}")
            only_print_once = True
        for col in price_cols:
            assert col in the_X.index
            if args.convert_price_level_with_baseline == "fraction":
                the_X[col] = the_X[col] / baseline_price
            elif args.convert_price_level_with_baseline == "return":
                the_X[col] = (the_X[col] / baseline_price) - 1  # -1 --> Returns instead of ratios
        the_d_data.append(past_df.index[-1])
        the_x_data.append(the_X.values)
        the_y_data.append(the_Y.values)
    # Reverse the lists
    the_x_data = np.asarray(the_x_data[::-1])
    the_y_data = np.asarray(the_y_data[::-1])
    the_d_data = the_d_data[::-1]
    if args.verbose:
        print(f"There is {len(the_d_data)} data, ranging from {the_d_data[0].strftime('%Y-%m-%d')} to {the_d_data[-1].strftime('%Y-%m-%d')}.")
    assert the_d_data[0] < the_d_data[-1] and isinstance(the_d_data[0], pd.Timestamp), f"Make sure most recent data is at the end!"
    num_classes = len(np.unique(the_y_data))
    assert num_classes > 1, f"There shall be more than 1 class"
    if args.verbose:
        print(f"Unique classes found: {np.unique(the_y_data)} ({num_classes})")
    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores, f1_scores = [], []
    # Initialize lists to store per-class scores across folds ---
    fold_precision_scores_per_class = []
    fold_recall_scores_per_class = []
    fold_f1_scores_per_class = []
    base_models = ["xgb"]#, "lgb", "cat", "hgb"]
    for fold, (train_idx, val_idx) in enumerate(tscv.split(the_x_data)):
        X_train, X_val = copy.deepcopy(the_x_data[train_idx]), copy.deepcopy(the_x_data[val_idx])
        y_train, y_val = copy.deepcopy(the_y_data[train_idx]), copy.deepcopy(the_y_data[val_idx])
        # Flatten and Convert Labels to Integers ---
        # XGBoost expects 1D integer arrays for classification
        y_train = y_train.ravel().astype(int)
        y_val = y_val.ravel().astype(int)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        n_estimators = 500
        max_depth = 6
        learning_rate = 0.05
        random_state = 42
        classification_model_configs = {
            "xgb": {
                "class": xgb.XGBClassifier,
                "params": {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': random_state,
                    'objective': 'multi:softprob',  # Multi-class probability
                    'num_class': num_classes,  # Required
                    'eval_metric': 'mlogloss',
                    'tree_method': 'hist',
                    #'early_stopping_rounds': 20,
                    #'use_label_encoder': False,
                },
                "supports_early_stopping": True
            },
            "rf": {
                "class": RandomForestClassifier,
                "params": {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'class_weight': 'balanced'  # Helpful for imbalanced classes
                },
                "supports_early_stopping": False
            },
            "lgb": {
                "class": lgb.LGBMClassifier,
                "params": {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': random_state,
                    'objective': 'multiclass',  # Multi-class
                    'num_class': num_classes,  # Required
                    'metric': 'multi_logloss',
                    'verbosity': -1,
                    #'early_stopping_rounds': 20,
                },
                "supports_early_stopping": True
            },
            "cat": {
                "class": CatBoostClassifier,
                "params": {
                    'iterations': n_estimators,
                    'depth': max_depth,
                    'learning_rate': learning_rate,
                    'random_seed': random_state,
                    'verbose': False,
                    'loss_function': 'MultiClass',  # Multi-class
                    'classes_count': num_classes,  # Required
                    'eval_metric': 'MultiClass',
                    #'early_stopping_rounds': 20,
                },
                "supports_early_stopping": True
            },
            "hgb": {
                "class": HistGradientBoostingClassifier,
                "params": {
                    'max_iter': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'random_state': random_state,
                    'loss': 'log_loss',  # Supports multi-class automatically
                    'early_stopping': False,
                    'validation_fraction': 0.1,  # Uses portion of TRAIN for early stopping
                    'n_iter_no_change': 20,
                },
                "supports_early_stopping": True,
                "uses_validation_fraction": True  # Flag for custom logic
            }
        }
        fold_predictions_proba = []  # Store probabilities for ensemble
        for model_name in base_models:
            cfg = classification_model_configs[model_name]
            model = cfg["class"](**cfg["params"])
            fit_kwargs_init = {}
            model.fit(X_train_scaled, y_train, **fit_kwargs_init)

            # Store Probabilities for Ensemble
            pred_proba = model.predict_proba(X_val_scaled)

            # Ensure consistent shape (n_samples, num_classes) for all models ---
            # Models like HistGradientBoostingClassifier infer classes from y_train.
            # If a fold is missing some classes, pred_proba will have fewer columns.
            if pred_proba.shape[1] != num_classes:
                # Create a full probability matrix initialized to 0
                full_proba = np.zeros((pred_proba.shape[0], num_classes))
                # Map the predicted probabilities to the correct class indices
                # model.classes_ contains the specific class labels present in this training fold
                full_proba[:, model.classes_] = pred_proba
                fold_predictions_proba.append(full_proba)
            else:
                fold_predictions_proba.append(pred_proba)
        # --- Ensemble Averaging Probabilities ---
        avg_proba = np.mean(fold_predictions_proba, axis=0)
        y_pred_final = np.argmax(avg_proba, axis=1)

        # --- Multi-Class Metrics ---
        from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support

        acc = accuracy_score(y_val, y_pred_final)
        f1 = f1_score(y_val, y_pred_final, average='weighted')  # 'weighted' handles imbalance

        # --- Per-Class Scores ---
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_val, y_pred_final, labels=range(num_classes), zero_division=0
        )
        fold_precision_scores_per_class.append(precision)
        fold_recall_scores_per_class.append(recall)
        fold_f1_scores_per_class.append(f1_per_class)
        if args.verbose:
            print(f"Fold {fold + 1}: Acc={acc:.4f}, F1={f1:.4f}")
            # Print Per-Class F1 for this fold
            f1_str = ", ".join([f"{i}:{v:.3f}" for i, v in enumerate(f1_per_class)])
            print(f"  Per-Class F1: [{f1_str}]")

        acc_scores.append(acc)
        f1_scores.append(f1)
    if args.verbose:
        print(f"\nAverage Accuracy: {np.mean(acc_scores):.4f}")
        print(f"Average Weighted F1: {np.mean(f1_scores):.4f}")
        print(f"Precision\n"
              f"TP / (TP + FP)\n"
              f"Of all the instances I predicted as positive, how many were actually positive?")
        print(f"Recall\n"
              f"TP / (TP + FN)\n"
              f"Of all the actual positive instances, how many did I correctly identify?")
        print("Accuracy\n"
              "Accuracy = (True Positives + True Negatives) / Total Predictions\n"
              "         = (TP + TN) / (TP + TN + FP + FN)")
    # Average Per-Class Scores ---
    avg_precision = np.mean(fold_precision_scores_per_class, axis=0)
    avg_recall = np.mean(fold_recall_scores_per_class, axis=0)
    avg_f1 = np.mean(fold_f1_scores_per_class, axis=0)
    if args.verbose:
        print(f"\nAverage Per-Class Validation Scores ({len(fold_f1_scores_per_class)} folds):")
        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 44)
        for c in range(num_classes):
            print(f"{c:<8} {avg_precision[c]:<12.4f} {avg_recall[c]:<12.4f} {avg_f1[c]:<12.4f}")
    return f1_scores, acc_scores, avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--col", type=str, default='Close')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument("--look_ahead", type=int, default=1)
    #
    parser.add_argument('--step-back-range', type=int, default=99999,
                        help="Number of historical time windows to simulate (rolling backtest depth).")
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument("--epsilon", type=float, default=0.,
                        help="Threshold for neutral returns. Default: 0.")
    parser.add_argument("--target", type=str, default='POS_SEQ', choices=['POS_SEQ', 'NEG_SEQ'],
                        help="Target column for prediction. Options: POS_SEQ, NEG_SEQ. Default: POS_SEQ.")
    parser.add_argument("--convert-price-level-with-baseline", type=str, default='fraction', choices=['fraction', 'return'],
                        help="Method to convert price levels. 'fraction': price/baseline, 'return': (price/baseline)-1. Default: fraction.")
    parser.add_argument("--ema-windows", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9],
                        help="List of window sizes for Exponential Moving Average calculation. Default: 2 to 9.")
    parser.add_argument('--enable_ema', type=str2bool, default=False)
    parser.add_argument("--shift-ema-col", type=int, nargs='+', default=[],
                        help="List of shift periods for EMA. Default: None.")
    parser.add_argument("--sma-windows", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9],
                        help="List of window sizes for Moving Average calculation. Default: 2 to 9.")
    parser.add_argument('--enable_sma', type=str2bool, default=False)
    parser.add_argument("--shift-sma-col", type=int, nargs='+', default=[],
                        help="List of shift periods for SMA. Default: None.")
    parser.add_argument("--rsi-windows", type=int, nargs='+', default=[14],
                        help="List of window sizes for RSI calculation. Default: 14.")
    parser.add_argument("--shift-rsi-col", type=int, nargs='+', default=[],
                        help="List of shift periods for RSI. Default: None.")
    parser.add_argument('--enable_rsi', type=str2bool, default=False)
    parser.add_argument("--macd-params", type=str, default='{"fast": 12, "slow": 26, "signal": 9}',
                        help="JSON string for MACD parameters (fast, slow, signal). Default: 12, 26, 9.")
    parser.add_argument('--enable_macd', type=str2bool, default=False)
    parser.add_argument("--shift_macd_col", type=int, nargs='+', default=[],
                        help="List of shift periods for MACD. Default: None.")

    args = parser.parse_args()

    main(args)