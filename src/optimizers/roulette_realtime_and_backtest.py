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
import os
import sys
import numpy as np
import pandas as pd
from utils import DATASET_AVAILABLE, get_filename_for_dataset, str2bool, add_vwap_with_bands
from runners.streak_probability import add_sequence_columns, add_sequence_columns_vectorized, new_main
from runners.streak_probability import new_main as streak_probability
from argparse import Namespace
from tqdm import tqdm
import copy
import argparse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import warnings
import json
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# Ignore the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


# ---------------------------------------------------------
# HELPER FOR CONFIG MERGING
# ---------------------------------------------------------
def deep_update(base_dict, update_dict):
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


# ---------------------------------------------------------
# HELPER FUNCTIONS FOR MODEL INFERENCE
# ---------------------------------------------------------
def load_model_for_inference(model_path, verbose=True):
    """
    Load a saved model and preprocessing objects for inference.

    Args:
        model_path: Path to the saved .pkl model file
        verbose: Print loading information

    Returns:
        dict: Contains scaler, models, feature_list, and metadata
    """
    import pickle
    import os

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    if verbose:
        print("=" * 80)
        print("📦 Loading model for inference...")
        print("=" * 80)
        print(f"   Path: {model_path}")
        print(f"   Models: {list(model_data['models'].keys())}")
        print(f"   Features: {len(model_data['feature_list'])}")
        print(f"   Classes: {model_data['num_classes']}")
        print(f"   Training period: {model_data['training_date_range']['start']} to {model_data['training_date_range']['end']}")
        print(f"   CV Accuracy: {model_data['cv_scores']['accuracy_mean']:.4f} (+/- {model_data['cv_scores']['accuracy_std']:.4f})")
        print(f"   CV F1-Score: {model_data['cv_scores']['f1_mean']:.4f} (+/- {model_data['cv_scores']['f1_std']:.4f})")
        print("=" * 80)

    return model_data


def predict_with_saved_model(model_data, X_new, verbose=True):
    """
    Make predictions using a loaded model.

    Args:
        model_data: Dict from load_model_for_inference()
        X_new: New feature data (numpy array or DataFrame with same features)
        verbose: Print prediction information

    Returns:
        tuple: (predicted_classes, prediction_probabilities)
    """
    import numpy as np

    scaler = model_data['scaler']
    models = model_data['models']
    num_classes = model_data['num_classes']
    feature_list = model_data['feature_list']

    # Ensure X_new is numpy array
    if hasattr(X_new, 'values'):
        X_new = X_new.values

    # Select and order features correctly
    if X_new.shape[1] != len(feature_list):
        raise ValueError(f"Expected {len(feature_list)} features, got {X_new.shape[1]}")

    # Scale the data
    X_scaled = scaler.transform(X_new)

    # Get predictions from all models and ensemble
    all_proba = []
    for model_name, model in models.items():
        pred_proba = model.predict_proba(X_scaled)

        # Ensure consistent shape
        if pred_proba.shape[1] != num_classes:
            full_proba = np.zeros((pred_proba.shape[0], num_classes))
            full_proba[:, model.classes_] = pred_proba
            all_proba.append(full_proba)
        else:
            all_proba.append(pred_proba)

    # Average probabilities across models
    avg_proba = np.mean(all_proba, axis=0)
    predicted_classes = np.argmax(avg_proba, axis=1)

    if verbose:
        print(f"Made predictions for {len(predicted_classes)} samples")
        print(f"Predicted classes: {np.unique(predicted_classes, return_counts=True)}")

    return predicted_classes, avg_proba

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
    # ---------------------------------------------------------
    use_vix = False
    classification_model_configs = None
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
    if args.compiled_dataset_filename is None:
        # --- UPDATED: Load parameters from args instead of hardcoding ---
        macd_base_cols = ['MACD_Line', 'MACD_Signal', 'MACD_Hist']
        macd_params = json.loads(args.macd_params) if isinstance(args.macd_params, str) else args.macd_params
        SHIFTED_SEQ_COLS = []
        VWAP_COLS, VWAP_COLS_AS_PRICE = [], []
        def is_ema_enabled():
            return args.enable_ema

        def is_sma_enabled():
            return args.enable_sma

        def is_macd_enabled():
            return args.enable_macd

        def is_rsi_enabled():
            return args.enable_rsi

        def is_vwap_enabled():
            return args.enable_vwap

        def is_day_data_enabled():
            return args.enable_day_data
        assert args.look_ahead == 1.
        one_dataset_filename = get_filename_for_dataset(args.dataset_id, older_dataset=None)
        import pickle
        with open(one_dataset_filename, 'rb') as f:
            master_data_cache = pickle.load(f)
        open_col   = ("Open", args.ticker)
        high_col   = ("High", args.ticker)
        low_col    = ("Low", args.ticker)
        close_col  = ("Close", args.ticker)
        volume_col = ("Volume", args.ticker)
        vix__master_data_cache = copy.deepcopy(master_data_cache['^VIX'].sort_index()[('Close', '^VIX')])
        vix1d__master_data_cache = copy.deepcopy(master_data_cache['^VIX1D'].sort_index()[('Close', '^VIX1D')])
        vix3m__master_data_cache = copy.deepcopy(master_data_cache['^VIX3M'].sort_index()[('Close', '^VIX3M')])
        # ---------------------------------------------------------
        # Add POS/NEG Sequence
        # ---------------------------------------------------------
        assert args.epsilon >= 0.
        master_data_cache = add_sequence_columns_vectorized(df=master_data_cache[args.ticker].sort_index(), col_name=close_col, ticker_name=args.ticker, epsilon=args.epsilon)

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # <<master_data_cache>> is now the df
        # ---------------------------------------------------------
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # ADD MOVING AVERAGES
        # ---------------------------------------------------------
        if is_sma_enabled():
            for w in args.sma_windows:
                if 0 == w:
                    continue
                # Create column name tuple to match MultiIndex structure: ('MA_5', '^GSPC')
                ma_col_name = (f'SMA_{w}', args.ticker)
                # Calculate rolling mean on the close column
                master_data_cache[ma_col_name] = master_data_cache[close_col].rolling(window=w).mean()
            #
            for sw in args.shift_sma_col:
                if 0 == sw:
                    continue
                for w in args.sma_windows:
                    if 0 == w:
                        continue
                    ma_col_name = (f'SMA_{w}', args.ticker)
                    shift_ma_col_name = (f'SHIFTED_SMA{w}_{sw}', args.ticker)
                    master_data_cache[shift_ma_col_name] = master_data_cache[ma_col_name].shift(sw)
        if is_ema_enabled():
            for w in args.ema_windows:
                if 0 == w:
                    continue
                ma_col_name = (f'EMA_{w}', args.ticker)
                master_data_cache[ma_col_name] = master_data_cache[close_col].ewm(span=w, adjust=False).mean()
            for sw in args.shift_ema_col:
                if 0 == sw:
                    continue
                for w in args.ema_windows:
                    if 0 == w:
                        continue
                    ma_col_name = (f'EMA_{w}', args.ticker)
                    shift_ma_col_name = (f'SHIFTED_EMA{w}_{sw}', args.ticker)
                    master_data_cache[shift_ma_col_name] = master_data_cache[ma_col_name].shift(sw)

        # ---------------------------------------------------------
        # ADD RSI AND MACD (Like Moving Averages)
        # ---------------------------------------------------------
        # 1. Calculate Base RSI
        if is_rsi_enabled():
            for w in args.rsi_windows:
                if 0 == w:
                    continue
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
                if 0 == sw:
                    continue
                for w in args.rsi_windows:
                    if 0 == w:
                        continue
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

        # ---------------------------------------------------------
        # Adjust VIX
        # ---------------------------------------------------------
        if use_vix:
            if vix__master_data_cache.index[-1] != master_data_cache.index[-1]:
                if args.verbose:
                    print(f"Removing last element of VIX DF.")
                vix__master_data_cache = vix__master_data_cache.iloc[:-1]
            assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix__master_data_cache.index[-1].strftime('%Y-%m-%d')
            assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix1d__master_data_cache.index[-1].strftime('%Y-%m-%d')
            assert master_data_cache.index[-1].strftime('%Y-%m-%d') == vix3m__master_data_cache.index[-1].strftime('%Y-%m-%d')

        # ---------------------------------------------------------
        # Add VWAP
        # ---------------------------------------------------------
        vwap_bands, vwap_cols = (1, 2, 3), None
        if is_vwap_enabled():
            master_data_cache, vwap_cols = add_vwap_with_bands(
                df=master_data_cache,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                volume_col=volume_col,
                ticker=args.ticker,
                window=args.vwap_window,  # None = cumulative, 20 = rolling 20-day
                bands=vwap_bands,
                prefix=f"VWAP_{args.vwap_window}",
                use_hlc3=False,
                add_z_score_feature=True,
                add_scretch_condition=(True, True, True),
            )
            if args.add_only_vwap_z_and_vwap_triggers:
                VWAP_COLS += [vwap_cols['vwap_z']]
                for bb in vwap_bands:
                    VWAP_COLS += [vwap_cols[f'vwap_above_sigma_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_below_sigma_{bb}']]
            else:
                VWAP_COLS += [vwap_cols['vwap'], vwap_cols['vwap_std'], vwap_cols['vwap_z']]
                VWAP_COLS_AS_PRICE += [vwap_cols['vwap']]
                for bb in vwap_bands:
                    VWAP_COLS += [vwap_cols[f'vwap_uband_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_lband_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_above_sigma_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_below_sigma_{bb}']]
                    VWAP_COLS_AS_PRICE += [vwap_cols[f'vwap_uband_{bb}'], vwap_cols[f'vwap_lband_{bb}']]

        # ---------------------------------------------------------
        # Add Day data
        # ---------------------------------------------------------
        if is_day_data_enabled() and args.dataset_id == 'day':
            assert isinstance(master_data_cache.index, pd.DatetimeIndex)
            master_data_cache[('day_of_week', args.ticker)] = master_data_cache.index.dayofweek + 1  # Mon=1, Sun=7
            # 2. Cyclical Encoding (Sine/Cosine)
            # This helps models understand that day 7 is next to day 1
            days_in_week = 7
            master_data_cache[('day_sin', args.ticker)] = np.sin(2 * np.pi * master_data_cache[('day_of_week', args.ticker)] / days_in_week)
            master_data_cache[('day_cos', args.ticker)] = np.cos(2 * np.pi * master_data_cache[('day_of_week', args.ticker)] / days_in_week)

        # ---------------------------------------------------------
        # Shift SEQ
        # ---------------------------------------------------------
        if 0 != args.shift_seq_col:
            _pos_base_col, _neg_base_col = ('POS_SEQ', args.ticker), ('NEG_SEQ', args.ticker)
            for plm in range(1, args.shift_seq_col+1):
                _shift_pos_base_col = (f'SHIFTED_{plm}_POS_SEQ', args.ticker)
                _shift_neg_base_col = (f'SHIFTED_{plm}_NEG_SEQ', args.ticker)
                master_data_cache[_shift_pos_base_col] = master_data_cache[_pos_base_col].shift(plm)
                master_data_cache[_shift_neg_base_col] = master_data_cache[_neg_base_col].shift(plm)
                SHIFTED_SEQ_COLS.append(_shift_pos_base_col)
                SHIFTED_SEQ_COLS.append(_shift_neg_base_col)
        step_back_range = args.step_back_range if args.step_back_range < len(master_data_cache) else len(master_data_cache)

        pos_proba = streak_probability(Namespace(ticker=args.ticker, frequency=args.dataset_id, direction='pos', max_n=15, min_n=0, delta=0., verbose=False, debug_verify_speeding=False, epsilon=args.epsilon))
        neg_proba = streak_probability(Namespace(ticker=args.ticker, frequency=args.dataset_id, direction='neg', max_n=13, min_n=0, delta=0., verbose=False, debug_verify_speeding=False, epsilon=-args.epsilon))
        # 1. Create lookup dictionaries mapping streak length (x) to probability
        # We assume pos_proba/neg_proba are lists where index = streak length
        pos_map, neg_map, count_pos_map, count_neg_map = {}, {}, {}, {}
        #if args.dataset_id == 'day':
        count_pos_map = {x: pos_proba[x]['count'] for x in range(0, np.max(list(pos_proba.keys())) + 1)}
        pos_map       = {x: pos_proba[x]['prob']  for x in range(0, np.max(list(pos_proba.keys()))+1)}
        count_neg_map = {x: neg_proba[x]['count'] for x in range(0, np.max(list(neg_proba.keys())) + 1)}
        neg_map       = {x: neg_proba[x]['prob']  for x in range(0, np.max(list(neg_proba.keys()))+1)}
        if args.verbose:
            print(f"\n{count_pos_map=}\n{pos_map=}")
            print(f"\n{count_neg_map=}\n{neg_map=}")
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
                Xs = [(f'SMA_{w}', args.ticker) for w in args.sma_windows if w != 0]
            if 0 != len(args.shift_sma_col):
                for w in args.sma_windows:
                    if 0 == w:
                        continue
                    Xs += [(f'SHIFTED_SMA{w}_{sw}', args.ticker) for sw in args.shift_sma_col if sw != 0]

        if is_ema_enabled():
            if 0 != len(args.ema_windows):
                Xs = [(f'EMA_{w}', args.ticker) for w in args.ema_windows if w != 0]
            if 0 != len(args.shift_ema_col):
                for w in args.ema_windows:
                    if 0 == w:
                        continue
                    Xs += [(f'SHIFTED_EMA{w}_{sw}', args.ticker) for sw in args.shift_ema_col if sw != 0]

        # Add RSI Features
        if is_rsi_enabled():
            for w in args.rsi_windows:
                if 0 == w:
                    continue
                Xs += [(f'RSI_{w}', args.ticker)]
                if 0 != len(args.shift_rsi_col):
                    for sw in args.shift_rsi_col:
                        if 0 == sw:
                            continue
                        Xs += [(f'SHIFTED_RSI{w}_{sw}', args.ticker)]

        # Add MACD Features
        if is_macd_enabled():
            for base_name in macd_base_cols:
                Xs += [(base_name, args.ticker)]
                for sw in args.shift_macd_col:
                    Xs += [(f'SHIFTED_{base_name}_{sw}', args.ticker)]

        # Add vwap features
        Xs += VWAP_COLS

        if is_day_data_enabled() and args.dataset_id == 'day':
            Xs += [('day_sin', args.ticker), ('day_cos', args.ticker)]

        Xs += SHIFTED_SEQ_COLS

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
            price_cols = [col for col in Xs if ('SMA' in col[0] or 'EMA' in col[0] or col == close_col) and col[1] == args.ticker]
            if is_vwap_enabled():
                price_cols += VWAP_COLS_AS_PRICE
            if args.verbose and not only_print_once:
                print(f"\nConvert price levels to <<{args.convert_price_level_with_baseline}>>, relative to baseline: {price_cols}")
                only_print_once = True
            for col in price_cols:
                assert col in the_X.index
                if args.convert_price_level_with_baseline == "fraction":
                    the_X[col] = the_X[col] / baseline_price
                elif args.convert_price_level_with_baseline == "return":
                    the_X[col] = (the_X[col] / baseline_price) - 1  # -1 --> Returns instead of fractions
                else:
                    assert False, f"{args.convert_price_level_with_baseline=}"
            the_d_data.append(past_df.index[-1])
            the_x_data.append(the_X.values)
            the_y_data.append(the_Y.values)
        the_x_data = np.asarray(the_x_data[::-1])
        the_y_data = np.asarray(the_y_data[::-1])
        the_d_data = np.asarray(the_d_data[::-1])

        if args.verbose:
            print(f"There is {len(the_d_data)} data, ranging from {the_d_data[0].strftime('%Y-%m-%d')} to {the_d_data[-1].strftime('%Y-%m-%d')}.")

        assert the_d_data[0] < the_d_data[-1] and isinstance(the_d_data[0], pd.Timestamp), f"Make sure most recent data is at the end!"

        # Ensure the_y_data is 1D
        the_y_data = the_y_data.ravel()
        if -1 != args.min_percentage_to_keep_class:
            # Calculate class distribution
            unique_classes, counts = np.unique(the_y_data, return_counts=True)
            total_samples = len(the_y_data)
            class_percentages = counts / total_samples * 100

            if args.verbose:
                print(f"Class distribution before filtering:")
                for cls, cnt, pct in zip(unique_classes, counts, class_percentages):
                    print(f"  Class {cls}: {cnt} samples ({pct:.2f}%)")

            # Identify classes with >= X% of samples
            valid_classes = unique_classes[class_percentages >= args.min_percentage_to_keep_class]
            invalid_classes = unique_classes[class_percentages < args.min_percentage_to_keep_class]
            if len(invalid_classes) > 0:
                if args.verbose:
                    print(f"Removing {len(invalid_classes)} class(es) with < {args.min_percentage_to_keep_class}% samples: {invalid_classes}")
                # Create mask for valid samples
                mask = np.isin(the_y_data, valid_classes)
                # Filter data - index axis 0 explicitly for 2D arrays
                the_x_data = the_x_data[mask, :] if the_x_data.ndim > 1 else the_x_data[mask]
                the_y_data = the_y_data[mask]
                the_d_data = the_d_data[mask]
                if args.verbose:
                    print(f"Removed {np.sum(~mask)} samples. Remaining: {len(the_y_data)}")
            # Recalculate num_classes after filtering
            num_classes = len(np.unique(the_y_data))
            assert num_classes > 1, f"There shall be more than 1 class"
            if args.verbose:
                unique_classes_new, counts_new = np.unique(the_y_data, return_counts=True)
                print(f"Unique classes after filtering: {unique_classes_new} ({num_classes})")
                print(f"Count per class:")
                for cls, cnt in zip(unique_classes_new, counts_new):
                    print(f"  Class {int(cls)}: {int(cnt)} samples")
                print(f"There is {len(the_d_data)} data, ranging from {the_d_data[0].strftime('%Y-%m-%d')} to {the_d_data[-1].strftime('%Y-%m-%d')}.")
        # --- Filter by specified classes (e.g., [0, 1, 2]) ---
        if len(args.specific_wanted_class) > 0:
            initial_mask = np.isin(the_y_data, args.specific_wanted_class)
            the_x_data = the_x_data[initial_mask]
            the_y_data = the_y_data[initial_mask]
            the_d_data = the_d_data[initial_mask]
            if args.verbose:
                print(f"Kept only specified classes {args.specific_wanted_class}. Samples remaining: {len(the_y_data)}")
            # Recalculate num_classes after filtering by classes
            num_classes = len(np.unique(the_y_data))
            assert num_classes > 1, f"There shall be more than 1 class"
        assert the_d_data[0] < the_d_data[-1] and isinstance(the_d_data[0], pd.Timestamp), f"Make sure most recent data is at the end!"
        if args.save_dataset_to_file_and_exit is not None:
            if args.verbose:
                print(f"Saving data arrays to disk...")
                print(f"  the_x_data shape: {the_x_data.shape}, dtype: {the_x_data.dtype}")
                print(f"  the_y_data shape: {the_y_data.shape}, dtype: {the_y_data.dtype}")
                print(f"  the_d_data shape: {the_d_data.shape}, dtype: {the_d_data.dtype}")
            args_array = np.array(vars(args), dtype=object)
            np.savez_compressed(args.save_dataset_to_file_and_exit, x=the_x_data, y=the_y_data, d=the_d_data, args=args_array, num_classes=num_classes, xs=Xs, ys=Ys)
            file_size = os.path.getsize(args.save_dataset_to_file_and_exit) / (1024 * 1024)  # Size in MB
            if args.verbose:
                print(f"Saved to {args.save_dataset_to_file_and_exit} ({file_size:.2f} MB)")
            sys.exit(0)
    else:
        # Load the .npz file
        data = np.load(args.compiled_dataset_filename, allow_pickle=True)

        # Access each array by the key you used when saving
        the_x_data  = data['x']
        the_y_data  = data['y']
        the_d_data  = data['d']
        num_classes = data['num_classes']
        Xs = data['xs']
        Ys = data['ys']
        # If the 'd' array contains datetime objects, you may need to restore them
        if the_d_data.dtype == 'datetime64[ns]':
            the_d_data = pd.to_datetime(the_d_data)
        args_dict = data['args'].item()
        loaded_args = Namespace(**args_dict)
        if args.verbose:
            print("\t🔧 Arguments used to generate the loaded data:")
            for arg, value in vars(loaded_args).items():
                if 'master_data_cache' in arg:
                    print(f"\t    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to {value.index[-1].strftime('%Y-%m-%d')} ({args.one_dataset_filename})")
                    continue
                print(f"\t    {arg:.<40} {value}")
            print("-" * 80, flush=True)
            unique_classes_new, counts_new = np.unique(the_y_data, return_counts=True)
            print(f"Unique classes after filtering: {unique_classes_new} ({num_classes})")
            print(f"Count per class:")
            for cls, cnt in zip(unique_classes_new, counts_new):
                print(f"  Class {int(cls)}: {int(cnt)} samples")
            print(f"There is {len(the_d_data)} data, ranging from {the_d_data[0].strftime('%Y-%m-%d')} to {the_d_data[-1].strftime('%Y-%m-%d')}.")
            print(f"Xs={Xs}    Ys={Ys}")
    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores, f1_scores = [], []
    # Initialize lists to store per-class scores across folds ---
    fold_precision_scores_per_class = []
    fold_recall_scores_per_class = []
    fold_f1_scores_per_class = []
    # ---------------------------------------------------------
    # MODEL CONFIGURATION (UPDATED WITH ARGS)
    # ---------------------------------------------------------
    # 1. Parse Model Overrides
    try:
        model_overrides = json.loads(args.model_overrides)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON for --model_overrides. Example: '{\"xgb\": {\"params\": {\"n_estimators\": 1000}}}'")

    ## 2. Define Base Configs using Global Args
    # We use a factory pattern to ensure args are applied consistently
    def get_base_config(model_name, model_class, params_template):
        cfg = {
            "class": model_class,
            "params": params_template.copy(),
            "supports_early_stopping": False,
            "uses_validation_fraction": False
        }
        # Apply global args to params if keys exist
        if 'n_estimators' in cfg["params"]: cfg["params"]["n_estimators"] = args.n_estimators
        if 'max_depth' in cfg["params"]: cfg["params"]["max_depth"] = args.max_depth
        if 'learning_rate' in cfg["params"]: cfg["params"]["learning_rate"] = args.learning_rate
        if 'max_iter' in cfg["params"]: cfg["params"]["max_iter"] = args.n_estimators # Map n_estimators to max_iter for some models
        if 'iterations' in cfg["params"]: cfg["params"]["iterations"] = args.n_estimators # Map for CatBoost
        if 'depth' in cfg["params"]: cfg["params"]["depth"] = args.max_depth # Map for CatBoost

        # Special flags
        if model_name in ["xgb", "lgb", "cat", "hgb", "mlp"]:
            cfg["supports_early_stopping"] = False
        if model_name in ["hgb", "mlp"]:
            cfg["uses_validation_fraction"] = False
        return cfg

    classification_model_configs = {
        "xgb": get_base_config("xgb", xgb.XGBClassifier, {
            'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'objective': 'multi:softprob', 'num_class': num_classes,
            'eval_metric': 'mlogloss', 'tree_method': 'hist',
        }),
        "rf": get_base_config("rf", RandomForestClassifier, {
            'n_estimators': 500, 'max_depth': 6, 'random_state': 42,
            'n_jobs': -1, 'class_weight': 'balanced'
        }),
        "lgb": get_base_config("lgb", lgb.LGBMClassifier, {
            'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
            'objective': 'multiclass', 'num_class': num_classes,
            'metric': 'multi_logloss', 'verbosity': -1,
        }),
        "cat": get_base_config("cat", CatBoostClassifier, {
            'iterations': 500, 'depth': 6, 'learning_rate': 0.05,
            'random_seed': 42, 'verbose': False,
            'loss_function': 'MultiClass', 'classes_count': num_classes,
            'eval_metric': 'MultiClass',
        }),
        "hgb": get_base_config("hgb", HistGradientBoostingClassifier, {
            'max_iter': 500, 'max_depth': 6, 'learning_rate': 0.05,
            'random_state': 42, 'loss': 'log_loss',
            'early_stopping': False, 'n_iter_no_change': 20,
        }),
        "et": get_base_config("et", ExtraTreesClassifier, {
            'n_estimators': 500, 'max_depth': 6, 'random_state': 42,
            'n_jobs': -1, 'class_weight': 'balanced', 'bootstrap': False
        }),
        "svm": get_base_config("svm", SVC, {
            'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'random_state': 42,
            'probability': True, 'class_weight': 'balanced',
            'decision_function_shape': 'ovr'
        }),
        "knn": get_base_config("knn", KNeighborsClassifier, {
            'n_neighbors': 5, 'weights': 'distance', 'metric': 'minkowski',
            'p': 2, 'n_jobs': 1
        }),
        "mlp": get_base_config("mlp", MLPClassifier, {
            'hidden_layer_sizes': (100, 50), 'activation': 'relu',
            'solver': 'adam', 'alpha': 0.0001, 'batch_size': 'auto',
            'learning_rate': 'adaptive', 'learning_rate_init': 0.05,
            'max_iter': 500, 'early_stopping': False,
            'validation_fraction': 0.1, 'n_iter_no_change': 20,
            'random_state': 42,
        }),
        "lr": get_base_config("lr", LogisticRegression, {
            'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs',
            'max_iter': 500, 'random_state': 42,
            'class_weight': 'balanced', 'n_jobs': -1
        }),
        "dt": get_base_config("dt", DecisionTreeClassifier, {
            'max_depth': 6, 'random_state': 42, 'class_weight': 'balanced',
            'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 1
        })
    }
    # 3. Apply JSON Overrides
    for model_name, override_params in model_overrides.items():
        if model_name in classification_model_configs:
            deep_update(classification_model_configs[model_name], override_params)
            if args.verbose:
                print(f"⚙️ Applied overrides for {model_name}: {override_params}")
        else:
            print(f"⚠️ Warning: Model '{model_name}' in overrides not found in config.")
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

        fold_predictions_proba = []  # Store probabilities for ensemble
        for model_name in args.base_models:
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
    # ---------------------------------------------------------
    # SAVE MODEL AND PREPROCESSING OBJECTS
    # ---------------------------------------------------------
    if args.save_model_path is not None:
        if args.verbose:
            print("\n" + "=" * 80)
            print("💾 Saving model and preprocessing objects for later inference...")
            print("=" * 80)

        # Train final models on ALL data (not just CV folds)
        if args.verbose:
            print(f"Training final models on full dataset ({len(the_x_data)} samples)...")

        # Fit scaler on full dataset
        final_scaler = RobustScaler()
        X_full_scaled = final_scaler.fit_transform(the_x_data)
        y_full = the_y_data.ravel().astype(int)

        # Train final ensemble models
        final_models = {}
        for model_name in args.base_models:
            cfg = classification_model_configs[model_name]
            model = cfg["class"](**cfg["params"])

            if args.verbose:
                print(f"  Training {model_name}...")

            # Handle early stopping if supported
            fit_kwargs = {}
            if cfg.get("uses_validation_fraction", False):
                fit_kwargs['validation_fraction'] = 0.1
                fit_kwargs['n_iter_no_change'] = 20

            model.fit(X_full_scaled, y_full, **fit_kwargs)
            final_models[model_name] = model

        # Prepare metadata for inference
        model_metadata = {
            'scaler': final_scaler,
            'models': final_models,
            'feature_list': Xs,
            'feature_names': [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col) for col in Xs],
            'num_classes': num_classes,
            'target_column': Ys,
            'args': vars(args),
            'class_distribution': {int(cls): int(cnt) for cls, cnt in zip(unique_classes_new, counts_new)},
            'training_date_range': {
                'start': the_d_data[0].strftime('%Y-%m-%d') if hasattr(the_d_data[0], 'strftime') else str(the_d_data[0]),
                'end': the_d_data[-1].strftime('%Y-%m-%d') if hasattr(the_d_data[-1], 'strftime') else str(the_d_data[-1])
            },
            'cv_scores': {
                'accuracy_mean': float(np.mean(acc_scores)),
                'accuracy_std': float(np.std(acc_scores)),
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
                'per_class_f1_mean': avg_f1.tolist(),
                'per_class_precision_mean': avg_precision.tolist(),
                'per_class_recall_mean': avg_recall.tolist(),
            },
            'preprocessing_config': {
                'convert_price_level_with_baseline': args.convert_price_level_with_baseline,
                'enable_ema': args.enable_ema,
                'enable_sma': args.enable_sma,
                'enable_rsi': args.enable_rsi,
                'enable_macd': args.enable_macd,
                'enable_vwap': args.enable_vwap,
                'enable_day_data': args.enable_day_data,
                'shift_seq_col': args.shift_seq_col,
                'ema_windows': args.ema_windows,
                'sma_windows': args.sma_windows,
                'rsi_windows': args.rsi_windows,
                'macd_params': macd_params if 'macd_params' in locals() else args.macd_params,
                'vwap_window': args.vwap_window,
                'add_only_vwap_z_and_vwap_triggers': args.add_only_vwap_z_and_vwap_triggers,
                'epsilon': args.epsilon,
                'look_ahead': args.look_ahead,
            }
        }

        # Save using pickle
        import pickle
        os.makedirs(os.path.dirname(args.save_model_path) if os.path.dirname(args.save_model_path) else '.', exist_ok=True)

        with open(args.save_model_path, 'wb') as f:
            pickle.dump(model_metadata, f)

        file_size = os.path.getsize(args.save_model_path) / (1024 * 1024)
        if args.verbose:
            print(f"\n✅ Model saved successfully!")
            print(f"   Path: {args.save_model_path}")
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Models saved: {list(final_models.keys())}")
            print(f"   Features: {len(Xs)}")
            print(f"   Classes: {num_classes}")
            print(f"   CV Accuracy: {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
            print(f"   CV F1-Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
            # ---------------------------------------------------------
            # Added Per-Class Performance Table
            # ---------------------------------------------------------
            print(f"\n📊 Per-Class CV Performance (Average across {len(fold_f1_scores_per_class)} folds):")
            print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 44)
            for c in range(num_classes):
                print(f"{c:<8} {avg_precision[c]:<12.4f} {avg_recall[c]:<12.4f} {avg_f1[c]:<12.4f}")
            print("=" * 80)
    return np.mean(f1_scores), np.mean(acc_scores), avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ticker", type=str, default='^GSPC')
    parser.add_argument("--older_dataset", type=str, default="None")
    parser.add_argument("--dataset_id", type=str, default="day", choices=DATASET_AVAILABLE)
    parser.add_argument("--look_ahead", type=int, default=1)
    #
    parser.add_argument('--step_back_range', type=int, default=99999,
                        help="Number of historical time windows to simulate (rolling backtest depth).")
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument("--epsilon", type=float, default=0.,
                        help="Threshold for neutral returns. Default: 0.")
    parser.add_argument("--target", type=str, default='POS_SEQ', choices=['POS_SEQ', 'NEG_SEQ'],
                        help="Target column for prediction. Options: POS_SEQ, NEG_SEQ. Default: POS_SEQ.")
    parser.add_argument("--convert_price_level_with_baseline", type=str, default='fraction', choices=['fraction', 'return'],
                        help="Method to convert price levels. 'fraction': price/baseline, 'return': (price/baseline)-1. Default: fraction.")
    parser.add_argument("--ema_windows", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9],
                        help="List of window sizes for Exponential Moving Average calculation. Default: 2 to 9.")
    parser.add_argument('--enable_ema', type=str2bool, default=False)
    parser.add_argument("--shift_ema_col", type=int, nargs='+', default=[],
                        help="List of shift periods for EMA. Default: None.")
    parser.add_argument("--sma_windows", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9],
                        help="List of window sizes for Moving Average calculation. Default: 2 to 9.")
    parser.add_argument('--enable_sma', type=str2bool, default=False)
    parser.add_argument("--shift_sma_col", type=int, nargs='+', default=[],
                        help="List of shift periods for SMA. Default: None.")
    parser.add_argument("--rsi_windows", type=int, nargs='+', default=[14],
                        help="List of window sizes for RSI calculation. Default: 14.")
    parser.add_argument("--shift_rsi_col", type=int, nargs='+', default=[],
                        help="List of shift periods for RSI. Default: None.")
    parser.add_argument('--enable_rsi', type=str2bool, default=False)
    parser.add_argument("--macd_params", type=str, default='{"fast": 12, "slow": 26, "signal": 9}',
                        help="JSON string for MACD parameters (fast, slow, signal). Default: 12, 26, 9.")
    parser.add_argument('--enable_macd', type=str2bool, default=False)
    parser.add_argument("--shift_macd_col", type=int, nargs='+', default=[],
                        help="List of shift periods for MACD. Default: None.")
    parser.add_argument('--enable_vwap', type=str2bool, default=False)
    parser.add_argument('--vwap_window', type=int, default=20)
    parser.add_argument('--add_only_vwap_z_and_vwap_triggers', type=str2bool, default=False)
    parser.add_argument('--enable_day_data', type=str2bool, default=True,
                        help="Add column for the day")
    parser.add_argument('--compiled_dataset_filename', type=str, default=None,
                        help="Skip the dataframe build process and use the one provided.")
    parser.add_argument('--save_dataset_to_file_and_exit', type=str, default=None,
                        help="Save to disk the dataset created and used for ML.")
    parser.add_argument('--min_percentage_to_keep_class', type=float, default=4.,
                        help="Minimum percentage of class target data in Y. Default: 4. -1 to disabled.")
    parser.add_argument("--specific_wanted_class", type=int, nargs='+', default=[],
                        help="List of classes to keep. Discard others. Default: None.")
    parser.add_argument(
        "--base_models",
        type=str,
        nargs="+",  # Accept one or more values
        default=["xgb"],
        choices=["xgb", "lgb", "cat", "hgb", "rf", "et", "svm", "knn", "mlp", "lr", "dt"],
        help="Base model(s) to use for training. Can specify multiple. "
             "Options: xgb (XGBoost), lgb (LightGBM), cat (CatBoost), hgb (HistGradientBoosting), "
             "rf (Random Forest), et (Extra Trees), svm (SVM), knn (K-Nearest Neighbors), "
             "mlp (Multi-Layer Perceptron), lr (Logistic Regression), dt (Decision Tree). "
             "Example: --base_models xgb lgb cat"
    )
    parser.add_argument('--save_model_path', type=str, default=None,
                        help="Path to save the trained model(s) and preprocessing objects for later inference.")
    # ---------------------------------------------------------
    # MODEL HYPERPARAMETER ARGUMENTS
    # ---------------------------------------------------------
    parser.add_argument('--n_estimators', type=int, default=500,
                        help="Global default for n_estimators/iterations/max_iter across models. (Default: 500)")
    parser.add_argument('--max_depth', type=int, default=6,
                        help="Global default for max_depth/depth across models. (Default: 6)")
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help="Global default for learning_rate across models. (Default: 0.05)")
    parser.add_argument('--model_overrides', type=str, default='{}',
                        help='JSON string to override specific model params. Structure: {"model_name": {"params": {"param": value}}}. '
                             'Example: \'{"xgb": {"params": {"subsample": 0.5}}, "rf": {"params": {"n_estimators": 100}}}\'')
    # ---------------------------------------------------------

    parser.add_argument("--shift_seq_col", type=int, default=3,
                        help="")
    args = parser.parse_args()

    main(args)