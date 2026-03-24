"""
================================================================================
POS/NEG Sequence Classification Model
================================================================================
This script trains classification models to predict streak sequences (POS_SEQ, NEG_SEQ)
using technical indicators and ensemble methods.
================================================================================
"""

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
import sys
import pickle
import json
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
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
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")

# Import utility functions
from utils import DATASET_AVAILABLE, get_filename_for_dataset, str2bool, add_vwap_with_bands
from runners.streak_probability import new_main as streak_probability
from runners.streak_probability import add_sequence_columns_vectorized

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def deep_update(base_dict, update_dict):
    """
    Recursively update a dictionary with values from another dictionary.

    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with values to update

    Returns:
        Updated base_dict
    """
    for key, value in update_dict.items():
        if (key in base_dict and
                isinstance(base_dict[key], dict) and
                isinstance(value, dict)):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def encode_tuple_target(pos_seq, neg_seq, tuple_to_class):
    """
    Encode (pos_seq, neg_seq) tuple into sequential class label.

    Args:
        pos_seq: Positive sequence length
        neg_seq: Negative sequence length
        tuple_to_class: Dictionary mapping tuples to class IDs

    Returns:
        Class ID (int)
    """
    return tuple_to_class.get((pos_seq, neg_seq), -1)


def decode_tuple_target(class_id, class_to_tuple):
    """
    Decode sequential class label back to (pos_seq, neg_seq) tuple.

    Args:
        class_id: Class ID to decode
        class_to_tuple: Dictionary mapping class IDs to tuples

    Returns:
        Tuple of (pos_seq, neg_seq)
    """
    return class_to_tuple.get(class_id, (0, 0))


def create_tuple_label_mapping(df, pos_col, neg_col):
    """
    Create mapping between tuple values and SEQUENTIAL class IDs.

    Args:
        df: DataFrame containing pos and neg columns
        pos_col: Column name for positive sequences
        neg_col: Column name for negative sequences

    Returns:
        tuple_to_class: Dict mapping (pos, neg) -> class_id
        class_to_tuple: Dict mapping class_id -> (pos, neg)
        num_classes: Total number of unique classes
    """
    # Get all unique (pos, neg) tuples from the data
    unique_tuples = df[[pos_col, neg_col]].dropna().apply(lambda row: (int(row[pos_col]), int(row[neg_col])), axis=1).unique()

    # Sort tuples for consistent ordering
    unique_tuples = sorted(unique_tuples, key=lambda x: (x[0], x[1]))

    # Create SEQUENTIAL mapping (0, 1, 2, 3, ...)
    tuple_to_class = {}
    class_to_tuple = {}
    for i, (pos, neg) in enumerate(unique_tuples):
        tuple_to_class[(pos, neg)] = i
        class_to_tuple[i] = (pos, neg)

    num_classes = len(tuple_to_class)
    return tuple_to_class, class_to_tuple, num_classes


def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        series: Price series
        window: Lookback window (default: 14)

    Returns:
        RSI series
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator.

    Args:
        series: Price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        macd_line, signal_line, histogram
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def load_model_for_inference(model_path, verbose=True):
    """
    Load a saved model and preprocessing objects for inference.

    Args:
        model_path: Path to the saved .pkl model file
        verbose: Print loading information

    Returns:
        dict: Contains scaler, models, feature_list, and metadata
    """
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
        print(f"   Training period: {model_data['training_date_range']['start']} to "
              f"{model_data['training_date_range']['end']}")
        print(f"   CV Accuracy: {model_data['cv_scores']['accuracy_mean']:.4f} "
              f"(+/- {model_data['cv_scores']['accuracy_std']:.4f})")
        print(f"   CV F1-Score: {model_data['cv_scores']['f1_mean']:.4f} "
              f"(+/- {model_data['cv_scores']['f1_std']:.4f})")
        print("=" * 80)

    return model_data


def predict_with_saved_model(model_data, X_new, verbose=True):
    """
    Make predictions using a loaded model.

    Args:
        model_data: Dict from load_model_for_inference()
        X_new: New feature data (numpy array or DataFrame)
        verbose: Print prediction information

    Returns:
        tuple: (predicted_classes, prediction_probabilities, decoded_tuples)
    """
    scaler = model_data['scaler']
    models = model_data['models']
    num_classes = model_data['num_classes']
    feature_list = model_data['feature_list']
    class_to_tuple = model_data.get('class_to_tuple', None)

    # Ensure X_new is numpy array
    if hasattr(X_new, 'values'):
        X_new = X_new.values

    # Validate feature count
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

    # Decode tuples if mapping exists
    decoded_tuples = None
    if class_to_tuple is not None:
        decoded_tuples = [class_to_tuple.get(cls, (0, 0)) for cls in predicted_classes]

    if verbose:
        print(f"Made predictions for {len(predicted_classes)} samples")
        if decoded_tuples:
            print(f"Decoded tuples: {decoded_tuples[:10]}...")

    return predicted_classes, avg_proba, decoded_tuples


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(args):
    """
    Main function for training classification models on POS/NEG sequence data.

    Args:
        args: Command line arguments
    """
    # Force classification strategy (regression removed)
    args.optimization_strategy = 'classification'

    # Print arguments if verbose
    if args.verbose:
        print("🔧 Arguments:")
        for arg, value in vars(args).items():
            if 'master_data_cache' in str(arg):
                if hasattr(value, 'index'):
                    print(f"    {arg:.<40} {value.index[0].strftime('%Y-%m-%d')} to "
                          f"{value.index[-1].strftime('%Y-%m-%d')}")
                continue
            print(f"    {arg:.<40} {value}")
        print("-" * 80, flush=True)

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------
    the_x_data, the_y_data, the_d_data = [], [], []
    num_classes, my_label_encoder, tuple_to_class, class_to_tuple = None, None, None, None
    if args.compiled_dataset_filename is None:
        # Load parameters
        macd_base_cols = ['MACD_Line', 'MACD_Signal', 'MACD_Hist']
        macd_params = (json.loads(args.macd_params)
                       if isinstance(args.macd_params, str)
                       else args.macd_params)

        # Initialize column lists
        SHIFTED_SEQ_COLS = []
        CLOSE_DIFF_COLS = []
        VWAP_COLS = []
        VWAP_COLS_AS_PRICE = []

        # Load dataset
        one_dataset_filename = get_filename_for_dataset(args.dataset_id,
                                                        older_dataset=None)
        with open(one_dataset_filename, 'rb') as f:
            master_data_cache = pickle.load(f)

        # Define column tuples
        open_col = ("Open", args.ticker)
        high_col = ("High", args.ticker)
        low_col = ("Low", args.ticker)
        close_col = ("Close", args.ticker)
        volume_col = ("Volume", args.ticker)

        # Add POS/NEG sequence columns
        assert args.epsilon >= 0.
        master_data_cache = add_sequence_columns_vectorized(
            df=master_data_cache[args.ticker].sort_index(),
            col_name=close_col,
            ticker_name=args.ticker,
            epsilon=args.epsilon
        )

        # Add close difference
        if args.add_close_diff:
            master_data_cache[("Close_diff", args.ticker)] = master_data_cache[close_col].diff() / master_data_cache[close_col]
            CLOSE_DIFF_COLS.append(("Close_diff", args.ticker))

        # Add Moving Averages (SMA)
        if args.enable_sma:
            for w in args.sma_windows:
                if w == 0:
                    continue
                ma_col_name = (f'SMA_{w}', args.ticker)
                master_data_cache[ma_col_name] = (
                    master_data_cache[close_col].rolling(window=w).mean()
                )
                for sw in args.shift_sma_col:
                    if sw == 0:
                        continue
                    shift_ma_col_name = (f'SHIFTED_SMA{w}_{sw}', args.ticker)
                    master_data_cache[shift_ma_col_name] = (
                        master_data_cache[ma_col_name].shift(sw)
                    )

        # Add Moving Averages (EMA)
        if args.enable_ema:
            for w in args.ema_windows:
                if w == 0:
                    continue
                ma_col_name = (f'EMA_{w}', args.ticker)
                master_data_cache[ma_col_name] = (
                    master_data_cache[close_col].ewm(span=w, adjust=False).mean()
                )
                for sw in args.shift_ema_col:
                    if sw == 0:
                        continue
                    shift_ma_col_name = (f'SHIFTED_EMA{w}_{sw}', args.ticker)
                    master_data_cache[shift_ma_col_name] = (
                        master_data_cache[ma_col_name].shift(sw)
                    )

        # Add RSI
        if args.enable_rsi:
            for w in args.rsi_windows:
                if w == 0:
                    continue
                rsi_col_name = (f'RSI_{w}', args.ticker)
                master_data_cache[rsi_col_name] = calculate_rsi(
                    master_data_cache[close_col], window=w
                )
                for sw in args.shift_rsi_col:
                    if sw == 0:
                        continue
                    shift_rsi_col_name = (f'SHIFTED_RSI{w}_{sw}', args.ticker)
                    master_data_cache[shift_rsi_col_name] = (
                        master_data_cache[rsi_col_name].shift(sw)
                    )

        # Add MACD
        if args.enable_macd:
            macd_line, signal_line, histogram = calculate_macd(
                master_data_cache[close_col],
                fast=macd_params['fast'],
                slow=macd_params['slow'],
                signal=macd_params['signal']
            )
            master_data_cache[('MACD_Line', args.ticker)] = macd_line
            master_data_cache[('MACD_Signal', args.ticker)] = signal_line
            master_data_cache[('MACD_Hist', args.ticker)] = histogram

            for sw in args.shift_macd_col:
                for base_name in macd_base_cols:
                    base_col_name = (base_name, args.ticker)
                    shift_macd_col_name = (f'SHIFTED_{base_name}_{sw}', args.ticker)
                    master_data_cache[shift_macd_col_name] = (
                        master_data_cache[base_col_name].shift(sw)
                    )

        # Add VWAP
        if args.enable_vwap:
            vwap_bands = (1, 2, 3)
            master_data_cache, vwap_cols = add_vwap_with_bands(
                df=master_data_cache,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                volume_col=volume_col,
                ticker=args.ticker,
                window=args.vwap_window,
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
                VWAP_COLS += [vwap_cols['vwap'], vwap_cols['vwap_std'],
                              vwap_cols['vwap_z']]
                VWAP_COLS_AS_PRICE += [vwap_cols['vwap']]
                for bb in vwap_bands:
                    VWAP_COLS += [vwap_cols[f'vwap_uband_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_lband_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_above_sigma_{bb}']]
                    VWAP_COLS += [vwap_cols[f'vwap_below_sigma_{bb}']]
                    VWAP_COLS_AS_PRICE += [vwap_cols[f'vwap_uband_{bb}'],
                                           vwap_cols[f'vwap_lband_{bb}']]

        # Add Day data
        if args.enable_day_data and args.dataset_id == 'day':
            assert isinstance(master_data_cache.index, pd.DatetimeIndex)
            master_data_cache[('day_of_week', args.ticker)] = (
                    master_data_cache.index.dayofweek + 1
            )
            days_in_week = 7
            master_data_cache[('day_sin', args.ticker)] = np.sin(
                2 * np.pi * master_data_cache[('day_of_week', args.ticker)] /
                days_in_week
            )
            master_data_cache[('day_cos', args.ticker)] = np.cos(
                2 * np.pi * master_data_cache[('day_of_week', args.ticker)] /
                days_in_week
            )

        # Shift SEQ columns
        if args.shift_seq_col != 0:
            _pos_base_col = ('POS_SEQ', args.ticker)
            _neg_base_col = ('NEG_SEQ', args.ticker)
            for plm in range(1, args.shift_seq_col + 1):
                _shift_pos_base_col = (f'SHIFTED_{plm}_POS_SEQ', args.ticker)
                _shift_neg_base_col = (f'SHIFTED_{plm}_NEG_SEQ', args.ticker)
                master_data_cache[_shift_pos_base_col] = (
                    master_data_cache[_pos_base_col].shift(plm)
                )
                master_data_cache[_shift_neg_base_col] = (
                    master_data_cache[_neg_base_col].shift(plm)
                )
                SHIFTED_SEQ_COLS.append(_shift_pos_base_col)
                SHIFTED_SEQ_COLS.append(_shift_neg_base_col)

        # Calculate streak probabilities
        step_back_range = args.step_back_range if args.step_back_range < len(master_data_cache) else len(master_data_cache)

        pos_proba = streak_probability(
            Namespace(ticker=args.ticker, frequency=args.dataset_id,
                      direction='pos', max_n=15, min_n=0, delta=0.,
                      verbose=False, debug_verify_speeding=False,
                      epsilon=args.epsilon))
        neg_proba = streak_probability(
            Namespace(ticker=args.ticker, frequency=args.dataset_id,
                      direction='neg', max_n=13, min_n=0, delta=0.,
                      verbose=False, debug_verify_speeding=False,
                      epsilon=-args.epsilon))

        # Create probability mappings
        pos_map = {x: pos_proba[x]['prob'] for x in range(0, np.max(list(pos_proba.keys())) + 1)}
        neg_map = {x: neg_proba[x]['prob'] for x in range(0, np.max(list(neg_proba.keys())) + 1)}

        if args.verbose:
            print(f"POS Probability Map: {pos_map}")
            print(f"NEG Probability Map: {neg_map}")

        # Map sequence columns to probability columns
        master_data_cache[('positive_probability', args.ticker)] = master_data_cache[('POS_SEQ', args.ticker)].map(pos_map)
        master_data_cache[('negative_probability', args.ticker)] = master_data_cache[('NEG_SEQ', args.ticker)].map(neg_map)

        # Add future information (next step within low/high)
        if args.drop_when_out_of_range:
            master_data_cache[("next_step_within_low_high", args.ticker)] = master_data_cache[close_col].shift(-1).between(master_data_cache[low_col],master_data_cache[high_col],inclusive="both")

        # -------------------------------------------------------------------------
        # BUILD FEATURE LIST (Xs)
        # -------------------------------------------------------------------------
        Xs = []
        Xs += CLOSE_DIFF_COLS

        if args.enable_sma and len(args.sma_windows) > 0:
            Xs += [(f'SMA_{w}', args.ticker) for w in args.sma_windows if w != 0]
            if len(args.shift_sma_col) > 0:
                for w in args.sma_windows:
                    if w == 0:
                        continue
                    Xs += [(f'SHIFTED_SMA{w}_{sw}', args.ticker) for sw in args.shift_sma_col if sw != 0]

        if args.enable_ema and len(args.ema_windows) > 0:
            Xs += [(f'EMA_{w}', args.ticker) for w in args.ema_windows if w != 0]
            if len(args.shift_ema_col) > 0:
                for w in args.ema_windows:
                    if w == 0:
                        continue
                    Xs += [(f'SHIFTED_EMA{w}_{sw}', args.ticker)
                           for sw in args.shift_ema_col if sw != 0]

        if args.enable_rsi:
            for w in args.rsi_windows:
                if w == 0:
                    continue
                Xs += [(f'RSI_{w}', args.ticker)]
                for sw in args.shift_rsi_col:
                    if sw == 0:
                        continue
                    Xs += [(f'SHIFTED_RSI{w}_{sw}', args.ticker)]

        if args.enable_macd:
            for base_name in macd_base_cols:
                Xs += [(base_name, args.ticker)]
                for sw in args.shift_macd_col:
                    Xs += [(f'SHIFTED_{base_name}_{sw}', args.ticker)]

        Xs += VWAP_COLS

        if args.enable_day_data and args.dataset_id == 'day':
            Xs += [('day_sin', args.ticker), ('day_cos', args.ticker)]

        Xs += SHIFTED_SEQ_COLS

        # Add sequence columns and close price
        Xs += [("POS_SEQ", args.ticker), ("NEG_SEQ", args.ticker), ("STREAK_SEQ", args.ticker), close_col]

        # Target columns (POS/NEG tuple)
        Ys = [('POS_SEQ', args.ticker), ('NEG_SEQ', args.ticker)]

        if args.verbose:
            print(f"Xs ({len(Xs)}) are: {Xs}")
            print(f"Ys ({len(Ys)}) are: {Ys}")

        # -------------------------------------------------------------------------
        # PREPARE TRAINING DATA
        # -------------------------------------------------------------------------
        # Remove invalid rows (both POS and NEG non-zero)
        invalid_mask = ((master_data_cache[("POS_SEQ", args.ticker)] != 0) & (master_data_cache[("NEG_SEQ", args.ticker)] != 0))
        invalid_count = invalid_mask.sum()

        if invalid_count > 0 and args.epsilon == 0:
            if args.verbose:
                print(f"⚠️ Dropping {invalid_count} rows where POS_SEQ and NEG_SEQ were both non-zero.")
            invalid_rows = master_data_cache.index[invalid_mask]
            print(f"Mutual exclusivity violated at indices: {invalid_rows.tolist()}")
            master_data_cache = master_data_cache[~invalid_mask]

        # Create tuple label mapping
        tuple_to_class, class_to_tuple, _ = create_tuple_label_mapping(master_data_cache, ("POS_SEQ", args.ticker), ("NEG_SEQ", args.ticker))
        assert len(tuple_to_class) == len(class_to_tuple)
        if args.verbose:
            print(f"Tuple to Class Mapping: {tuple_to_class}")

        # Build dataset with step_back iteration
        for step_back in (tqdm(range(0, step_back_range + 1)) if args.verbose else range(0, step_back_range + 1)):
            if step_back == 0:
                past_df = master_data_cache.dropna()
                future_df = None
                if not args.real_time_only:
                    continue
            else:
                assert not args.real_time_only
                past_df = master_data_cache.iloc[:-step_back].dropna()
                future_df = master_data_cache.iloc[-step_back:].dropna()

                if not args.real_time_only and (args.look_ahead > len(future_df) or len(past_df) == 0):
                    continue

            if not args.real_time_only:
                assert past_df.index.intersection(future_df.index).empty, "Indices must be disjoint"

            # Extract features for the last row of past_df
            the_X = past_df.iloc[-1][Xs].copy().sort_index()

            # Encode tuple target into single class
            if not args.real_time_only:
                pos_val = int(future_df.iloc[args.look_ahead - 1][("POS_SEQ", args.ticker)])
                neg_val = int(future_df.iloc[args.look_ahead - 1][("NEG_SEQ", args.ticker)])
                the_Y = encode_tuple_target(pos_val, neg_val, tuple_to_class)
            else:
                the_Y = None

            # Normalize price levels for stationarity
            baseline_price = the_X[close_col]
            if isinstance(baseline_price, pd.Series):
                assert len(baseline_price) == 1
                baseline_price = baseline_price.values[0]
            assert baseline_price > 0, f"{baseline_price=}   {close_col=}"

            price_cols = [col for col in Xs
                          if ('SMA' in col[0] or 'EMA' in col[0] or
                              col == close_col) and col[1] == args.ticker]

            if args.enable_vwap:
                price_cols += VWAP_COLS_AS_PRICE

            for col in price_cols:
                assert col in the_X.index
                if args.convert_price_level_with_baseline == "fraction":
                    the_X[col] = the_X[col] / baseline_price
                elif args.convert_price_level_with_baseline == "return":
                    the_X[col] = (the_X[col] / baseline_price) - 1
                else:
                    raise ValueError(f"Invalid conversion method: "
                                     f"{args.convert_price_level_with_baseline}")

            the_d_data.append(past_df.index[-1])
            the_x_data.append(the_X.values)

            if not args.real_time_only:
                the_y_data.append(the_Y)

            if args.real_time_only:
                break

        # Convert to numpy arrays (reverse to get chronological order)
        the_x_data = np.asarray(the_x_data[::-1])
        the_y_data = np.asarray(the_y_data[::-1]) if not args.real_time_only else None
        the_d_data = np.asarray(the_d_data[::-1])

        if args.verbose:
            print(f"There is {len(the_d_data)} data, ranging from {the_d_data[0].strftime('%Y-%m-%d')} to {the_d_data[-1].strftime('%Y-%m-%d')}.")

        if not args.real_time_only:
            the_y_data = the_y_data.ravel() # Ensure y_data is 1D
            # Filter classes by minimum percentage
            if args.min_percentage_to_keep_class != -1:
                unique_classes, counts = np.unique(the_y_data, return_counts=True)
                total_samples = len(the_y_data)
                class_percentages = counts / total_samples * 100

                if args.verbose:
                    print("Class distribution before filtering:")
                    for cls, cnt, pct in zip(unique_classes, counts, class_percentages):
                        print(f"  Class {cls}: {cnt} samples ({pct:.2f}%)")

                valid_classes = unique_classes[class_percentages >= args.min_percentage_to_keep_class]
                invalid_classes = unique_classes[class_percentages < args.min_percentage_to_keep_class]

                if len(invalid_classes) > 0:
                    if args.verbose:
                        print(f"Removing {len(invalid_classes)} class(es) with < {args.min_percentage_to_keep_class}% samples: {invalid_classes}")

                    mask = np.isin(the_y_data, valid_classes)
                    the_x_data = the_x_data[mask, :] if the_x_data.ndim > 1 else the_x_data[mask]
                    the_y_data = the_y_data[mask]
                    the_d_data = the_d_data[mask]
                    if args.verbose:
                        print(f"Removed {np.sum(~mask)} samples. Remaining: {len(the_y_data)}")
                    if args.verbose:
                        unique_classes_new, counts_new = np.unique(the_y_data, return_counts=True)
                        print(f"Unique classes after filtering: {unique_classes_new} ({num_classes})")
                        print("Count per class:")
                        for cls, cnt in zip(unique_classes_new, counts_new):
                            print(f"  Class {int(cls)}: {int(cnt)} samples")

            # Filter by specific classes if specified
            if len(args.specific_wanted_class) > 0:
                initial_mask = np.isin(the_y_data, args.specific_wanted_class)
                the_x_data = the_x_data[initial_mask]
                the_y_data = the_y_data[initial_mask]
                the_d_data = the_d_data[initial_mask]

                if args.verbose:
                    print(f"Kept only specified classes {args.specific_wanted_class}. Samples remaining: {len(the_y_data)}")

        # Make sure the labels are from 0 to num_classes-1
        my_label_encoder = LabelEncoder()
        the_y_data  = my_label_encoder.fit_transform(the_y_data)
        num_classes = len(np.unique(the_y_data))
        # Save dataset and exit if requested
        if args.save_dataset_to_file_and_exit is not None:
            if args.verbose:
                print("Saving data arrays to disk...")
                print(f"  the_x_data shape: {the_x_data.shape}, "
                      f"dtype: {the_x_data.dtype}")
                print(f"  the_y_data shape: {the_y_data.shape}, "
                      f"dtype: {the_y_data.dtype}")
                print(f"  the_d_data shape: {the_d_data.shape}, "
                      f"dtype: {the_d_data.dtype}")

            args_array = np.array(vars(args), dtype=object)
            np.savez_compressed(args.save_dataset_to_file_and_exit,
                x=the_x_data, y=the_y_data, d=the_d_data,label_encoder=my_label_encoder, tuple_to_class=tuple_to_class, class_to_tuple=class_to_tuple,
                args=args_array, num_classes=num_classes, xs=Xs, ys=Ys)

            if args.verbose:
                print(f"Saved to {args.save_dataset_to_file_and_exit}")
            return
    else:
        # Load compiled dataset if provided
        data = np.load(args.compiled_dataset_filename, allow_pickle=True)
        the_x_data = data['x']
        the_y_data = data['y']
        the_d_data = data['d']
        num_classes = data['num_classes']
        Xs = data['xs']
        Ys = data['ys']
        label_encoder, my_label_encoder, tuple_to_class = data['tuple_to_class'], data['class_to_tuple'], data['class_to_tuple']
        if the_d_data.dtype == 'datetime64[ns]':
            the_d_data = pd.to_datetime(the_d_data)

        args_dict = data['args'].item()
        loaded_args = Namespace(**args_dict)

        if args.verbose:
            print("\t🔧 Arguments used to generate the loaded data:")
            for arg, value in vars(loaded_args).items():
                if 'master_data_cache' in str(arg):
                    continue
                print(f"\t    {arg:.<40} {value}")
            print("-" * 80, flush=True)

        unique_classes_new, counts_new = np.unique(
            the_y_data, return_counts=True
        )
        print(f"Unique classes after filtering: "
              f"{unique_classes_new} ({num_classes})")
        print("Count per class:")
        for cls, cnt in zip(unique_classes_new, counts_new):
            if cls is not None:
                print(f"  Class {int(cls)}: {int(cnt)} samples")
    assert len(the_x_data) == len(the_d_data) == len(the_y_data)
    assert num_classes > 1 and np.max(the_y_data) <= num_classes
    # -------------------------------------------------------------------------
    # LOAD MODEL AND PREDICT (Real-time inference)
    # -------------------------------------------------------------------------
    if args.load_model_path is not None and args.real_time_only and args.compiled_dataset_filename is not None:
        if args.verbose:
            print("\n" + "=" * 80)
            print("🔮 LOADING MODEL FOR REAL-TIME PREDICTION")
            print("=" * 80)

        model_data = load_model_for_inference(
            args.load_model_path, verbose=args.verbose
        )

        assert len(the_x_data) == len(the_d_data)

        # Get the latest data point
        latest_X = the_x_data[-1:].copy()
        date_of_data_point = the_d_data[-1]

        if args.verbose:
            print(f"\n📅 Data point used: {date_of_data_point.strftime('%Y-%m-%d')}")
            print(f"📊 Input Features Shape: {latest_X.shape}")

        # Make predictions
        predicted_classes, prediction_probabilities, decoded_tuples = predict_with_saved_model(model_data, latest_X, verbose=args.verbose)

        # Display results
        if args.verbose:
            print("\n" + "=" * 80)
            print("📈 PREDICTION RESULTS")
            print("=" * 80)
            print(f"Date: {date_of_data_point.strftime('%Y-%m-%d')}")
            print(f"Predicted Class: {predicted_classes[0]}")
            if decoded_tuples:
                print(f"Decoded Target (POS_SEQ, NEG_SEQ): {decoded_tuples[0]}")
            print("\nClass Probabilities:")
            for i, prob in enumerate(prediction_probabilities[0]):
                if prob > 0.01:
                    decoded = model_data['class_to_tuple'].get(i, (0, 0))
                    print(f"  Class {i} → {decoded}: {prob:.4f} "
                          f"({prob * 100:.2f}%)")
            print("\n📦 Model Information:")
            print(f"  Training Period: "
                  f"{model_data['training_date_range']['start']} to "
                  f"{model_data['training_date_range']['end']}")
            print(f"  CV Accuracy: {model_data['cv_scores']['accuracy_mean']:.4f} "
                  f"(+/- {model_data['cv_scores']['accuracy_std']:.4f})")
            print(f"  CV F1-Score: {model_data['cv_scores']['f1_mean']:.4f} "
                  f"(+/- {model_data['cv_scores']['f1_std']:.4f})")
            print(f"  Number of Classes: {model_data['num_classes']}")
            print(f"  Models in Ensemble: {list(model_data['models'].keys())}")
            print("=" * 80)

        return predicted_classes[0], prediction_probabilities[0], \
            date_of_data_point, model_data

    # -------------------------------------------------------------------------
    # CLASSIFICATION TRAINING
    # -------------------------------------------------------------------------
    if args.verbose:
        print(f"\nBuilding classification model...")

    # Parse model overrides
    try:
        model_overrides = json.loads(args.model_overrides)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON for --model_overrides.")

    # Define base classification configs
    def get_base_config(model_name, model_class, params_template):
        cfg = {
            "class": model_class,
            "params": params_template.copy(),
            "supports_early_stopping": False,
            "uses_validation_fraction": False
        }
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

    # Apply JSON overrides
    for model_name, override_params in model_overrides.items():
        if model_name in classification_model_configs:
            deep_update(classification_model_configs[model_name], override_params)
            if args.verbose:
                print(f"⚙️ Applied overrides for {model_name}: {override_params}")
        else:
            print(f"⚠️ Warning: Model '{model_name}' in overrides not found in config.")

    # Initialize score lists
    acc_scores, f1_scores = [], []
    fold_precision_scores_per_class = []
    fold_recall_scores_per_class = []
    fold_f1_scores_per_class = []

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(the_x_data)):
        X_train, X_val = copy.deepcopy(the_x_data[train_idx]), copy.deepcopy(the_x_data[val_idx])
        y_train, y_val = copy.deepcopy(the_y_data[train_idx]), copy.deepcopy(the_y_data[val_idx])

        # Flatten and convert labels to integers
        y_train = y_train.ravel().astype(int)
        y_val = y_val.ravel().astype(int)

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        fold_predictions_proba = []
        for model_name in args.base_models:
            cfg = classification_model_configs[model_name]
            model = cfg["class"](**cfg["params"])  # New model

            fit_kwargs_init = {}
            model.fit(X_train_scaled, y_train, **fit_kwargs_init)

            # Store probabilities for ensemble
            pred_proba = model.predict_proba(X_val_scaled)

            # Ensure consistent shape
            if pred_proba.shape[1] != num_classes:
                full_proba = np.zeros((pred_proba.shape[0], num_classes))
                full_proba[:, model.classes_] = pred_proba
                fold_predictions_proba.append(full_proba)
            else:
                fold_predictions_proba.append(pred_proba)

        # Ensemble averaging
        avg_proba = np.mean(fold_predictions_proba, axis=0)
        y_pred_final = np.argmax(avg_proba, axis=1)

        # Calculate metrics
        acc = accuracy_score(y_val, y_pred_final)
        f1 = f1_score(y_val, y_pred_final, average='weighted')

        # Per-class scores
        precision, recall, f1_per_class, support = precision_recall_fscore_support(y_val, y_pred_final, labels=range(num_classes), zero_division=0)

        fold_precision_scores_per_class.append(precision)
        fold_recall_scores_per_class.append(recall)
        fold_f1_scores_per_class.append(f1_per_class)

        if args.verbose:
            print(f"Fold {fold + 1}: Acc={acc:.4f}, F1={f1:.4f}")
            f1_str = ", ".join([f"{i}:{v:.3f}" for i, v in enumerate(f1_per_class)])
            print(f"  Per-Class F1: [{f1_str}]")

        acc_scores.append(acc)
        f1_scores.append(f1)
    if args.verbose:
        print(f"\nAverage Accuracy: {np.mean(acc_scores):.4f}")
        print(f"Average Weighted F1: {np.mean(f1_scores):.4f}")
        print(f"Precision\nTP / (TP + FP)\n\tOf all the instances I predicted as positive, how many were actually positive?")
        print(f"Recall\nTP / (TP + FN)\n\tOf all the actual positive instances, how many did I correctly identify?")
        print("Accuracy\nAccuracy = (True Positives + True Negatives) / Total Predictions\n         = (TP + TN) / (TP + TN + FP + FN)")
        # Average per-class scores
        avg_precision = np.mean(fold_precision_scores_per_class, axis=0)
        avg_recall = np.mean(fold_recall_scores_per_class, axis=0)
        avg_f1 = np.mean(fold_f1_scores_per_class, axis=0)

        print(f"\nAverage Per-Class Validation Scores "
              f"({len(fold_f1_scores_per_class)} folds):")
        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 44)
        real__2__tx = dict(zip(my_label_encoder.classes_, my_label_encoder.transform(my_label_encoder.classes_)))
        tx__2__real = dict(zip(my_label_encoder.transform(my_label_encoder.classes_), my_label_encoder.classes_))
        for c in range(num_classes):
            cc = str(decode_tuple_target(tx__2__real[c], class_to_tuple))
            print(f"{cc:<8} {avg_precision[c]:<12.4f} {avg_recall[c]:<12.4f} {avg_f1[c]:<12.4f}")

    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------
    if args.save_model_path is not None:
        if args.verbose:
            print("\n" + "=" * 80)
            print("💾 Saving model and preprocessing objects...")
            print("=" * 80)
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
            'feature_names': [f"{col[0]}_{col[1]}" if isinstance(col, tuple)
                              else str(col) for col in Xs],
            'num_classes': num_classes,
            'target_type': 'tuple',
            'target_columns': Ys,
            'class_to_tuple': class_to_tuple,
            'tuple_to_class': tuple_to_class,
            'args': vars(args),
            'class_distribution': {int(cls): int(cnt)
                                   for cls, cnt in zip(unique_classes_new,
                                                       counts_new)},
            'training_date_range': {
                'start': the_d_data[0].strftime('%Y-%m-%d') if hasattr(
                    the_d_data[0], 'strftime') else str(the_d_data[0]),
                'end': the_d_data[-1].strftime('%Y-%m-%d') if hasattr(
                    the_d_data[-1], 'strftime') else str(the_d_data[-1])
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
                'convert_price_level_with_baseline':
                    args.convert_price_level_with_baseline,
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
                'macd_params': macd_params if 'macd_params' in locals()
                else args.macd_params,
                'vwap_window': args.vwap_window,
                'add_only_vwap_z_and_vwap_triggers':
                    args.add_only_vwap_z_and_vwap_triggers,
                'epsilon': args.epsilon,
                'look_ahead': args.look_ahead,
            }
        }

        # Save using pickle
        os.makedirs(
            os.path.dirname(args.save_model_path) if
            os.path.dirname(args.save_model_path) else '.',
            exist_ok=True
        )

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
            print(f"   CV Accuracy: {np.mean(acc_scores):.4f} "
                  f"(+/- {np.std(acc_scores):.4f})")
            print(f"   CV F1-Score: {np.mean(f1_scores):.4f} "
                  f"(+/- {np.std(f1_scores):.4f})")
            print("\n📊 Per-Class CV Performance:")
            print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} "
                  f"{'F1-Score':<12}")
            print("-" * 44)
            for c in range(num_classes):
                print(f"{c:<8} {avg_precision[c]:<12.4f} {avg_recall[c]:<12.4f} "
                      f"{avg_f1[c]:<12.4f}")
            print("=" * 80)

        return np.mean(f1_scores), np.mean(acc_scores), avg_precision, \
            avg_recall, avg_f1

    return np.mean(f1_scores)

# =============================================================================
# ARGUMENT PARSER
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="POS/NEG Sequence Classification Model"
    )

    # Data arguments
    parser.add_argument("--ticker", type=str, default='^GSPC',
                        help="Stock ticker symbol")
    parser.add_argument("--older_dataset", type=str, default="None",
                        help="Older dataset filename")
    parser.add_argument("--dataset_id", type=str, default="day",
                        choices=DATASET_AVAILABLE,
                        help="Dataset frequency (default: day)")
    parser.add_argument("--look_ahead", type=int, default=1,
                        help="Look ahead periods (default: 1)")

    # Mode arguments
    parser.add_argument("--real-time-only", action="store_true",
                        help="Skip training; run only real-time prediction")
    parser.add_argument("--real-time-only-num-classes", type=int, default=None,
                        help="Number of classes for real-time mode")
    parser.add_argument('--drop_when_out_of_range', type=str2bool, default=False,
                        help="Drop rows outside low/high range")
    parser.add_argument('--step_back_range', type=int, default=99999,
                        help="Historical windows for rolling backtest")
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help="Enable verbose output")

    # Target arguments
    parser.add_argument("--epsilon", type=float, default=0.,
                        help="Threshold for neutral returns (default: 0)")
    parser.add_argument("--convert_price_level_with_baseline", type=str,
                        default='fraction', choices=['fraction', 'return'],
                        help="Price normalization method")
    parser.add_argument('--add_close_diff', type=str2bool, default=True,
                        help="Add close difference feature")

    # Technical indicator arguments
    parser.add_argument("--ema_windows", type=int, nargs='+', default=[],
                        help="EMA window sizes")
    parser.add_argument('--enable_ema', type=str2bool, default=False,
                        help="Enable EMA features")
    parser.add_argument("--shift_ema_col", type=int, nargs='+', default=[],
                        help="EMA shift periods")

    parser.add_argument("--sma_windows", type=int, nargs='+', default=[],
                        help="SMA window sizes")
    parser.add_argument('--enable_sma', type=str2bool, default=False,
                        help="Enable SMA features")
    parser.add_argument("--shift_sma_col", type=int, nargs='+', default=[],
                        help="SMA shift periods")

    parser.add_argument("--rsi_windows", type=int, nargs='+', default=[],
                        help="RSI window sizes")
    parser.add_argument("--shift_rsi_col", type=int, nargs='+', default=[],
                        help="RSI shift periods")
    parser.add_argument('--enable_rsi', type=str2bool, default=False,
                        help="Enable RSI features")

    parser.add_argument("--macd_params", type=str,
                        default='{"fast": 12, "slow": 26, "signal": 9}',
                        help="MACD parameters (JSON)")
    parser.add_argument('--enable_macd', type=str2bool, default=False,
                        help="Enable MACD features")
    parser.add_argument("--shift_macd_col", type=int, nargs='+', default=[],
                        help="MACD shift periods")

    parser.add_argument('--enable_vwap', type=str2bool, default=False,
                        help="Enable VWAP features")
    parser.add_argument('--vwap_window', type=int, default=20,
                        help="VWAP window size")
    parser.add_argument('--add_only_vwap_z_and_vwap_triggers', type=str2bool,
                        default=True,
                        help="Add only VWAP z-score and triggers")

    parser.add_argument('--enable_day_data', type=str2bool, default=True,
                        help="Add day of week features")

    # Dataset arguments
    parser.add_argument('--compiled_dataset_filename', type=str, default=None,
                        help="Load pre-compiled dataset")
    parser.add_argument('--save_dataset_to_file_and_exit', type=str, default=None,
                        help="Save dataset and exit")
    parser.add_argument('--min_percentage_to_keep_class', type=float, default=-1.,
                        help="Minimum class percentage to keep (-1 to disable)")
    parser.add_argument("--specific_wanted_class", type=int, nargs='+',
                        default=[],
                        help="Specific classes to keep")

    # Model arguments
    parser.add_argument("--base_models", type=str, nargs="+", default=["xgb"],
                        choices=["xgb", "lgb", "cat", "hgb", "rf", "et",
                                 "svm", "knn", "mlp", "lr", "dt"],
                        help="Base models for ensemble")
    parser.add_argument('--save_model_path', type=str, default=None,
                        help="Path to save trained model")
    parser.add_argument('--load_model_path', type=str, default=None,
                        help="Path to load saved model for inference")
    parser.add_argument('--model_overrides', type=str, default='{}',
                        help='JSON overrides for model parameters')
    parser.add_argument("--shift_seq_col", type=int, default=3,
                        help="Number of shifted sequence columns")

    args = parser.parse_args()
    main(args)