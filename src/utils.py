import numpy as np
import re
from types import SimpleNamespace
import platform
from torcheval.metrics import BinaryAccuracy
import os
import yfinance as yf
from hurst import compute_Hc
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import numpy as np
from datetime import datetime, timedelta
import random


os_name = platform.system()
IS_RUNNING_ON_WINDOWS = True
IS_RUNNING_ON_CASIR   = False

# Function to check if two dictionaries are equal, considering NumPy arrays
def dicts_are_equal(d1, d2):
    if d1.keys() != d2.keys():
        return False
    for key in d1:
        if isinstance(d1[key], np.ndarray) and isinstance(d2[key], np.ndarray):
            if not np.array_equal(d1[key], d2[key]):
                return False
        elif d1[key] != d2[key]:
            return False
    return True


# Function to check if all dictionaries are the same
def all_dicts_equal(dicts, reference_dict):
    for d in dicts:
        if not dicts_are_equal(d, reference_dict):
            return False
    return True


def namespace_to_dict(ns):
    return {k: namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v for k, v in ns.__dict__.items()}


def dict_to_namespace(d):
    return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def get_index_from_date(_df, _specific_date, inc=True, max_missing_days=31):
    counted_missing_days = 0
    while counted_missing_days < max_missing_days:
        # Check if the specific date is in the index
        if _specific_date in _df.index:
            _index_number = _df.index.get_loc(_specific_date)
            # print(f"The index number for {_specific_date} is {_index_number}.")
            return _index_number
        else:
            # print(f"{_specific_date} is not in the index.")
            counted_missing_days += 1
            if inc:  # Increment the date by one day
                _specific_date = (datetime.strptime(_specific_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                _specific_date = (datetime.strptime(_specific_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    assert False, f"Cannot find an index for date {_specific_date}"

def get_df_SPY_and_VIX(interval="1d"):
    df_vix    = yf.download("^VIX", period="max", interval=interval, auto_adjust=False)
    df_vix    = df_vix.drop("Volume", axis=1)
    df_spy    = yf.download("SPY", period="max", interval="1d", auto_adjust=False)
    merged_df = pd.merge(df_spy, df_vix, on='Date', how='left')

    merged_df['day_of_week']  = merged_df.index.dayofweek + 1
    merged_df['week_of_year'] = merged_df.index.isocalendar().week + 1
    merged_df['unique_week']  = merged_df.index.year * 1000 + merged_df['week_of_year']
    merged_df[('Close_direction', 'SPY')]  = merged_df.apply(lambda row: 1 if row[('Close', 'SPY')] > row[('Open', 'SPY')] else -1, axis=1)
    merged_df[('Close_direction', '^VIX')] = merged_df.apply(lambda row: 1 if row[('Close', '^VIX')] > row[('Open', '^VIX')] else -1, axis=1)

    for window_size in [5]:  # Define the window size for the moving average
        do_ma_on_those = [('Close', 'SPY'), ('High', 'SPY'), ('Low', 'SPY'), ('Open', 'SPY'), ('Volume', 'SPY'),
                          ('Close', '^VIX'), ('High', '^VIX'), ('Low', '^VIX'), ('Open', '^VIX')]
        new_cols = []
        for col in merged_df.columns:
            if not col in do_ma_on_those:
                continue
            col_title = (col[0] + f'_MA{window_size}', col[1])
            merged_df[col_title] = merged_df[col].rolling(window=window_size, center=True).mean()
            merged_df[col_title] = merged_df[col_title].shift(window_size // 2)
            new_cols.append(col_title)
        #print(new_cols)
    merged_df = merged_df.dropna()

    merged_df = merged_df.sort_index(ascending=True)

    # merged_df[['column1', 'column2']] = merged_df[['column1', 'column2']].astype(int)
    # float_cols = merged_df.select_dtypes(include=['float64']).columns
    # merged_df[float_cols] = merged_df[float_cols].astype(int)

    return merged_df, 'spy_vix_multicol_reverse_rc1__direction'


def _get_root_dir():
    # Get the current working directory of the running program
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This file is utils.py, which is in the src/ directory
    return os.path.abspath(os.path.join(current_dir, '..'))


def get_stub_dir():
    a_dir = os.path.join(_get_root_dir(), "stubs")
    return a_dir


def generate_indices_basic_style(df, dates, x_seq_length, y_seq_length, just_x_no_y=False):
    # Simply takes N days to predict the next P days. Only the "P days" shall be in the date range specified
    indices = []
    assert pd.to_datetime(dates[0]) <= pd.to_datetime(dates[1])
    if just_x_no_y:
        for idx in reversed(range(0, len(df) + 1)):
            idx1, idx2 = idx - x_seq_length, idx
            if idx1 < 0 or idx2 < 0:
                continue
            if len(df.iloc[idx1:idx2]) != x_seq_length:
                continue
            assert idx2 > idx1
            indices.append((idx1, idx2))
            break
    else:
        for idx in reversed(range(0, len(df)+1)):
            idx1, idx2, idx3 = idx-y_seq_length-x_seq_length, idx-y_seq_length, idx
            assert df.iloc[idx2:idx3].index.intersection(df.iloc[idx1:idx2].index).empty
            if idx1 <0 or idx2<0 or idx3<0:
                continue
            # Make sure that y is in the range
            t1 = df.iloc[idx2:idx3].index[0]
            if t1 < pd.to_datetime(dates[0]) or t1 > pd.to_datetime(dates[1]):
                continue
            t1 = df.iloc[idx2:idx3].index[-1]
            if t1 < pd.to_datetime(dates[0]) or t1 > pd.to_datetime(dates[1]):
                continue
            assert pd.to_datetime(dates[0]) <= t1 <= pd.to_datetime(dates[1])
            if len(df.iloc[idx1:idx2]) != x_seq_length:
                continue
            if len(df.iloc[idx2:idx3]) != y_seq_length:
                continue
            assert idx3 > idx2 > idx1
            indices.append((idx1, idx2, idx3))
    return indices, df.copy()


def generate_indices_naked_monday_style(df, seq_length, ignore_data_before_this_date=None):
    # For each monday, predict next tuesday, wednesday and thursday based on the last K data
    target_length                         = 3
    cutoff_day                            = 1  # Monday
    flush_target_week_with_wrong_sequence = False  # Means that y shall be tuesday, wednesday and thursday
    indices                               = []
    assert 0 == seq_length%5
    monday_indexes = [index for index, row in df[df.day_of_week == cutoff_day].iterrows()]
    for idx in monday_indexes:
        idx = df.index.get_loc(idx) + 1
        i1, i2, i3 = idx - seq_length, idx, idx+target_length
        if i1 <0:
            continue
        the_x, the_y = df.iloc[i1:i2], df.iloc[i2:i3]
        if len(the_x) != seq_length or len(the_y) != target_length:
            continue
        if ignore_data_before_this_date is not None:
            if the_y.index[-1] < pd.to_datetime(ignore_data_before_this_date):
                continue
        assert target_length == len(the_y)
        assert the_x.iloc[-1].day_of_week.values[0] == 1
        assert 1 == len(list(set([the_y.iloc[uu].week_of_year.values[0] for uu in range(0, target_length)])))
        assert len(list(set([the_x.iloc[uu].week_of_year.values[0] for uu in range(0, len(the_x))]))) in [seq_length/5,seq_length/5+1,seq_length/5+2]
        dismiss = False
        if flush_target_week_with_wrong_sequence:
            for uu in range(0, target_length):
                if uu +2 != the_y.iloc[uu].day_of_week.values[0]:
                    dismiss = True
                #if 2 != the_y.iloc[0].day_of_week or 3 != the_y.iloc[1].day_of_week or 4 != the_y.iloc[2].day_of_week:
        if dismiss:
            continue
        indices.append((i1,i2,i3))
    return indices, df


def generate_indices_with_multiple_cutoff_day(_df, _dates, x_seq_length, y_seq_length, cutoff_days, split_ratio=-1., shuffle=False):
    _indices = []
    for cutoff_day in cutoff_days:
        train_indices, train_df = generate_indices_with_cutoff_day(cutoff_day=cutoff_day, _df=_df.copy(), _dates=_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
        _indices.extend(train_indices)
    return _indices, _df.copy()


def generate_indices_with_cutoff_day(_df, _dates, x_seq_length, y_seq_length, cutoff_day=1, split_ratio=-1., shuffle=False):
    # For all "cutoff_day" , generate indices a, b, c where b is the cutoff day, a-b] is x_seq and ]b->c is the y_seq
    _indices = []
    _end_idx = -1 if '2099' in _dates[1] else get_index_from_date(_df=_df, _specific_date=_dates[1])
    _df = _df[:_end_idx]
    cutoff_day_indexes = [index for index, row in _df[_df.day_of_week == cutoff_day].iterrows()]
    for idx in cutoff_day_indexes:
        idx = _df.index.get_loc(idx) + 1
        i1, i2, i3 = idx - x_seq_length, idx, idx + y_seq_length
        if i1 < 0:
            continue
        the_X, the_y = _df.iloc[i1:i2], _df.iloc[i2:i3]
        if len(the_X) != x_seq_length or len(the_y) != y_seq_length:
            continue
        if the_y.index[-1] < pd.to_datetime(_dates[0]):
            continue
        the_X, the_y = _df.iloc[i1:i2], _df.iloc[i2:i3]
        assert len(the_X) == x_seq_length and len(the_y) == y_seq_length
        assert cutoff_day in [1,2,3,4,5]
        if cutoff_day in [1, 2, 3, 4]:
            assert the_y.iloc[0].day_of_week.values[0] >= cutoff_day+1  # the first y is the next day or "next next" one if there was a holiday
        else:
            assert the_y.iloc[0].day_of_week.values[0] == 1
        assert the_X.iloc[-1].day_of_week.values[0] == cutoff_day
        assert the_y.iloc[0].day_of_week.values[0] != cutoff_day
        for a_df in [the_X, the_y]:
            for uu in range(0, len(a_df)):
                assert a_df.iloc[uu].day_of_week.values[0] in [1,2,3,4,5]
        _indices.append((i1, i2, i3))
    if shuffle:
        random.shuffle(_indices)
    if -1 != split_ratio:
        return _indices[:int(len(_indices)*split_ratio)], _indices[int(len(_indices)*split_ratio):], _df
    else:
        return _indices, _df


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate metrics for binary classification.

    Args:
    y_true (torch.Tensor): Ground truth labels.
    y_pred (torch.Tensor): Predicted probabilities.

    Returns:
    dict: Dictionary containing accuracy, precision, recall, F1 score, and AUC-ROC.
    """

    metric = BinaryAccuracy(threshold=0.5)
    assert y_pred.squeeze().shape == y_true.squeeze().shape
    metric.update(y_pred.squeeze(), y_true.squeeze())
    accuracy = metric.compute()

    return {'accuracy': accuracy}


def extract_info_from_filename(filename):
    """
    Extracts information from a filename.

    Args:
        filename (str): The filename to extract information from.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    pattern     = r"best__(?P<metric1_name>test_accuracy|test_loss)_(?P<metric1_value>\d+\.\d+)__with__(?P<metric2_name>test_accuracy|test_loss)_(?P<metric2_value>\d+\.\d+)__at_(?P<epoch>\d+)\.pt"
    pattern_alt = r"best__(?P<metric1_name>test_accuracy)_(?P<metric1_value>\d+\.\d+)__with__(?P<metric2_name>loss|test_loss)_(?P<metric2_value>\d+\.\d+)_at_(?P<epoch>\d+)\.pt"
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    else:
        match = re.match(pattern_alt, filename)
        if match:
            return match.groupdict()
        else:
            return None


def previous_weekday(date):
    previous_day = date - pd.Timedelta(1, unit='days')
    while previous_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        previous_day -= pd.Timedelta(1, unit='days')
    return previous_day


def next_weekday(date):
    next_day = date + pd.Timedelta(1, unit='days')
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += pd.Timedelta(1, unit='days')
    return next_day