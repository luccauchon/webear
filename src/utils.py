import numpy as np
from pathlib import Path
import sys
import re
from types import SimpleNamespace
import platform
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import os
import yfinance as yf
from hurst import compute_Hc
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import glob
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


def get_df_SPY_and_VIX_virgin_at_minutes():
    df_vix       = yf.download("^VIX", period="max", interval='1m', auto_adjust=False)
    df_vix       = df_vix.drop("Volume", axis=1)
    df_vix.index = df_vix.index.tz_convert('US/Eastern')

    df_spy       = yf.download("SPY", period="max", interval='1m', auto_adjust=False)
    df_spy.index = df_spy.index.tz_convert('US/Eastern')

    return  df_spy, df_vix


def get_df_SPY_and_VIX_virgin_at_30minutes():
    df_vix       = yf.download("^VIX", period="max", interval='30m', auto_adjust=False)
    df_vix       = df_vix.drop("Volume", axis=1)
    df_vix.index = df_vix.index.tz_convert('US/Eastern')

    df_spy       = yf.download("SPY", period="max", interval='30m', auto_adjust=False)
    df_spy.index = df_spy.index.tz_convert('US/Eastern')

    return  df_spy, df_vix


def get_df_SPY_and_VIX(interval="1d", add_moving_averages=True, _window_sizes=(2,3,4,5)):
    df_vix    = yf.download("^VIX", period="max", interval="1d", auto_adjust=False)
    df_vix    = df_vix.drop("Volume", axis=1)
    df_spy    = yf.download("SPY", period="max", interval="1d", auto_adjust=False)
    merged_df = pd.merge(df_spy, df_vix, on='Date', how='left')
    assert 0 == np.sum(merged_df.isna().sum().values)
    if interval in ['1mo', '1wk']:
        # Define the list of symbols
        symbols = ['SPY', '^VIX']
        # Define the aggregation functions for each column type
        agg_funcs = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',}
        agg_dict = {(col, symbol): agg_funcs[col] for symbol in symbols for col in agg_funcs}
        df_1jil = None
        if interval == '1wk':
            df_1jil = merged_df.resample('W-FRI').agg(agg_dict).copy()
            # Resample the volume data into 5-minute mean volumes
            volume_candles = merged_df['Volume'].resample('W-FRI').mean()
            # Add the volume column to the candles DataFrame
            df_1jil['Volume'] = volume_candles
            if merged_df.index[-1].weekday() < 5:  # 5 represents Saturday
                df_1jil = df_1jil.drop(df_1jil.index[-1])
        if interval == '1mo':
            df_1jil = merged_df.resample('ME').agg(agg_dict).copy()
            # Resample the volume data into monthly mean volumes
            volume_candles = merged_df['Volume'].resample('ME').mean()
            # Add the volume column to the candles DataFrame
            df_1jil['Volume'] = volume_candles
            if merged_df.index[-1].day < 28:
                df_1jil = df_1jil.drop(df_1jil.index[-1])
        assert 0 == np.sum(df_1jil.isna().sum().values)
        merged_df = df_1jil.copy()
    merged_df['day_of_week']  = merged_df.index.dayofweek + 1
    merged_df['week_of_year'] = merged_df.index.isocalendar().week + 1
    merged_df['unique_week']  = merged_df.index.year * 1000 + merged_df['week_of_year']
    merged_df['month_of_year'] = merged_df.index.month
    #merged_df[('Close_direction', 'SPY')]  = merged_df.apply(lambda row: 1 if row[('Close', 'SPY')] > row[('Open', 'SPY')] else -1, axis=1)
    #merged_df[('Close_direction', '^VIX')] = merged_df.apply(lambda row: 1 if row[('Close', '^VIX')] > row[('Open', '^VIX')] else -1, axis=1)

    if add_moving_averages:
        for window_size in _window_sizes:  # Define the window size for the moving average
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

                col_title = (col[0] + f'_EMA{window_size}', col[1])
                merged_df[col_title] = merged_df[col].ewm(span=window_size, adjust=False).mean()
                new_cols.append(col_title)

            #print(new_cols)
        merged_df = merged_df.dropna()

    merged_df = merged_df.sort_index(ascending=True)

    # merged_df[['column1', 'column2']] = merged_df[['column1', 'column2']].astype(int)
    # float_cols = merged_df.select_dtypes(include=['float64']).columns
    # merged_df[float_cols] = merged_df[float_cols].astype(int)

    return merged_df.copy(), f'spy_vix_multicol_reverse_rc1__direction_at_{interval}'


def _get_root_dir():
    # Get the current working directory of the running program
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This file is utils.py, which is in the src/ directory
    return os.path.abspath(os.path.join(current_dir, '..'))


def get_stub_dir():
    a_dir = os.path.join(_get_root_dir(), "stubs")
    return a_dir


def _get_data_dir():
    a_dir = os.path.join(_get_root_dir(), 'data')
    return a_dir



def get_spy_and_vix_data_dir():
    a_dir = os.path.join(_get_data_dir(), "spy_and_vix")
    return a_dir


def get_latest_spy_and_vix_dataframe():
    data_dir = get_spy_and_vix_data_dir()
    files = glob.glob(os.path.join(data_dir, "spy_and_vix__*.pkl"))

    if not files:
        raise FileNotFoundError("No files found in the directory.")

    latest_file = max(files, key=lambda x: datetime.strptime(os.path.basename(x).split("__")[1].split(".")[0], "%Y-%m-%d"))
    return pd.read_pickle(latest_file)


def get_spy_and_vix_df_from_data_dir(date):
    a_file = os.path.join(get_spy_and_vix_data_dir(), f"spy_and_vix__{datetime.strptime(date, '%Y-%m-%d')}.pkl")
    if os.path.exists(a_file):
        return pd.read_pickle(a_file)
    return None


def generate_indices_basic_style(df, dates, x_seq_length, y_seq_length, jump_ahead=0, just_x_no_y=False, data_interval='1d'):
    # Simply takes N days to predict the next P days. Only the "P days" shall be in the date range specified
    indices = []
    tt1 = pd.to_datetime(dates[0]).replace(hour=0, minute=0, second=0)
    tt2 = pd.to_datetime(dates[1]).replace(hour=23, minute=59, second=59)
    assert tt1 <= tt2
    #ts1 = tt1 - pd.Timedelta(2 * (x_seq_length + y_seq_length) + jump_ahead, unit='days')
    #ts2 = tt2 + pd.Timedelta(2 * (x_seq_length + y_seq_length) + jump_ahead, unit='days')
    #df = df.loc[ts1:ts2]  # Reduce length of dataframe to make the processing faster
    if just_x_no_y:
        assert tt1.date() == tt2.date()
        for idx in reversed(range(0, len(df) + 1)):
            idx1, idx2 = idx - x_seq_length, idx
            if idx1 < 0 or idx2 < 0:
                continue
            if len(df.iloc[idx1:idx2]) != x_seq_length:
                continue
            assert jump_ahead==0
            if data_interval=='1d':
                # We want to predict tomorrow, so we need data today
                if next_weekday(df.iloc[idx1:idx2].index[-1].date()) != tt1.date():
                    continue
            if data_interval == '1wk':
                # We want to predict this week, so we need data from last weeks
                # So, if the predicting date is not in the week followwing idx1:idx2, pass.
                if not is_it_next_week_after_last_week_of_df(df=df.iloc[idx1:idx2], date=tt1):
                    continue
            assert idx2 > idx1
            indices.append((idx1, idx2))
            break
    else:
        for idx in reversed(range(0, len(df)+1)):
            idx1, idx2 = idx-x_seq_length-1, idx-1
            idx3, idx4 = idx+jump_ahead-1, idx+jump_ahead+y_seq_length-1
            if idx1 <0 or idx4 > len(df):
                continue
            assert df.iloc[idx1:idx2].index.intersection(df.iloc[idx3:idx4].index).empty
            # Make sure that y is in the range
            t1_y, t2_y = df.iloc[idx3:idx4].index[0], df.iloc[idx3:idx4].index[-1]
            if t1_y < tt1 or t1_y > tt2:
                continue
            if t2_y < tt1 or t2_y > tt2:
                continue
            for tk_y in list(df.iloc[idx3:idx4].index):
                assert tt1 <= tk_y <= tt2
            if len(df.iloc[idx1:idx2]) != x_seq_length:
                continue
            if len(df.iloc[idx3:idx4]) != y_seq_length:
                continue
            assert idx4 > idx3 >= idx2 > idx1
            indices.append((idx1, idx2, idx3, idx4))
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


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate metrics for regression.

    Args:
    y_true (torch.Tensor): Ground truth values.
    y_pred (torch.Tensor): Predicted values.

    Returns:
    dict: Dictionary containing Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).
    """

    assert y_pred.squeeze().shape == y_true.squeeze().shape

    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()
    r2_metric = R2Score()

    if 1 == len(y_pred):
        mse_metric.update(y_pred[0], y_true[0])
        mae_metric.update(y_pred[0], y_true[0])
        r2_metric.update(y_pred[0], y_true[0])
    else:
        mse_metric.update(y_pred.squeeze(), y_true.squeeze())
        mae_metric.update(y_pred.squeeze(), y_true.squeeze())
        r2_metric.update(y_pred.squeeze(), y_true.squeeze())

    mse = mse_metric.compute()
    mae = mae_metric.compute()
    r2 = r2_metric.compute()

    return {'MSE': mse, 'MAE': mae, 'R2': r2}


def calculate_binary_classification_metrics(y_true, y_pred):
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
    if 1 == len(y_pred):
        metric.update(y_pred[0], y_true[0])
    else:
        metric.update(y_pred.squeeze(), y_true.squeeze())
    accuracy = metric.compute()

    return {'accuracy': accuracy}


def calculate_multiclass_classification_metrics(y_true, y_pred, num_classes):
    """


    Args:
    y_true (torch.Tensor): Ground truth labels.
    y_pred (torch.Tensor): Predicted probabilities.

    Returns:
    dict: Dictionary containing accuracy, precision, recall, F1 score, and AUC-ROC.
    """

    metric = MulticlassAccuracy()
    assert y_pred.shape[1] == num_classes and y_true.shape[0] == y_pred.shape[0]
    metric.update(y_pred, y_true)
    accuracy = metric.compute()

    return {'accuracy': accuracy}


def get_all_checkpoints(_a_directory):
    return [file for file in Path(_a_directory).rglob('*.pt') if 'checkpoints' in str(file)]


def extract_info_from_filename(filename):
    """
    Extracts information from a filename.

    Args:
        filename (str): The filename to extract information from.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    pattern     = r"best__(?P<metric1_name>val_accuracy|val_loss)_(?P<metric1_value>\d+\.\d+)__with__(?P<metric2_name>val_accuracy|val_loss)_(?P<metric2_value>\d+\.\d+)__at_(?P<epoch>\d+)\.pt"
    pattern_alt = r"best__(?P<metric1_name>val_accuracy)_(?P<metric1_value>\d+\.\d+)__with__(?P<metric2_name>loss|val_loss)_(?P<metric2_value>\d+\.\d+)_at_(?P<epoch>\d+)\.pt"
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    else:
        match = re.match(pattern_alt, filename)
        if match:
            return match.groupdict()
        else:
            return None


def previous_weekday_with_check(date, df):
    yesterday = previous_weekday(date)
    while yesterday not in df.index:
        yesterday = previous_weekday(yesterday)
    return yesterday


def previous_weekday(date):
    previous_day = date - pd.Timedelta(1, unit='days')
    while previous_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        previous_day -= pd.Timedelta(1, unit='days')
    return previous_day


def next_weekday_with_check(date, df, max_look_ahead=999):
    next_day = next_weekday(date)
    count = 0
    while next_day not in df.index:
        next_day = next_weekday(next_day)
        count += 1
        if count == max_look_ahead:
            return None
    return next_day


def next_weekday(date):
    next_day = date + pd.Timedelta(1, unit='days')
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += pd.Timedelta(1, unit='days')
    return next_day


def is_weekday(date):
    """
    Returns True if date is a weekday, False otherwise.
    """
    return date.weekday() < 5  # 5 represents Saturday, 6 represents Sunday


def find_next_sunday(date: datetime) -> datetime:
    """
    Finds the next Sunday given a datetime object.

    Args:
    date (datetime): The date from which to find the next Sunday.

    Returns:
    datetime: The next Sunday.
    """
    # Calculate the difference between the given date's weekday and Sunday (6)
    days_to_sunday = 6 - date.weekday()

    # If the given date is already Sunday, move to the next week
    if days_to_sunday == 0:
        days_to_sunday = 7

    # Add the calculated days to the given date
    next_sunday = date + timedelta(days=days_to_sunday)
    assert 6 == next_sunday.weekday()
    return next_sunday


def find_next_saturday(date: datetime, move_to_next_week=True) -> datetime:
    """
    Finds the next Saturday given a datetime object.

    Args:
    date (datetime): The date from which to find the next Saturday.

    Returns:
    datetime: The next Saturday.
    """
    # Calculate the difference between the given date's weekday and Saturday (5)
    days_to_saturday = 5 - date.weekday()
    if 0 == days_to_saturday and not move_to_next_week:
        return date
    # If the given date is already Saturday, move to the next week
    if days_to_saturday == 0:
        days_to_saturday = 7

    # Add the calculated days to the given date
    next_saturday = date + timedelta(days=days_to_saturday)
    assert 5 == next_saturday.weekday()
    return next_saturday


def find_previous_saturday(date: datetime) -> datetime:
    """
    Finds the previous Saturday given a datetime object.

    Args:
    date (datetime): The date from which to find the previous Saturday.

    Returns:
    datetime: The previous Saturday.
    """
    # Calculate the difference between the given date's weekday and Saturday (5)
    days_to_saturday = date.weekday() - 5

    # If the given date is already Saturday, move to the previous week
    if days_to_saturday <= 0:
        days_to_saturday = -1 * (7 + days_to_saturday)
    else:
        days_to_saturday = -1 * days_to_saturday

    # Subtract the calculated days from the given date
    previous_saturday = date + timedelta(days=days_to_saturday)
    assert 5 == previous_saturday.weekday()
    return previous_saturday


def string_to_bool(s):
    """
    Converts a string representation of a boolean to a boolean type.

    Args:
        s (str): The string representation of a boolean.

    Returns:
        bool: The boolean equivalent of the input string.

    Raises:
        ValueError: If the input string is not a valid boolean representation.
    """
    if isinstance(s, bool):
        return s
    s = s.lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise ValueError("Invalid boolean representation")


def is_it_next_week_after_last_week_of_df(df, date):
    last_date = df.index[-1]
    assert 4 == last_date.weekday()  # always a friday
    day_next_week = last_date + pd.Timedelta(days=7)
    assert 4 == day_next_week.weekday()  # always a friday
    return (last_date <= date <= day_next_week) and date.weekday() < 5
