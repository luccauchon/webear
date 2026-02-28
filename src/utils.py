import numpy as np
from pathlib import Path
from constants import FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_QUARTER, FYAHOO__OUTPUTFILENAME_YEAR
import sys
import re
from types import SimpleNamespace
import platform
import os
import yfinance as yf
#from hurst import compute_Hc
import pandas as pd
import warnings
import glob
import numpy as np
from datetime import datetime, timedelta, date
import random
from pathlib import Path
import copy


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

    df_spx = yf.download("^GSPC", period="max", interval='1m', auto_adjust=False)
    df_spx.index = df_spx.index.tz_convert('US/Eastern')

    return  df_spy, df_spx, df_vix


def get_df_SPY_and_VIX_virgin_at_30minutes():
    # Calculate start date (30 days ago)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    df_vix       = yf.download("^VIX", start=start_date, end=end_date, interval='30m', auto_adjust=False)
    df_vix       = df_vix.drop("Volume", axis=1)
    df_vix.index = df_vix.index.tz_convert('US/Eastern')

    df_spy       = yf.download("SPY", start=start_date, end=end_date, interval='30m', auto_adjust=False)
    df_spy.index = df_spy.index.tz_convert('US/Eastern')

    df_spx       = yf.download("^GSPC", start=start_date, end=end_date, interval='30m', auto_adjust=False)
    df_spx.index = df_spx.index.tz_convert('US/Eastern')

    return  df_spy, df_spx, df_vix


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
            df_1jil = copy.deepcopy(merged_df.resample('W-FRI').agg(agg_dict))
            # Resample the volume data into 5-minute mean volumes
            volume_candles = merged_df['Volume'].resample('W-FRI').mean()
            # Add the volume column to the candles DataFrame
            df_1jil['Volume'] = volume_candles
            if merged_df.index[-1].weekday() < 5:  # 5 represents Saturday
                df_1jil = df_1jil.drop(df_1jil.index[-1])
        if interval == '1mo':
            df_1jil = copy.deepcopy(merged_df.resample('ME').agg(agg_dict))
            # Resample the volume data into monthly mean volumes
            volume_candles = merged_df['Volume'].resample('ME').mean()
            # Add the volume column to the candles DataFrame
            df_1jil['Volume'] = volume_candles
            if merged_df.index[-1].day < 28:
                df_1jil = df_1jil.drop(df_1jil.index[-1])
        assert 0 == np.sum(df_1jil.isna().sum().values)
        merged_df = copy.deepcopy(df_1jil)
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

    return copy.deepcopy(merged_df), f'spy_vix_multicol_reverse_rc1__direction_at_{interval}'


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
    return indices, copy.deepcopy(df)


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
        train_indices, train_df = generate_indices_with_cutoff_day(cutoff_day=cutoff_day, _df=copy.deepcopy(_df), _dates=_dates, x_seq_length=x_seq_length, y_seq_length=y_seq_length)
        _indices.extend(train_indices)
    return _indices, copy.deepcopy(_df)


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
    from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()
    r2_metric = R2Score()

    if 1 == len(y_pred):
        mse_metric.update(y_pred[0], y_true[0])
        mae_metric.update(y_pred[0], y_true[0])
        #r2_metric.update(y_pred[0], y_true[0])
    else:
        mse_metric.update(y_pred.squeeze(), y_true.squeeze())
        mae_metric.update(y_pred.squeeze(), y_true.squeeze())
        #r2_metric.update(y_pred.squeeze(), y_true.squeeze())

    mse = mse_metric.compute()
    mae = mae_metric.compute()
    r2 = 0. #r2_metric.compute()

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
    from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
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
    from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
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


def next_weekday(input_date, nn=1):
    # Also:
    # import pandas as pd
    # from pandas.tseries.offsets import BDay
    #
    # def next_weekday(input_date, nn=1):
    #     # BDay(nn) adds 'nn' business days (Mon-Fri)
    #     return input_date + BDay(nn)

    _next_day = input_date
    count = 0
    while count < nn:
        _next_day += pd.Timedelta(days=1)
        # Only increment count if Mon(0) to Fri(4)
        if _next_day.weekday() < 5:
            count += 1
    return _next_day


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


def calculate_bollinger_bands(df, window=20, num_std=2, col_name='Close_SPY'):
    df[f'{col_name}__SMA'] = df[col_name].rolling(window).mean()
    df[f'{col_name}__std'] = df[col_name].rolling(window).std()
    df[f'{col_name}__Upper_BB'] = df[f'{col_name}__SMA'] + (df[f'{col_name}__std'] * num_std)
    df[f'{col_name}__Lower_BB'] = df[f'{col_name}__SMA'] - (df[f'{col_name}__std'] * num_std)
    df.dropna(inplace=True)
    return df


def calculate_rsi(df, window=14, col_name='Close_SPY'):
    delta = df[col_name].diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean().abs()
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    df[f'{col_name}__RSI'] = RSI
    df.dropna(inplace=True)
    return df


def calculate_macd(df, slow=26, fast=12, signal=9, col_name='Close_SPY'):
    ema_slow = df[col_name].ewm(span=slow, adjust=False).mean()
    ema_fast = df[col_name].ewm(span=fast, adjust=False).mean()
    df[f'{col_name}__MACD'] = ema_fast - ema_slow
    df[f'{col_name}__Signal'] = df[f'{col_name}__MACD'].ewm(span=signal, adjust=False).mean()
    return df


def format_execution_time(execution_time):
    """Format a duration in seconds into a zero-padded HHhMMmSSs string.

    Always includes hours, minutes, and seconds, padded with leading zeros
    to ensure two digits for each unit.

    Examples:
        format_execution_time(3663)  -> '01h01m03s'
        format_execution_time(61)    -> '00h01m01s'
        format_execution_time(5)     -> '00h00m05s'
        format_execution_time(0)     -> '00h00m00s'

    Args:
        execution_time (float or int): A non-negative duration in seconds.

    Returns:
        str: A zero-padded string in the format 'HHhMMmSSs'.
    """
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    mseconds = int((execution_time % 1) * 1000)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{mseconds:03d}ms"


def get_weekdays(today=None, number_of_days=3):
    if today is None:
        today = date.today()

    weekdays = []
    current = today
    while len(weekdays) < number_of_days:
        if current.weekday() < 5:  # Mon–Fri
            weekdays.append(current)
        current -= timedelta(days=1)
    return tuple(weekdays)


def transform_path(line, date_str):
    """
    Transforms a path assignment line by inserting a date folder before the filename.

    Args:
        line (str): Original line like 'FYAHOO__OUTPUTFILENAME_WEEK = r"D:...snapshot_week.pkl"'
        date_str (str or None): Date string in format YYYY.MM.DD (e.g., "2025.10.31")

    Returns:
        str: Transformed line with date folder inserted
    """
    # Pattern to match the assignment and capture the path parts
    return os.path.join(Path(line).parent, date_str, Path(line).name)


def format_duration(seconds):
    """Convert a duration in seconds to a compact human-readable string.

    Formats the input duration as a concatenation of hours, minutes, and seconds,
    omitting any units with zero values, except when the total duration is less
    than a minute—in that case, seconds are always shown (including '0s' for zero).

    Examples:
        format_duration(3663)  -> '1h1m3s'
        format_duration(61)    -> '1m1s'
        format_duration(5)     -> '5s'
        format_duration(0)     -> '0s'

    Args:
        seconds (float or int): A non-negative duration in seconds. Will be
            converted to an integer (truncated toward zero).

    Returns:
        str: A compact time string using 'h', 'm', and 's' suffixes.
    """
    seconds = int(seconds)  # Ensure it's an integer
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:  # Always show seconds if duration is <1 min, or if it's 0s
        parts.append(f"{secs}s")

    return ''.join(parts)


def is_monday(dt=None):
    """Return True if the given date is a Monday. If no date is given, use today."""
    if dt is None:
        dt = date.today()
    elif isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
    return dt.weekday() == 0  # Monday is 0 in Python


def is_tuesday(dt=None):
    """Return True if the given date is a Tuesday. If no date is given, use today."""
    if dt is None:
        dt = date.today()
    elif isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
    return dt.weekday() == 1  # Tuesday is 1


def is_wednesday(dt=None):
    """Return True if the given date is a Wednesday. If no date is given, use today."""
    if dt is None:
        dt = date.today()
    elif isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
    return dt.weekday() == 2  # Wednesday is 2


def is_thursday(dt=None):
    """Return True if the given date is a Thursday. If no date is given, use today."""
    if dt is None:
        dt = date.today()
    elif isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
    return dt.weekday() == 3  # Thursday is 3


def is_friday(dt=None):
    """Return True if the given date is a Friday. If no date is given, use today."""
    if dt is None:
        dt = date.today()
    elif isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
    return dt.weekday() == 4  # Friday is 4


def get_weekday_name(dt):
    if is_friday(dt):
        return "friday"
    if is_thursday(dt):
        return "thursday"
    if is_wednesday(dt):
        return "wednesday"
    if is_tuesday(dt):
        return "tuesday"
    return "monday"


def str2bool(v):
    return string_to_bool(v)


DATASET_AVAILABLE = ['day', 'week', 'month', 'quarter', 'year']
def get_filename_for_dataset(dataset_choice, older_dataset=None):
    mapping = {
        'day': FYAHOO__OUTPUTFILENAME_DAY,
        'week': FYAHOO__OUTPUTFILENAME_WEEK,
        'month': FYAHOO__OUTPUTFILENAME_MONTH,
        'quarter': FYAHOO__OUTPUTFILENAME_QUARTER,
        'year': FYAHOO__OUTPUTFILENAME_YEAR,
    }
    if older_dataset is not None and older_dataset != '':
        return transform_path(mapping[dataset_choice], older_dataset)
    return mapping[dataset_choice]


from datetime import datetime, date, timedelta
import calendar


def next_day(input_date, skip_weekends=True):
    """
    Get the next calendar day (or next weekday if skip_weekends=True).

    Args:
        input_date (str | date | datetime): Input date in "%Y-%m-%d" format or date/datetime object
        skip_weekends (bool): If True, skip Saturday/Sunday and return next Monday

    Returns:
        date: Next day as a date object
    """
    # Normalize input to date object
    if isinstance(input_date, str):
        dt = datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime):
        dt = input_date.date()
    else:  # date object
        dt = input_date

    next_dt = dt + timedelta(days=1)

    if skip_weekends:
        while next_dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_dt += timedelta(days=1)

    return next_dt


def next_week(input_date, weekday=None):
    """
    Get date for next week (same weekday by default, or specific weekday).

    Args:
        input_date (str | date | datetime): Input date
        weekday (int | None): Target weekday (0=Monday, 6=Sunday). If None, use same weekday as input.

    Returns:
        date: Date in next week
    """
    # Normalize input
    if isinstance(input_date, str):
        dt = datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime):
        dt = input_date.date()
    else:
        dt = input_date

    if weekday is None:
        weekday = dt.weekday()

    # Get next occurrence of target weekday (at least 1 day ahead)
    days_ahead = weekday - dt.weekday()
    if days_ahead <= 0:  # Target day already passed this week
        days_ahead += 7

    return dt + timedelta(days=days_ahead)


def next_month(input_date, day=None, preserve_month_end=True):
    """
    Get date in next month with smart day handling.

    Args:
        input_date (str | date | datetime): Input date
        day (int | None): Specific day of month (1-31). If None, use same day as input.
        preserve_month_end (bool): If True and input is last day of month, return last day of next month.

    Returns:
        date: Date in next month
    """
    # Normalize input
    if isinstance(input_date, str):
        dt = datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime):
        dt = input_date.date()
    else:
        dt = input_date

    # Determine target day
    if day is None:
        day = dt.day

        # Special handling for month-end dates
        if preserve_month_end:
            _, last_day = calendar.monthrange(dt.year, dt.month)
            if dt.day == last_day:
                # Input is last day of month → return last day of next month
                next_month = dt.month % 12 + 1
                next_year = dt.year + (dt.month // 12)
                _, last_day_next = calendar.monthrange(next_year, next_month)
                day = last_day_next

    # Calculate next month/year
    next_month = dt.month % 12 + 1
    next_year = dt.year + (dt.month // 12)

    # Handle invalid days (e.g., Jan 31 → Feb 31 doesn't exist)
    try:
        return date(next_year, next_month, day)
    except ValueError:
        # Fall back to last day of next month
        _, last_day = calendar.monthrange(next_year, next_month)
        return date(next_year, next_month, last_day)


from datetime import date, datetime, timedelta

from datetime import date, datetime, timedelta

def get_next_day_range(input_date=None):
    """
    Returns a formatted string showing the next week-calendar day and the date object of that day.

    Args:
        input_date (str | date | datetime | None):
            - If None: uses today's date
            - If str: expects "YYYY-MM-DD" format
            - If datetime/date: uses the given date

    Returns:
        tuple: (str, date)
            - str: "Next day is YYYY-MM-DD"
            - date: date object representing the next calendar day

    Examples:
        >>> get_next_day_range("2025-02-05")  # Wednesday
        ('Next day is 2025-02-06', datetime.date(2025, 2, 6))

        >>> get_next_day_range("2025-02-28")  # Non-leap year end of February
        ('Next day is 2025-03-01', datetime.date(2025, 3, 1))

        >>> get_next_day_range("2024-12-31")
        ('Next day is 2025-01-01', datetime.date(2025, 1, 1))

        >>> get_next_day_range(datetime(2025, 2, 16, 14, 30))  # Sunday with time
        ('Next day is 2025-02-17', datetime.date(2025, 2, 17))
    """
    # Normalize input to date object
    if input_date is None:
        dt = date.today()
    elif isinstance(input_date, str):
        dt = datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime):
        dt = input_date.date()
    else:  # date object
        dt = input_date

    next_day = next_weekday(dt)
    return f"Returned day is {next_day.strftime('%Y-%m-%d')}", next_day


def get_next_week_range(input_date=None):
    """
    Returns a formatted string showing the next Monday-to-Friday week range.

    Args:
        input_date (str | date | datetime | None):
            - If None: uses today's date
            - If str: expects "YYYY-MM-DD" format
            - If datetime/date: uses the given date

    Returns:
        str: "Next week is from YYYY-MM-DD to YYYY-MM-DD"

    Examples:
        >>> get_next_week_range("2025-02-05")  # Wednesday
        'Next week is from 2025-02-10 to 2025-02-14'

        >>> get_next_week_range("2025-02-14")  # Friday
        'Next week is from 2025-02-17 to 2025-02-21'

        >>> get_next_week_range("2025-02-16")  # Sunday
        'Next week is from 2025-02-17 to 2025-02-21'
    """
    # Normalize input to date object
    if input_date is None:
        dt = date.today()
    elif isinstance(input_date, str):
        dt = datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime):
        dt = input_date.date()
    else:  # date object
        dt = input_date

    # Find Monday of current week (Monday = weekday 0)
    monday_this_week = dt - timedelta(days=dt.weekday())

    # Next week's Monday is 7 days after current week's Monday
    monday_next_week = monday_this_week + timedelta(days=7)
    friday_next_week = monday_next_week + timedelta(days=4)  # Monday + 4 days = Friday

    return (f"Returned week is from {monday_next_week.strftime('%Y-%m-%d')} "
            f"to {friday_next_week.strftime('%Y-%m-%d')}"), friday_next_week


from datetime import date, datetime, timedelta
import calendar


def get_next_month_range(input_date=None):
    """
    Returns a formatted string showing the next full calendar month range (1st to last day).

    Args:
        input_date (str | date | datetime | None):
            - If None: uses today's date
            - If str: expects "YYYY-MM-DD" format
            - If datetime/date: uses the given date

    Returns:
        tuple: (formatted_string, last_day_date)
            - formatted_string (str): "Next month is from YYYY-MM-DD to YYYY-MM-DD"
            - last_day_date (date): date object representing the last day of next month

    Examples:
        >>> get_next_month_range("2025-01-15")
        ('Next month is from 2025-02-01 to 2025-02-28', datetime.date(2025, 2, 28))

        >>> get_next_month_range("2024-01-15")  # Leap year February
        ('Next month is from 2024-02-01 to 2024-02-29', datetime.date(2024, 2, 29))

        >>> get_next_month_range("2025-12-20")
        ('Next month is from 2026-01-01 to 2026-01-31', datetime.date(2026, 1, 31))

        >>> get_next_month_range("2025-02-28")  # End of short month
        ('Next month is from 2025-03-01 to 2025-03-31', datetime.date(2025, 3, 31))
    """
    # Normalize input to date object
    if input_date is None:
        dt = date.today()
    elif isinstance(input_date, str):
        dt = datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime):
        dt = input_date.date()
    else:  # date object
        dt = input_date

    # Calculate next month (handling December → January rollover)
    if dt.month == 12:
        next_month = 1
        next_year = dt.year + 1
    else:
        next_month = dt.month + 1
        next_year = dt.year

    # First day is always the 1st
    first_day = date(next_year, next_month, 1)

    # Get last day of next month using calendar.monthrange
    _, days_in_month = calendar.monthrange(next_year, next_month)
    last_day = date(next_year, next_month, days_in_month)

    formatted_string = (f"Returned month is from {first_day.strftime('%Y-%m-%d')} "
                        f"to {last_day.strftime('%Y-%m-%d')}")

    return formatted_string, last_day


def add_vwap_with_bands(
        df: pd.DataFrame,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
        window,
        ticker,
        bands=(1, 2, 3),
        add_scretch_condition=(True, True, True),
        prefix: str = "VWAP",
        use_hlc3=False,
        add_z_score_feature=False,
):
    """
    Adds VWAP and multi-band levels to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must contain High, Low, Close, Volume columns)
    open_col ,high_col, low_col, close_col, volume_col :
        Column names (supports MultiIndex columns like ('High', 'SPY'))
    window : int or None
        None  -> cumulative VWAP
        int   -> rolling VWAP window
    bands : tuple
        Standard deviation multipliers (e.g. (1,2,3))
    prefix : str
        Prefix name for generated columns

    Returns
    -------
    pd.DataFrame
        DataFrame with VWAP and bands added
    """

    df = df.copy()

    # --- Typical Price ---
    if use_hlc3:
        tp = (df[high_col] + df[low_col] + df[close_col]) / 3.0
    else:  # ohlc/4
        tp = (df[open_col] + df[high_col] + df[low_col] + df[close_col]) / 4.0

    if window is None:
        # ===== CUMULATIVE VWAP =====
        cum_vol = df[volume_col].cumsum()
        cum_tpv = (tp * df[volume_col]).cumsum()
        cum_tp2v = ((tp ** 2) * df[volume_col]).cumsum()

        vwap = cum_tpv / cum_vol
        var = (cum_tp2v / cum_vol) - (vwap ** 2)

    else:
        # ===== ROLLING VWAP =====
        vol_sum = df[volume_col].rolling(window).sum()
        tpv_sum = (tp * df[volume_col]).rolling(window).sum()
        tp2v_sum = ((tp ** 2) * df[volume_col]).rolling(window).sum()

        vwap = tpv_sum / vol_sum
        var = (tp2v_sum / vol_sum) - (vwap ** 2)

    std = np.sqrt(var.clip(lower=0))  # avoid negative floating noise
    col_bands_lst, col_features_lst = [], []
    col_dict = {'vwap':     (f"{prefix}", ticker),
                'vwap_std': (f"{prefix}_std", ticker),
                }
    if add_z_score_feature:
        col_dict.update({'vwap_z': (f"{prefix}_z", ticker),})
    for b in bands:
        col_dict.update({f'vwap_uband_{b}'      : (f"{prefix}_upper_{b}", ticker),
                         f'vwap_lband_{b}'      : (f"{prefix}_lower_{b}", ticker),
                         f'vwap_above_sigma_{b}': (f"{prefix}_above_{b}sigma", ticker),
                         f'vwap_below_sigma_{b}': (f"{prefix}_below_{b}sigma", ticker),
                         })

    # --- Store Base VWAP ---
    df[col_dict['vwap']] = vwap
    df[col_dict['vwap_std']] = std

    # --- Multi Bands ---
    for b in bands:
        df[col_dict[f'vwap_uband_{b}']] = vwap + b * std
        df[col_dict[f'vwap_lband_{b}']] = vwap - b * std
        col_bands_lst.append(col_dict[f'vwap_uband_{b}'])
        col_bands_lst.append(col_dict[f'vwap_lband_{b}'])

    if add_z_score_feature:
        df[col_dict[f'vwap_z']] = (df[close_col] - df[col_dict['vwap']]) / df[col_dict['vwap_std']]
        col_features_lst.append(col_dict['vwap_z'])

    if add_scretch_condition:
        assert len(add_scretch_condition) == len(bands)
        for b in bands:
            df[col_dict[f'vwap_above_sigma_{b}']] = df[close_col] > df[col_dict[f'vwap_uband_{b}']]
            df[col_dict[f'vwap_below_sigma_{b}']] = df[close_col] < df[col_dict[f'vwap_lband_{b}']]

    return df, col_dict


def get_growth_function(y_min, y_max):
    """
    Returns a function that maps:
    10  -> y_min
    20  -> 1.0
    100 -> y_max
    """
    x_min, x_mid, x_max = 10, 20, 100
    y_mid = 1.0

    # Solve for curvature 'k' based on the ratio of growth
    # (y_max - y_min) / (y_mid - y_min) = ((x_max - x_min) / (x_mid - x_min))^k
    ratio_y = (y_max - y_min) / (y_mid - y_min)
    ratio_x = (x_max - x_min) / (x_mid - x_min)  # This is 90 / 10 = 9

    k = np.log(ratio_y) / np.log(ratio_x)
    C = (y_mid - y_min) / ((x_mid - x_min) ** k)

    def growth_func(x):
        # Clip x to the bounds [10, 100] to avoid errors or unexpected values
        x = np.clip(x, x_min, x_max)
        return C * (x - x_min) ** k + y_min

    return growth_func


def abrupt_growth(x, y_min=0.1, y_max=10.0, sharpness=5):
    """
    x: input (10 to 100)
    y_min: value at x=10
    y_max: value at x=100
    sharpness: higher values = flatter start, more abrupt finish
    """
    x_min, x_mid, x_max = 10, 20, 100
    y_mid = 1.0

    # Normalize x to a 0-1 scale for the math
    # 0 at x=10, ~0.11 at x=20, 1 at x=100
    norm_x = (x - x_min) / (x_max - x_min)

    # Apply exponential growth: (e^(s*x) - 1) / (e^s - 1)
    # This creates a curve that starts at 0 and ends at 1
    curve = (np.exp(sharpness * norm_x) - 1) / (np.exp(sharpness) - 1)

    # Rescale the curve to fit y_min and y_max
    # Note: This might not hit 1.0 at exactly 20 unless we solve for
    # the specific 'sharpness' that passes through (20, 1).
    return y_min + (y_max - y_min) * curve
