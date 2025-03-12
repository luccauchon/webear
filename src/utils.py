import numpy as np
from types import SimpleNamespace
import platform
import os
import yfinance as yf
from hurst import compute_Hc
import pandas as pd
import warnings
from datetime import datetime, timedelta
import numpy as np
from datetime import datetime, timedelta


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


def get_df_SPY_and_VIX(interval="1d"):
    df_vix    = yf.download("^VIX", period="max", interval=interval)
    df_vix    = df_vix.drop("Volume", axis=1)
    df_spy    = yf.download("SPY", period="max", interval="1d")
    merged_df = pd.merge(df_spy, df_vix, on='Date', how='left')

    merged_df['day_of_week']  = merged_df.index.dayofweek + 1
    merged_df['week_of_year'] = merged_df.index.isocalendar().week + 1
    merged_df['unique_week']  = merged_df.index.year * 1000 + merged_df['week_of_year']
    merged_df[('Close_direction', 'SPY')]  = merged_df.apply(lambda row: 1 if row[('Close', 'SPY')] > row[('Open', 'SPY')] else -1, axis=1)
    merged_df[('Close_direction', '^VIX')] = merged_df.apply(lambda row: 1 if row[('Close', '^VIX')] > row[('Open', '^VIX')] else -1, axis=1)

    merged_df = merged_df.sort_index(ascending=True)
    return merged_df, 'spy_vix_multicol_reverse_rc1__direction'


def _get_root_dir():
    # Get the current working directory of the running program
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This file is utils.py, which is in the src/ directory
    return os.path.abspath(os.path.join(current_dir, '..'))


def get_stub_dir():
    return os.path.join(_get_root_dir(), "stubs")


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
        the_X, the_y = df.iloc[i1:i2], df.iloc[i2:i3]
        if len(the_X) != seq_length or len(the_y) != target_length:
            continue
        if ignore_data_before_this_date is not None:
            if the_y.index[-1] < pd.to_datetime(ignore_data_before_this_date):
                continue
        assert target_length == len(the_y)
        assert the_X.iloc[-1].day_of_week.values[0] == 1
        assert 1 == len(list(set([the_y.iloc[uu].week_of_year.values[0] for uu in range(0, target_length)])))
        assert len(list(set([the_X.iloc[uu].week_of_year.values[0] for uu in range(0, len(the_X))]))) in [seq_length%5,seq_length/5+1]
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