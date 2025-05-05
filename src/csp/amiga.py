import time
from utils import string_to_bool, namespace_to_dict, dict_to_namespace, get_stub_dir, get_df_SPY_and_VIX, get_all_checkpoints, calculate_binary_classification_metrics, extract_info_from_filename, generate_indices_basic_style, previous_weekday, next_weekday
import sys
from loguru import logger
import talib
from multiprocessing import Lock, Process, Queue, Value, freeze_support
import numpy as np
import yfinance as yf
import pandas as pd
import pandas as pd
import glob
import os
import plotly.graph_objects as go
from datetime import date, timedelta
from talib import abstract
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from fetchers.download_minutes import dump_dataframes_to_disk


def plot2(df2, patterns, fct_name):
    # PLotting of Data
    title = f'{fct_name} : SPY between {df2.index[0].strftime("%Y-%m-%d")} to {df2.index[-1].strftime("%Y-%m-%d")}'

    fig = go.Figure()

    # data preparation for plotting
    bull = np.where(patterns > 0, df2['Close_SPY'].values, np.nan)
    bear = np.where(patterns < 0, df2['Close_SPY'].values, np.nan)

    # plotting
    fig.add_trace(go.Candlestick(x=df2.index, open=df2['Open_SPY'], high=df2['High_SPY'], low=df2['Low_SPY'], close=df2['Close_SPY']))
    fig.add_trace(go.Scatter(x=df2.index, y=bull, mode='markers', name='bull', marker_symbol='triangle-up', marker_color='yellow', marker_size=15))
    fig.add_trace(go.Scatter(x=df2.index, y=bear, mode='markers', name='bear', marker_symbol='triangle-down', marker_color='lightskyblue', marker_size=15))

    fig.update_layout(template='plotly_dark', autosize=False, width=1200, height=600)
    fig.update_layout(title=title, xaxis_title='date', yaxis_title='prices')
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()


def run(configuration):
    ###########################################################################
    #
    ###########################################################################
    # Specify the directory containing pickle files
    directory   = configuration.get('csp__daily_directory', r'D:\Finance\data\daily')
    output_file = configuration.get('csp__daily_output_file', r'D:\Finance\data\daily_fusionned.pkl')
    # Initialize an empty list to store DataFrames
    dfs = []
    uu, cf = 0, 0
    merged_df = None
    # Dump today data first
    dump_dataframes_to_disk(output_dir=directory)

    # Iterate over all pickle files in the directory
    for file in glob.glob(os.path.join(directory, '*.pkl')):
        # Read the pickle file into a DataFrame
        df = pd.read_pickle(file)
        # print(f"Reading {file}")
        # Append the DataFrame to the list
        dfs.append(df)
        uu += 1
        if 2 == uu:
            if merged_df is None:
                merged_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
            else:
                tmp = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
                merged_df = pd.concat([merged_df, tmp])
            dfs = []
            uu = 0
        cf += 1
    merged_df = merged_df[~merged_df.index.duplicated()]
    assert 0 == np.count_nonzero(merged_df.index.duplicated())
    logger.debug(f"Read {cf} files... > {merged_df.index[0]} to {merged_df.index[-1]} for a total of {len(merged_df)} rows")
    merged_df.to_pickle(output_file)

    logger.debug(f"Output file is located @{output_file}")

    ###########################################################################
    #
    ###########################################################################
    def get_weekday_date():
        today = date.today()
        return today - timedelta(days=max(0, today.weekday() - 4))

    today_date = get_weekday_date().isoformat()
    logger.debug(f"Today is {today_date}")

    ###########################################################################
    #
    ###########################################################################
    df = pd.read_pickle(output_file)
    logger.debug(f"Data range from {df.index[0]} to {df.index[-1]}")
    df = df.loc[today_date]
    # print(df.index)
    # Convert MultiIndex columns to simple columns by concatenating levels
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    resample_params = ('5min', 0, 18)
    resample_params = ('1min', 60, 100)
    # Resample data
    df = df.resample(resample_params[0]).agg({
        'Open_SPY': 'first',
        'High_SPY': 'max',
        'Low_SPY': 'min',
        'Close_SPY': 'last',
        'Volume_SPY': 'sum'})

    ###########################################################################
    #
    ###########################################################################
    i_range_pattern, j_range_pattern = int(configuration.get('csp__ir', resample_params[1])), int(configuration.get('csp__jr', resample_params[2]))
    logger.debug(f"Looking for patterns between {df.index[i_range_pattern]} and {df.index[j_range_pattern]}")

    ###########################################################################
    #
    ###########################################################################
    is_bullish = string_to_bool(configuration.get('csp__bull', True))
    is_bearish = not is_bullish

    ###########################################################################
    #
    ###########################################################################
    # dict of functions by group
    for group, names in talib.get_function_groups().items():
        if group == 'Pattern Recognition':
            # print(group)
            for name in names:
                # print(f"  {name}")
                # or by name
                fct = abstract.Function(name)
                # print(fct)
                patterns = fct(df['Open_SPY'].values, df['High_SPY'].values, df['Low_SPY'].values, df['Close_SPY'].values)
                assert len(patterns) == len(df)
                patterns = patterns[i_range_pattern:j_range_pattern]
                df2 = df.iloc[i_range_pattern:j_range_pattern].copy()
                if is_bullish:
                    if 0 != np.count_nonzero(patterns > 0):
                        logger.debug(f"  {name}")
                        plot2(df2=df2, patterns=patterns, fct_name=name)
                else:
                    if 0 != np.count_nonzero(patterns < 0):
                        logger.debug(f"  {name}")

if __name__ == '__main__':
    freeze_support()

    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]
    namespace = {}
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            globals()[key] = val
    config = {k: globals()[k] for k in config_keys}
    tmp = {k: namespace[k] for k in [k for k, v in namespace.items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), dict, tuple, list))]}
    config.update({k: tmp[k] for k, v in config.items() if k in tmp})
    config.update({k: globals()[k] for k in globals() if k.startswith("csp__") or k in ['device', 'seed_offset', 'stub_dir']})
    configuration = dict_to_namespace(config)
    # -----------------------------------------------------------------------------
    logger.remove()
    logger.add(sys.stdout, level=namespace_to_dict(configuration).get("csp__debug_level", "INFO"))
    run(configuration = namespace_to_dict(configuration))