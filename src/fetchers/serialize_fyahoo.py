try:
    from version import sys__name, sys__version
except:
    import sys
    import os
    import pathlib

    # Get the current working directory
    current_dir = pathlib.Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    # print(parent_dir)
    # Add the current directory to sys.path
    sys.path.insert(0, str(parent_dir))
    from version import sys__name, sys__version
from colorama import init, Fore, Style
import pandas as pd
import copy
import pickle
import yfinance as yf
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from constants import FYAHOO__OUTPUTFILENAME_YEAR, FYAHOO__OUTPUTFILENAME, MY_TICKERS, TOP10_SP500_TICKERS
from constants import FYAHOO_TICKER__OUTPUTFILENAME, FYAHOO__OUTPUTFILENAME_DAY, FYAHOO__OUTPUTFILENAME_MONTH, FYAHOO__OUTPUTFILENAME_WEEK, FYAHOO__OUTPUTFILENAME_QUARTER


def entry():
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date_m1 = (datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    tickers = sorted(list(set(MY_TICKERS)), reverse=True)
    data_cache ={}
    ###########################################################################
    # 1 hour
    ###########################################################################
    print(f"{len(tickers)} tickers found , {start_date=} {end_date=} , interval=1h , ", flush=True)
    for ticker in tqdm(tickers):
        if ticker == '^SKEW':
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False, ignore_tz=True, progress=False)
        else:
            data = yf.download(ticker, start=start_date, end=end_date, interval='1h', auto_adjust=False, ignore_tz=True, progress=False)
        data_cache[ticker] = data
        # time.sleep(0.1)
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Data saved to {FYAHOO__OUTPUTFILENAME}")
    # Deserialize
    with open(FYAHOO__OUTPUTFILENAME, 'rb') as f:
        loaded_data = pickle.load(f)

    ###########################################################################
    # 1 day
    ###########################################################################
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = "1960-01-01"  # (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    print(f"{len(tickers)} tickers found , {start_date=} {end_date=} , interval=1d , ", flush=True)
    for ticker in tqdm(tickers):
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d', auto_adjust=False, ignore_tz=True, progress=False)
        data_cache[ticker] = data
        # time.sleep(0.1)
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Data saved to {FYAHOO__OUTPUTFILENAME_DAY}")

    ###########################################################################
    # 1 week
    ###########################################################################
    # end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    # start_date = "1980-01-01"  # (datetime.today() - timedelta(days=20 * 365)).strftime('%Y-%m-%d')
    # print(f"{len(tickers)} tickers found , {start_date=} {end_date=} , interval=1wk , ", flush=True)
    # for ticker in tqdm(tickers):
    #     data = yf.download(ticker, start=start_date, end=end_date, interval='1wk', auto_adjust=False, ignore_tz=True, progress=False)
    #     data_cache[ticker] = data
    #     # time.sleep(0.1)
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
        daily_data_cache = pickle.load(f)
    data_cache = {}
    for ticker in tqdm(tickers):
        data_cache[ticker] = copy.deepcopy(daily_data_cache[ticker].resample('W-FRI').agg({
            ('Open', ticker): 'first',
            ('High', ticker): 'max',
            ('Low', ticker): 'min',
            ('Close', ticker): 'last',
            ('Volume', ticker): 'sum'
        }))
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME_WEEK, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Weekly data saved to {FYAHOO__OUTPUTFILENAME_WEEK}")

    ###########################################################################
    # 1 month
    ###########################################################################
    # end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    # start_date = "1980-01-01"  # (datetime.today() - timedelta(days=40 * 365)).strftime('%Y-%m-%d')
    # print(f"{len(tickers)} tickers found , {start_date=} {end_date=} , interval=1mo , ", flush=True)
    # for ticker in tqdm(tickers):
    #     data = yf.download(ticker, start=start_date, end=end_date, interval='1mo', auto_adjust=False, ignore_tz=True, progress=False)
    #     data_cache[ticker] = data
    #     # time.sleep(0.1)
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
        daily_data_cache = pickle.load(f)
    data_cache = {}
    for ticker in tqdm(tickers):
        data_cache[ticker] = copy.deepcopy(daily_data_cache[ticker].resample('ME').agg({
            ('Open', ticker): 'first',
            ('High', ticker): 'max',
            ('Low', ticker): 'min',
            ('Close', ticker): 'last',
            ('Volume', ticker): 'sum'
        }))
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME_MONTH, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Monthly data saved to {FYAHOO__OUTPUTFILENAME_MONTH}")

    ###########################################################################
    # 1 quarter
    ###########################################################################
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
        daily_data_cache = pickle.load(f)
    data_cache = {}
    for ticker in tqdm(tickers):
        data_cache[ticker] = copy.deepcopy(daily_data_cache[ticker].resample('QE').agg({
            ('Open', ticker): 'first',
            ('High', ticker): 'max',
            ('Low', ticker): 'min',
            ('Close', ticker): 'last',
            ('Volume', ticker): 'sum'
        }))
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME_QUARTER, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Monthly data saved to {FYAHOO__OUTPUTFILENAME_QUARTER}")

    ###########################################################################
    # 1 year
    ###########################################################################
    with open(FYAHOO__OUTPUTFILENAME_DAY, 'rb') as f:
        daily_data_cache = pickle.load(f)
    data_cache = {}
    for ticker in tqdm(tickers):
        data_cache[ticker] = copy.deepcopy(daily_data_cache[ticker].resample('YE').agg({
            ('Open', ticker): 'first',
            ('High', ticker): 'max',
            ('Low', ticker): 'min',
            ('Close', ticker): 'last',
            ('Volume', ticker): 'sum'
        }))
    # Serialize
    with open(FYAHOO__OUTPUTFILENAME_YEAR, 'wb') as f:
        pickle.dump(data_cache, f)
    print(f"Yearly data saved to {FYAHOO__OUTPUTFILENAME_YEAR}")

    ###########################################################################
    # Economic
    ###########################################################################
    print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Starting Download Tracker...{Style.RESET_ALL}\n", flush=True)
    result = {}
    for ticker in tqdm(TOP10_SP500_TICKERS):
        # Get stock data
        stock = yf.Ticker(ticker)
        result[ticker] = {
            'info': stock.info,
            'earnings_dates': stock.earnings_dates,
            'history': stock.history(period='2y')
        }
    # Serialize
    with open(FYAHOO_TICKER__OUTPUTFILENAME, 'wb') as f:
        pickle.dump(result, f)

    print(f"{Fore.GREEN}✅ Data saved to {FYAHOO_TICKER__OUTPUTFILENAME}{Style.RESET_ALL}")


if __name__ == "__main__":
    entry()